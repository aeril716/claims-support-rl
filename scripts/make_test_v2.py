"""Held-out test set test_v2: 40 fresh tasks, hard by construction.

    python scripts/make_test_v2.py --plan     # draw the combinations and show the first prompt; no API calls
    python scripts/make_test_v2.py            # write data/v3_kb_definitions/tasks_test_v2.jsonl

A thin wrapper over scripts/make_sft_tasks.py: the same Sonnet writer, the same three v3 review
checks plus check D (duplicate of any sentence already in data/v3_kb_definitions/, which now
includes tasks_sft_new103.jsonl), the same rewrite-then-drop loop, the same record builder.
make_sft_tasks.py is not changed; this file swaps four of its module-level settings (output
file, scratch folder, seed, buckets) and one function (draw_for) before calling its main().

"Hard by construction" means the buckets fix the combination, never a model's score:
    ask_question            6 with the device stated and the cause hidden, 2 with the device hidden
    refer_to_manufacturer   4 with claims_last_12m at the device limit, 2 under it
    escalate                3 with the warranty still active (enrolled < 365) and an accidental
                            peril (drop, liquid, surge), so the claim limit decides, not the warranty
    explain_waiting_period  3 with battery or surge as the peril
    explain_not_covered     3 with claims_last_12m at the device limit
    file_claim 5, tech_support 5, and the remainder of each route above, unconstrained.
A bucket is filled by backward sampling as before: draw the usual components, keep the draw only
if its route and its hard-case condition both hold.

Task ids are h001-h040, assigned in route order. Scratch output goes to out/test_v2_gen/.
"""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_sft_tasks as M                   # noqa: E402  (imports components, generate_tasks, target_distribution)

C, G = M.C, M.G

ACCIDENT_PERILS = {"drop", "liquid", "surge"}

# (route, hard-case tag) -> number of tasks. None means no extra condition.
BUCKETS = {
    ("ask_question", "cause_hidden"): 6,
    ("ask_question", "device_hidden"): 2,
    ("file_claim", None): 5,
    ("explain_not_covered", "at_limit"): 3,
    ("explain_not_covered", None): 2,
    ("escalate", "warranty_active_accident"): 3,
    ("escalate", None): 3,
    ("refer_to_manufacturer", "at_limit"): 4,
    ("refer_to_manufacturer", "under_limit"): 2,
    ("explain_waiting_period", "battery_or_surge"): 3,
    ("explain_waiting_period", None): 2,
    ("tech_support", None): 5,
}
assert sum(BUCKETS.values()) == 40

HARD = {
    None: lambda spec: True,
    "cause_hidden": lambda spec: spec["stated_device"] and not spec["stated_peril"],
    "device_hidden": lambda spec: not spec["stated_device"],
    "at_limit": M.at_limit,
    "under_limit": lambda spec: not M.at_limit(spec),
    "warranty_active_accident": lambda spec: (spec["enrolled_days_ago"] < G.WARRANTY_DAYS
                                              and spec["peril"] in ACCIDENT_PERILS),
    "battery_or_surge": lambda spec: spec["peril"] in {"battery", "surge"},
}


def draw_for(bucket, rng, cells):
    """One combination whose route lands in the bucket and meets its hard-case condition."""
    route, tag = bucket
    while True:
        if route == "ask_question":
            if tag == "cause_hidden":
                pool = [(cell, n) for cell, n in cells if cell[0] == ["device"]]
            else:
                pool = [(cell, n) for cell, n in cells if "device" not in cell[0]]
            stated, peril = G.weighted_choice(rng, pool)
        else:
            stated, peril = ["device", "peril"], rng.choice(C.PERILS)
        spec = G.draw_components(rng, stated, peril)
        if G.spec_route(spec) != route or not HARD[tag](spec):
            continue
        spec["generator"] = M.WRITER_MODEL
        spec["bucket"] = bucket
        return spec


_make_record = G.make_record


def make_record_h(spec, sentence):
    """make_sft_tasks.main assigns s001..; this set uses h001.."""
    spec["task_id"] = "h" + spec["task_id"][1:]
    return _make_record(spec, sentence)


def hard_case_counts(records):
    """How many tasks of each route meet its hard-case condition, read off the finished records."""
    by_route = collections.defaultdict(list)
    for r in records:
        by_route[r["route_answer"]].append(r)
    limit = lambda r: r["account"]["claims_last_12m"] >= C.CLAIM_LIMITS[r["device"]]   # noqa: E731
    return {
        "ask_question: device stated, cause hidden":
            sum(1 for r in by_route["ask_question"] if r["device"] and not r["peril"]),
        "ask_question: device hidden":
            sum(1 for r in by_route["ask_question"] if not r["device"]),
        "refer_to_manufacturer: claims at the device limit":
            sum(1 for r in by_route["refer_to_manufacturer"] if limit(r)),
        "escalate: warranty active (enrolled < 365) and accidental peril (drop, liquid, surge)":
            sum(1 for r in by_route["escalate"]
                if r["account"]["enrolled_days_ago"] < G.WARRANTY_DAYS and r["peril"] in ACCIDENT_PERILS),
        "explain_waiting_period: battery or surge":
            sum(1 for r in by_route["explain_waiting_period"] if r["peril"] in {"battery", "surge"}),
        "explain_not_covered: claims at the device limit":
            sum(1 for r in by_route["explain_not_covered"] if limit(r)),
    }


def main():
    M.OUT_FILE = ROOT / "data" / "v3_kb_definitions" / "tasks_test_v2.jsonl"
    M.SCRATCH = ROOT / "out" / "test_v2_gen"
    M.SEED = 20260915
    M.BUCKETS = BUCKETS
    M.draw_for = draw_for
    G.make_record = make_record_h
    M.main()
    if "--plan" in sys.argv:
        return

    records = [json.loads(line) for line in M.OUT_FILE.open() if line.strip()]
    assert [r["task_id"] for r in records] == [f"h{i:03d}" for i in range(1, 41)]
    mapping_file = M.SCRATCH / "candidate_ids.json"
    mapping = json.loads(mapping_file.read_text())
    mapping_file.write_text(json.dumps({"h" + k[1:]: v for k, v in mapping.items()}, indent=1))

    print("\n| hard case | tasks |")
    print("|---|---|")
    for name, count in hard_case_counts(records).items():
        print(f"| {name} | {count} |")


if __name__ == "__main__":
    main()
