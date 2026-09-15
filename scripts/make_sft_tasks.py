"""Fresh tasks for the SFT arm, disjoint from everything in data/v3_kb_definitions/.

    python scripts/make_sft_tasks.py --plan     # draw the combinations and show the first prompt; no API calls
    python scripts/make_sft_tasks.py            # write data/v3_kb_definitions/tasks_sft_new103.jsonl

Every task comes from the same code path as the v3 data: data/generate_tasks.py draws the
components (draw_components), writes the sentence with the v3 writer prompt (build_prompt: KB
definitions and confusion-pair boundaries, no example sentences), runs the three v3 review
checks (validate: A parrot in code, B label-body and C cause leakage via the judge), and builds
the record, account, route_answer and rubric (make_record). Only two things differ from a v3 run:
  - the writer is claude-sonnet-5 alone (the r3 teacher's model string), not three providers
  - the tasks are drawn per route bucket instead of per cell, so the route mix is set here

Route buckets (label from route_answer, current rule order):
    ask_question 13; file_claim, explain_not_covered, escalate, refer_to_manufacturer,
    explain_waiting_period, tech_support 15 each.
A bucket is filled by backward sampling, as in generate_tasks.build_topup_specs: draw the usual
components and keep the draw only if its route lands in the bucket, so every free component keeps
its own distribution. ask_question draws its cell (what the customer hides, and the peril) with
the weights of the three hidden-fact columns in data/target_distribution.py; every other route
has both facts stated. refer_to_manufacturer is split so that 6 of its 15 tasks have
claims_last_12m at the device limit (tasks_train.jsonl has 11 of 30), because those are the
tasks where the warranty rule has to win over the claim limit.

A fourth check is added after the three v3 checks: D, the sentence matches a customer sentence
already in data/v3_kb_definitions/ (or one accepted earlier in this run), exactly or after
lowercasing and removing whitespace. A rejected sentence is rewritten for the same combination,
up to five attempts, as in v3. A combination still rejected after five attempts is dropped and a
fresh combination is drawn for the same bucket, until every bucket is full.

Task ids are s001-s103, assigned at the end in route order. While running, combinations carry
candidate ids (c0001, ...), which is what the rejection log shows; the s-to-c mapping is written
next to it. Scratch output (rejections.jsonl, candidate_ids.json) goes to out/sft_new103_gen/.
"""

import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "scripts"))

import components as C                       # noqa: E402
import generate_tasks as G                   # noqa: E402
import target_distribution as T              # noqa: E402
from reward.judge import JUDGE_URL           # noqa: E402

V3_DIR = ROOT / "data" / "v3_kb_definitions"
OUT_FILE = V3_DIR / "tasks_sft_new103.jsonl"
SCRATCH = ROOT / "out" / "sft_new103_gen"

WRITER_MODEL = "claude-sonnet-5"             # same model string as the r3 teacher
PRICE = (2.00, 10.00)                        # $ per million input / output tokens, as in make_teacher_sft.PRICE
SEED = 20260914                              # v3 used generate_tasks.SEED = 7

# (route, claim-limit condition) -> number of tasks. None means the limit is not constrained.
BUCKETS = {
    ("ask_question", None): 13,
    ("file_claim", None): 15,
    ("explain_not_covered", None): 15,
    ("escalate", None): 15,
    ("refer_to_manufacturer", "at_limit"): 6,
    ("refer_to_manufacturer", "under_limit"): 9,
    ("explain_waiting_period", None): 15,
    ("tech_support", None): 15,
}
ROUTE_ORDER = ["ask_question", "file_claim", "explain_not_covered", "escalate",
               "refer_to_manufacturer", "explain_waiting_period", "tech_support"]

USAGE = {"calls": 0, "input": 0, "output": 0}


def normalize(sentence):
    """The form used by check D: lowercase, all whitespace removed."""
    return re.sub(r"\s+", "", sentence.lower())


def existing_sentences():
    """Every customer sentence in data/v3_kb_definitions/, normalized. The output file itself is
    left out so a re-run does not compare the new tasks with themselves."""
    seen = set()
    for path in sorted(glob.glob(str(V3_DIR / "*.jsonl"))):
        if Path(path) == OUT_FILE:
            continue
        for line in open(path):
            if line.strip():
                seen.add(normalize(json.loads(line)["customer"]))
    return seen


def ask_cells():
    """The hidden-fact cells with their v3 target counts as weights: ((stated, peril), count)."""
    cells, counts = G.build_cells()
    return [(cell, count) for cell, count in zip(cells, counts)
            if count > 0 and cell[0] != ["device", "peril"]]


def at_limit(spec):
    return spec["claims_last_12m"] >= C.CLAIM_LIMITS[spec["device"]]


def draw_for(bucket, rng, cells):
    """One combination whose route lands in the bucket. Draws that miss are discarded."""
    route, limit = bucket
    while True:
        if route == "ask_question":
            stated, peril = G.weighted_choice(rng, cells)
        else:
            stated, peril = ["device", "peril"], rng.choice(C.PERILS)
        spec = G.draw_components(rng, stated, peril)
        if G.spec_route(spec) != route:
            continue
        if limit is not None and at_limit(spec) != (limit == "at_limit"):
            continue
        spec["generator"] = WRITER_MODEL
        spec["bucket"] = bucket
        return spec


def call_writer(prompt):
    """generate_tasks.call_anthropic with the Sonnet model string, plus token counting."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def go():
        reply = client.messages.create(
            model=WRITER_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        USAGE["calls"] += 1
        USAGE["input"] += reply.usage.input_tokens
        USAGE["output"] += reply.usage.output_tokens
        return "".join(block.text for block in reply.content if block.type == "text")

    return G.with_retry("anthropic", go)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true",
                        help="draw the first round of combinations, print them and the first writer "
                             "prompt, and exit; no API or judge calls")
    args = parser.parse_args()

    rng = random.Random(SEED)
    cells = ask_cells()
    seen = existing_sentences()
    print(f"{len(seen)} distinct existing sentences in {V3_DIR.relative_to(ROOT)}", flush=True)
    print(f"writer {WRITER_MODEL} | judge URL {JUDGE_URL}", flush=True)

    next_candidate = 1

    def draw(bucket, count):
        nonlocal next_candidate
        specs = []
        for _ in range(count):
            spec = draw_for(bucket, rng, cells)
            spec["task_id"] = f"c{next_candidate:04d}"
            next_candidate += 1
            specs.append(spec)
        return specs

    if args.plan:
        specs = [s for bucket, count in BUCKETS.items() for s in draw(bucket, count)]
        print("\n| bucket | tasks | devices | perils | enrolled_days_ago | claims at limit |")
        print("|---|---|---|---|---|---|")
        for bucket in BUCKETS:
            mine = [s for s in specs if s["bucket"] == bucket]
            devices = collections.Counter(s["device"] + ("" if s["stated_device"] else " (hidden)") for s in mine)
            perils = collections.Counter(s["peril"] + ("" if s["stated_peril"] else " (hidden)") for s in mine)
            enrolled = collections.Counter(s["enrolled_days_ago"] for s in mine)
            limit = sum(1 for s in mine if at_limit(s))
            print(f"| {bucket[0]}{'/' + bucket[1] if bucket[1] else ''} | {len(mine)} | {dict(devices)} | "
                  f"{dict(perils)} | {dict(sorted(enrolled.items()))} | {limit} |")
        print("\nfirst writer prompt:\n")
        print(G.build_prompt(list(enumerate(specs[:G.BATCH_SIZE], start=1))))
        return

    assert not OUT_FILE.exists(), f"{OUT_FILE} already exists; move it away to regenerate"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    G.CALLERS[WRITER_MODEL] = call_writer
    start = time.time()

    accepted = []                              # (spec, sentence), in acceptance order
    filled = collections.Counter()
    dropped = []
    check_counts = collections.Counter()
    round_number = 0
    while any(filled[b] < n for b, n in BUCKETS.items()):
        round_number += 1
        pending = [s for b, n in BUCKETS.items() for s in draw(b, n - filled[b])]
        print(f"\n=== fill round {round_number}: {len(pending)} new combinations ===", flush=True)
        attempts = collections.Counter()
        for attempt in range(1, G.MAX_ATTEMPTS + 1):
            if not pending:
                break
            print(f"attempt {attempt}/{G.MAX_ATTEMPTS}: {len(pending)} to write", flush=True)
            rejected = []
            for first in range(0, len(pending), G.BATCH_SIZE):
                numbered = list(enumerate(pending[first:first + G.BATCH_SIZE], start=1))
                written = G.write_batch(WRITER_MODEL, numbered)
                for number, spec in numbered:
                    sentence = written[number]
                    attempts[spec["task_id"]] += 1
                    problem = G.validate(spec, sentence, use_judge=True)
                    if problem is None and normalize(sentence) in seen:
                        problem = ("D_duplicate", "matches an existing or already accepted sentence")
                    if problem is None:
                        seen.add(normalize(sentence))
                        accepted.append((spec, sentence))
                        filled[spec["bucket"]] += 1
                    else:
                        check_counts[problem[0]] += 1
                        G.log_rejection(SCRATCH, spec, sentence, attempts[spec["task_id"]], *problem)
                        rejected.append(spec)
            pending = rejected
        for spec in pending:
            print(f"  DROPPED {spec['task_id']} {spec['bucket']} peril={spec['peril']} "
                  f"device={spec['device']} after {G.MAX_ATTEMPTS} attempts", flush=True)
        dropped += pending

    # s-ids in route order, acceptance order within a route
    accepted.sort(key=lambda pair: ROUTE_ORDER.index(pair[0]["bucket"][0]))
    records, mapping = [], {}
    for index, (spec, sentence) in enumerate(accepted, start=1):
        mapping[f"s{index:03d}"] = spec["task_id"]
        spec["task_id"] = f"s{index:03d}"
        records.append(G.make_record(spec, sentence))

    # checks on the finished set
    routes = collections.Counter(r["route_answer"] for r in records)
    for (route, _), count in BUCKETS.items():
        assert routes[route] == sum(n for (r, _), n in BUCKETS.items() if r == route), routes
    assert len({normalize(r["customer"]) for r in records}) == len(records)
    for r in records:
        if r["account"] and r["account"]["enrolled_days_ago"] < 31:
            assert r["account"]["claims_last_12m"] == 0, r["task_id"]
        assert (r["account"] == {}) == (r["device"] is None), r["task_id"]

    with OUT_FILE.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (SCRATCH / "candidate_ids.json").write_text(json.dumps(mapping, indent=1))

    refer_limit = sum(1 for r in records if r["route_answer"] == "refer_to_manufacturer"
                      and r["account"]["claims_last_12m"] >= C.CLAIM_LIMITS[r["device"]])
    cost = (USAGE["input"] * PRICE[0] + USAGE["output"] * PRICE[1]) / 1e6
    print(f"\nwrote {OUT_FILE.relative_to(ROOT)}: {len(records)} tasks")
    print("\n| route | tasks |")
    print("|---|---|")
    for route in ROUTE_ORDER:
        print(f"| {route} | {routes[route]} |")
    print(f"\nrefer_to_manufacturer at the claim limit: {refer_limit}")
    print(f"rejections by check: {dict(check_counts)}; combinations dropped after "
          f"{G.MAX_ATTEMPTS} attempts: {len(dropped)}")
    print(f"writer {WRITER_MODEL}: {USAGE['calls']} calls, {USAGE['input']} input + {USAGE['output']} "
          f"output tokens, ${cost:.2f}")
    print(f"judge: {G.TIMING['judge_calls']} calls, {G.TIMING['judge_s']/60:.1f} min; "
          f"wall {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()
