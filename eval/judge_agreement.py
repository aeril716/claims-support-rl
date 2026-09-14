"""Score the same completions with the active judge backend and compare two such runs.

    JUDGE_BACKEND=anthropic python eval/judge_agreement.py score --sft data/sft/train_v8reason_r3.jsonl \
        --tasks data/v3_kb_definitions/tasks_train.jsonl --sample 40 --seed 0 --out out/verdicts_r3_haiku.jsonl
    JUDGE_BACKEND=ollama python eval/judge_agreement.py score --outputs out/eval_base14b_v8_test3/outputs.json \
        --tasks data/v3_kb_definitions/tasks_test.jsonl --out out/verdicts_base_ollama.jsonl
    python eval/judge_agreement.py compare out/verdicts_r3_haiku.jsonl out/verdicts_r3_ollama.jsonl --label-a haiku --label-b ollama

`score` takes either an SFT jsonl (task_id + completion; --sample N with --seed picks the same
N items as the earlier rubric samples, random.Random(seed).sample) or an eval outputs.json
(task_id + raw_output, all items), runs reward.score on each with whatever JUDGE_BACKEND is
set, and writes one line per item: task_id, gold, rubric_score, and every rubric item's id,
answer, passed. `compare` joins two such files on (task_id, item id) and prints, per rubric
item, each judge's pass rate and the agreement (both yes or both no), flagging items under 70%.
"""

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FLAG_BELOW = 0.70


def load_items(args):
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(args.tasks) if l.strip()}
    if args.sft:
        rows = [json.loads(l) for l in open(args.sft) if l.strip()]
        if args.sample:
            rows = random.Random(args.seed).sample(rows, args.sample)
        return [(tasks[r["task_id"]], r["completion"], r["task_id"]) for r in rows]
    rows = json.loads(Path(args.outputs).read_text())
    return [(tasks[r["task_id"]], r["raw_output"], r["task_id"]) for r in rows]


def score(args):
    from reward import judge, rubric_wording
    from reward.reward import score as score_one
    rubric_wording.set_version(args.rubric)
    items = load_items(args)
    only = set(args.only_ids.split(",")) if args.only_ids else None
    print(f"judge {judge.judge_info()} | rubric wording {rubric_wording.active_version()} | {len(items)} completions"
          + (f" | only {sorted(only)}" if only else ""), flush=True)
    t0 = time.time()
    with open(args.out, "w") as f:
        for i, (task, text, tid) in enumerate(items, start=1):
            if only:
                task = {**task, "rubric": [it for it in task["rubric"] if it["id"] in only]}
                if not task["rubric"]:
                    continue
            reward, detail = score_one(task, text)
            if only:
                detail["items"] = [it for it in detail.get("items", []) if it["id"] in only]
                detail["rubric_score"] = None
            rec = {"task_id": tid, "gold": task["route_answer"], "parsed": detail["parsed"],
                   "route": detail.get("route"), "route_score": detail["route_score"],
                   "rubric_score": detail.get("rubric_score"),
                   "items": [{"id": it["id"], "answer": it["answer"], "passed": it["passed"]} for it in detail.get("items", [])],
                   "judge": judge.judge_info(), "rubric_wording": rubric_wording.active_version()}
            f.write(json.dumps(rec) + "\n")
            if i % 10 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)
    print(f"wrote {args.out} in {time.time() - t0:.0f} s, judge_failures {judge.judge_failures()}")


def compare(args):
    a = {r["task_id"]: r for r in map(json.loads, open(args.file_a))}
    b = {r["task_id"]: r for r in map(json.loads, open(args.file_b))}
    common = [t for t in a if t in b]
    per = collections.defaultdict(lambda: {"n": 0, "a_pass": 0, "b_pass": 0, "agree": 0})
    tot = {"n": 0, "a_pass": 0, "b_pass": 0, "agree": 0}
    ra, rb = [], []
    for t in common:
        ia = {it["id"]: it for it in a[t]["items"]}
        ib = {it["id"]: it for it in b[t]["items"]}
        for k in ia:
            if k not in ib:
                continue
            for d in (per[k], tot):
                d["n"] += 1
                d["a_pass"] += ia[k]["passed"]
                d["b_pass"] += ib[k]["passed"]
                d["agree"] += ia[k]["answer"] == ib[k]["answer"]
        if a[t]["rubric_score"] is not None and b[t]["rubric_score"] is not None:
            ra.append(a[t]["rubric_score"]); rb.append(b[t]["rubric_score"])
    means = f"rubric mean {args.label_a} {sum(ra)/len(ra):.3f} vs {args.label_b} {sum(rb)/len(rb):.3f} | " if ra else ""
    print(f"{len(common)} completions, {tot['n']} item verdicts | {means}overall agreement {tot['agree']/tot['n']:.3f}\n")
    print(f"| rubric item | n | {args.label_a} pass | {args.label_b} pass | agreement |")
    print("|---|---|---|---|---|")
    for k in sorted(per):
        d = per[k]
        flag = "  **< 70%**" if d["agree"] / d["n"] < FLAG_BELOW else ""
        print(f"| {k} | {d['n']} | {d['a_pass']}/{d['n']} | {d['b_pass']}/{d['n']} | {d['agree']/d['n']:.2f}{flag} |")
    print(f"| **all** | {tot['n']} | {tot['a_pass']}/{tot['n']} | {tot['b_pass']}/{tot['n']} | {tot['agree']/tot['n']:.2f} |")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--sft", default=None); s.add_argument("--outputs", default=None)
    s.add_argument("--tasks", required=True); s.add_argument("--sample", type=int, default=None)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", required=True)
    s.add_argument("--rubric", default="v2", help="rubric wording version (default v2)")
    s.add_argument("--only-ids", default=None,
                   help="comma-separated rubric item ids: score only these (others skipped; rubric_score is then None)")
    c = sub.add_parser("compare")
    c.add_argument("file_a"); c.add_argument("file_b")
    c.add_argument("--label-a", default="a"); c.add_argument("--label-b", default="b")
    args = parser.parse_args()
    if args.cmd == "score":
        assert bool(args.sft) != bool(args.outputs), "give exactly one of --sft or --outputs"
        score(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
