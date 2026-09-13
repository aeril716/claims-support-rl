"""Build the small rebalanced training subset used by run4 and run5.

Takes the full training split, groups tasks by route_answer, and draws a fixed number of
ask_question tasks plus an even per-route split of the remaining non-ask tasks, using one
seeded random generator so the same seed always gives the same file.

    python scripts/make_train_sub35.py \
        --src data/v3_kb_definitions/tasks_train.jsonl \
        --out data/v3_kb_definitions/tasks_train_sub35.jsonl \
        --seed 42 --n-ask 9 --n-total 35

With the defaults this reproduces data/v3_kb_definitions/tasks_train_sub35.jsonl byte for byte.
The draw order matters for reproducibility and is kept exactly as first run: the extra routes
for the remainder, then the ask tasks, then each non-ask route in sorted order, then a shuffle.
"""
import argparse
import collections
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/v3_kb_definitions/tasks_train.jsonl",
                        help="task file to sample from")
    parser.add_argument("--out", default="data/v3_kb_definitions/tasks_train_sub35.jsonl",
                        help="task file to write")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ask", type=int, default=9, help="number of ask_question tasks")
    parser.add_argument("--n-total", type=int, default=35, help="total number of tasks")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in open(args.src) if line.strip()]
    by = collections.defaultdict(list)
    for t in tasks:
        by[t["route_answer"]].append(t)

    rng = random.Random(args.seed)
    non_ask = sorted(r for r in by if r != "ask_question")
    n_non_ask = args.n_total - args.n_ask
    counts = {r: n_non_ask // len(non_ask) for r in non_ask}
    for r in rng.sample(non_ask, n_non_ask - sum(counts.values())):  # the remainder, seeded
        counts[r] += 1
    picked = rng.sample(by["ask_question"], args.n_ask)
    for r in non_ask:
        picked += rng.sample(by[r], counts[r])
    rng.shuffle(picked)

    with open(args.out, "w") as f:
        for t in picked:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    c = collections.Counter(t["route_answer"] for t in picked)
    print(f"wrote {args.out}: {len(picked)} tasks")
    for r, n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {r:<24}{n:>3}   (available in {args.src}: {len(by[r])})")
    print("task ids:", " ".join(t["task_id"] for t in picked))


if __name__ == "__main__":
    main()
