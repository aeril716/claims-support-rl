"""Bucket the tasks of a pass@k jsonl (from eval/pass_at_k.py) by how many of the k samples hit
the gold route: 0, 1-3, 4-12, 13-15, all k. The middle bucket is where a GRPO group has both
correct and incorrect completions, so those tasks are listed with their gold route.

    python eval/passk_buckets.py out/passk_base14b_v5_test3.jsonl [more.jsonl ...]
"""

import collections
import json
import sys
from pathlib import Path


def buckets_for(k):
    """(label, lo, hi) inclusive ranges, scaled from the 16-sample cut points."""
    return [("0", 0, 0), ("1-3", 1, 3), ("4-12", 4, 12), ("13-15", 13, k - 1), (f"{k}/{k}", k, k)]


def report(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    k = len(rows[0]["samples"])
    hits = {r["task_id"]: sum(1 for s in r["samples"] if s == r["gold"]) for r in rows}
    gold = {r["task_id"]: r["gold"] for r in rows}
    print(f"{path}: {len(rows)} tasks, k={k}"
          + (f", adapter {rows[0]['adapter']}" if rows[0].get("adapter") else "")
          + f", prompt {rows[0].get('prompt_version')}")
    print(f"| gold hits of {k} | tasks | by gold route |")
    print("|---|---|---|")
    for label, lo, hi in buckets_for(k):
        ids = [t for t, h in hits.items() if lo <= h <= hi]
        by_route = collections.Counter(gold[t] for t in ids)
        print(f"| {label} | {len(ids)} | " + ", ".join(f"{r} {n}" for r, n in by_route.most_common()) + " |")
    middle = [t for t, h in hits.items() if 4 <= h <= 12]
    if middle:
        print("\n4-12 (mixed groups):")
        for t in sorted(middle, key=lambda t: (gold[t], t)):
            print(f"  {t}  {gold[t]:<24} {hits[t]}/{k}")
    else:
        print("\n4-12: none")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for i, path in enumerate(sys.argv[1:]):
        if i:
            print()
        report(path)


if __name__ == "__main__":
    main()
