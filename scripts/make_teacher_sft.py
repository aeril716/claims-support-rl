"""Teacher samples for a distillation SFT baseline, with rejection sampling on the route.

    python scripts/make_teacher_sft.py --model claude-sonnet-5 --n 3 --temperature 0.7 \
        --out-samples out/teacher_v8reason_train.jsonl --out-sft data/sft/train_v8reason.jsonl

The teacher receives exactly what the student will see: the v8 prompt with the reasoning
field on, as system and user messages built by train/train_grpo.py's load_tasks. It never
sees the gold route or any other label. Each task is sampled n times; a sample is kept when
it parses and its route equals the gold route. The kept set holds at most --max-kept samples
per task, identical completions removed, with "prompt" being the chat-template rendering the
student trains on (train_grpo.render_prompt) and "completion" the teacher's JSON object.

--teacher-suffix appends text to the teacher's system prompt only (used to ask for short
replies); the student prompt written to --out-sft is untouched. --fact-check also requires the
three fact fields of a v8 answer to match the account (fact_truth), and the report counts the
samples that passed the route check but failed it.

The constant system prompt is sent with cache_control so the ~3,900 shared tokens are read
from cache after the first call. Spend is tracked from the usage fields and the run stops
before a call that would exceed --cap-usd.
"""

import argparse
import collections
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "data"))

from reward.reward import parse_record   # noqa: E402
import components as C                     # noqa: E402
import train_grpo                          # noqa: E402

WAITING_DAYS = 31        # as in data/generate_tasks.py route_answer (enrolled < 31)
WARRANTY_DAYS = 365      # generate_tasks.WARRANTY_DAYS


def fact_truth(task):
    """The fact fields a v8 answer should carry, from the account and the label rules: the
    waiting period is enrolled < 31, the limit is claims >= CLAIM_LIMITS[device]. An empty
    account (device unknown) has no number and "unknown" for both flags."""
    a = task["account"]
    if not a:
        return {"enrolled_days_ago": None, "inside_waiting_period": "unknown", "claim_limit_reached": "unknown"}
    return {"enrolled_days_ago": a["enrolled_days_ago"],
            "inside_waiting_period": "yes" if a["enrolled_days_ago"] < WAITING_DAYS else "no",
            "claim_limit_reached": "yes" if a["claims_last_12m"] >= C.CLAIM_LIMITS[task["device"]] else "no"}


def fact_mismatches(task, facts):
    """Names of the fact fields whose written value differs from fact_truth(task)."""
    truth = fact_truth(task)
    bad = []
    for name, want in truth.items():
        got = facts.get(name)
        if isinstance(got, str):
            got = got.strip().lower().strip('"')
        if name == "enrolled_days_ago":
            ok = got == want or (isinstance(got, (int, float)) and want is not None and int(got) == want) \
                 or (isinstance(got, str) and got.isdigit() and want is not None and int(got) == want)
        else:
            ok = got == want
        if not ok:
            bad.append(name)
    return bad

# $ per million tokens, Claude Sonnet 5 (first-party API): input, output, cache write, cache read
PRICE = {"claude-sonnet-5": (2.00, 10.00, 2.50, 0.20)}
STUDENT_TOKENIZER = "Qwen/Qwen2.5-14B-Instruct"   # same chat template as the 7B


class Spend:
    def __init__(self, model, cap):
        self.p, self.cap, self.lock = PRICE[model], cap, threading.Lock()
        self.tokens = collections.Counter()
        self.usd = 0.0

    def add(self, usage):
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        with self.lock:
            self.tokens.update({"input": usage.input_tokens, "output": usage.output_tokens,
                                "cache_write": cw, "cache_read": cr})
            self.usd += (usage.input_tokens * self.p[0] + usage.output_tokens * self.p[1]
                         + cw * self.p[2] + cr * self.p[3]) / 1e6

    def over(self, margin_usd=0.05):
        with self.lock:
            return self.usd + margin_usd >= self.cap


def object_span(text):
    """The teacher's JSON object as the student should emit it: the span from the first '{'
    to the last '}', which drops code fences or prose around it. Returns (span, stripped)."""
    text = text.strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return text, False
    span = text[a:b + 1]
    return span, span != text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sent via extra_body; dropped with a note if the model rejects sampling params")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--tasks", default="data/v3_kb_definitions/tasks_train.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="first N tasks only (dry run)")
    parser.add_argument("--task-ids", default=None,
                        help="comma-separated task ids to run instead of the whole file (stratified dry runs)")
    parser.add_argument("--out-samples", type=Path, default=Path("out/teacher_v8reason_train.jsonl"))
    parser.add_argument("--out-sft", type=Path, default=Path("data/sft/train_v8reason.jsonl"))
    parser.add_argument("--max-kept", type=int, default=2)
    parser.add_argument("--cap-usd", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-version", default="v8")
    parser.add_argument("--teacher-suffix", default=None, action="append",
                        help="line appended to the TEACHER's system prompt only (repeatable; one line each); "
                             "the student prompt written to --out-sft stays exactly what train_grpo.py renders")
    parser.add_argument("--fact-check", action="store_true",
                        help="keep a sample only if its fact fields also match the account (fact_truth)")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    train_grpo.PROMPT_VERSION = args.prompt_version
    train_grpo.REASONING = True
    rows = train_grpo.load_tasks(args.tasks)
    if args.task_ids:
        wanted = [t.strip() for t in args.task_ids.split(",") if t.strip()]
        by_id = {r["task_id"]: r for r in rows}
        missing = [t for t in wanted if t not in by_id]
        assert not missing, f"task ids not in {args.tasks}: {missing}"
        rows = [by_id[t] for t in wanted]
    if args.limit:
        rows = rows[:args.limit]
    tasks = {t["task_id"]: t for t in (json.loads(l) for l in open(args.tasks) if l.strip())}
    system_text = rows[0]["prompt"][0]["content"]
    assert all(r["prompt"][0]["content"] == system_text for r in rows), "system prompt must be constant"
    assert '"reasoning": <one or two sentences' in system_text, "reasoning field is not in the prompt"
    suffix_lines = [line.strip() for line in (args.teacher_suffix or []) if line and line.strip()]
    teacher_suffix = "\n".join(suffix_lines)
    teacher_system = system_text + ("\n\n" + teacher_suffix if teacher_suffix else "")
    spend = Spend(args.model, args.cap_usd)
    state = {"temperature": args.temperature, "temperature_note": None}
    stop = threading.Event()

    def call(row, i):
        if stop.is_set() or spend.over():
            stop.set()
            return None
        kwargs = dict(model=args.model, max_tokens=args.max_tokens,
                      thinking={"type": "disabled"},
                      system=[{"type": "text", "text": teacher_system, "cache_control": {"type": "ephemeral"}}],
                      messages=[{"role": "user", "content": row["prompt"][1]["content"]}])
        if state["temperature"] is not None:
            kwargs["extra_body"] = {"temperature": state["temperature"]}
        resp = client.messages.create(**kwargs)
        spend.add(resp.usage)
        text = "".join(b.text for b in resp.content if b.type == "text")
        rec = parse_record(text)
        route_ok = rec["route"] == row["route_answer"]
        bad_facts = fact_mismatches(tasks[row["task_id"]], rec["facts"]) if rec["route"] is not None else []
        return {"task_id": row["task_id"], "sample": i, "gold": row["route_answer"], "route": rec["route"],
                "route_ok": route_ok, "fact_mismatches": bad_facts,
                "kept": route_ok and (not args.fact_check or not bad_facts),
                "facts": rec["facts"], "reasoning": rec["reasoning"],
                "repairs": rec["repairs"], "stop_reason": resp.stop_reason,
                "output_tokens": resp.usage.output_tokens, "raw_text": text}

    print(f"teacher {args.model} | temperature {args.temperature} | n={args.n} max_tokens={args.max_tokens} "
          f"| {len(rows)} tasks | cap ${args.cap_usd} | fact_check={args.fact_check} | "
          f"teacher suffix: {teacher_suffix or '(none)'}", flush=True)
    # One probe before the pool settles whether the model accepts a temperature at all
    # (Claude Sonnet 5 and later reject sampling parameters), so the workers never race on it.
    if state["temperature"] is not None:
        try:
            probe = client.messages.create(model=args.model, max_tokens=8, thinking={"type": "disabled"},
                                           extra_body={"temperature": state["temperature"]},
                                           messages=[{"role": "user", "content": "Reply with the word ok."}])
            spend.add(probe.usage)
        except anthropic.BadRequestError as exc:
            if "temperature" not in str(exc):
                raise
            state["temperature_note"] = f"model rejected temperature ({exc.message}); sampled at the API default"
            state["temperature"] = None
            print("  " + state["temperature_note"], flush=True)
    jobs = [(row, i) for row in rows for i in range(args.n)]
    samples, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, result in enumerate(pool.map(lambda j: call(*j), jobs), start=1):
            if result is not None:
                samples.append(result)
            if k % 60 == 0:
                print(f"  {k}/{len(jobs)} calls, ${spend.usd:.2f}", flush=True)
    elapsed = time.time() - t0
    if stop.is_set():
        print(f"STOPPED at the spend cap: ${spend.usd:.2f} after {len(samples)} samples", flush=True)
    samples.sort(key=lambda s: (rows.index(next(r for r in rows if r["task_id"] == s["task_id"])), s["sample"]))

    args.out_samples.parent.mkdir(parents=True, exist_ok=True)
    with args.out_samples.open("w") as f:
        for s in samples:
            f.write(json.dumps({**s, "model": args.model, "temperature": state["temperature"],
                                "teacher_suffix": teacher_suffix, "fact_check": args.fact_check},
                               ensure_ascii=False) + "\n")

    # kept set: route == gold, at most --max-kept per task, identical completions removed
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER)
    by_task = collections.defaultdict(list)
    for s in samples:
        if s["kept"]:
            by_task[s["task_id"]].append(s)
    kept, stripped = [], 0
    args.out_sft.parent.mkdir(parents=True, exist_ok=True)
    with args.out_sft.open("w") as f:
        for row in rows:
            seen = set()
            for s in by_task.get(row["task_id"], []):
                span, was_stripped = object_span(s["raw_text"])
                if span in seen or len(seen) >= args.max_kept:
                    continue
                seen.add(span)
                stripped += was_stripped
                rec = {"task_id": row["task_id"], "prompt": train_grpo.render_prompt(row["prompt"], tok), "completion": span}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept.append({**s, "completion": span})

    # report
    routes = ["ask_question", "file_claim", "explain_not_covered", "escalate", "refer_to_manufacturer",
              "explain_waiting_period", "tech_support"]
    first = [s for s in samples if s["sample"] == 0]
    print(f"\nteacher {args.model}; " + (state["temperature_note"] or f"temperature {state['temperature']}"))
    print(f"calls {len(samples)} | wall {elapsed/60:.1f} min | tokens {dict(spend.tokens)} | cost ${spend.usd:.2f}")
    print("\n| gold route | first-sample accuracy | tasks with >=1 kept | kept samples |")
    print("|---|---|---|---|")
    for r in routes:
        f_r = [s for s in first if s["gold"] == r]
        if not f_r:
            continue
        tasks_r = {s["task_id"] for s in first if s["gold"] == r}
        print(f"| {r} | {sum(s['route'] == r for s in f_r)}/{len(f_r)} | "
              f"{sum(1 for t in tasks_r if t in by_task)}/{len(tasks_r)} | {sum(1 for k in kept if k['gold'] == r)} |")
    print(f"| **total** | {sum(s['route'] == s['gold'] for s in first)}/{len(first)} | "
          f"{len(by_task)}/{len(rows)} | {len(kept)} |")
    if args.fact_check:
        print("\n| gold route | route right, facts wrong | of route-right samples |")
        print("|---|---|---|")
        for r in routes:
            rr = [s for s in samples if s["gold"] == r and s["route_ok"]]
            if rr:
                print(f"| {r} | {sum(1 for s in rr if s['fact_mismatches'])} | {len(rr)} |")
        rr = [s for s in samples if s["route_ok"]]
        print(f"| **total** | {sum(1 for s in rr if s['fact_mismatches'])} | {len(rr)} |")
        field_counts = collections.Counter(f for s in rr for f in s["fact_mismatches"])
        print("mismatched fields among route-right samples:", dict(field_counts))
    from reward.rubric_checks import word_count
    words = [word_count(parse_record(k["completion"])["reply"]) for k in kept]
    if words:
        print(f"\nkept replies: mean {sum(words)/len(words):.0f} words, over 60 words: {sum(1 for w in words if w > 60)}/{len(words)}")
    parsed = sum(1 for s in samples if s["route"] is not None)
    lens = [s["output_tokens"] for s in samples]
    print(f"\nparse rate {parsed}/{len(samples)}; kept completions needing fence/prose stripping: {stripped}; "
          f"mean output {sum(lens)/len(lens):.0f} tokens, max {max(lens)}; "
          f"hit max_tokens: {sum(1 for s in samples if s['stop_reason'] == 'max_tokens')}")
    print("wrote", args.out_samples, "and", args.out_sft)


if __name__ == "__main__":
    main()
