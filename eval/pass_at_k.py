"""pass@k on route choice: sample k completions per task and ask how often the gold route is
among the first j of them. Route only; the judge is never called.

    python eval/pass_at_k.py --model Qwen/Qwen2.5-14B-Instruct --prompt-version v5 --k 16 \
        --out out/passk_base14b_v5_test3.jsonl [--greedy out/eval_base14b_v5_test3/summary.json]
    python eval/pass_at_k.py --model Qwen/Qwen2.5-14B-Instruct --adapter <run>/checkpoint-35 \
        --prompt-version v5 --k 16 --out out/passk_run8_ckpt35_v5_test3.jsonl
    python eval/pass_at_k.py --model Qwen/Qwen2.5-14B-Instruct --prompt-version v8 --reasoning \
        --tasks data/v3_kb_definitions/tasks_train.jsonl --k 4 --max-new-tokens 768 --save-raw \
        --out out/student_v8reason_train.jsonl        # raw texts kept, for DPO pairs

--reasoning renders the prompt with the reasoning field the way train/train_grpo.py --reasoning
does; --save-raw stores every completion's raw text under "texts" next to the parsed routes.

One generate() call per task with num_return_sequences=k, sampling on (temperature as given,
top_p 1.0, no top_k). If that call does not fit in GPU memory the k samples are drawn in
smaller chunks on the same prompt (see sample_routes); the final line of the report says how
many sequences each call held. max_new_tokens is the same as
eval/before_after.py (512), and the route is read with the same parser and repairs, so the
parse rate is comparable with the greedy runs.

Writes one JSON line per task, then prints pass@1, 2, 4, 8, 16 per gold route and in total,
the parse rate, and (with --greedy) the greedy accuracy of the matching before_after run.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "data"))

from reward.reward import parse_record   # noqa: E402
import train_grpo                          # noqa: E402

MAX_NEW_TOKENS = 512     # default; --max-new-tokens overrides (768 with the reasoning field)
KS = (1, 2, 4, 8, 16)
ROUTES = ["ask_question", "file_claim", "explain_not_covered", "escalate",
          "refer_to_manufacturer", "explain_waiting_period", "tech_support"]


CHUNK = {"size": None}   # sequences per generate() call; starts at k, halved on CUDA OOM


def _generate(model, tok, inputs, n, temperature, max_new_tokens):
    import torch
    with torch.no_grad():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=temperature, top_p=1.0, top_k=0,
                              num_return_sequences=n, pad_token_id=tok.eos_token_id)


def sample_routes(model, tok, text, k, temperature, device, max_new_tokens=MAX_NEW_TOKENS):
    """k sampled completions for one rendered prompt: (routes, token lengths, reasonings, texts),
    route None when no route could be parsed, length = completion tokens before the first
    end-of-sequence token, reasoning None unless the completion carried the field.

    Asks for all k in one generate() call. If that call runs out of GPU memory (14B fp16 with a
    3,200-token prompt and 16 sequences did on the 48 GB card: the prefill needs a 23.5 GiB
    block), the chunk size is halved and the samples are drawn in several calls on the same
    prompt. Each call samples independently, so the k samples are the same draw either way."""
    import torch
    inputs = tok(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    if CHUNK["size"] is None:
        CHUNK["size"] = k
    texts, lengths = [], []
    eos_ids = {tok.eos_token_id, tok.pad_token_id} | set(getattr(model.generation_config, "eos_token_id", None) or [])
    while len(texts) < k:
        n = min(CHUNK["size"], k - len(texts))
        try:
            out = _generate(model, tok, inputs, n, temperature, max_new_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if CHUNK["size"] == 1:
                raise
            CHUNK["size"] = max(1, CHUNK["size"] // 2)
            print(f"  CUDA OOM with {n} sequences per call; retrying with chunks of {CHUNK['size']}", flush=True)
            continue
        for seq in out:
            ids = seq[prompt_len:].tolist()
            stop = next((i for i, t in enumerate(ids) if t in eos_ids), len(ids))
            lengths.append(stop)
            texts.append(tok.decode(ids[:stop], skip_special_tokens=True))
        del out
    parsed = [parse_record(t) for t in texts]
    return [r["route"] for r in parsed], lengths, [r["reasoning"] for r in parsed], texts


def pass_table(records, greedy=None):
    """pass@j per gold route and in total; j capped at the number of samples."""
    ks = [j for j in KS if j <= len(records[0]["samples"])]
    by_gold = collections.defaultdict(list)
    for r in records:
        by_gold[r["gold"]].append(r)
    head = "| gold route | n | " + " | ".join(f"pass@{j}" for j in ks) + (" | greedy |" if greedy else " |")
    print(head)
    print("|---" * (len(ks) + 2 + (1 if greedy else 0)) + "|")

    def row(name, rows):
        cells = []
        for j in ks:
            hits = sum(1 for r in rows if r["gold"] in r["samples"][:j])
            cells.append(f"{hits}/{len(rows)}")
        g = ""
        if greedy:
            if name == "total":
                g = f" | {round(greedy['route_accuracy'] * greedy['tasks'])}/{greedy['tasks']}"
            elif name in greedy["per_route"]:
                g = f" | {greedy['per_route'][name]['correct']}/{greedy['per_route'][name]['total']}"
            else:
                g = " | -"
        print(f"| {name} | {len(rows)} | " + " | ".join(cells) + g + " |")

    for route in ROUTES:
        if by_gold.get(route):
            row(route, by_gold[route])
    row("total", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", default="v5",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7run7", "v8"])
    parser.add_argument("--tasks", default=str(train_grpo.TASKS), help="task file (default: the test split)")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature; the GRPO trainer uses 1.0 (train/train_grpo.py)")
    parser.add_argument("--out", type=Path, required=True, help="jsonl, one line per task")
    parser.add_argument("--greedy", type=Path, default=None,
                        help="summary.json of the matching eval/before_after.py run, for the reference column")
    parser.add_argument("--reasoning", action="store_true",
                        help="ask for a reasoning field before the route, as train/train_grpo.py --reasoning")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"completion cap (default {MAX_NEW_TOKENS}; use 768 with --reasoning)")
    parser.add_argument("--save-raw", "--save-text", dest="save_raw", action="store_true",
                        help="also store the raw completion text of every sample under \"texts\" (DPO pair pool)")
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint directory to load on the base, as eval/before_after.py --adapter")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 instead of float16 (A100/H100)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    train_grpo.PROMPT_VERSION = args.prompt_version
    train_grpo.REASONING = args.reasoning
    rows = train_grpo.load_tasks(args.tasks)
    device = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    print(f"{args.model} {dtype} {device} sdpa" + (f" + adapter {args.adapter}" if args.adapter else "")
          + f" | prompt {args.prompt_version}{' + reasoning' if args.reasoning else ''} | k={args.k} "
          f"temperature={args.temperature} top_p=1.0 max_new_tokens={args.max_new_tokens} | {len(rows)} tasks", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records, start = [], time.time()
    with args.out.open("w") as handle:
        for i, row in enumerate(rows, start=1):
            text = train_grpo.render_prompt(row["prompt"], tok)
            samples, lengths, reasonings, texts = sample_routes(model, tok, text, args.k, args.temperature,
                                                                device, args.max_new_tokens)
            record = {"task_id": row["task_id"], "gold": row["route_answer"], "model": args.model,
                      "adapter": args.adapter, "prompt_version": args.prompt_version,
                      "reasoning": args.reasoning, "temperature": args.temperature,
                      "max_new_tokens": args.max_new_tokens, "samples": samples,
                      "completion_tokens": lengths,
                      "hit_cap": sum(1 for n in lengths if n >= args.max_new_tokens)}
            if args.reasoning:
                record["reasonings"] = reasonings
            if args.save_raw:
                record["texts"] = texts
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records.append(record)
            hit = row["route_answer"] in samples
            print(f"  {i}/{len(rows)} {row['task_id']} {row['route_answer']:<24} gold in {sum(1 for s in samples if s == row['route_answer'])}/{args.k}"
                  + ("" if hit else "   MISS"), flush=True)
    elapsed = time.time() - start

    greedy = json.loads(args.greedy.read_text()) if args.greedy else None
    print()
    pass_table(records, greedy)
    total = sum(len(r["samples"]) for r in records)
    parsed = sum(1 for r in records for s in r["samples"] if s is not None)
    print(f"\nparse rate: {parsed}/{total} samples carried a route ({parsed / total:.3f})")
    chosen = collections.Counter(str(s) for r in records for s in r["samples"])
    print("routes sampled overall:", dict(chosen.most_common()))
    all_len = [n for r in records for n in r["completion_tokens"]]
    print(f"completion length: mean {sum(all_len) / len(all_len):.1f} tokens, max {max(all_len)}; "
          f"{sum(r['hit_cap'] for r in records)}/{total} hit the {args.max_new_tokens}-token cap")
    if device == "cuda":
        print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"wall {elapsed / 60:.1f} min for {len(rows)} tasks x {args.k} samples; "
          f"{CHUNK['size']} sequences per generate() call")
    first_ok = sum(1 for r in records if r["samples"][0] == r["gold"])
    mixed = sum(1 for r in records if r["gold"] in r["samples"] and any(s != r["gold"] for s in r["samples"]))
    print(f"first-sample accuracy {first_ok}/{len(records)}; tasks with at least one right and one wrong sample: {mixed}/{len(records)}")


if __name__ == "__main__":
    main()
