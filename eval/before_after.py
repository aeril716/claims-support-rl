"""Route accuracy of a policy on the held-out test split. Generation only: no reward, no judge.

    python eval/before_after.py --model Qwen/Qwen2.5-1.5B-Instruct --out <dir>
    python eval/before_after.py --model ollama:qwen2.5:32b-instruct --out <dir>
    python eval/before_after.py --model Qwen/Qwen2.5-7B-Instruct --adapter out/grpo_run3_full/checkpoint-320 --score --out <dir>

--adapter loads a LoRA checkpoint on top of the base model (the "after"). --score also runs
every output through reward.score, so the rubric pass rate can sit next to route accuracy;
that needs the judge and is off by default.

Greedy decoding, so a model gives the same numbers every time. The prompt is built by the same
code the trainer uses. Reports the JSON parse rate raw and after repair, and route accuracy
overall and per gold route. A model that does not fit in memory as Hugging Face weights can be
run through Ollama with the "ollama:" prefix; that is the quantised build Ollama serves, not the
full-precision weights, and the report says so.
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

from reward.reward import parse_record, score   # noqa: E402
from reward.judge import judge_info             # noqa: E402
import train_grpo                                # noqa: E402  (prompt builder and task loader)

MAX_NEW_TOKENS = 512


def generate_hf(model_id, rows, adapter=None, bf16=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if bf16 and device == "cuda":
        dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    outputs = []
    for i, row in enumerate(rows, start=1):
        text = train_grpo.render_prompt(row["prompt"], tok)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        outputs.append(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  {i}/{len(rows)} {row['task_id']}", flush=True)
    how = f"hf {dtype} {device} greedy" + (f" + adapter {adapter}" if adapter else "")
    return outputs, how


def generate_ollama(model_name, rows):
    import ollama
    outputs = []
    for i, row in enumerate(rows, start=1):
        reply = ollama.chat(model=model_name, messages=row["prompt"],
                            options={"temperature": 0, "num_predict": MAX_NEW_TOKENS,
                                     "num_ctx": 8192})
        outputs.append(reply["message"]["content"])
        print(f"  {i}/{len(rows)} {row['task_id']}", flush=True)
    return outputs, "ollama quantised greedy (temperature 0)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", default=None, help="LoRA checkpoint directory to load on the base")
    parser.add_argument("--score", action="store_true", help="also grade every output with reward.score")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7run7", "v8"], default="v1",
                        help="policy prompt version, as in train/train_grpo.py (v1 = run1-run6)")
    parser.add_argument("--reasoning", action="store_true",
                        help="ask for a leading reasoning field, as train/train_grpo.py --reasoning")
    parser.add_argument("--bf16", action="store_true", help="load the model in bfloat16 (A100/H100)")
    args = parser.parse_args()
    train_grpo.PROMPT_VERSION = args.prompt_version
    train_grpo.REASONING = args.reasoning
    args.out.mkdir(parents=True, exist_ok=True)

    rows = train_grpo.load_tasks()
    start = time.time()
    if args.model.startswith("ollama:"):
        outputs, how = generate_ollama(args.model[len("ollama:"):], rows)
    else:
        outputs, how = generate_hf(args.model, rows, args.adapter, args.bf16)
    elapsed = time.time() - start

    tasks = {t["task_id"]: t for t in (json.loads(l) for l in open(train_grpo.TASKS) if l.strip())}
    records, raw_ok, repaired_ok = [], 0, 0
    per_route = collections.defaultdict(lambda: [0, 0])
    chosen_hist = collections.Counter()
    item_pass = collections.defaultdict(lambda: [0, 0])
    for row, text in zip(rows, outputs):
        record_parsed = parse_record(text)
        route, reply, reasoning, repairs = (record_parsed["route"], record_parsed["reply"],
                                            record_parsed["reasoning"], record_parsed["repairs"])
        parsed = route is not None
        raw_ok += parsed and not repairs
        repaired_ok += parsed
        correct = parsed and route == row["route_answer"]
        per_route[row["route_answer"]][1] += 1
        per_route[row["route_answer"]][0] += correct
        chosen_hist[str(route)] += 1
        record = {"task_id": row["task_id"], "gold": row["route_answer"], "chosen": route,
                  "correct": bool(correct), "repairs": repairs, "reasoning": reasoning,
                  "facts": record_parsed["facts"], "raw_output": text}
        if args.score:
            reward, detail = score(tasks[row["task_id"]], text)
            record["reward"], record["rubric_score"] = reward, detail.get("rubric_score")
            for item in detail["items"]:
                item_pass[item["id"]][1] += 1
                item_pass[item["id"]][0] += item["passed"]
        records.append(record)

    n = len(rows)
    summary = {"model": args.model, "how": how, "tasks": n, "prompt_version": args.prompt_version,
               "reasoning": args.reasoning, "judge": judge_info() if args.score else None,
               "parsed_raw": raw_ok, "parsed_after_repair": repaired_ok,
               "route_accuracy": sum(r["correct"] for r in records) / n,
               "per_route": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(per_route.items())},
               "chosen_histogram": dict(chosen_hist),
               "seconds": round(elapsed)}
    if args.score:
        scored = [r["rubric_score"] for r in records if r.get("rubric_score") is not None]
        summary["rubric_pass_rate_mean"] = sum(scored) / len(scored) if scored else None
        summary["rubric_scored_outputs"] = len(scored)
        summary["item_pass"] = {k: {"passed": v[0], "total": v[1]} for k, v in sorted(item_pass.items())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "outputs.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))

    print(f"\n{args.model}  ({how})  {elapsed/60:.1f} min")
    print(f"  parsed raw {raw_ok}/{n}   after repair {repaired_ok}/{n}")
    print(f"  route accuracy {sum(r['correct'] for r in records)}/{n}")
    for route, (c, t) in sorted(per_route.items()):
        print(f"    {route:<24}{c:>3}/{t}")
    print(f"  chosen routes: {dict(chosen_hist)}")
    if args.score:
        print(f"  rubric pass rate (mean over {summary['rubric_scored_outputs']} parsed outputs): "
              f"{summary['rubric_pass_rate_mean']:.3f}")


if __name__ == "__main__":
    main()
