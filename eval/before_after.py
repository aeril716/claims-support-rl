"""Route accuracy of a policy on the held-out test split. Generation only: no reward, no judge.

    python eval/before_after.py --model Qwen/Qwen2.5-1.5B-Instruct --out <dir>
    python eval/before_after.py --model ollama:qwen2.5:32b-instruct --out <dir>
    python eval/before_after.py --model Qwen/Qwen2.5-7B-Instruct --adapter out/grpo_run3_full/checkpoint-320 --score --out <dir>
    python eval/before_after.py --rescore --out <dir>        # judge a saved outputs.json again, no generation

--rubric both (the default) scores under wording v2 and re-judges only the reworded items
under v1, so summary.json carries rubric_v1_pass_rate_mean and rubric_v2_pass_rate_mean plus
item_pass_v1 / item_pass_v2; rubric_pass_rate_mean and item_pass keep the primary (v2) values.

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
from reward.judge import JUDGE_URL, judge_info  # noqa: E402
from reward import rubric_wording               # noqa: E402
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


def parse_records(rows, outputs):
    """Route parsing and per-route counts for generated outputs; no judge involved."""
    records, raw_ok, repaired_ok = [], 0, 0
    per_route = collections.defaultdict(lambda: [0, 0])
    chosen_hist = collections.Counter()
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
        records.append({"task_id": row["task_id"], "gold": row["route_answer"], "chosen": route,
                        "correct": bool(correct), "repairs": repairs, "reasoning": reasoning,
                        "facts": record_parsed["facts"], "raw_output": text})
    return records, raw_ok, repaired_ok, per_route, chosen_hist


def write(out, summary, records):
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "outputs.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="policy model id (required unless --rescore)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", default=None, help="LoRA checkpoint directory to load on the base")
    parser.add_argument("--score", action="store_true", help="also grade every output with reward.score")
    parser.add_argument("--rescore", action="store_true",
                        help="skip generation: load <out>/outputs.json and run the judge on it")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7run7", "v8"], default="v1",
                        help="policy prompt version, as in train/train_grpo.py (v1 = run1-run6)")
    parser.add_argument("--reasoning", action="store_true",
                        help="ask for a leading reasoning field, as train/train_grpo.py --reasoning")
    parser.add_argument("--bf16", action="store_true", help="load the model in bfloat16 (A100/H100)")
    parser.add_argument("--rubric", choices=sorted(rubric_wording.VERSIONS) + ["both"], default="both",
                        help="rubric wording to score with: v1, v2, or both (default; scores under v2 and re-judges "
                             "only the reworded items under v1, reporting a rubric column per version)")
    args = parser.parse_args()
    print(f"judge URL: {JUDGE_URL}", flush=True)
    train_grpo.PROMPT_VERSION = args.prompt_version
    train_grpo.REASONING = args.reasoning
    versions = ["v2", "v1"] if args.rubric == "both" else [args.rubric]
    rubric_wording.set_version(versions[0])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = train_grpo.load_tasks()
    n = len(rows)

    if args.rescore:
        # Scoring only. The outputs and the generation summary are read back from <out>; the
        # prompt version and model are taken from the saved summary, not the flags.
        saved = json.loads((args.out / "outputs.json").read_text())
        old_summary = json.loads((args.out / "summary.json").read_text())
        by_id = {r["task_id"]: r for r in saved}
        assert set(by_id) == {r["task_id"] for r in rows}, "saved outputs do not match the current test file"
        outputs = [by_id[row["task_id"]]["raw_output"] for row in rows]
        how, elapsed = old_summary["how"], old_summary["seconds"]
        args.model, args.prompt_version = old_summary["model"], old_summary["prompt_version"]
        args.reasoning = old_summary.get("reasoning", False)
        args.score = True
        print(f"rescoring {len(outputs)} saved outputs from {args.out} with judge {judge_info()}")
    else:
        if not args.model:
            parser.error("--model is required unless --rescore")
        start = time.time()
        if args.model.startswith("ollama:"):
            outputs, how = generate_ollama(args.model[len("ollama:"):], rows)
        else:
            outputs, how = generate_hf(args.model, rows, args.adapter, args.bf16)
        elapsed = time.time() - start

    records, raw_ok, repaired_ok, per_route, chosen_hist = parse_records(rows, outputs)
    summary = {"model": args.model, "how": how, "tasks": n, "prompt_version": args.prompt_version,
               "reasoning": args.reasoning, "judge": None, "rubric_wording": args.rubric,
               "parsed_raw": raw_ok, "parsed_after_repair": repaired_ok,
               "route_accuracy": sum(r["correct"] for r in records) / n,
               "per_route": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(per_route.items())},
               "chosen_histogram": dict(chosen_hist),
               "seconds": round(elapsed)}
    # Saved before any judge call, so a scoring failure never costs the generation.
    write(args.out, summary, records)

    print(f"\n{args.model}  ({how})  {elapsed/60:.1f} min")
    print(f"  parsed raw {raw_ok}/{n}   after repair {repaired_ok}/{n}")
    print(f"  route accuracy {sum(r['correct'] for r in records)}/{n}")
    for route, (c, t) in sorted(per_route.items()):
        print(f"    {route:<24}{c:>3}/{t}")
    print(f"  chosen routes: {dict(chosen_hist)}")

    if args.score:
        tasks = {t["task_id"]: t for t in (json.loads(l) for l in open(train_grpo.TASKS) if l.strip())}
        primary = versions[0]
        item_pass = {v: collections.defaultdict(lambda: [0, 0]) for v in versions}
        t0 = time.time()
        for i, record in enumerate(records, start=1):
            task = tasks[record["task_id"]]
            rubric_wording.set_version(primary)
            reward, detail = score(task, record["raw_output"])
            record["reward"], record[f"rubric_score_{primary}"] = reward, detail.get("rubric_score")
            items = [{"id": it["id"], "answer": it["answer"], "passed": it["passed"],
                      "reasoning": it.get("reasoning")} for it in detail["items"]]
            record["items"] = items
            for it in items:
                item_pass[primary][it["id"]][1] += 1
                item_pass[primary][it["id"]][0] += it["passed"]
            if len(versions) > 1 and detail["parsed"]:
                # The other version differs only on the reworded items: re-judge just those
                # and rebuild that version's rubric score over the same item set.
                other = versions[1]
                changed = set(rubric_wording.VERSIONS[primary]) | set(rubric_wording.VERSIONS[other])
                sub = {**task, "rubric": [it for it in task["rubric"] if it["id"] in changed]}
                rubric_wording.set_version(other)
                other_items = {}
                if sub["rubric"]:
                    _r, d2 = score(sub, record["raw_output"])
                    other_items = {it["id"]: it for it in d2["items"] if it["id"] in changed}
                rubric_wording.set_version(primary)
                passed = 0
                for it in items:
                    o = other_items.get(it["id"])
                    ok = o["passed"] if o else it["passed"]
                    passed += ok
                    item_pass[other][it["id"]][1] += 1
                    item_pass[other][it["id"]][0] += ok
                    if o:
                        it[f"answer_{other}"], it[f"passed_{other}"] = o["answer"], o["passed"]
                record[f"rubric_score_{other}"] = passed / len(items) if items else None
            if i % 10 == 0 or i == n:
                print(f"  scored {i}/{n}", flush=True)
        summary["judge"] = judge_info()
        summary["judge_failures"] = summary["judge"]["judge_failures"]
        summary["scoring_seconds"] = round(time.time() - t0)
        for v in versions:
            scored = [r[f"rubric_score_{v}"] for r in records if r.get(f"rubric_score_{v}") is not None]
            summary[f"rubric_{v}_pass_rate_mean"] = sum(scored) / len(scored) if scored else None
            summary[f"item_pass_{v}"] = {k: {"passed": x[0], "total": x[1]} for k, x in sorted(item_pass[v].items())}
        # backwards-compatible names carry the primary version
        summary["rubric_pass_rate_mean"] = summary[f"rubric_{primary}_pass_rate_mean"]
        summary["rubric_scored_outputs"] = sum(1 for r in records if r.get(f"rubric_score_{primary}") is not None)
        summary["item_pass"] = summary[f"item_pass_{primary}"]
        for r in records:
            r["rubric_score"] = r.get(f"rubric_score_{primary}")
        write(args.out, summary, records)
        cols = "   ".join(f"rubric {v} {summary[f'rubric_{v}_pass_rate_mean']:.3f}" for v in versions
                          if summary[f"rubric_{v}_pass_rate_mean"] is not None)
        print(f"  {cols} (mean over {summary['rubric_scored_outputs']} parsed outputs)   judge {summary['judge']}"
              f"   {summary['scoring_seconds']} s   judge_failures {summary['judge_failures']}")


if __name__ == "__main__":
    main()
