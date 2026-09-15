"""Supervised fine-tuning (SFT) arm on the GPU server: teach the policy by imitating teacher replies.

    nohup .venv/bin/python train/sft.py > out/sft_r3_14b.log 2>&1 &    # GPU server (sm_75)
    python train/sft.py --bf16                                          # Colab A100
    python train/sft.py --data data/sft/train_v8reason_r3_new103.jsonl --out out/sft_r3_new103_14b

Policy: Qwen2.5-14B-Instruct with a LoRA adapter (r=16, same seven projection modules as
train/train_grpo.py). The GPU is a Quadro RTX 8000 (compute capability 7.5), which has no
bf16, so the base is loaded in fp16 with SDPA attention and the LoRA weights stay in fp32.
With --bf16 (for GPUs that have it, such as the Colab A100) the base is loaded in bfloat16
instead and training uses bf16 mixed precision, which needs no grad scaler. Every other
setting is the same in both modes.

Data: --data, default data/sft/train_v8reason_r3.jsonl (teacher samples, already filtered to the
gold route). The training set is built in two steps:
  1. at most 2 samples per task; the second is one whose "reasoning" text differs from the
     first when such a sample exists
  2. at most 50 samples per gold route; every task's first sample is taken before any task's
     second sample, so the cap drops second samples before it drops tasks
The per-route counts are printed before training, and the samples used are written to
data/sft/used_<name>.jsonl, where <name> is the data file name without "train_v8reason_" (so
used_r3.jsonl for the default, used_r3_new103.jsonl for the new103 file). Gold routes come from
data/v3_kb_definitions/tasks_train.jsonl and tasks_sft_new103.jsonl (t- and s-ids).

Loss is computed on the completion tokens only: prompt tokens get label -100, which the loss
ignores. The completion is followed by <|im_end|> so the model learns where to stop.

Out-of-memory fallback: training runs in a child process. If the fp16 (or bf16) LoRA child runs
out of GPU memory it exits with code 3, and the parent starts a second child that loads the base
in 4-bit (QLoRA) with the same LoRA settings. A fresh process guarantees the GPU is empty.

Output: --out, default out/sft_r3_14b/: the adapter plus run_stats.json (mode, peak VRAM, wall time,
loss at every step).
"""

import argparse
import collections
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-14B-Instruct"
TASK_FILES = [ROOT / "data" / "v3_kb_definitions" / "tasks_train.jsonl",
              ROOT / "data" / "v3_kb_definitions" / "tasks_sft_new103.jsonl"]
DEFAULT_DATA = ROOT / "data" / "sft" / "train_v8reason_r3.jsonl"
DEFAULT_OUT = ROOT / "out" / "sft_r3_14b"

# Set from --data and --out in main(), in the parent and in every child.
SFT_DATA = DEFAULT_DATA
USED = None
OUT = DEFAULT_OUT

PER_TASK = 2
PER_ROUTE = 50
SEED = 0
OOM_EXIT = 3
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def used_path(data):
    """data/sft/train_v8reason_r3.jsonl -> data/sft/used_r3.jsonl"""
    return ROOT / "data" / "sft" / ("used_" + data.stem.removeprefix("train_v8reason_") + ".jsonl")


def build_training_set():
    """Pick the samples to train on and write them to USED. Returns the per-route counts."""
    gold = {}
    for path in TASK_FILES:
        if not path.exists():
            continue
        for line in open(path):
            row = json.loads(line)
            assert row["task_id"] not in gold, f"task id {row['task_id']} appears in two task files"
            gold[row["task_id"]] = row["route_answer"]

    by_task = collections.OrderedDict()
    mismatched = 0
    for line in open(SFT_DATA):
        rec = json.loads(line)
        parsed = json.loads(rec["completion"])
        if parsed["route"] != gold[rec["task_id"]]:
            mismatched += 1
            continue
        rec["gold_route"] = gold[rec["task_id"]]
        rec["reasoning"] = parsed["reasoning"]
        by_task.setdefault(rec["task_id"], []).append(rec)
    print(f"samples whose route differs from gold (skipped): {mismatched}")

    # Step 1: at most 2 per task, second one with different reasoning when possible.
    firsts, seconds = [], []
    for samples in by_task.values():
        first = samples[0]
        rest = [s for s in samples[1:] if s["completion"] != first["completion"]]
        different = [s for s in rest if s["reasoning"] != first["reasoning"]]
        firsts.append(first)
        if different:
            seconds.append(different[0])
        elif rest:
            seconds.append(rest[0])

    # Step 2: cap each gold route, all first samples before any second sample.
    rng = random.Random(SEED)
    rng.shuffle(firsts)
    rng.shuffle(seconds)
    used = []
    counts = collections.Counter()
    for pick, group in ((1, firsts), (2, seconds)):
        for s in group:
            if counts[s["gold_route"]] < PER_ROUTE:
                counts[s["gold_route"]] += 1
                used.append({"task_id": s["task_id"], "gold_route": s["gold_route"], "pick": pick,
                             "prompt": s["prompt"], "completion": s["completion"]})

    with open(USED, "w") as f:
        for u in used:
            f.write(json.dumps(u) + "\n")

    tasks_used = collections.defaultdict(set)
    for u in used:
        tasks_used[u["gold_route"]].add(u["task_id"])
    print("| route | samples | tasks |")
    print("|---|---|---|")
    for route in sorted(counts):
        print(f"| {route} | {counts[route]} | {len(tasks_used[route])} |")
    print(f"| total | {sum(counts.values())} | {len({u['task_id'] for u in used})} |")
    print(f"rows after the per-task and per-route caps: {len(used)}")
    print(f"written: {USED}", flush=True)
    return counts


def train(mode, bf16):
    """Train one LoRA adapter. mode is "fp16", "bf16" or "qlora"; bf16 picks the half-precision type."""
    import torch
    half = torch.bfloat16 if bf16 else torch.float16
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainerCallback, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in open(USED)]
    examples = []
    for r in rows:
        prompt_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(r["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        examples.append({"input_ids": prompt_ids + completion_ids,
                         "labels": [-100] * len(prompt_ids) + completion_ids})
    print(f"[{mode}] examples: {len(examples)}, longest: {max(len(e['input_ids']) for e in examples)} tokens",
          flush=True)

    def collate(batch):
        # batch size is 1, so no padding is needed
        return {"input_ids": torch.tensor([b["input_ids"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch])}

    load_kwargs = {"attn_implementation": "sdpa", "device_map": {"": 0}}
    load_kwargs["dtype"] = half
    if mode == "qlora":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=half)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **load_kwargs)
    model.config.use_cache = False
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=TARGET_MODULES)
    model = get_peft_model(model, lora)
    # mixed precision keeps the trainable weights in fp32 (fp16 requires it; bf16 keeps it identical)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    losses = []

    class LossLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                losses.append({"step": state.global_step, "loss": logs["loss"]})

    args = TrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate,
                      callbacks=[LossLog()])

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start
    model.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))

    stats = {"mode": mode, "half_dtype": str(half), "model": MODEL, "examples": len(examples),
             "optimizer_steps": trainer.state.global_step,
             "train_seconds": round(train_seconds, 1),
             "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
             "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
             "losses": losses}
    with open(OUT / "run_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{mode}] saved adapter to {OUT}", flush=True)
    print(json.dumps({k: v for k, v in stats.items() if k != "losses"}), flush=True)


def child(mode, bf16):
    import torch
    try:
        train(mode, bf16)
    except torch.OutOfMemoryError as e:
        print(f"[{mode}] OUT OF MEMORY: {e}", flush=True)
        sys.exit(OOM_EXIT)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[{mode}] OUT OF MEMORY: {e}", flush=True)
            sys.exit(OOM_EXIT)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=["fp16", "bf16", "qlora"])
    parser.add_argument("--bf16", action="store_true",
                        help="load and train in bfloat16 (A100 and newer) instead of fp16")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="teacher SFT samples (jsonl)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the adapter")
    args = parser.parse_args()
    global SFT_DATA, USED, OUT
    SFT_DATA, OUT = args.data.resolve(), args.out.resolve()
    USED = used_path(SFT_DATA)
    if args.child:
        child(args.child, args.bf16 or args.child == "bf16")
        return

    print(f"data: {SFT_DATA}\nused samples: {USED}\noutput dir: {OUT}", flush=True)
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    build_training_set()

    code = None
    child_args = ["--bf16"] if args.bf16 else []
    child_args += ["--data", str(SFT_DATA), "--out", str(OUT)]
    for mode in ("bf16" if args.bf16 else "fp16", "qlora"):
        print(f"=== starting {mode} run ===", flush=True)
        code = subprocess.call([sys.executable, __file__, "--child", mode] + child_args)
        if code != OOM_EXIT:
            break
        print(f"=== {mode} ran out of GPU memory; falling back ===", flush=True)
    print(f"=== finished: mode={mode} exit={code} wall={time.time() - start:.0f}s ===", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
