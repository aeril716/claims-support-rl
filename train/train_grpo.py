"""GRPO training run for the policy, on the GPU server.

    python train/train_grpo.py            # writes to out/grpo_run1 with the run1 settings
    python train/train_grpo.py --out out/grpo_run2_batch4 --batch 4 --accum 2 --steps 5
    python train/train_grpo.py --out out/grpo_run3_full --steps 320 --save-steps 40 --seed 42

Only the output directory, the micro-batch split, the step count, the checkpoint interval and
the seed can be overridden; every other setting is fixed in the file.

Policy: Qwen2.5-7B-Instruct with a LoRA adapter. Data: the training split, built by
data/build_dataset.py (prompt string + task JSON string). Reward: the two functions in
reward/trl_rewards.py, route first, rubric second. Every setting below is fixed on purpose;
see the "Training setup" section of README.md for the reason behind each non-default.

This file also holds the one definition of the policy prompt (system_prompt, load_tasks,
render_prompt), which data/build_dataset.py and eval/before_after.py import. It reads data/kb/
and the task files only, never data/judge_only/.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from reward import reward as R                    # noqa: E402
from reward import judge as J                     # noqa: E402
import components as C                            # noqa: E402
from transformers import TrainerCallback          # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TASKS = ROOT / "data" / "v3_kb_definitions" / "tasks_test.jsonl"
KB_DIR = ROOT / "data" / "kb"

TIMING = {"generate_s": 0.0, "generate_calls": 0}
RAW = []          # (step, completion text) for every completion the reward saw, for the report
LIVE_PATH = None  # set in main(): <out>/live_steps.jsonl, one line per step while training runs


def knowledge_base(version=None):
    """Stage 1 puts the whole KB in the prompt, one file after another. The directory depends
    on the prompt version (KB_DIRS), so older prompts reproduce byte for byte."""
    version = version or PROMPT_VERSION
    parts = []
    for path in sorted(KB_DIRS[version].glob("*.md")):
        parts.append(f"### {path.name}\n{path.read_text().strip()}")
    return "\n\n".join(parts)


PROMPT_VERSION = "v1"   # set from --prompt-version in main(); v1 is the run1-run6 prompt
REASONING = False       # set from --reasoning in main(); off = the run1-run6 answer format

ANSWER_FORMAT = '{"route": <one of the routes below>, "reply": <your message to the customer>}'
ANSWER_FORMAT_REASONING = ('{"reasoning": <one or two sentences: which device and incident you read, '
                           'and which rule applies>, "route": <one of the routes below>, '
                           '"reply": <your message to the customer>}')


# Prompt v5: six fact fields before the route, in place of the --reasoning field.
ANSWER_SECTION_V5 = """Answer with exactly one JSON object and nothing else. Fill every field, in this order:
{"device": <phone | laptop | tablet | watch | "not stated">,
 "incident": <short phrase, or "not stated">,
 "enrolled_days_ago": <the number from the account, or null>,
 "inside_waiting_period": <"yes" | "no" | "unknown">,
 "inside_manufacturer_warranty": <"yes" | "no" | "unknown">,
 "claim_limit_reached": <"yes" | "no" | "unknown">,
 "route": <one of the routes below>,
 "reply": <your message to the customer>}"""


# Prompt v7: device and incident only; the account carries the three flags instead.
ANSWER_SECTION_V7 = """Answer with exactly one JSON object and nothing else. Fill every field, in this order:
{"device": <phone | laptop | tablet | watch | "not stated">,
 "incident": <short phrase, or "not stated">,
 "route": <one of the routes below>,
 "reply": <your message to the customer>}"""


# Prompt v8: v5's fact fields without inside_manufacturer_warranty, which is now an input flag.
ANSWER_SECTION_V8 = ANSWER_SECTION_V5.replace(
    ' "inside_manufacturer_warranty": <"yes" | "no" | "unknown">,\n', "")
assert "inside_manufacturer_warranty" not in ANSWER_SECTION_V8 and ANSWER_SECTION_V8.count("\n") == ANSWER_SECTION_V5.count("\n") - 1


def answer_format(reasoning=None):
    """The one line that shows the JSON object the policy must emit. With reasoning on, the
    object gains a leading "reasoning" field; it is recorded, never scored."""
    reasoning = REASONING if reasoning is None else reasoning
    return ANSWER_FORMAT_REASONING if reasoning else ANSWER_FORMAT


def answer_section(version=None, reasoning=None):
    """The "Answer with ..." lines. v1-v4 share one line plus the object shape (with or without
    the reasoning field); v5 has its own block with the six fact fields and does not take
    --reasoning."""
    version = version or PROMPT_VERSION
    reasoning = REASONING if reasoning is None else reasoning
    if version in ("v5", "v6", "v7", "v7run7", "v8"):
        if reasoning:
            raise ValueError(f"prompt {version} has its own fact fields; --reasoning does not apply to it")
        if version in ("v7", "v7run7"):
            return ANSWER_SECTION_V7
        return ANSWER_SECTION_V8 if version == "v8" else ANSWER_SECTION_V5
    return "Answer with exactly one JSON object and nothing else:\n" + answer_format(reasoning)

ROUTES_V2 = """Routes (choose exactly one):
- ask_question — you do not yet know what the device is or what happened. Ask for the missing detail instead of deciding.
- file_claim — the incident is covered under the plan and the customer can file a claim now.
- tech_support — the problem is software (apps, OS, accounts, passwords). Tech support handles it at no charge; it is not a claim.
- refer_to_manufacturer — the problem is one the manufacturer's warranty still covers, so send the customer to the manufacturer.
- explain_waiting_period — the plan is active but the incident falls inside the waiting period, so coverage has not started.
- explain_not_covered — this kind of incident is not covered for this kind of device.
- escalate — the account has reached its claim limit. Hand the case to a person; do not say the claim is denied and do not explain why.

Use the knowledge base and the account fields to decide."""


ROUTES_V3 = """Routes. Decide the route first, then write the reply to match it.

- ask_question — the customer has not said what the device is, or what happened. Reply: ask for the missing detail only. Do not decide anything yet.
- explain_waiting_period — enrolled_days_ago is still inside the waiting period in the knowledge base. Reply: the plan is active, but coverage for repairs and replacements has not started yet; say when it begins.
- explain_not_covered — the knowledge base says this kind of incident is not covered for this kind of device (for example loss or theft of a non-phone). Reply: say it is not covered, cite the rule, and offer an alternative if one exists.
- tech_support — the problem is software (apps, OS, accounts, passwords, updates). Reply: say it is not a claim, tech support handles it at no charge, and give one concrete step to try.
- refer_to_manufacturer — the problem is a malfunction, wear, or dust/heat/humidity failure, and enrolled_days_ago is still inside the manufacturer warranty period in the knowledge base. Reply: send the customer to the manufacturer and say why.
- escalate — claims_last_12m has reached the claim limit for this device type in the knowledge base. Reply: say the case is being handed to a person. Do not say the claim is denied and do not explain why.
- file_claim — none of the above applies and the incident is covered. Reply: say it is covered, give the deductible for this price tier, how to start the claim, and what documents may be requested.

Before choosing, check enrolled_days_ago, claims_last_12m, and the device type against the knowledge base."""


ROUTES_V4 = """First, from the customer message and the account, write down:
- the device type (or "not stated")
- what happened (or "not stated")
- enrolled_days_ago: is it inside the waiting period? inside the manufacturer warranty period? (both periods are in the knowledge base)
- claims_last_12m: has it reached the claim limit for this device type? (limit is in the knowledge base)

Then choose the route. Check in this order; the first that applies wins. Write the reply to match the route.

- ask_question — device or incident not stated. Reply: ask for the missing detail only. Do not decide anything yet.
- explain_waiting_period — inside the waiting period. Reply: the plan is active, but coverage for repairs and replacements has not started yet; say when it begins.
- explain_not_covered — the knowledge base says this kind of incident is not covered for this kind of device (for example loss or theft of a non-phone). Reply: say it is not covered, cite the rule, and offer an alternative if one exists.
- tech_support — the problem is software (apps, OS, accounts, passwords, updates). Reply: say it is not a claim, tech support handles it at no charge, and give one concrete step to try.
- refer_to_manufacturer — malfunction, wear, or dust/heat/humidity failure, and still inside the manufacturer warranty period. Reply: send the customer to the manufacturer and say why.
- escalate — claim limit reached. Reply: say the case is being handed to a person. Do not say the claim is denied and do not explain why.
- file_claim — none of the above applies and the incident is covered. Reply: say it is covered, give the deductible for this price tier, how to start the claim, and what documents may be requested."""

# v5 keeps the v4 routes block exactly and drops the write-down paragraph and its bullets; the
# facts move into the answer object instead (ANSWER_SECTION_V5).
ROUTES_V5 = ROUTES_V4[ROUTES_V4.index("Then choose the route."):]

# v7run7 = the v6 block with the three account conditions named by the derived flags. This is
# the exact block run7 trained on (launched 2026-09-13 16:52); kept so its checkpoint can be
# evaluated with its own training prompt.
ROUTES_V7_RUN7 = (ROUTES_V5
                  .replace("- explain_waiting_period — inside the waiting period. Reply:",
                           "- explain_waiting_period — coverage_active is false. Reply:")
                  .replace("failure, and still inside the manufacturer warranty period. Reply:",
                           "failure, and manufacturer_warranty_active is true. Reply:")
                  .replace("- escalate — claim limit reached. Reply:",
                           "- escalate — claim_limit_reached is true. Reply:"))
assert ROUTES_V7_RUN7.count("coverage_active is false") == 1
assert ROUTES_V7_RUN7.count("manufacturer_warranty_active is true") == 1
assert ROUTES_V7_RUN7.count("claim_limit_reached is true") == 1

# v7 = v7run7 with the waiting-period rule restated and moved after escalate, matching the
# corrected label order in data/generate_tasks.py (2026-09-13): the waiting period only
# decides incidents the plan would otherwise cover from day 31.
_WAITING_RUN7 = ("- explain_waiting_period — coverage_active is false. Reply: the plan is active, but "
                 "coverage for repairs and replacements has not started yet; say when it begins.\n")
_WAITING_V7 = ("- explain_waiting_period — coverage_active is false and the incident is one the plan "
               "covers from day 31 (drop, cracked screen, liquid, power surge, battery, or phone "
               "loss/theft). Reply: the plan is active, but coverage for repairs and replacements has "
               "not started yet; say when it begins.\n")
_ESCALATE_V7 = ("- escalate — claim_limit_reached is true. Reply: say the case is being handed to a "
                "person. Do not say the claim is denied and do not explain why.\n")
assert ROUTES_V7_RUN7.count(_WAITING_RUN7) == 1 and ROUTES_V7_RUN7.count(_ESCALATE_V7) == 1
ROUTES_V7 = ROUTES_V7_RUN7.replace(_WAITING_RUN7, "").replace(_ESCALATE_V7, _ESCALATE_V7 + _WAITING_V7)
assert ROUTES_V7.index("- escalate") < ROUTES_V7.index("- explain_waiting_period") < ROUTES_V7.index("- file_claim")

# v8 = v5's block with the corrected label order (data/generate_tasks.py, 2026-09-13) and the
# refer_to_manufacturer and explain_waiting_period definitions restated.
def _v8_block():
    header, bullets = ROUTES_V5.split("\n\n", 1)
    lines = {l.split(" — ")[0][2:]: l for l in bullets.strip().splitlines()}
    assert set(lines) == set(C.ROUTES), sorted(lines)
    lines["refer_to_manufacturer"] = (
        "- refer_to_manufacturer — the problem is a malfunction, wear and tear, or dust/heat/humidity "
        "failure, and manufacturer_warranty_active is true. Route to the manufacturer. Reply: tell the "
        "customer to contact the manufacturer and say why (the warranty covers this kind of failure).")
    lines["explain_waiting_period"] = (
        "- explain_waiting_period — the account is inside the waiting period and the incident would "
        "otherwise be a claim (drop, cracked screen, liquid, power surge, battery, or phone loss/theft). "
        "Choose this even though the incident is a covered one. Reply: the plan is active, but coverage "
        "for repairs and replacements has not started yet; say when it begins.")
    order = ["ask_question", "explain_not_covered", "tech_support", "refer_to_manufacturer",
             "escalate", "explain_waiting_period", "file_claim"]
    return header + "\n\n" + "\n".join(lines[r] for r in order)

ROUTES_V8 = _v8_block()

# Which knowledge-base directory each prompt version reads. data/kb_v1 is the KB as it stood
# for run1-run6 and the v1-v3 baselines; data/kb is the live KB (file 10 gives the warranty
# as 365 days since 2026-09-13). The judge and the task generator read data/kb.
KB_DIRS = {"v1": ROOT / "data" / "kb_v1", "v2": ROOT / "data" / "kb_v1",
           "v3": ROOT / "data" / "kb_v1", "v4": ROOT / "data" / "kb", "v5": ROOT / "data" / "kb",
           "v6": ROOT / "data" / "kb_v6",   # v6 = v5 text over a trimmed, reordered KB
           "v7": ROOT / "data" / "kb_v6",   # v7 = v6 with derived account flags
           "v7run7": ROOT / "data" / "kb_v6",   # the v7 block as run7 trained on it
           "v8": ROOT / "data" / "kb_v6"}       # v8 = v5 + warranty flag, corrected rule order


def routes_block(version=None):
    """The one line (v1) or block (v2, v3) that names the routes. v1 lists the names only; v2 adds
    a one-line definition per route; v3 states the account condition and the required reply for
    each. Everything else in the prompt is the same bytes."""
    version = version or PROMPT_VERSION
    if version == "v1":
        return "Routes: " + ", ".join(C.ROUTES)
    if version == "v2":
        return ROUTES_V2
    if version == "v3":
        return ROUTES_V3
    if version == "v4":
        return ROUTES_V4
    if version in ("v5", "v6"):
        return ROUTES_V5
    if version == "v7":
        return ROUTES_V7
    if version == "v7run7":
        return ROUTES_V7_RUN7
    if version == "v8":
        return ROUTES_V8
    raise ValueError(f"unknown prompt version {version!r}")


def system_prompt(version=None, reasoning=None):
    return (
        "Read the customer message and reply as a support agent for a device protection plan. "
        "One turn only.\n\n"
        + answer_section(version, reasoning) + "\n\n"
        + routes_block(version) + "\n\n"
        "Knowledge base:\n\n" + knowledge_base(version)
    )


# The thresholds the route rules read, as in data/generate_tasks.py (31 is written there
# inline, WARRANTY_DAYS is 365, limits come from components.CLAIM_LIMITS). Prompt v7 renders
# them as three derived flags in the account; nothing else in training reads them.
WAITING_DAYS = 31
WARRANTY_DAYS = 365


def render_account(task, version=None):
    """The account JSON for the user turn. v1-v6: the stored account as-is. v7: the same fields
    plus three flags derived from them and the rule constants, in a fixed order. An empty
    account renders as {} in every version."""
    version = version or PROMPT_VERSION
    account = task["account"]
    if version not in ("v7", "v7run7", "v8") or not account:
        return json.dumps(account)
    enrolled, claims = account["enrolled_days_ago"], account["claims_last_12m"]
    if version == "v8":      # one derived flag only; the waiting period and the limit stay with the policy
        return json.dumps({
            "enrolled_days_ago": enrolled,
            "manufacturer_warranty_active": enrolled < WARRANTY_DAYS,
            "claims_last_12m": claims,
            "device_price_tier": account["device_price_tier"],
        })
    derived = {
        "enrolled_days_ago": enrolled,
        "coverage_active": enrolled >= WAITING_DAYS,
        "manufacturer_warranty_active": enrolled < WARRANTY_DAYS,
        "claims_last_12m": claims,
        "claim_limit_reached": claims >= C.CLAIM_LIMITS[task["device"]],
        "device_price_tier": account["device_price_tier"],
    }
    return json.dumps(derived)


def load_tasks(path=TASKS):
    """One row per task: the conversational prompt plus the columns the reward reads."""
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = []
    for task in tasks:
        user = f"Customer message: {task['customer']}\n\nAccount: {render_account(task)}"
        rows.append({
            "prompt": [{"role": "system", "content": system_prompt()},
                       {"role": "user", "content": user}],
            "task_id": task["task_id"],
            "customer": task["customer"],
            "route_answer": task["route_answer"],
            "rubric_json": json.dumps(task["rubric"]),
        })
    return rows


def render_prompt(messages, tokenizer):
    """The exact string the policy sees: the chat template applied to the messages with the
    generation prompt appended. This is the one definition of the rendered training prompt;
    eval/before_after.py and data/build_dataset.py both call it. TRL renders a conversational
    prompt column the same way, which was verified byte for byte."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def completion_text(completion):
    """Conversational completions arrive as a list of messages; take the text."""
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion)
    return completion


def time_generation():
    """Wrap generate() so the report can split wall time between generation and judging."""
    from transformers.generation.utils import GenerationMixin
    original = GenerationMixin.generate

    def timed(self, *args, **kwargs):
        start = time.time()
        try:
            return original(self, *args, **kwargs)
        finally:
            TIMING["generate_s"] += time.time() - start
            TIMING["generate_calls"] += 1

    GenerationMixin.generate = timed


class StepTiming(TrainerCallback):
    """Per-step wall time, split into generation (patched generate), judge (reward TIMING),
    and the rest (forward/backward/optimizer), from deltas of the running counters."""

    def __init__(self):
        self.rows = []
        self._t0 = self._g0 = self._j0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0, self._g0, self._j0 = time.time(), TIMING["generate_s"], R.TIMING["judge_s"]

    def on_step_end(self, args, state, control, **kwargs):
        wall = time.time() - self._t0
        gen = TIMING["generate_s"] - self._g0
        judge = R.TIMING["judge_s"] - self._j0
        self.rows.append({"step": state.global_step, "wall_s": round(wall, 1),
                          "generate_s": round(gen, 1), "judge_s": round(judge, 1),
                          "other_s": round(wall - gen - judge, 1)})
        if state.global_step == 2:
            import torch
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
            mean = sum(r["wall_s"] for r in self.rows) / len(self.rows)
            print(f"after step 2: {mean:.1f} s/step (gen {gen:.0f} s, judge {judge:.0f} s this step), "
                  f"peak allocated {peak:.2f} GiB", flush=True)


def recording(reward_func):
    """Same function, same name, same behaviour; also keeps every completion it saw, tagged
    with the training step, so the report can show whole completions from chosen steps."""
    @functools.wraps(reward_func)
    def wrapped(prompts, completions, **kwargs):
        from reward import trl_rewards
        step = getattr(kwargs.get("trainer_state"), "global_step", None)
        result = reward_func(prompts, completions, **kwargs)
        if reward_func.__name__ == "route_reward":
            # Diagnostic: on a step whose task is not ask_question, the fraction of completions
            # that chose the correct route; NaN on ask_question steps. Logged through trl's own
            # log_metric hook, so it appears next to the built-in metrics.
            gold = json.loads(kwargs["task"][0])["route_answer"]
            if gold == "ask_question":
                non_ask = float("nan")
            else:
                non_ask = sum(d["route_score"] for d in trl_rewards.DETAILS) / len(trl_rewards.DETAILS)
            if "log_metric" in kwargs:
                kwargs["log_metric"]("non_ask_accuracy", non_ask)
            # One line per step for a live view while the run is going. Append-only, in the
            # run's output directory; nothing reads it back during training.
            if LIVE_PATH is not None:
                task = json.loads(kwargs["task"][0])
                with LIVE_PATH.open("a") as handle:
                    handle.write(json.dumps({
                        "step": (step or 0) + 1, "task_id": task["task_id"], "route_answer": gold,
                        "chosen": [d.get("route") for d in trl_rewards.DETAILS],
                        "route_reward": sum(d["route_score"] for d in trl_rewards.DETAILS) / len(trl_rewards.DETAILS),
                        "rubric_reward": sum((d.get("rubric_score") or 0.0) for d in trl_rewards.DETAILS) / len(trl_rewards.DETAILS),
                        "non_ask_accuracy": None if gold == "ask_question" else non_ask,
                    }) + "\n")
            # route_reward is the scoring pass; its stored details carry every rubric verdict,
            # so keep them here and the per-item pass rates need no rescoring later.
            for completion, detail in zip(completions, trl_rewards.DETAILS):
                RAW.append({"step": step, "text": completion_text(completion),
                            "parsed": detail["parsed"], "route": detail.get("route"),
                            "reasoning": detail.get("reasoning"), "facts": detail.get("facts"),
                            "route_score": detail["route_score"],
                            "rubric_score": detail.get("rubric_score"),
                            "items": [{"id": i["id"], "passed": i["passed"], "answer": i["answer"],
                                       "reasoning": i.get("reasoning")} for i in detail["items"]]})
        return result
    return wrapped


def main():
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import build_dataset
    from reward import trl_rewards

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out/grpo_run1")
    parser.add_argument("--model", default=MODEL, help="policy model id (default Qwen2.5-7B-Instruct)")
    parser.add_argument("--bf16", action="store_true",
                        help="train in bfloat16 instead of float16 (A100/H100; the Quadro RTX 8000 has no bf16)")
    parser.add_argument("--batch", type=int, default=2, help="per_device_train_batch_size")
    parser.add_argument("--accum", type=int, default=4, help="gradient_accumulation_steps")
    parser.add_argument("--gens", type=int, default=8, help="num_generations (completions per prompt)")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient (0 = off)")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7run7", "v8"], default="v1",
                        help="v1 = route names only (run1-run6); v2 = names plus one-line definitions; "
                             "v3 = rule-ordered definitions with the reply each route requires; "
                             "v4 = v3 plus a write-down-the-facts step first (reads data/kb, not kb_v1); "
                             "v5 = v4 with the facts as six answer fields instead (no --reasoning); "
                             "v6 = v5 over the trimmed, reordered KB in data/kb_v6; "
                             "v7 = v6 with derived account flags in the user turn, answer = device, incident, route, reply")
    parser.add_argument("--reasoning", action="store_true",
                        help="ask the policy for a leading \"reasoning\" field (recorded, never scored)")
    parser.add_argument("--steps", type=int, default=10, help="max_steps")
    parser.add_argument("--save-steps", type=int, default=5, help="save_steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="passed explicitly so seed-matched comparisons are possible")
    parser.add_argument("--tasks", default="data/v3_kb_definitions/tasks_train.jsonl",
                        help="task file the dataset is built from")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    global LIVE_PATH, PROMPT_VERSION, REASONING
    LIVE_PATH = out / "live_steps.jsonl"
    PROMPT_VERSION = args.prompt_version
    REASONING = args.reasoning
    print(f"prompt version: {PROMPT_VERSION}   reasoning field: {REASONING}", flush=True)
    time_generation()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"policy model: {args.model}   dtype: {dtype}   judge: {J.judge_info()}", flush=True)
    pad_note = "pad token present"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_note = "pad token was missing; set to eos"
    print(f"tokenizer: padding_side={tokenizer.padding_side!r}  pad_token={tokenizer.pad_token!r}  "
          f"({pad_note})", flush=True)

    dataset = build_dataset.build([ROOT / args.tasks], tokenizer, PROMPT_VERSION, REASONING)
    # Guard against the prompt silently falling back to another version: the rendered prompt
    # must contain this run's routes block and answer section, and its length is printed.
    first = dataset[0]["prompt"]
    assert routes_block(PROMPT_VERSION) in first, "dataset prompt does not carry the requested routes block"
    assert answer_section(PROMPT_VERSION, REASONING) in first, "dataset prompt does not carry the requested answer section"
    print(f"dataset prompt check: version {PROMPT_VERSION} confirmed in rendered prompt; "
          f"row 0 is {len(tokenizer(first)['input_ids'])} tokens", flush=True)
    print(f"dataset: {len(dataset)} rows, columns {dataset.column_names}", flush=True)

    config = GRPOConfig(
        output_dir=str(out),
        fp16=not args.bf16,
        bf16=args.bf16,
        model_init_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
        gradient_checkpointing=True,
        num_generations=args.gens,
        max_completion_length=256,
        temperature=1.0,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_steps=args.steps,
        seed=args.seed,
        learning_rate=1e-5,
        beta=args.beta,
        scale_rewards="none",
        mask_truncated_completions=True,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_steps=args.save_steps,
        report_to="none",
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    timing = StepTiming()
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[recording(trl_rewards.route_reward), recording(trl_rewards.rubric_reward)],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[timing],
    )
    print(f"trainer tokenizer: padding_side={trainer.processing_class.padding_side!r}", flush=True)
    print(f"model dtype {next(trainer.model.parameters()).dtype}, device {next(trainer.model.parameters()).device}",
          flush=True)
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    wall = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2**30

    summary = {"wall_s": round(wall, 1), "peak_gpu_gib": round(peak, 2),
               "config": {"model": args.model, "dtype": str(dtype), "prompt_version": PROMPT_VERSION,
                          "reasoning": REASONING, "tasks": args.tasks, "num_generations": args.gens,
                          "beta": args.beta, "seed": args.seed, "max_steps": args.steps},
               "generate": TIMING,
               "judge": {**J.judge_info(), **{k: v for k, v in R.TIMING.items() if k != "judge_each_s"}},
               "judge_failures": J.judge_failures(),
               "judge_each_s": R.TIMING["judge_each_s"],
               "steps": timing.rows, "log_history": trainer.state.log_history,
               "repairs": R.REPAIRS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "raw_completions.json").write_text(json.dumps(RAW, indent=1, ensure_ascii=False))
    print(f"\nDONE wall {wall/60:.1f} min | peak GPU {peak:.2f} GiB | generate {TIMING['generate_s']/60:.1f} min "
          f"| judge {R.TIMING['judge_s']/60:.1f} min ({R.TIMING['judge_calls']} calls) | completions {len(RAW)}",
          flush=True)


if __name__ == "__main__":
    main()
