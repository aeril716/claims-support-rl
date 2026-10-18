# claims-support-rl

This project trains an open model (Qwen2.5-14B-Instruct) to handle one turn of a
device-insurance support chat. The model reads a customer message, picks one of
seven routes (for example: file a claim, ask for missing information, or send the
customer to the manufacturer), and writes a reply. The reward checks the route
with code and the reply with an LLM judge.

I compared four setups on a held-out test set of 40 tasks: the base model,
RL only (GRPO), SFT only, and SFT followed by RL.

## Key findings

- SFT moved the model the most. Route accuracy went from 23/40 (base) to 35/40.
- RL on top of SFT added a little more: 36/40. The average over 16 samples per
  task went from 526/640 to 557/640 (noise is about ±10).
- RL made correct answers more likely. It did not make new
  ones possible.
- RL alone did almost nothing: 24/40 against the base model's 23/40. A follow-up
  run (RL from the base model, on the same training set as SFT+RL) also stayed
  at base level (23/40). So the gain came from starting at the SFT model, not
  from the training set.
- RL only learns from tasks where the model is sometimes right and sometimes
  wrong. If all samples for a task get the same reward, there is nothing to learn.
  After SFT, the old RL training set was almost all solved, so it had to be
  chosen again.

## The task

### Input
The model gets one customer message and the customer's account.

    Customer message: Watch won't stay charged more than an hour.
    Casing looks puffed up on the back.

    Account: {"enrolled_days_ago": 600, "manufacturer_warranty_active": false,
              "claims_last_12m": 0, "device_price_tier": "250-399"}

The system prompt holds the full knowledge base (KB): 11 short policy documents
written from Asurion's public FAQ pages. It is not a real policy.

### Output
One JSON object. The model first writes down the facts it read, then picks a
route, then writes the reply. This is the final model (SFT+RL) on test task h010:

    {"device": "watch",
     "incident": "battery failure (swollen battery)",
     "enrolled_days_ago": 600,
     "inside_waiting_period": "no",
     "claim_limit_reached": "no",
     "reasoning": "Watch with a swollen battery is a battery failure, which the
       plan covers from day 31 regardless of manufacturer warranty status;
       enrolled 600 days ago so well past the waiting period.",
     "route": "file_claim",
     "reply": "This is covered under your plan as a battery failure. The
       deductible for a watch claim is a flat $49. Start your claim online
       (about 10 minutes) with your order number and a description of the
       issue. Photos of the swelling may be requested."}

### Routes
| route | when |
|---|---|
| ask_question | the customer did not say the device or what happened |
| explain_not_covered | the plan does not cover it (e.g. loss or theft of a non-phone) |
| tech_support | a software problem; not a claim |
| refer_to_manufacturer | a defect while the manufacturer warranty is still active |
| escalate | the device is at its claim limit; a person has to look at past claims |
| explain_waiting_period | the plan is less than 31 days old |
| file_claim | none of the above |

The correct route comes from a fixed rule list, checked top to bottom in the
order above. The first rule that matches wins. The order matters: a phone at
its claim limit still goes to ask_question if the customer has not said what
happened, because a person cannot pick up the case without that.

### Reward
Two reward functions, added together:
- **Route reward** (code): 1 if the route matches the answer key, 0 if not.
  Output that cannot be read as JSON gets 0.
- **Rubric reward** (LLM judge): each task has 8-11 pass/fail questions about
  the reply, such as "Does the reply ask what happened?" or "Does the reply
  state the deductible amount?". One question is one judge call. The score is
  the fraction that pass.

The judge was Claude Haiku 4.5 for every run reported here.

### Data
All tasks are synthetic. A script picks a device, an incident, an account, a
tone, and which facts the customer mentions. Then an LLM (Claude, GPT, or
Gemini) writes the customer message. The route is computed from those choices,
so the answer key never depends on the LLM.

| split | tasks | used for |
|---|---|---|
| train | 307 | RL training sets were picked from here |
| SFT tasks | 103 | new tasks for teacher answers, separate from train |
| validation | 40 | every design decision during development |
| test_v2 | 40 | held-out, built later and harder; scored once at the end |

## Results

All numbers are on test_v2: 40 held-out tasks, built after development ended
and scored once. Base model: Qwen2.5-14B-Instruct. Judge: Claude Haiku 4.5.

### Four setups

| setup | route, greedy | rubric (mean) | correct of 16 samples × 40 tasks | pass@16 |
|---|---|---|---|---|
| base | 23/40 | 0.776 | 385/640 | 32/40 |
| RL only (run10) | 24/40 | 0.775 | 393/640 | 33/40 |
| SFT only | 35/40 | 0.868 | 526/640 | 38/40 |
| SFT + RL (run12) | 36/40 | 0.904 | 557/640 | 38/40 |

- **greedy**: one answer per task, no randomness.
- **correct of 16 samples**: each task sampled 16 times at temperature 1.0.
  Counts how often the model picks the right route, not just whether it can.
- **pass@16**: tasks where at least one of the 16 samples is right.
  This is the ceiling: what the model can do at all.

SFT raised the ceiling: pass@16 went from 32/40 to 38/40. RL did not, from
either starting point. On top of SFT, RL made the model pick right answers
more often, but did not add new ones.

### Where the gain came from

RL was run from two starting points, on two training sets. Each training set
was picked for one starting model: tasks where that model was right on some
samples and wrong on others.

| start \ RL training set | picked for base (103 tasks) | picked for SFT (68 tasks) |
|---|---|---|
| base | run10: 24/40 | run13: 23/40* |
| SFT | run11: almost no signal (see below) | run12: 36/40 |

*run13 was trained twice with the same settings. The first adapter was lost
(saved outside Drive) and scored 24/40, pass@16 32/40. The number above is
the saved second adapter.

run13 uses the same training set as run12 but starts from the base model,
and it stays at base level. So the gain in run12 came from starting at SFT,
not from the training set.

run11 shows the other side. Its training set was picked for the base model,
and the SFT model already got almost all of it right: mean reward 1.81 of 2
in the first 5 steps, 1.90 in the last 5. Groups had little contrast, so the
updates were small (gradient norm 0.04-0.06, against 0.18-0.37 for run13).
After SFT, the RL training set has to be picked again.

### One task, before and after

Task h001. The customer names the phone but not what happened. The phone is
at its claim limit. The right move is to ask what happened first.

| model | route | reply |
|---|---|---|
| base | escalate ✗ | Your case is being handled by a specialist who will contact you shortly. |
| SFT + RL | ask_question ✓ | Sorry to hear your phone is having trouble! To help you, could you describe exactly what's happening with the device? Is it a software issue like an app crash, or a hardware problem like a cracked screen or liquid damage? |

The base model wrote "problem" as the incident and then acted on the claim
limit. run10 (RL only) gave the exact same output as base, byte for byte.

### A note on the validation set

On the 40 validation tasks used during development, SFT and SFT+RL both
scored 40/40, so the gain above was not visible there. That is why test_v2
was built: harder cases, chosen by task design, not by model scores.

## How this project evolved

### 1. 7B model: RL collapsed (run3-run7)
Started with Qwen2.5-7B-Instruct. run3 collapsed to ask_question. In run4-run6,
four routes (refer_to_manufacturer, explain_waiting_period,
explain_not_covered, escalate) stayed at 0 on every step. The base model never
picked them, so **every sample in a group got the same reward and there was
nothing to learn from.** run7 woke three of them up, but the model then leaned
toward refer_to_manufacturer. Changing the KL penalty did not help; KL stayed
near 0.

Eight prompt versions were tried to see what the base model could do. The 7B
model kept deciding from one visible clue and could not check two conditions
together (type of incident AND date or claim count). Moved to 14B.

Along the way, a labeling bug turned up: the rule order checked the waiting
period too early, above the not-covered, manufacturer-warranty, and tech
support rules. The KB settles all three of those regardless of the waiting
period. 19 tasks were relabeled.

### 2. 14B model: RL did not move (run8, run9)
Validation route accuracy went 29→28 and 31→32. The trainer's
zero-variance counter said groups were fine, but it was reading the summed
reward. The rubric part always varies a little, which hid groups where every
sample picked the same route.

Sampling each training task 16 times (pass@k buckets) showed the problem:
of 35 training tasks, 12 were never right and 12 were always right. Only 11
were in between, and only 5 near the middle (4-12 of 16). GRPO only learns
from tasks in between.

### 3. Better prompt, better training set (run10)
Adding a short "reasoning" field before the route raised the base model to
35/40 on validation. The training set was rebuilt from the full train split:
103 tasks where the base model was right on some samples and wrong on others.

run10 learned on its own training set (226→254 correct of 412 samples) but
validation stayed at 35/40. The tasks it still missed were ones the base
model never got right, not even once in 16 tries. RL cannot reinforce an
answer the model never produces.

### 4. SFT on teacher examples (SFT, run11)
To give the model answers it could not find itself, Claude Sonnet 5 wrote
4 example answers for each of 103 new tasks. Only answers with the correct
route and correct fact fields were kept (174 of 412). SFT on these reached
40/40 on validation.

run11 then ran RL from the SFT model on the old training set. The SFT model
already got almost all of it right, so there was little to learn.

### 5. Re-pick the training set for SFT (run12)
Sampled the SFT model on the train split again and kept the 68 mixed tasks.
run12 ran RL from SFT on these. On validation, SFT and run12 were both
40/40. The validation set was saturated and could not tell them apart.

### 6. A new held-out test (test_v2) and one more run (run13)
The validation set had been used for every decision, so it was no longer a
fair test. test_v2 was built with 40 new tasks, harder by design (more hidden
causes, more claim-limit and warranty edge cases), and scored once.

On test_v2 the differences showed up (see Results). run13 filled the last
cell of the 2×2: RL from the base model on run12's training set. It stayed
at base level, which showed the run12 gain came from the SFT start.

## Lessons learned

### Data
- **Check labels against the source documents, not against the model.**
  The rule order checked the waiting period before three rules the KB settles
  regardless of it, so 19 tasks had the wrong answer key. A correct route is
  not proof of a correct label either: one mislabeled task still reached the
  right route by a different rule path.
- **Examples in a writer prompt leak their words into the data.** With
  examples, the same nouns showed up everywhere. Without them, the writer
  copied the label word. KB definitions plus "not this" boundaries between
  similar incidents fixed both.

### Reward
- **Read each reward's spread on its own.** The trainer reported no
  zero-variance groups, but that was on the summed reward. The rubric part
  always varies a little and hid groups where every sample picked the same
  route.
- **A reward only checks what it asks about.** One answer picked the right
  route and then told the customer the opposite. Nothing caught it until a
  route-vs-reply consistency question was added.
- **Batching judge questions changes the verdicts.** Asking all rubric
  questions in one call was only 1.4x faster and changed 18% of verdicts.
  One question per call.

### Training
- **GRPO learns only from tasks the model sometimes gets right.** Sample each
  training task several times and bucket it (pass@k buckets). Never right:
  RL cannot help; it needs a prompt change or SFT. Sometimes right: RL can
  help. Always right: nothing to learn. Do this on the training set, not the
  test set; the test set pointed at the wrong route once.
- **RL cannot reinforce an answer the model never gives.** SFT on filtered
  teacher answers fixed the tasks RL could not. After SFT, the RL training set
  had to be picked again for the new model.

### Evaluation
- **A set used for decisions is a validation set.** The original test set was
  used for every prompt, rubric, and run decision. When both final models hit
  40/40 on it, a new held-out set (test_v2) was built and scored once.
- **Change one thing at a time, and use a 2×2 to separate two causes**
  (one-factor-at-a-time, factorial design). run12 changed both the starting
  model and the training set. run13 filled the missing cell and showed which
  one mattered.

## Setup and how to run

### Hardware
- **14B training and eval:** Google Colab, A100 80 GB, bf16. Peak memory for
  a 14B GRPO run was about 37 GB.
- **Data generation, 7B runs, some evals:** a home server with a Quadro RTX
  8000 (48 GB). This card has no bf16 or FlashAttention 2, so it runs fp16
  with SDPA attention.

### Libraries
transformers 5.17.0, trl 1.13.0, peft 0.20.0, anthropic 1.4.0
(torch 2.11 on the server; Colab's default torch).

### Model and adapter
Qwen2.5-14B-Instruct with a LoRA adapter: r=16, alpha=32, dropout 0, on all
attention and MLP projections (q, k, v, o, gate, up, down).

### Key settings
| | GRPO (run10-run13) | SFT |
|---|---|---|
| learning rate | 1e-5 | 1e-4 |
| batch | per-device 2, gradient accumulation 4 | per-device 1, gradient accumulation 8 |
| length | 52 steps (2 prompts per step; one pass over 103 tasks) | 3 epochs |
| samples per prompt | 8 | - |
| temperature | 1.0 | - |
| max new tokens | 256 | - |
| KL penalty (beta) | 0 | - |
| reward scaling | none (rewards already 0-1 each) | - |
| seed | 42 | 0 |

### Files
| file | what it is |
|---|---|
| `data/kb_v6/` | knowledge base used in the final prompt (v8) |
| `data/v3_kb_definitions/tasks_train.jsonl` | 307 training tasks |
| `data/v3_kb_definitions/tasks_test.jsonl` | 40 validation tasks (named "test" for historical reasons) |
| `data/v3_kb_definitions/tasks_test_v2.jsonl` | 40 held-out test tasks |
| `data/v3_kb_definitions/tasks_train_mixed.jsonl` | RL training set picked for the base model (103) |
| `data/v3_kb_definitions/tasks_train_mixed_sft.jsonl` | RL training set picked for the SFT model (68) |
| `data/sft/train_v8reason_r3_new103.jsonl` | filtered teacher answers for SFT (174) |
| `notebooks/grpo_colab.ipynb` | the Colab notebook as run, outputs kept |

### How to run
API keys go in the environment: `ANTHROPIC_API_KEY` for the judge and the
teacher. Every command below shares the same model and prompt options:

    COMMON="--model Qwen/Qwen2.5-14B-Instruct --bf16 --prompt-version v8 --reasoning"

1. Find the mixed tasks for a model (pass@k on the train split):

       python eval/pass_at_k.py $COMMON --k 4 --temperature 1.0 --max-new-tokens 768 \
         --tasks data/v3_kb_definitions/tasks_train.jsonl --out passk_train.jsonl

2. SFT on teacher answers:

       python train/sft.py --bf16 \
         --data data/sft/train_v8reason_r3_new103.jsonl --out sft_adapter

3. GRPO (drop `--init-adapter` to start from the base model):

       JUDGE_BACKEND=anthropic python train/train_grpo.py $COMMON \
         --init-adapter sft_adapter \
         --tasks data/v3_kb_definitions/tasks_train_mixed_sft.jsonl \
         --batch 2 --accum 4 --gens 8 --beta 0 --steps 52 --save-steps 26 --seed 42 \
         --out grpo_adapter

4. Evaluate (greedy + rubric, then 16 samples per task):

       JUDGE_BACKEND=anthropic python eval/before_after.py $COMMON --score \
         --adapter grpo_adapter/checkpoint-52 \
         --tasks data/v3_kb_definitions/tasks_test_v2.jsonl --out eval_out

       python eval/pass_at_k.py $COMMON --k 16 --temperature 1.0 --max-new-tokens 768 \
         --adapter grpo_adapter/checkpoint-52 \
         --tasks data/v3_kb_definitions/tasks_test_v2.jsonl --out passk_test.jsonl

## Limitations

