# claims-support-rl

## What this is

A small RL environment that trains a support agent to route and answer device-insurance
messages. The agent reads one customer message, decides how the case should be routed, and
writes one reply. A per-task rubric of atomic pass/fail checks is the reward function.

The design lives in [CLAUDE.md](CLAUDE.md).

## Data

Synthetic. The knowledge base under `data/kb/` is written from Asurion's public FAQ pages but
is not a real policy document — deductibles and limits were unified across device types for
this experiment.

### Scope note on the device question

This is modeled as a multi-device plan, where one account can cover a phone, laptop, tablet
and watch. The system knows the plan status (enrollment age, claims used, price tier) but not
which device the customer means, so asking which device is a valid step.

In Asurion's real flow the customer picks the device from their registered list, then supplies
the model, incident date and description.

### Scope note on single-turn tasks

The KB line about a software problem turning into a malfunction claim was removed. It only has
meaning across turns, and every task here is a single turn with one fixed peril. It goes back
in if the project moves to multi-turn.

## How to run

TBD

## Results

TBD

## Training setup

Hardware: one Quadro RTX 8000, 48 GB, compute capability sm_75; 125 GB RAM; 48 cores.
Libraries: torch 2.11.0+cu128, transformers 5.17.0, trl 1.13.0, peft 0.20.0.

Policy: Qwen2.5-7B-Instruct with a LoRA adapter (r=16, alpha=32, dropout 0, on q/k/v/o and
gate/up/down projections; 40.4M trainable parameters, 0.53% of the model). Reward: the two
functions in `reward/trl_rewards.py`, route first, rubric second. Data: `tasks_train.jsonl`,
built by `data/build_dataset.py`.

| setting | value | why, when not the library default |
|---|---|---|
| fp16 | True | sm_75 has no bf16 support; fp16 is the only half-precision option on this card |
| model_init_kwargs | dtype fp16, attn sdpa | flash-attn requires sm_80 or newer; sdpa is the fastest attention available on sm_75 |
| gradient_checkpointing | True | 7B weights plus 8 sequences of ~2.9k-token prompts do not fit activations otherwise |
| num_generations | 8 | library default |
| max_completion_length | 256 | measured maximum under training-style sampling was 117 tokens; 256 leaves headroom without paying for the 512 default |
| temperature | 1.0 | library default |
| per_device_train_batch_size | 2 | memory: 2 sequences per micro-step |
| gradient_accumulation_steps | 4 | 2 × 4 = 8, one full rollout group per optimizer step |
| max_steps | 10 | run1 plumbing-check value; real runs use 35 (sub35) or 52 (mixed-103) |
| learning_rate | 1e-5 | LoRA rate; the 1e-6 default is for full fine-tuning |
| beta | 0.0 | no KL term, so no reference model in memory; drift is watched through the reward curves instead |
| scale_rewards | "none" | rewards are already in [0, 1]; dividing by the group std would inflate small differences into full-size updates |
| mask_truncated_completions | True | a completion cut at the cap has no closing brace, so it parses as nothing; its tokens should not be trained on |
| logging_steps | 1 | every step is visible in a 10-step run |
| log_completions, num_completions_to_print | True, 2 | two full completions printed per step, to read what the policy actually emits |
| save_steps | 5 | run1 plumbing-check value (checkpoints at steps 5 and 10); real runs save at the last step, 35 or 52 |
| report_to | "none" | no tracker |

Design decisions that shaped the reward and the data are recorded in the Lessons learned
section of `CLAUDE.md`: the judge and its `format` schema, the rejected batched judging, the
judge model swap, and the sampling and rubric rules. They are not repeated here.

## Lessons learned

- Sampling components independently produced account states that cannot exist.
  `claims_last_12m` was drawn from a fixed list, but the claim limit differs by device
  (3 for phones, 2 for everything else), so a tablet could be given 3 past claims. The fix was
  a sampling order: draw device first, then draw `claims_last_12m` within that device's limit.
  A component whose valid range depends on another has to be drawn after it.
- A route named `deny_claim_limit` made the chatbot the party that refuses a claim, which is
  not what this model does. Reaching the claim limit is not automatically a refusal: repeated
  failures on the same device may be a defective unit or a manufacturer warranty matter, and
  that call needs a human who can see what the earlier claims were. The route was removed and
  those cases now go to `escalate`, while terms the policy settles outright were renamed to
  `explain_waiting_period` and `explain_not_covered`.
- The claim-limit escalation rule was first placed above the rule that asks for missing
  information, on the reasoning that a case going to a human does not need the chatbot to ask
  anything. That was wrong: the human cannot pick the case up without knowing the device and
  the cause, and the chatbot already has a turn in which to collect them. Rules that gather
  information belong above rules that decide where a case goes.
- Putting a peril in the KB's not-covered table was enough to make one route unreachable.
  `software` sits in that table, so the not-covered rule matched it first and `tech_support`
  was never a gold answer, even though the KB says software problems are handled by tech
  support for free. Not covered by the claim process and not handled at all are two different
  things, and the rule order has to keep them apart.
- Claim limits: started by unifying all devices at 2 claims per 12 months, reverted to
  per-device limits (phone 3, others 2) to match the KB terms.
- Route rules: moved the claim-limit rule lower in the rule order, below the rules that decide
  whether the case is a plan claim at all, because a denied claim does not count toward the
  limit.
- RAG: deferred to stage 2. Stage 1 puts the full KB in the prompt, so that the effect of each
  component stays visible before retrieval is added.
- Generator models: used three frontier models (Claude, GPT, Gemini) for customer messages.
  One model alone did not give enough variation in tone. A fourth model was dropped for weak
  instruction-following.
- Examples in the writer prompt: examples leak their own vocabulary into the dataset (v1: the
  planted nouns everywhere, turned, somewhere, halfway, passcode all showed up), but removing
  them lets the writer parrot the label and drift off it (v2: 4 of 33 stated-peril messages
  used the code word, 2 of 33 described a different peril). Replaced examples with KB
  definitions plus explicit boundaries between confusable perils, which constrains meaning
  without constraining wording.
- Wrong labels do not always produce wrong routes. t015 was labeled environment but described
  rain (liquid). Tracing the rule chain showed both perils reach escalate, because at 600 days
  the device is outside the 365-day warranty window either way, and the claim-limit rule fires
  afterwards regardless of peril. Had the same task been under 365 days, environment would stop
  at refer_to_manufacturer and liquid would continue to escalate. So route agreement is not
  evidence that a label is correct, and label quality has to be checked directly.
- Some peril and visibility combinations are structurally impossible. For loss and theft the
  cause and the observable state are the same fact, so a hidden-cause message cannot be written
  without leaking (v2 t031: "I no longer have it in my possession"). These combinations are
  excluded at the combination step rather than filtered afterwards.
- Bad generations are discarded and regenerated rather than fixed by prompt iteration, with an
  explicit target distribution and a retry cap so that discarding does not quietly change the
  distribution. A rejected sentence is regenerated for the exact same combination, because the
  route is a deterministic function of the combination and resampling would change the answer
  key.
- A reward function can score a self-contradictory answer highly. One completion picked the
  correct route and then told the customer the opposite in the reply body; the rubric checked
  route correctness and reply quality separately, so nothing caught the contradiction. Added a
  consistency criterion between the two.
- Rubric criteria that never fire contribute nothing. In the first working run, 6 of 18
  criteria passed 0% of the time and one passed 100%. A criterion that returns the same verdict
  for every completion adds no variance to the advantage, which is what GRPO learns from.
- Repairing malformed output is not neutral. The reward is computed on the repaired text but
  applied to the tokens the model emitted, so a syntax error attached to correct content gets
  reinforced. Accepted deliberately here — format compliance is not what is being measured —
  and logged per repair with its route correctness so the effect can be watched rather than
  assumed away.
- Judge latency was not the bottleneck on this hardware. Over a 35 minute run, judge calls took
  10 minutes and the forward/backward pass took 23. At this rate one epoch over 320 tasks would
  take roughly 8 hours locally, which is what moved training to a CUDA box.
- GRPO on the 14B base did not move: run8 (prompt v5) went 29 to 28 of 40 on route
  accuracy, run9 (prompt v8) 31 to 32. The trainer's `frac_reward_zero_std` stayed at 0 for
  the whole of both runs, which looked like healthy groups. It was not: the rubric reward
  adds variance on top of the route reward, so the summed reward almost never has zero
  spread even when every completion in the group chose the same route. The last step of run9
  is the plain case, all 8 completions choosing `file_claim` on a watch at its claim limit,
  reward std above zero from the rubric alone. We now read the route-reward std per step on
  its own. The cause is GRPO's structure, not a bug: when all 8 completions in a group get
  the same reward the advantage is zero and nothing is learned, whether the group is all
  right or all wrong.
- We added `eval/pass_at_k.py`, which samples each task 16 times at the trainer's
  temperature (1.0) and buckets tasks by how many samples hit the gold route: 0/16, 1-3,
  4-12, 13-15, 16/16. On the sub35 training set under v8, 12 tasks sit at 0/16, all five
  escalate tasks among them, 12 at 16/16, and only 5 in the 4-12 bucket, so GRPO had contrast
  on roughly 10 of its 35 steps. The bucket also says which lever applies. A task at 0/16
  needs a prompt or data change, because there is nothing in the sampling distribution to
  reinforce. A task at 1-3 responds to more generations, a higher temperature, or more
  steps. A task at 4-12 should learn under the default setup, and 13 and above needs
  nothing. The warranty flag added between v5 and v8 is an example of the first lever:
  `refer_to_manufacturer` went from 25 of 640 test samples to 131, at the cost of
  `explain_not_covered` bleeding into refer.
- We first ran pass@k on the test split and predicted from it that run9 would improve
  `explain_not_covered`, because that route sat in the 4-12 bucket on test. It did not move.
  On the training set the same route has no mixed tasks (16, 13, 2, and 0 of 16); the mixed
  tasks there are `ask_question`, `explain_waiting_period`, and `file_claim`, and the one
  route that did move on test was `explain_waiting_period`. Diagnose on the distribution the
  policy actually trains on. The test split is a different sample of the same generator and
  can point at the wrong route.
- run9 adapter pass@k on sub35: Sampling the run9 adapter 16 times on sub35 against the
  base: of the five mixed tasks, t047 went 9->12 and t060 5->9; the other three stayed
  within ±1. Two 0/16 tasks became nonzero (t148 0->2, t024 0->1). pass@16 total 23->25,
  pass@1 19->19. With 16 samples the noise on a mid-range count is about ±2, so only the +3
  and +4 count. Learning happened where the base had contrast, but each task was seen about
  twice in 35 steps, so the gain was small; the bigger lever was reselecting the training
  set, which is what run10 does.

## Status

Design complete, data generation not built yet.
