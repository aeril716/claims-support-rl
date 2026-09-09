# support-rl-env

A small RL post-training experiment: train a small open model to write one good
customer-support turn for a device-insurance company, using a rubric-based reward.
Built as an interview project. The design lives here; do not redesign it.

## Working style (read first)

- The user designs. Claude Code types. Do not build ahead of what the user asks for.
- Work one file at a time. Finish, run, show output, stop.
- Explain what a file does as if the user has no coding background. Name the concept, then show it.
- When showing outputs or data structures, show the real, full thing. No `...` truncation.
- Do not swap tools or change the folder layout without asking.
- Everything in the repo is in English (code, comments, data, docs). Chat can be in Korean.
- Keep code plain and readable over clever.

## What we are building (the "environment")

Following the Asurion practicum framing, an RL environment has three parts:

1. **Task** — one short instruction: "Read the customer message and reply as a support agent. One turn only."
2. **World** — a closed set of knowledge-base (KB) documents the model can look at. ~15 markdown files.
   Relevant facts are mixed in with irrelevant docs.
3. **Rubric** — a list of atomic pass/fail checks per task.
   `reward = 0.5 * route_score + 0.5 * (rubric_passed / len(rubric))`, a number in [0, 1].
   This rubric IS the reward function.

The model being trained plays the role of a small "specialist" (like Asurion's contributors),
not a general assistant. Single turn only for now; no multi-turn.

## Domain

Device protection insurance (phones, laptops, tablets, watches).
Ten perils, defined in `data/kb/04_perils.md`.

Each task must let the reward check four things the real system cares about:
what the customer said, which peril it is, what info is still missing, and whether the reply
follows the KB (e.g. suggest charging before force-restart if the KB says so).

## Data generation (random component mix, not free-form LLM generation)

Components are fixed lists; a script samples one value from each and only then asks an LLM
to write the customer sentence in that style. This keeps the peril distribution balanced.

```python
perils           = ["crack", "drop", "liquid", "surge", "battery", "malfunction", "wear", "environment", "loss", "theft"]
devices          = ["phone", "laptop", "tablet", "watch"]
tones            = ["terse", "frustrated", "normal", "rambling"]
typos            = [True, False]
volunteered_info = ["nothing", "device_only", "device_and_date", "device_date_and_cause"]
```

- `terse` = short fragments ("screen broke. covered?"). `rambling` = long, includes unnecessary detail.
- `account` is always present in every task, independent of `volunteered_info`: it is what the
  system knows about the customer, not what the customer says.

Target: 200 tasks, 20 per peril. Customer messages should read like real people
(typos, missing grammar, emotion), e.g. "It's broken. Can't see nothing. The screen is black."
Do not always state the peril plainly; some messages should require inferring it.

- Customer sentences are written by the Claude API (anthropic SDK in .venv, key in .env, both gitignored).
- Sentence style: normal support-ticket tone, starts like "Hi, ..." — no dramatic openers.
  This applies to `normal`, `frustrated`, and `rambling` only; `terse` is the exception.

`known_facts` and `missing_info` are filled by code from the `volunteered_info` value, not by the LLM.
Rubric items come from a per-peril template plus items that depend on `volunteered_info`
(e.g. "does not re-ask stated info", "asks for the missing X").

### Task record format (`data/tasks.jsonl`, one JSON object per line)

```json
{"task_id": "t007",
 "device": "laptop",
 "peril": "liquid",
 "tone": "normal",
 "typos": false,
 "volunteered_info": "device_only",
 "account": {"enrolled_days_ago": 95, "claims_last_12m": 0, "device_price_tier": "700+"},
 "goal": "how_much_will_it_cost",
 "customer": "Hi, my laptop's keyboard has stopped working and there is some sticky residue around the keys. I'm on the protection plan. How much would it cost to get this fixed?",
 "known_facts": {"device": "laptop", "event_date": "unknown", "cause": "unknown"},
 "missing_info": ["event_date", "cause"],
 "route_answer": "file_claim",
 "rubric": ["states the laptop deductible amount",
            "does not tell the customer to contact the manufacturer",
            "does not re-ask which device it is",
            "asks what happened to the laptop or when it happened",
            "asks at most one question",
            "60 words or fewer"]}
```

## Model output

The policy model must output one JSON object: `{"route": <one of ROUTES>, "reply": <text to customer>}`.

`ROUTES = ["file_claim", "tech_support", "refer_to_manufacturer", "deny_waiting_period", "deny_claim_limit", "deny_not_covered", "escalate"]`

Traps are not a separate list: they follow from `account` and `peril`
(e.g. enrolled_days_ago < 31 → deny_waiting_period; loss on a non-phone → deny_not_covered).

## Reward

- Code checks (deterministic): word count, number of questions.
- Judge checks: every other rubric item is asked to a judge model as a single yes/no question.
  One rubric item = one judge call = one pass/fail. Never ask the judge for an overall score.
- Route check (deterministic): 1 if output route == route_answer, else 0.
- Reply checks: code checks + judge yes/no per rubric item as before.
- Combined as in the formula under "What we are building". Weights may change.

Judge: `Qwen2.5-14B-Instruct` served locally on the user's Mac via Ollama (M5 Pro, 48 GB).
Training runs on Colab; the Colab notebook calls the Mac judge over HTTP through a tunnel
(ngrok or similar). This wiring is not built yet.

<!-- Copy this block verbatim into README.md when the README is written. -->
## Rubric rule: peril AND volunteered_info together
A rubric item must never reward a claim the customer gave no evidence for.
Rubric items come from two inputs, not one:
- `peril` decides which coverage facts apply.
- `volunteered_info` decides which of those facts the model is allowed to state.
Example: peril = liquid, volunteered_info = device_only. The customer did not say what happened.
- Wrong rubric item: "states that liquid damage is covered." A reply that guesses "liquid"
  would score higher than one that asks, so the model learns to guess causes.
- Right rubric item: "asks what happened to the device." The coverage item is only added
  when volunteered_info includes the cause.
This is the most common way a rubric-based reward gets gamed, so the rubric template code
must branch on both fields.

## Training

- Policy model: `Qwen2.5-0.5B-Instruct` first; move to 1.5B only if 0.5B clearly works.
- Algorithm: GRPO via the `trl` library. Generate ~8 replies per task, score each with the reward, update.
- Hardware: Google Colab, A100 40GB (paid compute units; budget is fine). Training does NOT run on the Mac.
- Evaluate on a held-out split (e.g. 40 of the 200 tasks) that training never sees.

## Repo layout

```
support-rl-env/
  README.md              what this is, how reward works, results
  data/
    kb/                  ~15 knowledge-base docs (.md)
    components.py        the fixed component lists above
    generate_tasks.py    samples components, calls an LLM for the sentence, writes tasks.jsonl
    tasks.jsonl          200 task records
  reward/
    rubric_checks.py     code checks: word count, question count
    judge.py             yes/no judge call per rubric item
    reward.py            combines both into one number
  train/
    train_grpo.py        training script imported by the Colab notebook
    grpo_colab.ipynb
  eval/
    before_after.py      score base vs trained model on held-out tasks
  results/
    reward_hacks.md      what the model did to game the rubric, and how the rubric was changed
```

`README.md` and `results/reward_hacks.md` are the files a reviewer will actually read.

## Not decided yet

- Trap ratio in the 200 tasks.
- How the 40 held-out tasks are chosen (random vs stratified by device × peril).
- Reward weight between route and reply.
- Tunnel setup between Colab and the Mac judge.

## Isolation
- `data/judge_only/` is read ONLY by `reward/judge.py` and `data/generate_tasks.py`.
- Nothing under `train/` or in any policy-model prompt may read or reference `data/judge_only/`.
- If a task needs scenario answers inside the policy prompt, stop and ask; do not copy from judge_only.
