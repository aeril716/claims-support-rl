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
Eleven perils, listed as `PERILS` in `data/components.py`. Those labels are ours, internal to
this project; `data/kb/04_perils.md` describes the same incident types in customer-facing
language and carries no such labels. Ten of them are covered perils. The eleventh, `software`,
corresponds to the app / OS / virus / account line in that file's not-covered list; it is
included because software problems are a common real inquiry, and without them `tech_support`
would never be the answer for any task.

Each task must let the reward check four things the real system cares about:
what the customer said, which peril it is, what info is still missing, and whether the reply
follows the KB (e.g. suggest charging before force-restart if the KB says so).

## Data generation (random component mix, not free-form LLM generation)

Components are fixed lists; a script samples one value from each and only then asks an LLM
to write the customer sentence in that style. This keeps the peril distribution balanced.

```python
perils           = ["crack", "drop", "liquid", "surge", "battery", "malfunction", "wear", "environment", "loss", "theft", "software"]
devices          = ["phone", "laptop", "tablet", "watch"]
tones            = ["terse", "frustrated", "normal", "rambling"]
typos            = [True, False]
customer_states  = [(["device", "peril"], 50),          # (stated facts, weight)
                    (["device"], 25),
                    (["peril"], 15),
                    ([], 10)]
```

- `terse` = short fragments ("screen broke. covered?"). `rambling` = long, includes unnecessary detail.
- `account` is what the system knows about the customer at the time of the message, not what
  the customer says. An account can have several devices enrolled, so the system does not know
  which device the message is about until the customer names it.
  - When `device` is null, `account` is an empty object `{}` — not a
    reduced set of fields, empty. Rule 1 settles those tasks before any other rule is read, so
    no account field takes part in the answer. Any field left in would only invite the model to
    decide instead of asking, and would leak the device category.
  - When `device` is stated but `peril` is null, `account` keeps every field. A real agent sees
    the account the moment the customer names the device, and the point of these tasks is that
    the model holds off and asks what happened even with a waiting-period or claim-limit number
    in front of it.

Target: 400 tasks. A cell is one customer_states outcome × one peril (4 × 11 = 44 cells).
Cell sizes differ because customer_states is weighted, so the split is proportional stratified:
the same fraction (80% / 10% / 10%) is taken from every cell, so train, val, and test all have
the same distribution as the whole set.
- `tasks_train.jsonl` (320): GRPO training.
- `tasks_val.jsonl` (40): for checking after a rubric change and re-run. May be looked at many times.
- `tasks_test.jsonl` (40): looked at once, at the end. The before/after numbers come from here.

Customer messages should read like real people
(typos, missing grammar, emotion), e.g. "It's broken. Can't see nothing. The screen is black."
Do not always state the peril plainly; some messages should require inferring it.

- Customer sentences are written by the Claude API (anthropic SDK in .venv, key in .env, both gitignored).
- Sentence style: normal support-ticket tone, starts like "Hi, ..." — no dramatic openers.
  This applies to `normal`, `frustrated`, and `rambling` only; `terse` is the exception.

A task record stores `device` and `peril` as `null` when the customer did not identify them.
`null` means the customer never said it, not a hidden truth the grader knows. Customers do not
follow an order: they may say what happened without naming the device ("it got wet, is that
covered?" leaves `device` null and `peril` set). Nothing downstream needs the unstated value:
route rule 1 settles those tasks before any other rule reads a field.

A field is null exactly when the customer's message does not contain that fact, and carries its
value exactly when the message does. The two must always agree: if the sentence says the device
got wet, `peril` is `"liquid"`, not null.

The generator draws a real device and a real peril in step 1, because the customer sentence
cannot be written without them. It then decides which of the two the customer states, and from
that point the two values are treated differently:
- The stated fact keeps its value. It is in the sentence, so it is in the record.
- The unstated fact is given to the sentence writer only so that the sentence can avoid naming
  it, and is then discarded. Only this one is discarded, and it must not reach the record, the
  prompt, or the grader.

Worked example: device = watch, peril = liquid, the customer states the peril only.
- The sentence writer is told the device is a watch so it can avoid naming it, and is told to
  state the liquid damage.
- The record stores `device: null`, `peril: "liquid"`.
- `watch` is discarded; `liquid` is kept.

Rubric items are assembled by code from the fields that are not null; see the assembly rules
under Reward.

### Distribution
Most components are sampled uniformly at random. There are two exceptions, and they are
different in kind:
- `customer_states` is a weighted list: every outcome carries a weight next to it.
  Uniform sampling over the four combinations would make 75% of tasks `ask_question`, so the
  model would mostly
  learn to ask back and real policy decisions would appear in only 25% of tasks; the weights
  bring `ask_question` to about 50%.
- `claims_last_12m` is not a list but a rule: its range depends on another component, the
  device, and within that range the draw is weighted toward 0. See Sampling order below.

There is no separate trap-ratio parameter. The intended distribution lives in the component
definitions themselves, so changing a distribution means editing a weighted list or the
sampling rule in components.py.

A "trap" is simply a task whose `route_answer` is not `file_claim` — a policy explanation, a
referral, an escalation, or a case that needs more information (`ask_question`). Traps are
never shown to the policy model; they follow from `account`, `peril`, and `device`.

### Sampling order
Some components constrain the range of others, so they are not all drawn at once:
- step 1: `device`, `enrolled_days_ago`, `peril`, `tone`, `typos`, `device_price_tier`, and
  which of `device` and `peril` the customer states
- step 2: `claims_last_12m`

`claims_last_12m` is drawn last because its range depends on the two fields above it.
This replaces the fixed list it used before.
- `claims_last_12m` is 0 when the device is inside the waiting period — see Account consistency.
- otherwise draw from 0 up to that device's limit (phone 0-3, laptop / tablet / watch 0-2)
- weight the draw so 0 comes up more often, keeping at-limit tasks a minority

What this does not exclude. Situations that are real but not covered must still be generated,
because they are the gold cases for several routes:
- watch + loss, tablet + theft → `explain_not_covered`
- device under one year old + malfunction / wear / environment → `refer_to_manufacturer`
- any peril + `enrolled_days_ago` < 31 → `explain_waiting_period`

Only impossible account states are excluded, never uncovered situations.

### Target distribution, validation, and refill
Cell counts do not come from the weights at run time. They are written out explicitly in
`data/target_distribution.py`: v1's 400-task distribution, with loss and theft removed from the
two hidden-peril columns (for those perils the cause and the observable state are the same
fact, so a hidden-cause message cannot be written without leaking) and their share spread over
the other nine. The weights above still describe the intent; the file is the number.

Every generated sentence passes three checks before it is accepted, and every rejection is
logged to `rejections.jsonl` next to the output with the task id, the check, the peril, and the
text:
- A, code: the peril code word or its KB label phrasing appears in the message, allowing for
  misspellings ("malfuctioning", "enviromental", "serge").
- B, judge: the message is not consistent with its peril label, given the KB definition of the
  label and of its nearest confusable perils.
- C, judge, hidden-peril tasks only: the cause can be determined from the message. A symptom
  ("won't turn on") passes; a cause ("I dropped it") fails.

A rejected sentence is rewritten for the exact same combination — device, account, peril, what
the customer stated — and only the writer runs again. The route is a deterministic function of
the combination, so resampling any part of it would change the answer key. Five attempts per
combination; a combination still short after that is reported with its cell and the checks it
failed, no split files are written, and nothing is substituted.

The writer prompt gives no example sentences. For a stated peril it gives the KB definition,
verbatim from `data/kb/04_perils.md`, the nearest confusable peril with its own KB definition as
"not this", and a ban on the code word and its label phrasing. All three readers of a
definition — the writer, check A, and the judge — get it from `data/peril_definitions.py`, which
reads the KB file rather than restating it.

### Account consistency
Account fields are sampled independently, so some combinations are impossible and must be
corrected after sampling:
- If `enrolled_days_ago` < 31, `claims_last_12m` is set to 0. Coverage does not start until
  day 31, so no claim can exist yet.

generate_tasks.py applies these corrections right after sampling, before computing route_answer.
Sampling components independently can produce other impossible combinations later, so
generate_tasks.py also ends with a check pass over the generated 400 tasks that counts and
prints any rule violations.

### route_answer rule
Code sets `route_answer` by checking the rules top to bottom. The first match wins and the
remaining rules are not evaluated. This is the order in `data/generate_tasks.py` and the one
the task files in the repo were labeled with (relabeled on 2026-09-13; the waiting-period rule
used to sit second).
1. `device` is null or `peril` is null → `ask_question`
   (there is no date field: the incident date is collected when the claim is actually filed,
   and `account.enrolled_days_ago` already fixes the timing)
2. peril the plan can handle neither as a claim nor through tech support
   (e.g. loss / theft on a non-phone) → `explain_not_covered`
   (that is the definition of the rule, not a list with an exception attached. Software
   falls outside it: `data/kb/13_data_and_software.md` states that software problems are
   not claims and are handled by tech support at no charge, so software has a route of its
   own rather than being a dead end)
3. software problem → `tech_support`
   (`data/kb/12_tech_support.md` and `data/kb/13_data_and_software.md` both say software
   problems are handled by tech support at no charge, and `03_enrollment_and_waiting_period.md`
   says tech support starts the day of enrollment with no waiting period. Not covered by the
   claim process and not handled at all are two different things)
4. peril is malfunction / wear / environment and the device is still inside the manufacturer's
   warranty (`enrolled_days_ago` < 365; warranty is 12 months from purchase and enrollment
   happens within 30 days of purchase, so enrolled_days_ago is a close proxy) → `refer_to_manufacturer`
   (`data/kb/10_manufacturer_warranty_interaction.md`: the manufacturer is responsible for
   defects while its warranty is active, regardless of plan coverage)
5. `claims_last_12m` at or over the limit for that device → `escalate`
   (phone 3, laptop / tablet / watch 2, per `data/kb/07_claim_limits.md`; read the limit from
   the task's device, never hardcode 3. The limit only applies to claims the plan would
   actually take. A peril the plan does not cover, or one the manufacturer is responsible for,
   never consumes a claim: `data/kb/07_claim_limits.md` states that a denied claim does not
   count toward the limit. So the rules that decide whether this is a plan claim at all are
   checked before the limit is read)
6. `enrolled_days_ago` < 31 → `explain_waiting_period`
   (only reached when the incident would otherwise be a covered claim: physical perils, and
   loss or theft on a phone. Software, warranty defects, and non-phone loss/theft have their
   own answers above and do not depend on the waiting period, which the KB confirms: tech
   support has no waiting period, the manufacturer handles warranty defects, and non-phone
   loss/theft is excluded permanently. Account consistency sets `claims_last_12m` to 0
   whenever `enrolled_days_ago` < 31, so rules 5 and 6 never both apply)
7. otherwise → `file_claim`

Why this order. Reaching the claim limit is not automatically a refusal: repeated failures on
the same device may be a defective unit or a manufacturer warranty matter, and deciding that
requires seeing what the earlier claims were, which the chatbot cannot do. So the case goes to
a human. But asking for the missing device and peril comes first, because a human cannot pick
the case up without them, and the chatbot can collect them in the turn it already has.
Collecting information comes before deciding where a case goes. The waiting period comes last
among the decisions because it only defers a claim the plan would take; every other rule
settles the case on grounds the waiting period does not change. What the reply asks and how it
is worded is not decided here; that is the rubric's job.

### Task record format (`data/tasks_*.jsonl`, one JSON object per line)

```json
{"task_id": "t142",
 "device": null,
 "peril": "loss",
 "tone": "frustrated",
 "typos": true,
 "account": {},
 "goal": "is_this_covered",
 "customer": "Hi, ive looked everywhere for it and its just gone, pretty sure it fell out of my bag at the gym. ive been paying for this plan for months, is it covered?",
 "route_answer": "ask_question",
 "rubric": [
   {"id": "common.word_count", "check": "code", "question": "Is the reply 60 words or fewer?", "expect": "yes"},
   {"id": "common.question_cap", "check": "code", "question": "Does the reply ask 2 questions or fewer?", "expect": "yes"},
   {"id": "common.no_reask", "check": "judge", "question": "Does the reply ask for information the customer already gave?", "expect": "no"},
   {"id": "common.on_topic", "check": "judge", "question": "Does the reply say anything unrelated to the customer's message?", "expect": "no"},
   {"id": "ask_question.asks_missing", "check": "judge", "question": "Does the reply ask for the information missing from the customer's message?", "expect": "yes"},
   {"id": "ask_question.no_assertion", "check": "judge", "question": "Does the reply assert something it cannot know yet?", "expect": "no"},
   {"id": "peril.loss", "check": "judge", "question": "Does the reply say that loss coverage applies to phones only?", "expect": "yes"},
   {"id": "missing.device", "check": "judge", "question": "Does the reply ask which device it is?", "expect": "yes"}
 ]}
```

The rubric above is the complete assembly for this combination, not a sample of it: four common
items, two from the gold route, one peril item, and one missing-information item. No exclusion
item is attached, because `device` is null. `account` is `{}` for the same reason. The device
the sentence was written around is not on the record at all.

Only two fields of a task record reach the model: `customer` and `account`. `route_answer`,
`rubric`, `device`, `peril` and `tone` never do. They are generator and scoring metadata and
stay out of the prompt.

## Model output

The policy model must output one JSON object: `{"route": <one of ROUTES>, "reply": <text to customer>}`.

`ROUTES = ["ask_question", "file_claim", "tech_support", "refer_to_manufacturer", "explain_waiting_period", "explain_not_covered", "escalate"]`

- `ask_question` = not enough information to decide yet; the correct reply asks for the
  missing detail instead of routing.
- The chatbot never approves or denies a claim. `explain_*` routes state a policy term that
  the documents settle outright and that has no exceptions. `escalate` is for cases where the
  outcome depends on facts only a human can look at.
- `escalate` was removed earlier on the grounds that a route which never appears as a gold
  answer only teaches the model to pick it at random. Route rule 2 makes it a gold answer,
  so that reason no longer holds.
- There is no `deny_claim_limit`. Reaching the limit is not automatically a refusal, so those
  cases go to `escalate`.

Traps are not a separate list: they follow from `account`, `peril`, and `device`
via the route_answer rule under Data generation
(e.g. enrolled_days_ago < 31 → explain_waiting_period; loss on a non-phone → explain_not_covered).

## Reward

- Code checks (deterministic): word count, number of questions.
- Judge checks: every other rubric item is asked to a judge model as a single yes/no question.
  One rubric item = one judge call = one pass/fail. Never ask the judge for an overall score.
- Route check (deterministic): 1 if output route == route_answer, else 0.
- Reply checks: code checks + judge yes/no per rubric item as before.
- Combined as in the formula under "What we are building". Weights may change.

Judge: `qwen3:30b-a3b`, served by Ollama on a separate machine on the local network at
`192.168.88.59:11434`, called through `/api/generate` with a `format` JSON schema that
constrains every reply to `{"reasoning": <string>, "verdict": "yes" | "no"}`. One rubric item
is one call. The judge was `Qwen2.5-32B-Instruct` on the Mac until the swap measured in Lessons
learned. Training runs on Colab; how the Colab notebook reaches the judge is not built yet.

<!-- Copy this block verbatim into README.md when the README is written. -->
## Rubric rule: a task never carries a fact the customer did not state
A rubric item must never reward a claim the customer gave no evidence for.
An earlier design kept the true peril on every task and relied on the rubric assembly to leave
it alone when the customer had not stated it. Storing `null` instead removes the possibility:
when the customer did not name the device or the peril, the record holds no value to leak, so
no rubric item, prompt field, or grader check can reach for one.
- Wrong rubric item: "states that liquid damage is covered", on a task where the customer only
  said the device stopped working. A reply that guesses "liquid" would score higher than one
  that asks, so the model learns to guess.
- Right rubric item: "asks what happened". The coverage item exists only when `peril` is not
  null.
This is the most common way a rubric-based reward gets gamed, so the fact never enters the task
in the first place.

## Rubric

A rubric is a list of atomic pass/fail items attached to one task. The fraction of items that
pass is the reply half of the reward. Items are concatenated from three bundles: common items
on every task, items fixed by the gold route, and items that vary per task.

### Item format (proposed)

Every item is one object with four fields:

```json
{"id": "escalate.no_reason",
 "check": "judge",
 "question": "Does the reply explain why the case is being handed to a person?",
 "expect": "no"}
```

- `question` is always phrased positively, as a yes/no question about the reply. Negative items
  are not written as negated text; they are written as a positive question with `expect: "no"`.
- `expect` is the answer that makes the item pass, `"yes"` or `"no"`. This is the expected
  direction. An item passes when the answer equals `expect`, so both directions score the same
  way and no item needs special handling.
- `check` is `"code"` or `"judge"`, and decides which half of `reward/` evaluates the item.
  `code` items go to `rubric_checks.py`, `judge` items to `judge.py` as one yes/no call each.
- `id` names the bundle and the item, so a reward hack can be traced to the item that paid for it.

### Common items (every task)

| question | check | expect |
|---|---|---|
| Is the reply 60 words or fewer? | code | yes |
| Does the reply ask 2 questions or fewer? | code | yes |
| Does the reply ask for information the customer already gave? | judge | no |
| Does the reply say anything unrelated to the customer's message? | judge | no |

Exception, carried over from the earlier assembly rules: when both `device` and `peril` are
null the customer has stated nothing, so the "already gave" item is not attached. An item that
always passes is free credit and inflates the reward.

One more judge item is added at scoring time to every task, after the stored rubric:

| question | check | expect |
|---|---|---|
| The agent chose the route R, which means M. Does the reply tell the customer something consistent with that, rather than the opposite? | judge | yes |

It is not stored on the task because it depends on the route the model chose, which does not
exist until the completion does. R is the chosen route and M its meaning from `ROUTE_MEANING`
in `reward/reward.py`. It was added after a completion chose `file_claim` correctly and then
told the customer the incident was not covered; route correctness and reply quality were judged
separately, so nothing caught the contradiction. A route that is not in `ROUTES` fails this
item without a judge call.

**Reading the model's output.** The policy must emit one JSON object. Output that is
recognisably that object with one of three syntax slips — an unquoted string value, a stray `]`,
a missing closing brace — is repaired and scored on the repaired text. Output that is not a JSON
object at all scores 0; the parser never scans loose text for a route name, since that would
let it invent an answer the model never committed to. Every repair is logged with the original
text and whether the recovered route was correct, because the advantage lands on the tokens the
model actually emitted, so a syntax slip on correct content gets reinforced with it. No format
penalty for now; the repair rate is watched across runs instead.

### Per-route items

Every item below is a `judge` item.

| route | question | expect |
|---|---|---|
| ask_question | Does the reply ask for the information missing from the customer's message? | yes |
| ask_question | Does the reply assert something it cannot know yet? | no |
| file_claim | Does the reply say the incident is covered? | yes |
| file_claim | Does the reply state the deductible amount? | yes |
| file_claim | Does the reply say how to start the claim? | yes |
| file_claim | Does the reply mention documentation that may be requested? | yes |
| escalate | Does the reply say the case is being handed to a person? | yes |
| escalate | Does the reply say the claim is denied? | no |
| escalate | Does the reply explain why the case is being handed to a person? | no |
| explain_waiting_period | Does the reply state the 31-day rule? | yes |
| explain_waiting_period | Does the reply say the plan itself is active? | yes |
| explain_waiting_period | Does the reply say when coverage begins? | yes |
| explain_not_covered | Does the reply say it is not covered? | yes |
| explain_not_covered | Does the reply ground that in the policy? | yes |
| explain_not_covered | Does the reply offer an alternative, if one exists? | yes |
| refer_to_manufacturer | Does the reply direct the customer to the manufacturer warranty? | yes |
| refer_to_manufacturer | Does the reply explain why? | yes |
| tech_support | Does the reply identify the problem as a software problem and not a claim? | yes |
| tech_support | Does the reply say tech support handles it at no charge? | yes |
| tech_support | Does the reply give a concrete step the customer can try? | yes |

### Per-task items

Four generators, all producing `judge` items.

**Peril.** One item naming the specific coverage for that peril. It is attached only when all
three of these hold:
- `peril` is not null. When the customer has not said what happened, the reply cannot be
  expected to name a coverage it has no way to know, and the record does not hold the value
  either.
- `device` is not null. Coverage detail differs by device, so it cannot be stated while the
  device is unknown.
- the gold route is `file_claim`. On any other route, stating coverage contradicts the route's
  own items: `explain_not_covered` already says it is not covered, `refer_to_manufacturer`
  sends the customer elsewhere, `escalate` must not explain anything, and `ask_question` must
  not assert what it cannot know yet.

There is no `software` row. The tech_support item "identifies the problem as a software problem
and not a claim" already covers it, and keeping both would pay twice for one sentence.

Exact phrasings:

| peril | phrasing |
|---|---|
| crack | cracked screen coverage |
| drop | accidental damage coverage |
| liquid | liquid damage coverage |
| surge | power surge coverage |
| battery | battery failure coverage |
| malfunction | mechanical failure coverage |
| wear | wear and tear coverage |
| environment | environmental damage coverage |
| loss | loss coverage, phones only |
| theft | theft coverage, phones only |

**Device and price.** The deductible figure the answer should state comes from
`device_price_tier` for phones (`data/kb/05_deductibles_phone.md`) and is flat for laptop,
tablet, and watch (`data/kb/06_deductibles_other_devices.md`).

A phone task has two amounts, because repair and replacement differ, and one item asks for
both: "Does the reply state both deductible amounts for this price tier, $X for repair and $Y
for replacement?" Not "X or Y" — that splits the judge's verdict and the item stops being
atomic. The reply states both because `data/kb/05_deductibles_phone.md` says the plan decides
between repair and replacement and the customer cannot choose, so neither amount is settled at
the time of the reply. Laptop, tablet and watch keep their single flat amount.

Model-specific pricing is out of scope. `data/kb/05_deductibles_phone.md` also carries a flat
$29 screen repair for eligible models, but a task has no model field, so that line can never be
applied and must not appear in any rubric item. `model` is not a component.

Loss and theft on a phone are the one case where no figure can be named, because the same file
tiers those by model. The item asks instead: "Does the reply say the exact deductible depends on
the phone model and is shown when the claim is started?" That stays in scope while the $29
screen repair does not, because `data/kb/05_deductibles_phone.md` states the tier is shown when
the claim is started, so the reply can say so without knowing the model.

Carried over from the earlier assembly rules: an amount may only be demanded when both `device`
and `peril` are not null, since naming a figure without knowing the device is guessing. On
`file_claim` route rule 1 already guarantees both, so this condition is met by construction
there.

**Missing information.** One item per null field, naming that field. When neither field is null
there is nothing left to ask for, so instead one item is attached asking whether the reply asked
any question at all, with `expect: "no"`.

**Device-specific exclusion.** Added when the peril is excluded for that device, e.g. loss on a
watch: the answer should say loss coverage is phones only. Attached only when `device` is not
null, for the same reason as the peril item: a reply that has not been told the device cannot be
expected to name an exclusion that depends on it.

## Training

- Policy model: `Qwen2.5-0.5B-Instruct` first; move to 1.5B only if 0.5B clearly works.
- Algorithm: GRPO via the `trl` library. Generate ~8 replies per task, score each with the reward, update.
- Hardware: Google Colab, A100 40GB (paid compute units; budget is fine). Training does NOT run on the Mac.
- Evaluate on `tasks_test.jsonl` (40 tasks) that training never sees.

## Repo layout

```
support-rl-env/
  README.md              what this is, how reward works, results
  data/
    kb/                  ~15 knowledge-base docs (.md)
    components.py        the fixed component lists above
    generate_tasks.py    samples components, calls an LLM for the sentence, writes the three splits
    tasks_train.jsonl    320 task records
    tasks_val.jsonl      40 task records
    tasks_test.jsonl     40 task records
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

## Lessons learned

Short factual notes, each with the measurement behind it. The README carries the ones a
reviewer needs; these are the ones that changed how the pieces here are built.

- Ollama's `think: false` option is silently ignored on the judge box, because its models were
  imported from raw GGUF and the thinking switch is not in the template. Appending " /think" to
  the prompt narrowed output from a 10x spread to about 3.7x but did not eliminate it: 29 of 176
  calls still exceeded the expected range, up to 4,013 tokens. Why the string works in the
  opposite direction to its name was never established.
- Constraining the judge with Ollama's `format` JSON schema solved what prompt wording could
  not. Output tokens went from a 206–4,013 range to 139–318, latency from a 4.2s median to about
  0.9s, and the 8 unparseable replies out of 176 became structurally impossible because
  `verdict` is an enum. On this box the constrained output lands in the reply's `thinking`
  field, not `response`.
- Batching all rubric items into one judge call was measured and rejected. Only 1.4x faster
  (23 calls at 112.6s against 176 at 155.1s), and 18% of verdicts changed. Same model, same
  temperature, same questions; the only difference was other questions sharing the prompt.
  Items with open-ended judgement moved most (`ask_question.no_assertion` 38% agreement,
  `missing.none` 47%); items checking for specific content in the reply stayed at 100%. Batched
  and per-item scoring should be treated as two different graders. `ask_batch` stays in
  `reward/judge.py`, unused.
- The judge model swap (`qwen2.5:32b` local to `qwen3:30b-a3b` remote) agreed 88.1% over 168
  verdicts. Route-specific rubric items agreed 100%; the disagreement concentrated in
  `common.no_reask` (64%) and `common.on_topic` (70%). On `no_reask` the new judge flipped all 6
  disagreements from no to yes, so it applies a broader standard for "asked something the
  customer already said".

## Current state

Where the training track stands, machines, run names, tooling, and the next step are kept in
`NOTES.md` (uncommitted, dated at the top). Read it before touching `train/`, `reward/`, or the
GPU server.

## Not decided yet

- Exact wording of the per-peril item sets (11 perils).
- How many rubric items a task should carry, and whether every generator above fires on
  every route.
- Reward weight between route and reply.
- Tunnel setup between Colab and the Mac judge.

## Isolation
- `data/judge_only/` is read ONLY by `reward/judge.py` and `data/generate_tasks.py`.
- Nothing under `train/` or in any policy-model prompt may read or reference `data/judge_only/`.
- If a task needs scenario answers inside the policy prompt, stop and ask; do not copy from judge_only.
