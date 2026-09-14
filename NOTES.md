# Working notes (state as of 2026-09-13 19:30, uncommitted)

Continuity notes for the training track. CLAUDE.md is the design; this file is where things
stand. Nothing here is committed; last commit is 158c951 (data generator + v1/v2 datasets).

## Machines
- GPU server: `aeril@192.168.88.172` (host aeri-desktop), Quadro RTX 8000 48 GB, sm_75. Project
  at `~/RL` (rsynced from this checkout, not a git clone — git is not installed there), venv at
  `~/RL/.venv` (torch 2.11 cu128, transformers 5.17.0, trl 1.13.0, peft 0.20.0). Key auth with
  `~/.ssh/id_ed25519`. Sync with rsync; `.env`, `.venv`, `.git`, scratch files, v1/v2 data and
  `data/judge_only` are excluded on purpose.
- Judge box: `192.168.88.59:11434`, Ollama `qwen3:30b-a3b`, called by `reward/judge.py` with a
  `format` JSON schema. Never run the policy model there.

## Runs (all under `~/RL/out/` on the server; fetched copies in the session scratchpad)
- run1 `grpo_run1`: 10-step plumbing check, 7B + LoRA, fp16, batch 2×4. Completed.
- run2 `grpo_run2_batch4`: same with batch 4×2, 5 steps. Same peak memory (33.4 GiB); no gain.
- run3 `grpo_run3_full`: 320 steps, one pass over tasks_train.jsonl, seed 42. Completed 8.5 h.
  Result: policy collapsed to `ask_question` by step ~26-34 (last non-ask correct at step 18);
  from step 41 on it never matched a non-ask route. Eval with checkpoint-320: 16/40 route
  accuracy but all 40 outputs `ask_question` (16 = number of ask gold tasks). Training order
  rebuilt from trl's RepeatSampler (seed 42) and validated 320/320.
- run4 `grpo_run4_rebalance` (completed 2026-09-13 10:43): the data-rebalancing
  test. 35 tasks from `data/v3_kb_definitions/tasks_train_sub35.jsonl` (9 ask_question / 26
  non-ask, 4-5 per route, seed 42; built by `scripts/make_train_sub35.py`, which reproduces
  the file byte for byte from `tasks_train.jsonl` with its defaults), everything else identical to run3, `max_steps=35`,
  `save_steps=35`. Extra logged metric `non_ask_accuracy` (fraction of the 8 completions with
  the correct route on non-ask steps; NaN on ask steps). The question it answers: after step
  ~26, are completions still 8/8 ask_question? Interim: non_ask_accuracy was non-zero on
  steps 2-7 (two steps at 8/8), then zeros from step 8; not yet conclusive.
  RESULT (2026-09-13 10:43): no collapse to ask_question, but collapse to file_claim /
  tech_support instead. The four routes refer_to_manufacturer / explain_waiting_period /
  explain_not_covered / escalate scored 0/8 on every one of their 21 steps ("dead routes"), and
  the 9 ask steps went 2/72. Wall 60.6 min, peak 33.55 GiB allocated (36.1 GiB in nvidia-smi).
  Report: `out/grpo_run4_rebalance/completions_report.md`.
- run5 `grpo_run5_gen12` (launched 2026-09-13 12:03, completed 13:33, wall 89.6 min, peak
  42.97 GiB allocated / 45.7 GiB nvidia-smi): run4 with one
  change, `num_generations` 8 -> 12 (`--gens 12`, new trainer flag, default 8). To keep one
  prompt per step the batch became `--batch 4 --accum 3` (trl derives generation_batch_size
  12, steps_per_generation 3; task order verified identical to run4). reward_weights stays at
  trl's default None (= 1,1), same as run4 — note the trainer never set [2,1]. First run whose
  recorder carries judge reasoning. nvidia-smi peak 45.3 GiB over steps 1-2 (~3.8 GiB headroom
  on the 48 GB card); ~152 s/step. Question: on the four dead routes, does any step get
  >= 1/12 correct? Post-run chain (background job): fetch -> completions_report.md (also saved
  on the server) -> `scratch_compare_run4.py` (now three columns, run3/run4/run5, N-aware).
  RESULT: one step only (step 29, explain_waiting_period, 1/12); the other 20 dead-route
  steps 0/12. Same file_claim / tech_support collapse as run4; ask steps 4/108. Three invented
  route strings appeared (examine_claim, clarify_details, "file Claim"), each once.
  Report with judge reasoning: `out/grpo_run5_gen12/completions_report.md`.
- run6 `grpo_run6_beta004` (launched 2026-09-13 13:37, completed 14:42, wall 64.4 min, ~109
  s/step, peak 33.55 GiB allocated): run4 with one
  change, `beta` 0 -> 0.04 (`--beta` flag, new, default 0). 8 generations, batch 2x4, seed 42,
  35 steps, sub35 data, default reward weights. With PEFT and beta != 0 trl keeps
  `ref_model = None` and gets reference log-probs by disabling the adapter (one extra forward
  pass, no second model copy). trl logs `kl` per step. Questions: does the file_claim +
  tech_support share in steps 16-35 stay below run4's 0.90, and does route_reward on the four
  dead routes move at all (run4 0/136)? Post-run chain armed; `scratch_compare_run4.py` now
  has run6 as a fourth column plus the share metric, the dead-route rate, and a kl table.
  RESULT: no change from run4. Dead routes 0/136 (no step >= 1/8); file_claim+tech_support
  share in steps 16-35 was 0.93 (run4 0.90); kl stayed between 0.0001 and 0.013 throughout.
  One unparsed completion (step 27). Report: `out/grpo_run6_beta004/completions_report.md`.
- run7 (ON HOLD, not launched; the user decides after reading the baselines): the candidate is
  run4 with the policy prompt changed. Two new prompt switches exist on the trainer and on
  `eval/before_after.py`, both defaulting to the run1-run6 prompt (verified byte-identical):
  `--prompt-version v2` replaces the "Routes: ..." line with one-line route definitions
  (`ROUTES_V2` in train/train_grpo.py); `--reasoning` changes the answer-format line so the
  object starts with a "reasoning" field. reasoning is parsed (`reward.parse_fields`), stored
  by the recorder and by the eval outputs, and never scored: code items and judge items get
  `reply` only (verified with an intercepted judge call). A queued job runs the base model on
  test 40 four ways after run6's chain: `out/eval_base_v1`, `out/eval_base_v1r` (v1 +
  reasoning), `out/eval_base_v2`, `out/eval_base_v2r` (v2 + reasoning). The earlier v1
  baseline (12/40) was run on the Mac in fp32; these four run on the server in fp16.
  RESULT (2026-09-13 14:57, 7B base, greedy, fp16 on the server): v1 12/40 (same as the Mac
  fp32 figure, per-route identical), v1+reasoning 12/40, v2 17/40, v2+reasoning 20/40. All
  40/40 parsed in every run. The dead routes on test 40: refer/waiting/not_covered stay 0 in
  all four; escalate reaches 2/4 only with v2+reasoning. Details in
  `scratchpad/eval_base_*/` and `out/eval_base_*` on the server. run7 still not launched.
  Prompt v3 added (2026-09-13 15:15; `ROUTES_V3` in train/train_grpo.py, choices on both
  scripts now v1/v2/v3, v1 and v2 verified unchanged): rule-ordered route definitions naming
  the account field each rule reads and the reply each route requires. Base model, v3 +
  reasoning, test 40: 21/40 (`out/eval_base_v3r`); explain_not_covered 2/4 appears for the
  first time; refer_to_manufacturer and explain_waiting_period still 0. run7 not launched.
  Prompt v4 added (2026-09-13 15:35; `ROUTES_V4`): v3 plus a write-down-the-facts step
  (device, incident, enrolled_days_ago vs both periods, claims_last_12m vs limit) and an
  explicit first-match-wins order. With it, one KB edit: `data/kb/10_manufacturer_warranty_
  interaction.md` now says "12 months (365 days) from enrollment" instead of "usually 12
  months". KB versions per prompt version (`KB_DIRS` in train/train_grpo.py): v1, v2, v3 read
  `data/kb_v1/` (the KB as it stood for run1-run6 and the v1-v3 baselines, a full copy, only
  file 10 differs); v4 reads `data/kb/` (live). The judge reads no KB; the generator and
  filters read only `data/kb/04_perils.md`; route labels come from `WARRANTY_DAYS = 365` in
  generate_tasks.py and all 400 stored labels are reproduced unchanged.
  Base model, v4 + reasoning, test 40: 20/40 (`out/eval_base_v4r`); ask_question 12/16 (best
  so far), refer_to_manufacturer and explain_waiting_period still 0/3 and 0/4. run7 not
  launched.
  Prompt v5 added (2026-09-13 15:55; `ROUTES_V5` = the v4 block from "Then choose the
  route." on, `ANSWER_SECTION_V5`): the write-down paragraph is gone and the answer object
  starts with six fact fields (device, incident, enrolled_days_ago, inside_waiting_period,
  inside_manufacturer_warranty, claim_limit_reached) before route and reply. v5 reads
  `data/kb`; `--reasoning` is refused with v5. `reward.parse_record` returns the facts
  (missing or malformed -> None, never a parse failure); the recorder and the eval outputs
  store them; nothing scores them. Base model, v5, test 40: `out/eval_base_v5`, 19/40 with
  35/40 parsed: 5 outputs wrote bare multi-word values (`"device": not stated`) that the
  single-word bare-value repair does not cover, all five would have been correct
  ask_question. escalate 4/4 for the first time (claim_limit_reached is right 22/24).
  inside_manufacturer_warranty was never "yes" (0/24 with an account); enrolled_days_ago was
  copied for 20/45/95 and written null for every 200/400/600.
  Bare-value repair extended (2026-09-13 16:30): multi-word bare values (`"device": not
  stated`) are quoted too, applied only outside quoted strings (`_outside_strings` in
  reward/reward.py), same repair label. 700 run5/run6 completions parse unchanged. v5
  re-scored with it: 24/40, 40/40 parsed (`out/eval_base_v5/summary_rescored.json`, originals
  kept).
  Prompt v6 added (`data/kb_v6/`, 11 files, v5 text over a trimmed, reordered KB): threshold
  files first (waiting period, claim limits, warranty), then perils, coverage, devices, loss
  and theft, tech support, deductibles, filing. Dropped 09 repair options, 13 data and
  software, 14 cancellation. Four intro sentences cut, no rule or number changed; kept files
  otherwise byte-identical to data/kb. Rendered system prompt 13405 -> 10968 chars (3115 ->
  2603 tokens). The generator still reads data/kb/04_perils.md, untouched; 400/400 labels
  reproduce. Base model, v6, test 40: `out/eval_base_v6`, 22/40, 40/40 parsed (7 repaired
  bare values). ask_question 13/16 and escalate 4/4; refer_to_manufacturer,
  explain_waiting_period, explain_not_covered all 0. run7 not launched.
  Prompt v7 added (2026-09-13 16:45): v6 plus three derived flags rendered into the account
  JSON of the user turn by `render_account` in train/train_grpo.py (coverage_active =
  enrolled >= 31, manufacturer_warranty_active = enrolled < 365, claim_limit_reached =
  claims >= CLAIM_LIMITS[device]; order enrolled, coverage, warranty, claims, limit, tier;
  empty account stays {}). Answer object = device, incident, route, reply. The three route
  conditions name the flags. KB = kb_v6. Rendering only: task files untouched, flags agree
  with the stored route_answer on 400/400. Base model, v7, test 40: `out/eval_base_v7`.
- run7, first launch (16:47, killed at step 2, kept as `out/grpo_run7_v7_wrongprompt_v1`):
  trained on prompt v1 while logging "prompt version: v7". Cause: run as a script,
  train/train_grpo.py is `__main__`; data/build_dataset.py does `import train_grpo`, which
  loads a second module copy whose PROMPT_VERSION was still "v1", and that copy rendered the
  dataset. Caught because step 1 matched run4 exactly (loss, num_tokens 2.29e+04, mean
  completion length 68.38). Fix: `build_dataset.build(files, tokenizer, prompt_version,
  reasoning)` sets the version on its own copy, main() passes it, and main() asserts the
  rendered prompt contains the requested routes block and answer section and prints the
  token count. The evals were never affected (eval/before_after.py sets the version on the
  module it calls).
- run7 `grpo_run7_v7` (relaunched after the fix 16:52, completed 17:49, wall 56.7 min, peak
  31.48 GiB allocated): run4 settings
  (sub35, 8 gen, batch 2x4, beta 0, seed 42, 35 steps, default reward weights) with
  `--prompt-version v7`. Chain: steps 1-2 VRAM -> fetch -> completions report -> comparison
  (run7 column, dead-route section) -> `out/eval_run7_ckpt35` (checkpoint-35, v7, test 40).
  Questions: does test-40 after training beat the v7 base baseline, and how many dead-route
  steps get >= 1/8?
  RESULT: dead routes woke up: 7 of 21 dead-route steps >= 1/8 (refer 3,1,5,6 of 8;
  escalate 1 and 3; not_covered 1), 20/136 vs run4's 0/136. But the policy collapsed onto
  refer_to_manufacturer instead (steps 26-35: refer 39/80); file_claim 0/8 on all four of its
  steps, waiting 0/8 on all four, ask steps 34/72. On the new test 40 the checkpoint scores
  15/40 with its training prompt v7run7 and 17/40 with the updated v7, against base 15/40
  (v7, new test) and 24/40 (v5, new test). Report: `out/grpo_run7_v7/completions_report.md`.
- run8 `grpo_run8_v5` ABORTED (launched 18:12, killed at step 0 on the user's instruction to
  rebuild the test first; `out/grpo_run8_v5` left as is: train.log and an empty completions
  dir). Setup was run4 settings + `--prompt-version v5` on the relabeled sub35; the log
  confirmed v5 and 35 rows. Not rerun yet.

## Label fix and split rebuild (2026-09-13 17:15)
- `route_answer` in data/generate_tasks.py reordered: ask_question -> explain_not_covered ->
  tech_support -> refer_to_manufacturer -> escalate -> explain_waiting_period -> file_claim.
  Reason (also in the code comment): the KB gives tech support no waiting period (03, 12),
  routes manufacturer-warranty defects to the manufacturer regardless of plan coverage (10),
  and excludes non-phone loss/theft permanently (01, 11); so waiting_period is only the answer
  when the incident would otherwise be a covered claim (physical perils, phone loss/theft).
  CLAUDE.md's "route_answer rule" section still shows the old order; not edited (design doc).
- All 400 relabeled with the new order (customer text and accounts untouched; unchanged
  records reproduce byte for byte). 19 records changed, all enrolled_days_ago = 20: 9 warranty
  perils -> refer_to_manufacturer (t093 t096 t110 t117 t121 t126 t134 t135 t140, train), 6
  non-phone loss/theft -> explain_not_covered (t145 t154 t168 train; t172 t177 t178 val), 4
  software -> tech_support (t181 t189 t195 t196, old test). t110 is in sub35: run4-run7 were
  rewarded on explain_waiting_period for it (step 30); the comparison script says so.
- Old labeled files kept as `tasks_{train,val,test,train_sub35}_prelabelfix.jsonl`.
- Splits, final (17:30): the four relabeled software tasks (t181 t189 t195 t196) moved from
  test to val; three enrolled<31 train tasks still explain_waiting_period and not in sub35
  moved into test (t005 crack watch, t044 liquid laptop, t069 surge tablet), joining t170
  (phone theft) from val. A first attempt had moved val's three non-phone theft tasks
  (t172 t177 t178) into test; they are back in val. Sizes: train 317, val 43, test 40.
  Test routes: ask 16, file_claim 6, escalate 4, explain_not_covered 4,
  explain_waiting_period 4, refer 3, tech 3. Val: ask 14, tech 8, not_covered 8, file_claim
  6, escalate 4, refer 3. sub35 (relabeled, t110 only) stays the training set; run3's 320
  training tasks now include three test tasks (t005 t044 t069), so run3 cannot be scored on
  this test. Pre-fix split files kept as `*_prelabelfix.jsonl`.
- Test rebuilt again (18:25) as a route-stratified 40 ("test3"): ask 6, file_claim 6,
  explain_not_covered 6, escalate 6, refer 6, waiting 5, tech 5. Kept the existing test tasks
  where possible; 10 filled from train tasks outside sub35 (seed 42, preferring unseen
  device/peril pairs within a route): t013 t059 (escalate), t173 t423 (not_covered), t112 t119
  t122 (refer), t022 (waiting), t406 t415 (tech). 10 displaced ask tasks moved to val. The
  previous test is `tasks_test_ask16.jsonl`. Sizes: train 307, val 53, test 40. Evals on it
  carry the suffix `_test3`.
- Prompt v8 added (18:45): v5 with (1) one derived account flag,
  manufacturer_warranty_active = enrolled < 365, rendered after enrolled_days_ago (empty
  account stays {}); (2) inside_manufacturer_warranty dropped from the answer object, the other
  five fact fields kept; (3) v5's routes block in the corrected label order with the
  refer_to_manufacturer and explain_waiting_period definitions restated (the waiting one says
  "choose this even though the incident is a covered one"). KB = kb_v6. Base on the stratified
  test: `out/eval_base_v8_test3`: 19/40 (v5 21/40). refer_to_manufacturer 4/6 for the first
  time on a base prompt with the flag as input, escalate 6/6, ask 6/6, but refer was chosen
  19 times (every in-warranty non-software task), taking file_claim to 0/6 and
  explain_not_covered to 0/6; waiting still 0/5.
- Base-model size check (19:05): Qwen2.5-14B-Instruct (fp16, greedy, downloaded to the
  server's HF cache, 28 GB) on the stratified test: v5 30/40 and v8 30/40 (7B: 21 and 19),
  non-ask 24/34 both. 14B copies enrolled_days_ago 40/40 and gets inside_waiting_period
  right 38-40/40, so explain_waiting_period reaches 3/5 with either prompt; file_claim 6/6.
  Still not done by 14B: the 365-day warranty comparison (warranty field 0/34 under v5) and
  the refer-vs-escalate order on at-limit accounts. nvidia-smi peak 46.1-46.3 GiB (14B fp16
  weights ~28 GiB plus KV cache), 11.4 and 8.9 min per 40 tasks. `out/eval_base14b_*_test3`.
- Prompt v7 updated after run7 launched: the waiting-period rule now reads "coverage_active is
  false and the incident is one the plan covers from day 31 (...)" and sits after escalate.
  The block run7 trained on is kept as `--prompt-version v7run7`.

## Colab track (2026-09-13 19:30, prepared, nothing launched)
- `train/grpo_colab.ipynb`: A100 80GB notebook. Installs deps (transformers 5.17.0, trl
  1.13.0, peft 0.20.0, anthropic), clones the repo, lays `colab_bundle.tar.gz` from
  `MyDrive/claims-support-rl/` over it (the uncommitted code and data: reward/, train/, eval/,
  data/kb, kb_v1, kb_v6, the relabeled sub35, the stratified test), sets ANTHROPIC_API_KEY
  from a Colab secret, sends one test judge call, then: 14B base eval v5 -> run8 (14B bf16,
  LoRA r=16, v5, sub35, 8 gen, 2x4, beta 0, seed 42, 35 steps) -> checkpoint-35 eval ->
  the same three cells for v8 as run9. Outputs go to Drive. The notebook clones the public
  repo at commit a4d8602 (code, KB versions, relabeled splits all committed there); the
  install cell uninstalls Colab's torchao 0.10, which peft 0.20 rejects; peft needs it only
  for quantized layers and imports without it (verified on the server); the
  earlier Drive bundle step is gone.
- Judge backend switch in reward/judge.py: `JUDGE_BACKEND=ollama` (default, runs 1-7) or
  `anthropic` (Messages API, `claude-haiku-4-5`, no sampling parameters because anthropic 1.x
  removed `temperature` from messages.create, same prompt text, JSON parsed
  from the reply text, SDK retries 5, at most 4 calls in flight). `reward/trl_rewards.py`
  scores completions in a thread pool only on the anthropic backend. Every summary.json now
  records the judge backend and model (`judge.backend`, `judge.model`); the trainer also
  records model, dtype, prompt version and the run settings under `config`.
- Anthropic backend hardening (after the first Colab eval failed at scoring): the reply is
  fence-stripped and the first balanced JSON object that has a yes/no verdict is taken
  (`_first_object`), and the prompt gets "Reply with only the JSON object, no code fences"
  appended on that backend only. `eval/before_after.py` now writes outputs.json and an
  unscored summary.json right after generation, scores as a second phase, and has
  `--rescore --out <dir>` to judge a saved outputs.json without regenerating; per-item
  verdicts with reasoning are stored per record under `items`.
- Anthropic backend, second hardening (a Colab rescore died at 20/40 on a reply cut at
  max_tokens): max_tokens 1024 and "keep reasoning to one sentence" in the prompt suffix; a
  reply that does not parse or was cut off is asked once more with a stricter form; if that
  fails too the item gets verdict "no" with reasoning "judge_parse_failure" and the run goes
  on. The count is `judge_failures` in every summary.json (and inside the trainer's judge
  block); it is 0 on the ollama path by construction.
- Third hardening (run8 on Colab died at step 6 on a fenced reply cut at max_tokens; that
  runtime was on a checkout older than 325d057): `_call_anthropic` is wrapped so it never
  raises; an API error after the SDK's retries is also recorded (reasoning
  "judge_api_failure: <Exception>") and counted; a reply that opens an object and never
  closes it is treated as truncated and retried regardless of stop_reason. Offline tests in
  `reward/test_judge.py` (`python -m unittest reward.test_judge`) feed that exact reply.
- Runs 8-9 (Colab) use claude-haiku-4-5 as judge, so their rubric scores are not comparable
  with runs 3-7 (qwen3:30b-a3b); route accuracy is comparable.
- New trainer flags: `--model` (default Qwen2.5-7B-Instruct), `--bf16` (fp16 otherwise; the
  Quadro has no bf16); the timing callback prints s/step and peak VRAM after step 2. Eval:
  `--bf16`. Both default to the run1-run7 behaviour.

## Trainer and reward (uncommitted code)
- `train/train_grpo.py`: fixed settings (see README "Training setup"); overrides `--out
  --batch --accum --steps --save-steps --seed --tasks`. The `recording` wrapper keeps every
  completion with its per-item verdicts (`raw_completions.json`, written at run end), logs
  `non_ask_accuracy` via trl's `log_metric`, and (runs started after run4) appends
  `live_steps.jsonl` per step. From the next run on, each recorded item also carries the
  judge's answer and reasoning (added 2026-09-13 after run4 started; run4's files lack it).
- `reward/trl_rewards.py`: `route_reward` scores once and stores details; `rubric_reward`
  reads them (None on parse failure). Task column is a JSON string.
- `reward/judge.py`: remote schema-constrained judge; `ask_batch` exists but is unused
  (batching was measured and rejected, see CLAUDE.md Lessons learned).
- `eval/before_after.py --adapter <ckpt> --score`: after-eval with LoRA + rubric scoring.

## Post-run tooling (repo root, `scratch_*`, not committed)
- `scratch_postrun_run3.sh` + `scratch_postrun_analysis.py`: one-command report for run3
  (fetch, reward table + `rewards.png`, route histogram first/last 40, per-item pass rates,
  truncation/repairs, then eval on checkpoint-320 if the GPU is free). Edit RUN= for others.
- `scratch_compare_run4.py`: per-step table for run3 (steps 1-35), run4, run5, run6 (step,
  task, answer, correct/N, chosen routes) + per-5-step route histograms, dead-route check,
  file_claim+tech_support share for steps 16-35, and a kl table for runs with beta != 0.
  Skips any run whose `raw_completions.json` is not in the scratchpad.
- `scratch_completions_report.py <run dir> <task file> <out.md>`: per-step markdown
  (customer, rubric items, 8 completions with route ✓/✗, rubric, reply, item pass/fail).
  A chained job is waiting to generate `out/grpo_run4_rebalance/completions_report.md`.
- `scratch_dashboard.py <run dir> --tasks <file> --port 8765`: read-only live dashboard
  (polls train.log / live_steps.jsonl / raw_completions.json over ssh every 10 s). Running for
  run4 at http://127.0.0.1:8765/. Columns ask_rate/chosen fill from the recorder file, so for
  run4 they appear only after the run ends.
- The session scratchpad (`/private/tmp/claude-501/.../scratchpad`) holds fetched run outputs
  (`grpo_run3/`, `grpo_run4/`, baselines). It is per-session; the scripts above are copied to
  the repo root so they survive, the fetched data is not.

## Gotchas learned the hard way
- A module-level setting changed in `__main__` is not seen by code that `import`s the same
  file under its module name: the script and the import are two module objects. Pass such
  settings as arguments (see run7's first launch under Runs).
- `pgrep -f PATTERN` / `pkill -f PATTERN` inside an ssh command match the remote shell that
  is running that very command (its argv contains PATTERN). Effects seen: waiters that never
  exit ("still running" forever), a sampler loop that kept itself alive, and pkill killing its
  own ssh session (exit 255). Use `ps -eo args | grep -q "[p]attern"` (bracket trick) instead.
- Launching a long process over ssh: wrap it as `(setsid nohup CMD > log 2>&1 < /dev/null &)`
  in a subshell, otherwise ssh hangs until the process exits.
- scp with `{a,b}` brace expansion in a quoted remote path does not expand; copy files one
  at a time.
- Arrow struct columns unify schema across rows (account {} became a dict of Nones); the
  task column is therefore a JSON string.

## Lessons learned

- run3–run5: the policy prompt listed the seven route names only. Route meanings
  (ROUTE_MEANING) lived in reward.py and were shown only to the judge; the labeling rule chain
  (route_answer) lived in generate_tasks.py and was shown to no one. The policy was guessing
  routes from names. The three routes with self-explanatory names (ask_question, file_claim,
  tech_support) survived; the four that need a rule (refer_to_manufacturer,
  explain_waiting_period, explain_not_covered, escalate) went 0/8 on every step.
- With a 0/1 route reward, a group that is all wrong has zero variance and zero advantage —
  GRPO cannot learn a route the base model never hits (silent-group problem; DAPO calls this
  dynamic-sampling territory). Raising num_generations 8→12 (run5) did not help because the
  base hit rate on those routes was ≈0.4%. The fix is upstream of RL: give the policy the
  definitions (run7), and let it write a short reasoning before the route (run8).
- Rule of thumb: before an RL run, check that the base model hits every target class at least
  sometimes with the training prompt. If a class is at 0, RL will not teach it; fix the prompt
  or data first.
- The label rule chain put "enrolled < 31 → explain_waiting_period" ahead of software,
  warranty, and non-phone loss/theft rules, contradicting the KB for those cases. All four
  waiting-period tasks in the old test were software problems, so 0/4 on that route across
  six prompt versions measured a label–KB conflict, not the model; the base model was
  following the KB. Rule: when labels come from a rule chain, check each rule's order against
  the document the policy is given, case by case, before reading a zero as a model failure.

### Pre-run checklist

Before any training run (60 min), do these cheap checks first:
1. Base hit rate per class: run the base model on the eval set with the training prompt. Any
   class at 0 will not be learned by GRPO (no in-group contrast). Fix the prompt or data
   first. (Learned from run4/run5/run6: three runs spent on a signal that was never there.)
2. Label–document consistency: walk each rule in the label rule chain against the document
   the policy is given, case by case, in rule order. (Learned from the waiting-period label:
   six prompt versions measured a rule-order conflict, not the model.)
3. Instruction fits the output format: if the prompt asks the model to write something down,
   the answer format must have a field for it. (Learned from v4: a four-line checklist asked
   for in a one-sentence reasoning field was ignored.)
4. Manipulation check: after turning a knob, confirm in the first steps' logs that it
   actually engaged. (Learned from run6: beta 0.04 with KL at 0.001 was not a test of beta.)
5. Eval set class balance: if one class is 40% of the eval set, the score measures that
   class. Stratify before comparing prompts. (Learned from test 40 with 16 ask tasks.)

## Next planned step
1. When run4 finishes: `completions_report.md` (automatic), then run
   `scratch_compare_run4.py` and report the per-step table and 5-step histograms beside run3
   steps 1-35. The single question: does rebalancing alone stop the collapse to ask_question?
2. Then decide the collapse fix. The user planned a seed-matched `scale_rewards` none-vs-group
   comparison; nothing chosen yet. Candidates seen in the numbers: data mix, beta=0 (no KL),
   ask_question rubric items being the easiest to pass.
3. Nothing is committed since 158c951; a commit of the trainer/reward/eval code and docs is
   pending the user's go-ahead.
