# test_v2: held-out test set, hard by construction

`data/v3_kb_definitions/tasks_test_v2.jsonl`, 40 tasks, ids h001-h040, same schema as `tasks_test.jsonl`.
Built 2026-09-15 by `scripts/make_test_v2.py`, a thin wrapper over `scripts/make_sft_tasks.py` (unchanged):
the same code path as the v3 data (`data/generate_tasks.py` draws the components, writes the sentence with
the v3 writer prompt, runs the three review checks A parrot / B label-body / C cause leakage, builds the
record and rubric), the Sonnet writer (`claude-sonnet-5`), and check D, duplicate of any sentence already in
`data/v3_kb_definitions/` (the 400 v3 tasks and `tasks_sft_new103.jsonl`: 501 distinct sentences). Seed 20260915.

`tasks_test.jsonl` (40) was used for every development decision and is now the validation set. This file is
the held-out test, scored once at the end. No policy model, `before_after.py`, or `pass_at_k.py` has been run on it.

## Route counts

| route | tasks |
|---|---|
| ask_question | 8 |
| refer_to_manufacturer | 6 |
| escalate | 6 |
| explain_not_covered | 5 |
| file_claim | 5 |
| explain_waiting_period | 5 |
| tech_support | 5 |
| **total** | 40 |

## Hard cases by construction

The buckets fix the combination before any sentence is written; nothing here was selected by a model's score.
Backward sampling as in the SFT batch: the usual components are drawn and a draw is kept only when its route
and its hard-case condition both hold, so the free components keep their own distributions.

| route | hard case | target | tasks |
|---|---|---|---|
| ask_question | device stated, cause hidden | at least 6 | 6 |
| ask_question | device hidden | at least 2 | 2 |
| refer_to_manufacturer | claims_last_12m at the device limit (warranty active + defect/wear/humidity) | at least 4 | 4 |
| escalate | warranty active (enrolled < 365) + accidental peril (drop, liquid, surge) | at least 3 | 4 |
| explain_waiting_period | enrolled < 31 with battery or surge | at least 3 | 3 |
| explain_not_covered | non-phone loss/theft with claims_last_12m at the limit | at least 3 | 3 |

The remaining tasks of each route are unconstrained draws, as are file_claim and tech_support.

## Generation

- Fill round 1: 40 combinations, attempt 1 accepted 39, attempt 2 accepted the last one. Combinations dropped after 5 attempts: 0.
- Rejections: 1 (A_parrot on h030 (malfunction, watch): definition run: wont turn screen flickers).
  Log: `out/test_v2_gen/rejections.jsonl`; candidate-to-h id map next to it.
- Writer claude-sonnet-5: 15 calls, 13,136 input + 9,140 output tokens, $0.12. Judge: 40 calls, 0.7 min. Wall 2.6 min.
- A first run of the same command crashed before any judge verdict because the Mac was off the home LAN and the
  judge box was unreachable (five connection retries, then `Operation timed out`). Its 10 writer calls, about $0.08,
  produced nothing that was kept. The successful run reached the judge through an ssh tunnel via the GPU server and
  a local relay proxy set on the process with `http_proxy`; no repo code was changed for that. Since
  e706d57 the same situation is handled by `export JUDGE_URL=<judge Tailscale address>` (see NOTES.md).
- Tone: frustrated 11, normal 10, rambling 9, terse 10. Typos: 19 of 40. Generator field: claude-sonnet-5 on all 40.

## Tasks

| id | route | device | peril | enrolled_days_ago | claims_last_12m | hard case |
|---|---|---|---|---|---|---|
| h001 | ask_question | phone | - | 600 | 3 | cause hidden |
| h002 | ask_question | tablet | - | 45 | 0 | cause hidden |
| h003 | ask_question | laptop | - | 95 | 1 | cause hidden |
| h004 | ask_question | watch | - | 20 | 0 | cause hidden |
| h005 | ask_question | tablet | - | 45 | 1 | cause hidden |
| h006 | ask_question | laptop | - | 20 | 0 | cause hidden |
| h007 | ask_question | - | - | - | - | device hidden |
| h008 | ask_question | - | theft | - | - | device hidden |
| h009 | file_claim | phone | battery | 45 | 0 |  |
| h010 | file_claim | watch | battery | 600 | 0 |  |
| h011 | file_claim | phone | surge | 45 | 1 |  |
| h012 | file_claim | tablet | wear | 600 | 0 |  |
| h013 | file_claim | laptop | surge | 45 | 0 |  |
| h014 | explain_not_covered | watch | loss | 400 | 2 | at limit |
| h015 | explain_not_covered | laptop | theft | 600 | 2 | at limit |
| h016 | explain_not_covered | tablet | loss | 45 | 2 | at limit |
| h017 | explain_not_covered | laptop | loss | 600 | 1 |  |
| h018 | explain_not_covered | laptop | loss | 20 | 0 |  |
| h019 | escalate | watch | drop | 95 | 2 | warranty active + accident |
| h020 | escalate | watch | surge | 45 | 2 | warranty active + accident |
| h021 | escalate | laptop | liquid | 95 | 2 | warranty active + accident |
| h022 | escalate | phone | drop | 600 | 3 |  |
| h023 | escalate | phone | drop | 95 | 3 | warranty active + accident |
| h024 | escalate | phone | battery | 45 | 3 |  |
| h025 | refer_to_manufacturer | tablet | wear | 95 | 2 | at limit |
| h026 | refer_to_manufacturer | laptop | environment | 45 | 2 | at limit |
| h027 | refer_to_manufacturer | phone | wear | 200 | 3 | at limit |
| h028 | refer_to_manufacturer | tablet | malfunction | 20 | 0 |  |
| h029 | refer_to_manufacturer | phone | malfunction | 20 | 0 |  |
| h030 | refer_to_manufacturer | watch | malfunction | 200 | 2 | at limit |
| h031 | explain_waiting_period | phone | battery | 20 | 0 | battery/surge |
| h032 | explain_waiting_period | watch | surge | 20 | 0 | battery/surge |
| h033 | explain_waiting_period | phone | surge | 20 | 0 | battery/surge |
| h034 | explain_waiting_period | tablet | drop | 20 | 0 |  |
| h035 | explain_waiting_period | tablet | liquid | 20 | 0 |  |
| h036 | tech_support | laptop | software | 400 | 2 |  |
| h037 | tech_support | tablet | software | 200 | 0 |  |
| h038 | tech_support | tablet | software | 20 | 0 |  |
| h039 | tech_support | watch | software | 95 | 2 |  |
| h040 | tech_support | tablet | software | 20 | 0 |  |
