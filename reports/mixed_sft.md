# mixed-sft: RL training set re-selected against the SFT model

The SFT adapter `sft_new103_14b` (bf16 LoRA on Qwen2.5-14B-Instruct, trained on Colab from
the r4 teacher samples of the new103 batch) scored 40/40 greedy on test 40. Against that model
the old RL set, mixed-103 (`tasks_train_mixed.jsonl`, selected against the 14B base), no longer
gives contrast: run11 on Colab logged route_reward std 0 at most steps. This set is the
training tasks on which the SFT model is still mixed.

## How it was selected

- `eval/pass_at_k.py` on all 307 tasks of `data/v3_kb_definitions/tasks_train.jsonl`,
  `--adapter out/sft_new103_14b`, prompt v8 with the reasoning field, k=4, temperature 1.0
  (the GRPO trainer's value in `train/train_grpo.py`), top_p 1.0, max_new_tokens 768, route
  only (no judge).
- GPU server, Quadro RTX 8000, fp16 + SDPA. 278.9 min, peak allocated 42.37 GiB, 4 sequences
  per generate() call.
- Output: `out/passk_sft_new103_v8reason_train_k4.jsonl` (one line per task),
  log `out/passk_sft_new103_v8reason_train_k4.log`.
- Bucketed by how many of the 4 samples chose the gold route: 0 / 1-3 / 4. The 1-3 bucket is
  written to `data/v3_kb_definitions/tasks_train_mixed_sft.jsonl`: 68 tasks, the unchanged
  `tasks_train.jsonl` records in that file's order, same schema as `tasks_train_mixed.jsonl`.

## Buckets

| gold route | 0/4 | 1-3/4 | 4/4 | total |
|---|---|---|---|---|
| ask_question | 29 | 32 | 67 | 128 |
| file_claim | 2 | 9 | 34 | 45 |
| explain_not_covered | 0 | 6 | 30 | 36 |
| escalate | 0 | 9 | 19 | 28 |
| refer_to_manufacturer | 1 | 5 | 24 | 30 |
| explain_waiting_period | 0 | 3 | 10 | 13 |
| tech_support | 0 | 4 | 23 | 27 |
| total | 32 | 68 | 207 | 307 |

Within the 1-3 bucket: 3/4 correct 39 tasks, 2/4 21, 1/4 8.

pass@1 249/307, pass@2 265/307, pass@4 275/307. Parse rate 1218/1228 (10 samples with no
route, 1 with the invalid route `refer_to_customer_service`). Completion length mean 159
tokens, max 356, none at the cap.

1-3 task ids (gold hits of 4 in parentheses):

- ask_question (32): t202(3) t203(3) t207(1) t212(3) t220(2) t233(3) t241(2) t244(2) t245(2)
  t250(2) t251(1) t252(1) t253(2) t257(3) t260(3) t265(3) t281(2) t283(2) t290(3) t293(2)
  t294(1) t296(2) t318(3) t321(3) t324(2) t325(2) t328(2) t330(3) t348(1) t352(2) t354(1)
  t356(2)
- file_claim (9): t003(3) t025(1) t027(2) t036(2) t062(3) t073(3) t074(1) t078(3) t083(3)
- explain_not_covered (6): t145(3) t165(3) t167(3) t168(3) t433(3) t435(3)
- escalate (9): t020(3) t029(2) t033(3) t041(2) t049(3) t050(3) t056(2) t058(3) t079(2)
- refer_to_manufacturer (5): t110(3) t121(3) t129(3) t134(3) t136(3)
- explain_waiting_period (3): t026(3) t076(3) t085(3)
- tech_support (4): t191(3) t401(3) t412(3) t413(3)

0/4 task ids: t004 t084 (file_claim); t140 (refer_to_manufacturer); t201 t205 t213 t216 t219
t225 t226 t227 t228 t229 t230 t231 t232 t234 t240 t243 t246 t261 t268 t269 t276 t285 t286
t288 t289 t291 t297 t347 t353 (ask_question).

## Overlap with mixed-103

28 of the 68 tasks are also in mixed-103: t003 t026 t027 t058 t062 t079 t085 t110 t134 t136
t145 t191 t202 t203 t260 t265 t318 t321 t325 t330 t348 t352 t354 t356 t401 t413 t433 t435.

Read the other way, the 103 mixed-103 tasks fall into the SFT buckets as 0/4: 6, 1-3/4: 28,
4/4: 69. Two thirds of the old set is solved 4 of 4 by the SFT model, which matches the
route_reward std 0 seen in run11.

## Caveat: fp16 here, bf16 in training and eval

The adapter was trained in bf16 on an A100, and the 40/40 greedy result was measured there.
These samples were drawn in fp16 on the RTX 8000, which has no bf16. The two precisions give
slightly different logits, and at temperature 1.0 that can move a task between neighbouring
buckets, most likely at the 1/4 and 3/4 edges. So this set is for choosing RL training tasks
only. It is not an evaluation of the SFT model, and none of these numbers belong in the eval
table.
