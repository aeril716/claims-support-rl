40 completions, 366 item verdicts | rubric mean haiku 0.790 vs ollama 0.810 | overall agreement 0.902

| rubric item | n | haiku pass | ollama pass | agreement |
|---|---|---|---|---|
| ask_question.asks_missing | 6 | 6/6 | 6/6 | 1.00 |
| ask_question.no_assertion | 6 | 4/6 | 6/6 | 0.67  **< 70%** |
| common.no_reask | 34 | 24/34 | 34/34 | 0.71 |
| common.on_topic | 40 | 27/40 | 29/40 | 0.80 |
| common.question_cap | 40 | 40/40 | 40/40 | 1.00 |
| common.word_count | 40 | 33/40 | 33/40 | 1.00 |
| consistency.route_reply | 40 | 37/40 | 37/40 | 0.95 |
| escalate.handed_to_person | 6 | 4/6 | 2/6 | 0.67  **< 70%** |
| escalate.no_reason | 6 | 6/6 | 5/6 | 0.83 |
| escalate.not_denied | 6 | 6/6 | 6/6 | 1.00 |
| exclusion.phones_only | 6 | 1/6 | 1/6 | 1.00 |
| explain_not_covered.alternative | 6 | 1/6 | 0/6 | 0.83 |
| explain_not_covered.grounded | 6 | 1/6 | 1/6 | 0.67  **< 70%** |
| explain_not_covered.not_covered | 6 | 5/6 | 3/6 | 0.67  **< 70%** |
| explain_waiting_period.31_day_rule | 5 | 3/5 | 3/5 | 1.00 |
| explain_waiting_period.coverage_begins | 5 | 3/5 | 3/5 | 1.00 |
| explain_waiting_period.plan_active | 5 | 3/5 | 4/5 | 0.80 |
| file_claim.covered | 6 | 6/6 | 6/6 | 1.00 |
| file_claim.deductible | 6 | 0/6 | 0/6 | 1.00 |
| file_claim.documentation | 6 | 6/6 | 6/6 | 1.00 |
| file_claim.how_to_start | 6 | 6/6 | 6/6 | 1.00 |
| missing.device | 6 | 6/6 | 6/6 | 1.00 |
| missing.none | 34 | 32/34 | 34/34 | 0.94 |
| missing.peril | 6 | 4/6 | 3/6 | 0.83 |
| peril.environment | 2 | 1/2 | 0/2 | 0.50  **< 70%** |
| peril.loss | 1 | 1/1 | 0/1 | 0.00  **< 70%** |
| peril.theft | 2 | 1/2 | 1/2 | 1.00 |
| peril.wear | 1 | 0/1 | 0/1 | 1.00 |
| refer_to_manufacturer.to_manufacturer | 6 | 4/6 | 4/6 | 1.00 |
| refer_to_manufacturer.why | 6 | 0/6 | 0/6 | 1.00 |
| tech_support.concrete_step | 5 | 5/5 | 5/5 | 1.00 |
| tech_support.no_charge | 5 | 5/5 | 5/5 | 1.00 |
| tech_support.software_not_claim | 5 | 5/5 | 5/5 | 1.00 |
| **all** | 366 | 286/366 | 294/366 | 0.90 |
