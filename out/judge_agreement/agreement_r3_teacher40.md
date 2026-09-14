39 completions, 353 item verdicts | rubric mean haiku 0.927 vs ollama 0.967 | overall agreement 0.932

| rubric item | n | haiku pass | ollama pass | agreement |
|---|---|---|---|---|
| ask_question.asks_missing | 12 | 12/12 | 12/12 | 1.00 |
| ask_question.no_assertion | 12 | 11/12 | 12/12 | 0.92 |
| common.no_reask | 39 | 35/39 | 39/39 | 0.90 |
| common.on_topic | 39 | 33/39 | 38/39 | 0.87 |
| common.question_cap | 39 | 39/39 | 39/39 | 1.00 |
| common.word_count | 39 | 39/39 | 39/39 | 1.00 |
| consistency.route_reply | 39 | 39/39 | 39/39 | 1.00 |
| escalate.handed_to_person | 4 | 4/4 | 3/4 | 0.75 |
| escalate.no_reason | 4 | 4/4 | 3/4 | 0.75 |
| escalate.not_denied | 4 | 4/4 | 4/4 | 1.00 |
| exclusion.phones_only | 6 | 6/6 | 6/6 | 1.00 |
| explain_not_covered.alternative | 6 | 1/6 | 2/6 | 0.83 |
| explain_not_covered.grounded | 6 | 4/6 | 5/6 | 0.50  **< 70%** |
| explain_not_covered.not_covered | 6 | 6/6 | 6/6 | 1.00 |
| explain_waiting_period.31_day_rule | 3 | 3/3 | 3/3 | 1.00 |
| explain_waiting_period.coverage_begins | 3 | 3/3 | 3/3 | 1.00 |
| explain_waiting_period.plan_active | 3 | 3/3 | 3/3 | 1.00 |
| file_claim.covered | 6 | 6/6 | 6/6 | 1.00 |
| file_claim.deductible | 6 | 5/6 | 5/6 | 1.00 |
| file_claim.documentation | 6 | 6/6 | 6/6 | 1.00 |
| file_claim.how_to_start | 6 | 4/6 | 3/6 | 0.50  **< 70%** |
| missing.device | 2 | 2/2 | 2/2 | 1.00 |
| missing.none | 27 | 27/27 | 27/27 | 1.00 |
| missing.peril | 10 | 10/10 | 10/10 | 1.00 |
| peril.battery | 2 | 2/2 | 2/2 | 1.00 |
| peril.crack | 1 | 0/1 | 0/1 | 1.00 |
| peril.drop | 1 | 1/1 | 1/1 | 1.00 |
| peril.surge | 2 | 2/2 | 2/2 | 1.00 |
| refer_to_manufacturer.to_manufacturer | 4 | 4/4 | 4/4 | 1.00 |
| refer_to_manufacturer.why | 4 | 0/4 | 4/4 | 0.00  **< 70%** |
| tech_support.concrete_step | 4 | 4/4 | 4/4 | 1.00 |
| tech_support.no_charge | 4 | 3/4 | 4/4 | 0.75 |
| tech_support.software_not_claim | 4 | 4/4 | 4/4 | 1.00 |
| **all** | 353 | 326/353 | 340/353 | 0.93 |
