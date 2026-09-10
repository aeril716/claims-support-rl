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

## Status

Design complete, data generation not built yet.
