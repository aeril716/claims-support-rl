"""Fixed component lists and limits for task generation.

A task is built in two steps, not in one pass (see the "Sampling order" block in CLAUDE.md):
  step 1: draw the independent components - device, enrolled_days_ago, peril, tone, typos,
          device_price_tier, and which of device and peril the customer states.
  step 2: draw claims_last_12m, whose range depends on the device drawn in step 1 and on
          whether that device is still inside the waiting period.
Only then is an LLM asked to write the customer sentence in that style.
Keeping the components fixed keeps the peril distribution balanced.

These definitions mirror the "Data generation" and "Model output" sections of CLAUDE.md.
If something changes here, change it in CLAUDE.md too.

This file holds the components and the limits only. The drawing itself is generate_tasks.py.
"""

# Eleven perils, one per incident. This list is the source of truth for the labels: they are our
# internal names, not text taken from the knowledge base. data/kb/04_perils.md describes the same
# incident types in customer-facing language and carries no such labels.
# Ten of them are covered perils. "software" corresponds to the app / OS / virus / account line
# in that file's not-covered list. It sits in that list because a software problem is not a
# claim, not because the plan refuses to help: tech support handles it at no charge. It is here
# because software problems are a common real-world inquiry, and without it tech_support would
# never be the answer for any task.
PERILS = [
    "crack",
    "drop",
    "liquid",
    "surge",
    "battery",
    "malfunction",
    "wear",
    "environment",
    "loss",
    "theft",
    "software",
]

# Four device categories, matching data/kb/02_eligible_devices.md.
DEVICES = ["phone", "laptop", "tablet", "watch"]

# How the customer writes.
#   terse    = short fragments ("screen broke. covered?")
#   rambling = long, includes unnecessary detail
# The "normal support-ticket tone, starts like 'Hi, ...'" style applies to
# normal, frustrated, and rambling only; terse is the exception.
TONES = ["terse", "frustrated", "normal", "rambling"]

# Whether the customer message contains typos and broken grammar.
TYPOS = [True, False]

# Which of the two identifying facts the customer states, drawn as one of four outcomes.
# Customers do not follow an order: they may say what happened without naming the device
# ("it got wet, is that covered?"). Whatever is not stated is written into the task record as
# null. null means the customer never said it, not a hidden truth the grader knows.
# There is no "date": the incident date is collected when the claim is actually filed, so it
# never affects a route and never generates a rubric item.
#
# This list is NOT sampled uniformly. Each entry is (stated facts, weight).
# Uniform sampling over these four outcomes would make 75% of tasks ask_question, so the model
# would mostly learn to ask back, and real policy decisions (waiting period, claim limit,
# coverage) would appear in only 25% of tasks. With these weights ask_question is about 50%.
# Real customers most often state the device and what happened together.
CUSTOMER_STATES = [
    (["device", "peril"], 50),
    (["device"], 25),
    (["peril"], 15),
    ([], 10),
]

# `account` is what the system knows about the customer at the time of the message, not what the
# customer says. An account can have several devices enrolled, so the system does not know which
# device the message is about until the customer names it. When device is null in the record, the
# account block is an empty object -- not a reduced set of fields, empty. Route rule 1 settles
# those tasks before any other rule is read, so no account field takes part in the answer; a
# field left in would only invite the model to decide instead of asking, and would leak the
# device category. When device is stated but peril is null, every field stays: a real agent sees
# the account as soon as the customer names the device, and the task is there to check that the
# model still asks what happened.
# Each field left in this dict is drawn uniformly from its own list, so the list sets the
# distribution. claims_last_12m is deliberately NOT here: its range depends on device and on
# enrolled_days_ago, so it is drawn in step 2 against CLAIM_LIMITS below.
ACCOUNT_FIELDS = {
    # Coverage starts on day 31 (data/kb/03). Only 20 is inside the waiting period -> ~17% of tasks.
    # This field also constrains another one: when 20 is drawn the device is inside the waiting
    # period, so claims_last_12m is forced to 0. At-limit cases can therefore only come from the
    # other five values, on top of the weighting toward 0.
    "enrolled_days_ago": [20, 45, 95, 200, 400, 600],
    # Phone retail price bands from data/kb/05. Only affects phone deductibles; always present anyway.
    "device_price_tier": ["150-199", "200-249", "250-399", "400-699", "700+"],
}

# Maximum approved claims per rolling 12-month period, by device.
# These numbers also live in data/kb/07_claim_limits.md. Change both together, the same way
# this file and CLAUDE.md are kept in step. Keys must match DEVICES exactly.
# Used by route rule 2 (claim limit reached -> escalate) and by the step-2 draw of
# claims_last_12m, which ranges from 0 up to the limit for the device that was drawn.
CLAIM_LIMITS = {"phone": 3, "laptop": 2, "tablet": 2, "watch": 2}

# Weights for the step-2 draw of claims_last_12m, keyed by the device's claim limit so the three
# non-phone devices do not repeat. Each entry is (value, weight), the same shape as
# CUSTOMER_STATES. The shape is deliberate: 0 and the limit carry equal weight and the values
# between them share what is left. Most real accounts have no claims, and the at-limit cases are
# the ones that produce the escalate route, which lands near 10% of tasks with these numbers.
# A task inside the waiting period ignores this table: claims_last_12m is 0 there.
CLAIMS_LAST_12M_WEIGHTS = {
    3: [(0, 40), (1, 10), (2, 10), (3, 40)],   # phone
    2: [(0, 40), (1, 20), (2, 40)],            # laptop, tablet, watch
}

# The policy model must output {"route": <one of ROUTES>, "reply": <text>}.
# Traps are not a separate list: they follow from `account`, `peril`, and `device`
# via the route_answer rule in CLAUDE.md.
# ask_question = not enough information to decide yet; the reply asks for the missing detail.
# The model never approves or denies a claim. explain_* states a policy term that the documents
# settle outright and that has no exceptions; escalate is for cases whose outcome depends on
# facts only a human can see. Claim-limit cases go to escalate, not to a route of their own.
ROUTES = [
    "ask_question",
    "file_claim",
    "tech_support",
    "refer_to_manufacturer",
    "explain_waiting_period",
    "explain_not_covered",
    "escalate",
]


if __name__ == "__main__":
    print("PERILS          ", PERILS)
    print("DEVICES         ", DEVICES)
    print("TONES           ", TONES)
    print("TYPOS           ", TYPOS)
    print("CUSTOMER_STATES   (stated facts, weight; anything not stated is null)")
    for stated, weight in CUSTOMER_STATES:
        print("   ", stated, weight)
    print("    weights sum to", sum(w for _, w in CUSTOMER_STATES))
    print("ACCOUNT_FIELDS    (step 1, drawn uniformly)")
    for name, values in ACCOUNT_FIELDS.items():
        print("   ", name, values)
    print("CLAIM_LIMITS      (max approved claims per rolling 12 months)")
    for device in DEVICES:
        print("   ", device, CLAIM_LIMITS[device])
    print("claims_last_12m   (step 2; inside the waiting period it is always 0)")
    print("     device   outside waiting period, (value, weight)")
    for device in DEVICES:
        print("    ", device.ljust(8), CLAIMS_LAST_12M_WEIGHTS[CLAIM_LIMITS[device]])
    print("ROUTES          ", ROUTES)
