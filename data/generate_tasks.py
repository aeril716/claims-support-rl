"""Generate the task records described in the "Data generation" and "Reward" sections of
CLAUDE.md, and write them to data/tasks_train.jsonl / tasks_val.jsonl / tasks_test.jsonl.

Run a small sample first:   python data/generate_tasks.py --smoke
Write the full 400 tasks:   python data/generate_tasks.py

Three sentence writers take a third of the tasks each, rotating by component combination so
that no writer is concentrated on one route. Which one wrote a sentence is recorded in the
`generator` field, which is metadata: it never reaches the prompt or the grader.
"""

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import components as C
import peril_definitions as P
import target_distribution as T

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Model strings taken from each provider's current documentation, not from memory.
ANTHROPIC_MODEL = "claude-opus-5"
OPENAI_MODEL = "gpt-5.6-sol"
GOOGLE_MODEL = "gemini-3.5-flash-lite"
WRITERS = ["anthropic", "openai", "google"]

DEFAULT_OUT = HERE


def split_sizes(count):
    """80 / 10 / 10 of the run size. The writing loop walks records in cell order with a
    repeating 8 train / 1 val / 1 test pattern, so any multiple of ten lands exactly."""
    return [("tasks_train.jsonl", count * 8 // 10),
            ("tasks_val.jsonl", count // 10),
            ("tasks_test.jsonl", count // 10)]
BATCH_SIZE = 4
SEED = 7
PARSE_ATTEMPTS = 3

# A combination whose sentence keeps failing validation is rewritten this many times, always
# for the same combination, then reported as a failed cell. Never more, never a substitute.
MAX_ATTEMPTS = 5
REJECTION_LOG = "rejections.jsonl"

# Wall time and call counts, reported at the end of a run. Measurement only; nothing reads it.
TIMING = {"writer_s": 0.0, "writer_calls": 0, "judge_s": 0.0, "judge_calls": 0}
PARROT_RATIO = 0.8      # difflib ratio at which a misspelling still counts as the same word
DEFINITION_RUN = 4      # consecutive content words copied from a definition that count as copying

# Words skipped when a message is reduced to content words, so that "a power surge" and
# "power surge" are the same phrase.
FILLER = {"a", "an", "the", "and", "or", "of", "my", "is", "was", "it", "its", "to", "in", "on",
          "from", "for", "with", "this", "that", "has", "have", "had", "been", "be", "not",
          "no", "so", "out", "up", "at", "by", "i", "im", "ive", "me", "we", "you"}

# Sentences already written are kept beside the output, keyed by task_id, so a crash does not
# throw away what was already generated. The cache lives in the output directory because the
# same task_id means a different sentence under a different prompt. Gitignored.
CACHE_NAME = ".sentence_cache.json"

# Statuses worth trying again. Anything else is the provider telling us the request was wrong,
# and sending it again would only repeat the same answer.
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Words that would give away the device category if the customer is not stating it.
DEVICE_GIVEAWAYS = [
    "screen", "charger", "keyboard", "wrist", "pocket", "band", "case",
    "and any brand or model name",
]

# Words that would give away what happened if the customer is not stating the peril.
# Only the device list was specified; this one is the same idea applied to the other field.
PERIL_GIVEAWAYS = {
    "crack": ["cracked", "shattered", "smashed", "glass", "broke the display"],
    "drop": ["dropped", "fell", "drop", "impact", "knocked"],
    "liquid": ["wet", "water", "spilled", "spill", "soaked", "rain", "sink", "pool"],
    "surge": ["surge", "lightning", "storm", "outlet", "plugged in when"],
    "battery": ["battery", "charge", "swollen", "drains", "dies fast"],
    "malfunction": ["stopped working", "malfunction", "faulty", "defective"],
    "wear": ["worn", "wear", "loose", "sticks", "over time"],
    "environment": ["dust", "heat", "humidity", "sand", "hot car"],
    "loss": ["lost", "misplaced", "can't find", "left it", "gone"],
    "theft": ["stolen", "stole", "theft", "robbed", "taken"],
    "software": ["app", "update", "software", "virus", "passcode", "os"],
}

# Deductibles, from data/kb/05_deductibles_phone.md and 06_deductibles_other_devices.md.
PHONE_DEDUCTIBLES = {
    "150-199": (10, 29), "200-249": (19, 39), "250-399": (29, 49),
    "400-699": (39, 99), "700+": (49, 129),
}
FLAT_DEDUCTIBLES = {"laptop": 129, "tablet": 99, "watch": 49}

PHONE_ONLY_PERILS = {"loss", "theft"}
WARRANTY_PERILS = {"malfunction", "wear", "environment"}
WARRANTY_DAYS = 365

PER_ROUTE_ITEMS = {
    "ask_question": [
        ("asks_missing", "Does the reply ask for the information missing from the customer's message?", "yes"),
        ("no_assertion", "Does the reply assert something it cannot know yet?", "no"),
    ],
    "file_claim": [
        ("covered", "Does the reply say the incident is covered?", "yes"),
        ("deductible", None, "yes"),          # question is filled in per task
        ("how_to_start", "Does the reply say how to start the claim?", "yes"),
        ("documentation", "Does the reply mention documentation that may be requested?", "yes"),
    ],
    "escalate": [
        ("handed_to_person", "Does the reply say the case is being handed to a person?", "yes"),
        ("not_denied", "Does the reply say the claim is denied?", "no"),
        ("no_reason", "Does the reply explain why the case is being handed to a person?", "no"),
    ],
    "explain_waiting_period": [
        ("31_day_rule", "Does the reply state the 31-day rule?", "yes"),
        ("plan_active", "Does the reply say the plan itself is active?", "yes"),
        ("coverage_begins", "Does the reply say when coverage begins?", "yes"),
    ],
    "explain_not_covered": [
        ("not_covered", "Does the reply say it is not covered?", "yes"),
        ("grounded", "Does the reply ground that in the policy?", "yes"),
        ("alternative", "Does the reply offer an alternative, if one exists?", "yes"),
    ],
    "refer_to_manufacturer": [
        ("to_manufacturer", "Does the reply direct the customer to the manufacturer warranty?", "yes"),
        ("why", "Does the reply explain why?", "yes"),
    ],
    "tech_support": [
        ("software_not_claim", "Does the reply identify the problem as a software problem and not a claim?", "yes"),
        ("no_charge", "Does the reply say tech support handles it at no charge?", "yes"),
        ("concrete_step", "Does the reply give a concrete step the customer can try?", "yes"),
    ],
}

PERIL_COVERAGE_PHRASE = {
    "crack": "cracked screen coverage",
    "drop": "accidental damage coverage",
    "liquid": "liquid damage coverage",
    "surge": "power surge coverage",
    "battery": "battery failure coverage",
    "malfunction": "mechanical failure coverage",
    "wear": "wear and tear coverage",
    "environment": "environmental damage coverage",
    "loss": "loss coverage, which applies to phones only",
    "theft": "theft coverage, which applies to phones only",
}


# ---------------------------------------------------------------- small helpers

def weighted_choice(rng, pairs):
    """Pick one value from a list of (value, weight)."""
    values = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(values, weights=weights, k=1)[0]


def largest_remainder(total, weights):
    """Split `total` whole items across `weights`, keeping the proportions as close as the
    whole numbers allow. Used for the 44 cells and again for the train/val/test split."""
    share = sum(weights)
    exact = [total * w / share for w in weights]
    counts = [int(x) for x in exact]
    left = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[:left]:
        counts[i] += 1
    return counts


# ---------------------------------------------------------------- drawing a task

def build_cells():
    """One cell is one customer_states outcome by one peril. The count per cell comes from
    data/target_distribution.py; a zero there (loss and theft with the peril hidden) means the
    cell is never drawn."""
    cells, counts = [], []
    for key, stated in T.STATES.items():
        for peril in C.PERILS:
            cells.append((stated, peril))
            counts.append(T.cell_target(key, peril))
    return cells, counts


def draw_components(rng, stated, peril):
    """Step 1 draws everything independent; step 2 draws claims_last_12m inside the limit for
    the device that was drawn. A real device and peril are always drawn, because the sentence
    cannot be written without them; the unstated one is discarded after the sentence."""
    device = rng.choice(C.DEVICES)
    enrolled = rng.choice(C.ACCOUNT_FIELDS["enrolled_days_ago"])
    tier = rng.choice(C.ACCOUNT_FIELDS["device_price_tier"])
    tone = rng.choice(C.TONES)
    typos = rng.choice(C.TYPOS)

    if enrolled < 31:
        claims = 0                      # account consistency: coverage starts on day 31
    else:
        claims = weighted_choice(rng, C.CLAIMS_LAST_12M_WEIGHTS[C.CLAIM_LIMITS[device]])

    return {
        "device": device, "peril": peril, "enrolled_days_ago": enrolled,
        "device_price_tier": tier, "tone": tone, "typos": typos,
        "claims_last_12m": claims,
        "stated_device": "device" in stated, "stated_peril": "peril" in stated,
    }


def route_answer(device, peril, enrolled, claims):
    """Top to bottom, first match wins. `device` and `peril` are the record values, so they
    are None when the customer did not state them.

    Order, corrected 2026-09-13 (the waiting-period rule used to sit second). The KB gives tech
    support no waiting period (data/kb/03, 12), sends manufacturer-warranty defects to the
    manufacturer regardless of plan coverage (10), and excludes non-phone loss and theft
    permanently (01, 11). So explain_waiting_period is only the answer when the incident would
    otherwise be a covered claim: physical perils, and loss or theft on a phone. It therefore
    comes after the rules that settle software, warranty, and not-covered cases, and after the
    claim limit (which account consistency makes 0 inside the waiting period anyway)."""
    if device is None or peril is None:
        return "ask_question"
    # A peril the plan can handle neither as a claim nor through tech support. Across our
    # eleven perils that is loss or theft on anything but a phone.
    if peril in PHONE_ONLY_PERILS and device != "phone":
        return "explain_not_covered"
    if peril == "software":
        return "tech_support"
    if peril in WARRANTY_PERILS and enrolled < WARRANTY_DAYS:
        return "refer_to_manufacturer"
    if claims >= C.CLAIM_LIMITS[device]:
        return "escalate"
    if enrolled < 31:
        return "explain_waiting_period"
    return "file_claim"


# ---------------------------------------------------------------- rubric assembly

def deductible_question(device, peril, tier):
    """The figure the answer should state. Phones have a repair and a replacement amount and
    the plan picks between them, so both are named; loss and theft are tiered by model and the
    exact tier is only shown once the claim is started."""
    if device == "phone":
        if peril in PHONE_ONLY_PERILS:
            # In scope even though model is not a component: data/kb/05 says the tier is shown
            # when the claim is started, so the reply can say that without knowing the model.
            return ("Does the reply say the exact deductible depends on the phone model and is "
                    "shown when the claim is started?")
        repair, replace = PHONE_DEDUCTIBLES[tier]
        # Both amounts, not "either/or": data/kb/05 says the plan decides repair vs replacement
        # and the customer cannot choose, so neither is settled when the reply is written.
        # Asking for one of two would also split the judge's verdict and stop being atomic.
        return (f"Does the reply state both deductible amounts for this price tier, ${repair} "
                f"for repair and ${replace} for replacement?")
    return f"Does the reply state the ${FLAT_DEDUCTIBLES[device]} deductible?"


def build_rubric(device, peril, tier, route):
    items = []

    def add(item_id, check, question, expect):
        items.append({"id": item_id, "check": check, "question": question, "expect": expect})

    # common
    add("common.word_count", "code", "Is the reply 60 words or fewer?", "yes")
    add("common.question_cap", "code", "Does the reply ask 2 questions or fewer?", "yes")
    if device is not None or peril is not None:
        add("common.no_reask", "judge",
            "Does the reply ask for information the customer already gave?", "no")
    add("common.on_topic", "judge",
        "Does the reply say anything unrelated to the customer's message?", "no")

    # per route
    for suffix, question, expect in PER_ROUTE_ITEMS[route]:
        if question is None:                       # the deductible item is filled in per task
            question = deductible_question(device, peril, tier)
        add(f"{route}.{suffix}", "judge", question, expect)

    # per task: peril. Only on file_claim, and only with both fields known. Coverage detail
    # differs by device, and on any other route naming the coverage fights that route's own
    # items (already not covered, sent to the manufacturer, handed over without explanation,
    # or still being asked about).
    if (peril is not None and device is not None and route == "file_claim"
            and peril in PERIL_COVERAGE_PHRASE):
        add(f"peril.{peril}", "judge",
            f"Does the reply name {PERIL_COVERAGE_PHRASE[peril]}?", "yes")

    # per task: missing information
    missing = [name for name, value in (("device", device), ("peril", peril)) if value is None]
    if missing:
        for field in missing:
            question = ("Does the reply ask which device it is?" if field == "device"
                        else "Does the reply ask what happened?")
            add(f"missing.{field}", "judge", question, "yes")
    else:
        add("missing.none", "judge", "Does the reply ask any question?", "no")

    # per task: device-specific exclusion
    if device is not None and peril in PHONE_ONLY_PERILS and device != "phone":
        add("exclusion.phones_only", "judge",
            f"Does the reply say {peril} coverage applies to phones only?", "yes")

    return items


# ---------------------------------------------------------------- sentence prompt

def case_block(number, drawn):
    device, peril = drawn["device"], drawn["peril"]
    lines = [f"Case {number}:"]

    stated, withheld = [], []
    (stated if drawn["stated_device"] else withheld).append(f"the device, which is a {device}")
    (stated if drawn["stated_peril"] else withheld).append("what happened")

    lines.append("  States: " + ("; ".join(stated) if stated else
                                 "neither the device nor what happened"))
    lines.append("  Does not state: " + ("; ".join(withheld) if withheld else "nothing"))

    if not drawn["stated_device"]:
        lines.append("  Must not contain any word that would reveal the device: "
                     + ", ".join(DEVICE_GIVEAWAYS) + ". Refer to it only as 'it'.")
    if drawn["stated_peril"]:
        lines.append(f"  peril: {peril}")
        lines.append(f"  definition: {P.DEFINITION[peril]}")
        for other in P.NOT_THIS[peril]:
            lines.append(f"  not this:   {P.label(other)} — {P.DEFINITION[other]}")
        banned = ", ".join(f'"{w}"' for w in P.BANNED[peril])
        lines.append(f"  Do not use the word {banned}. Do not copy the definition.")
        lines.append("  Describe what happened in the customer's own words.")
    else:
        lines.append("  Must not contain any word that would reveal what happened: "
                     + ", ".join(PERIL_GIVEAWAYS[peril])
                     + ". Say only that there is a problem, without naming its cause.")

    lines.append(f"  Tone: {drawn['tone']}.")
    if drawn["tone"] == "terse":
        lines.append("  Terse means a short fragment or two, no greeting.")
    else:
        lines.append("  Start with \"Hi, \" and keep a normal support-ticket tone.")
    if drawn["tone"] == "rambling":
        lines.append("  The extra length must come from things unrelated to the failure: when "
                     "they got it, what they use it for, how they noticed. It must not be "
                     "filled by inventing a cause for the problem.")
    lines.append("  Typos and broken grammar: " + ("yes" if drawn["typos"] else "no") + ".")
    return "\n".join(lines)


def build_prompt(numbered):
    """`numbered` is a list of (case number, spec). The numbers are carried through to the reply
    so a single case can be re-requested later under its own number."""
    header = (
        "You write realistic first messages from customers to a device protection insurance "
        "support line. Write one message for each case below.\n\n"
        "Rules for every message:\n"
        "- This is the customer's very first message about this problem. There have been no "
        "earlier calls, no previous agents, and no claim already in progress. Do not refer to "
        "any of those.\n"
        "- Say only what the case says the customer states. Never mention a deductible, a "
        "policy rule, or a claim number.\n"
        "- One short paragraph, under 60 words unless the tone is rambling.\n"
        "- Make the messages sound like different people, not one person writing several "
        "times. Vary the openings, the sentence lengths and the vocabulary.\n\n"
    )
    cases = "\n\n".join(case_block(number, spec) for number, spec in numbered)
    example = ", ".join('{"case": %d, "text": "..."}' % number for number, _ in numbered)
    tail = ("\n\nReturn only this JSON object, with one entry per case and the case number "
            "copied from above:\n"
            "{\"messages\": [" + example + "]}\n"
            "No other text and no markdown fence.")
    return header + cases + tail


def parse_messages(text, wanted):
    """Pull {"messages": [{"case": n, "text": s}]} out of a reply.

    Returns the entries that are well formed and belong to a wanted case, plus a note about
    what went wrong. An entry that is malformed is dropped, not raised on: one bad item then
    costs one re-request instead of the whole batch and everything already written."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return {}, "no JSON object in the reply"
    try:
        payload = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return {}, f"reply was not valid JSON: {exc}"
    entries = payload.get("messages")
    if not isinstance(entries, list):
        return {}, "reply had no messages array"

    found = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        number, sentence = entry.get("case"), entry.get("text")
        if not isinstance(number, int) or number not in wanted:
            continue
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        found[number] = sentence.strip()

    missing = sorted(set(wanted) - set(found))
    note = f"cases {missing} missing or malformed" if missing else None
    return found, note


# ---------------------------------------------------------------- provider calls

def is_transient(exc):
    """True only for failures that a later identical request could survive: a timeout, a broken
    connection, or one of the retryable statuses. A 400 or a 401 is not transient, and sending
    the same request again would only get the same answer."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in TRANSIENT_STATUS


def with_retry(provider, call):
    """Wait and try again when a provider is throttling or briefly unavailable."""
    delays = [5, 15, 45, 90]
    for attempt, delay in enumerate([None] + delays):
        if delay is not None:
            print(f"  [{provider}] transient failure, waited {delay}s, retry "
                  f"{attempt}/{len(delays)}", flush=True)
            time.sleep(delay)
        try:
            return call()
        except Exception as exc:            # noqa: BLE001 - each SDK raises its own types
            if not is_transient(exc) or delay == delays[-1]:
                raise
    raise RuntimeError("unreachable")


def call_anthropic(prompt):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def go():
        reply = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in reply.content if block.type == "text")

    return with_retry("anthropic", go)


def call_openai(prompt):
    def go():
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": OPENAI_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "reasoning_effort": "high"},
            timeout=300.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return with_retry("openai", go)


def call_google(prompt):
    def go():
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent",
            headers={"x-goog-api-key": os.environ["GOOGLE_API_KEY"]},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=300.0,
        )
        response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)

    return with_retry("google", go)


CALLERS = {"anthropic": call_anthropic, "openai": call_openai, "google": call_google}


def timed_writer(writer, prompt):
    start = time.time()
    try:
        return CALLERS[writer](prompt)
    finally:
        TIMING["writer_s"] += time.time() - start
        TIMING["writer_calls"] += 1


# ---------------------------------------------------------------- validation

def content_words(text):
    """Lowercase words with the filler removed, so phrases can be compared as written."""
    words = re.findall(r"[a-z]+", text.lower().replace("'", "").replace("\u2019", ""))
    return [w for w in words if w not in FILLER]


def same_word(a, b):
    """Exact for short words, loose for long ones. Loose matching on four letters turns "sure"
    into "surge", so the length floor matters as much as the ratio."""
    if a == b:
        return True
    if len(a) >= 5 and len(b) >= 5:
        return difflib.SequenceMatcher(None, a, b).ratio() >= PARROT_RATIO
    return False


def run_found(message_words, phrase_words):
    """True when the message contains those words consecutively, in order, each matched by
    same_word."""
    span = len(phrase_words)
    for start in range(len(message_words) - span + 1):
        if all(same_word(message_words[start + i], phrase_words[i]) for i in range(span)):
            return True
    return False


def label_phrases(peril):
    """The short ways the label itself gets said. The KB label, split on "/" so
    "Drop / impact damage" gives two, plus the multi-word entries of the ban list, which are
    the same label phrased the way a customer would. Single-word labels are left out: "wear"
    and "battery" on their own are ordinary English."""
    phrases = [part.strip() for part in P.label(peril).split("/")]
    phrases += [term for term in P.BANNED[peril] if " " in term]
    return [words for words in (content_words(p) for p in phrases) if len(words) >= 2]


def parrot_hit(message, peril):
    """What the message copied from the prompt, or None.

    Two sources, matched as runs of consecutive words rather than as single tokens, because
    "stolen", "lost" and "battery" are ordinary English and customers say them. What is blocked
    is the customer categorising their own incident in the policy's terms.
    - the label phrasing, matched whole: "environmental damage", "liquid damage".
    - the definition text, matched at DEFINITION_RUN words: the writer copying the definition
      instead of describing the incident. A shorter run would fire on any accurate symptom,
      since the definitions are written out of symptom vocabulary.
    """
    words = content_words(message)
    for phrase in label_phrases(peril):
        if run_found(words, phrase):
            return "label phrase: " + " ".join(phrase)
    definition = content_words(P.DEFINITION[peril])
    for start in range(len(definition) - DEFINITION_RUN + 1):
        run = definition[start:start + DEFINITION_RUN]
        if run_found(words, run):
            return "definition run: " + " ".join(run)
    return None


def timed_judge(call):
    start = time.time()
    try:
        return call()
    finally:
        TIMING["judge_s"] += time.time() - start
        TIMING["judge_calls"] += 1


def judge_label_matches(message, peril):
    """Check B. The judge sees the KB definition of the label and of its confusable
    neighbours, then gives its reasoning and its verdict in one call."""
    from reward.judge import ask_with_reasoning
    lines = ["Definitions from the policy knowledge base:",
             f"- {P.label(peril)}: {P.DEFINITION[peril]}"]
    for other in P.NOT_THIS[peril]:
        lines.append(f"- {P.label(other)}: {P.DEFINITION[other]}")
    lines.append(f"\nCustomer message: \"{message}\"")
    return timed_judge(lambda: ask_with_reasoning(
        f"Is this message consistent with {P.label(peril)} as defined above?", "\n".join(lines)))


def judge_cause_visible(message):
    """Check C, hidden-peril tasks only. A symptom is allowed; a cause is not.

    The state wording is spelled out because the judge read "cause" as needing an event and let
    "I no longer have it in my possession" through: not having the device is the loss itself,
    however it is phrased."""
    from reward.judge import ask_with_reasoning
    return timed_judge(lambda: ask_with_reasoning(
        "Can the cause of the damage or loss be determined from this message?\n"
        "- A symptom of the device not working, such as 'won't turn on' or 'keeps freezing', "
        "is not a cause.\n"
        "- An event is a cause: dropping it, getting it wet, a power cut, a failed update, "
        "someone taking it.\n"
        "- A description of the device's current state is also a cause when that state is "
        "itself what happened. Saying the device is missing, gone, not in the customer's "
        "possession, or that they cannot find it, reveals loss or theft just as much as "
        "describing the event would.",
        f"Customer message: \"{message}\""))


def validate(spec, sentence, use_judge):
    """None when the sentence is accepted, else (check, detail)."""
    if spec["stated_peril"]:
        hit = parrot_hit(sentence, spec["peril"])
        if hit:
            return "A_parrot", hit
        if use_judge:
            verdict, reasoning = judge_label_matches(sentence, spec["peril"])
            if verdict == "no":
                return "B_label_body", reasoning
    elif use_judge:
        verdict, reasoning = judge_cause_visible(sentence)
        if verdict == "yes":
            return "C_cause_leak", reasoning
    return None


def log_rejection(out_dir, spec, sentence, attempt, check, detail):
    entry = {"task_id": spec["task_id"], "attempt": attempt, "check": check,
             "peril": spec["peril"], "peril_stated": spec["stated_peril"],
             "device": spec["device"], "device_stated": spec["stated_device"],
             "generator": spec["generator"], "detail": detail, "message": sentence}
    with (out_dir / REJECTION_LOG).open("a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  REJECT {spec['task_id']} [{spec['generator']}] {check} "
          f"peril={spec['peril']}{'' if spec['stated_peril'] else ' (hidden)'}: {detail}",
          flush=True)


# ---------------------------------------------------------------- assembly

def spec_route(spec):
    """The gold route for a spec, using the record values rather than the drawn ones."""
    device = spec["device"] if spec["stated_device"] else None
    peril = spec["peril"] if spec["stated_peril"] else None
    return route_answer(device, peril, spec["enrolled_days_ago"], spec["claims_last_12m"])


def build_specs():
    """The ordered list of component draws, one per task. The route is computed first and the
    three writers rotate within each route, so every route gets an even share of all three.
    Rotating over the combination order instead left file_claim at 13 / 17 / 27."""
    rng = random.Random(SEED)
    cells, counts = build_cells()
    specs = []
    for (stated, peril), count in zip(cells, counts):
        for _ in range(count):
            specs.append(draw_components(rng, stated, peril))

    seen = {}
    for index, spec in enumerate(specs, start=1):
        # task_id is the position in the full 400, so it is the same in a smoke run as in a
        # full one and can key the sentence cache.
        spec["task_id"] = f"t{index:03d}"
        route = spec_route(spec)
        position = seen.get(route, 0)
        spec["generator"] = WRITERS[position % len(WRITERS)]
        seen[route] = position + 1
    return specs


def build_topup_specs(route, count, perils, rng, first_index):
    """Backward sampling for one route.

    Forward sampling gives a route whatever share the component ratios happen to produce, which
    left tech_support at 14 of 400. Here the route is chosen first: only what the rule chain
    requires is fixed — the peril, and both facts stated, since every route except ask_question
    needs them — and everything else is drawn the usual way. A draw that lands on a different
    route is discarded rather than corrected, so the free components keep their own
    distributions instead of being forced.
    """
    specs = []
    while len(specs) < count:
        spec = draw_components(rng, ["device", "peril"], rng.choice(perils))
        if spec_route(spec) != route:
            continue
        spec["task_id"] = f"t{first_index + len(specs):03d}"
        specs.append(spec)
    for position, spec in enumerate(specs):
        spec["generator"] = WRITERS[position % len(WRITERS)]
    return specs


def make_record(drawn, sentence):
    device = drawn["device"] if drawn["stated_device"] else None
    peril = drawn["peril"] if drawn["stated_peril"] else None
    route = route_answer(device, peril, drawn["enrolled_days_ago"], drawn["claims_last_12m"])
    account = {} if device is None else {
        "enrolled_days_ago": drawn["enrolled_days_ago"],
        "claims_last_12m": drawn["claims_last_12m"],
        "device_price_tier": drawn["device_price_tier"],
    }
    return {
        "task_id": drawn["task_id"],
        "device": device,
        "peril": peril,
        "tone": drawn["tone"],
        "typos": drawn["typos"],
        "account": account,
        "customer": sentence,
        "route_answer": route,
        "rubric": build_rubric(device, peril, drawn["device_price_tier"], route),
        "generator": drawn["generator"],
    }


def load_cache(out_dir):
    path = out_dir / CACHE_NAME
    if path.exists():
        return json.loads(path.read_text())
    return {}


def write_batch(writer, numbered):
    """One batch. Whatever came back well formed is kept; each case that did not is re-requested
    on its own, so a single malformed entry costs one small call instead of the whole batch."""
    raw = timed_writer(writer, build_prompt(numbered))
    wanted = {number for number, _ in numbered}
    sentences, note = parse_messages(raw, wanted)
    if note:
        print(f"  [{writer}] {note}. Reply began: {raw.strip()[:200]!r}", flush=True)

    by_number = dict(numbered)
    for number in sorted(wanted - set(sentences)):
        for attempt in range(1, PARSE_ATTEMPTS + 1):
            print(f"  [{writer}] re-requesting case {number} on its own "
                  f"({attempt}/{PARSE_ATTEMPTS})", flush=True)
            single_raw = timed_writer(writer, build_prompt([(number, by_number[number])]))
            single, single_note = parse_messages(single_raw, {number})
            if number in single:
                sentences[number] = single[number]
                break
            print(f"  [{writer}] {single_note}. Reply began: "
                  f"{single_raw.strip()[:200]!r}", flush=True)
        else:
            raise RuntimeError(f"[{writer}] could not produce case {number} on its own after "
                               f"{PARSE_ATTEMPTS} tries")
    return sentences


def write_sentences(specs, out_dir, use_judge=True):
    """Write, validate, and refill. Returns (accepted, failed): accepted maps task_id to an
    accepted sentence; failed lists the specs still short after MAX_ATTEMPTS.

    A rejected sentence is rewritten for the exact same combination. Nothing about the
    combination is resampled, so the route stays what it was, and a combination that keeps
    failing shows up as such instead of being quietly replaced."""
    cache = load_cache(out_dir)
    attempts = {spec["task_id"]: 0 for spec in specs}
    pending = [spec for spec in specs if spec["task_id"] not in cache]
    print(f"{len(specs) - len(pending)} sentences from cache, {len(pending)} to write",
          flush=True)

    for round_number in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        print(f"\nround {round_number}/{MAX_ATTEMPTS}: {len(pending)} to write", flush=True)
        rejected = []
        for writer in WRITERS:
            mine = [spec for spec in pending if spec["generator"] == writer]
            for start in range(0, len(mine), BATCH_SIZE):
                chunk = mine[start:start + BATCH_SIZE]
                numbered = list(enumerate(chunk, start=1))
                print(f"[{writer}] writing {len(chunk)} sentences "
                      f"({start + len(chunk)}/{len(mine)})", flush=True)
                written = write_batch(writer, numbered)
                for number, spec in numbered:
                    sentence = written[number]
                    attempts[spec["task_id"]] += 1
                    problem = validate(spec, sentence, use_judge)
                    if problem is None:
                        cache[spec["task_id"]] = sentence
                    else:
                        log_rejection(out_dir, spec, sentence, attempts[spec["task_id"]],
                                      *problem)
                        rejected.append(spec)
                (out_dir / CACHE_NAME).write_text(json.dumps(cache, ensure_ascii=False))
        pending = rejected

    return cache, pending


def split_by_route(records, splits):
    """Stratify the split by route: take 80 / 10 / 10 out of each route separately, so every
    route appears in train, val and test.

    Splitting the whole set at once puts the same fractions in each file overall but not within
    a small route: on v3 that left refer_to_manufacturer with nothing in test and tech_support
    with nothing in val. Rounding each route on its own does not add up to 320 / 40 / 40 by
    itself, so a correction pass moves single records between files, always from a route that
    can spare one, until the totals are exact."""
    names = [name for name, _ in splits]
    targets = {name: size for name, size in splits}

    out = {name: [] for name in names}
    for route in sorted({record["route_answer"] for record in records}):
        mine = sorted((r for r in records if r["route_answer"] == route),
                      key=lambda r: r["task_id"])
        taken = 0
        for name, count in zip(names, largest_remainder(len(mine), [8, 1, 1])):
            out[name].extend(mine[taken:taken + count])
            taken += count

    # Correction: move one record at a time from an over-full file to a short one, taking it
    # from the largest route in that file so no route is emptied.
    for _ in range(len(records)):
        over = next((n for n in names if len(out[n]) > targets[n]), None)
        under = next((n for n in names if len(out[n]) < targets[n]), None)
        if over is None or under is None:
            break
        counts = {}
        for record in out[over]:
            counts.setdefault(record["route_answer"], []).append(record)
        donor = max(counts.values(), key=len)
        moved = donor[-1]
        out[over].remove(moved)
        out[under].append(moved)

    for name in names:
        assert len(out[name]) == targets[name], \
            f"{name}: {len(out[name])} records, expected {targets[name]}"
    return out


def report_failures(failed, out_dir):
    """Which cells are short after the retry cap, and what the log says went wrong."""
    print(f"\n{len(failed)} combinations still short after {MAX_ATTEMPTS} attempts. "
          f"No split files written. See {out_dir / REJECTION_LOG}.")
    reasons = {}
    for line in (out_dir / REJECTION_LOG).read_text().splitlines():
        entry = json.loads(line)
        reasons.setdefault(entry["task_id"], []).append(entry["check"])
    for spec in failed:
        states = ("device" if spec["stated_device"] else "-") + "/" + \
                 ("peril" if spec["stated_peril"] else "-")
        print(f"  {spec['task_id']}  cell {states} x {spec['peril']:<12} "
              f"route={spec_route(spec):<24} [{spec['generator']}]  "
              f"failed checks: {', '.join(reasons.get(spec['task_id'], []))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="generate two tasks per route, print them, write no files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory for the split files, the sentence cache, and the "
                             "rejection log")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the writer prompt for the first smoke batch and exit; "
                             "no API calls are made")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the two judge checks and run only the parrot check; for "
                             "testing when Ollama is not available")
    args = parser.parse_args()

    splits = split_sizes(T.total())
    args.out.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    if args.smoke:
        # Two tasks per route. The start position is offset per route, because taking the first
        # two of every route only ever lands on rotation positions 0 and 1 and would show two
        # of the three writers.
        offsets = {route: i % len(WRITERS) for i, route in enumerate(sorted(PER_ROUTE_ITEMS))}
        picked, seen = [], {}
        for index, spec in enumerate(specs):
            route = spec_route(spec)
            position = seen.get(route, 0)
            seen[route] = position + 1
            if offsets[route] <= position < offsets[route] + 2:
                picked.append(index)
        specs = [specs[i] for i in picked]

    if args.show_prompt:
        first = [spec for spec in specs if spec["generator"] == WRITERS[0]][:BATCH_SIZE]
        print(build_prompt(list(enumerate(first, start=1))))
        return

    if args.no_judge:
        print("WARNING: --no-judge set; checks B and C are skipped, only the parrot check runs",
              flush=True)
    accepted, failed = write_sentences(specs, args.out, use_judge=not args.no_judge)
    if failed:
        report_failures(failed, args.out)
        print(f"\nwriter: {TIMING['writer_calls']} calls, {TIMING['writer_s']/60:.1f} min")
        print(f"judge:  {TIMING['judge_calls']} calls, {TIMING['judge_s']/60:.1f} min")
        sys.exit(1)
    records = [make_record(spec, accepted[spec["task_id"]]) for spec in specs]

    if args.smoke:
        for record in records:
            print(json.dumps(record, indent=1, ensure_ascii=False))
        return

    out = split_by_route(records, splits)

    for name, expected in splits:
        path = args.out / name
        with path.open("w") as handle:
            for record in out[name]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{path}: {len(out[name])} records (target {expected})")

    print(f"\nwriter: {TIMING['writer_calls']} calls, {TIMING['writer_s']/60:.1f} min")
    print(f"judge:  {TIMING['judge_calls']} calls, {TIMING['judge_s']/60:.1f} min")


if __name__ == "__main__":
    main()
