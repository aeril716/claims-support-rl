"""Turn one policy output into one number.

    reward = 0.5 * route_score + 0.5 * (rubric_passed / len(rubric))

The policy must emit one JSON object {"route": ..., "reply": ...}. Output that is recognisably
that object with a syntax slip is repaired and scored on the repaired text; output that is not a
JSON object at all scores 0. The line is drawn at the object: the parser never scans loose text
for a route name, because that would let it invent an answer the model never committed to.
Every repair is logged with the original text and whether the recovered route was correct,
because the advantage is applied to the tokens the model actually emitted, so a syntax slip
attached to correct content gets reinforced along with the content.

Code items go to rubric_checks, judge items go to the judge one question at a time, and an item
passes when the answer equals its `expect`. One judge item is added at scoring time on every
task: whether the reply is consistent with the route the model chose. Batching the judge items
into one call per completion was measured and set aside; see the Lessons learned in CLAUDE.md.
"""

import json
import re
import threading
import time

from reward import rubric_checks
from reward import rubric_wording
from reward.judge import ask_with_reasoning

ROUTE_WEIGHT = 0.5

# Wall time spent inside judge calls, for the smoke-test report. Measurement only.
TIMING = {"judge_s": 0.0, "judge_calls": 0, "judge_each_s": []}

# Every repaired completion, appended by parse_output. The training script writes this out.
REPAIRS = []

# What each route commits the agent to saying. The consistency item reads the chosen route's
# meaning to the judge, so "consistent" has a definite referent.
ROUTE_MEANING = {
    "ask_question": "the agent does not yet have enough information and asks the customer for "
                    "the missing detail instead of deciding",
    "file_claim": "the incident is covered and the customer can file a claim",
    "tech_support": "the problem is a software problem that tech support handles at no charge; "
                    "it is not a claim",
    "refer_to_manufacturer": "the manufacturer's warranty is responsible, so the customer is "
                             "sent to the manufacturer",
    "explain_waiting_period": "the incident falls inside the 31-day waiting period, so "
                              "coverage has not started yet",
    "explain_not_covered": "this incident is not covered for this device",
    "escalate": "the case is being handed to a person",
}

# A bare (unquoted) string value after a colon: one word (`tech_support`) or several separated by
# single spaces (`not stated`), ending at the next comma, brace or bracket. Applied only to the
# parts of the text that are outside quoted strings (see _outside_strings), so a colon inside a
# reply never triggers it.
_BARE_VALUE = re.compile(r'(:\s*)([A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z0-9_]+)*)(\s*[,}\]])')


def _outside_strings(text):
    """Split `text` into (is_string, segment) pieces, honouring backslash escapes inside strings,
    so a repair can be applied to the parts that are JSON syntax and not to string contents."""
    pieces, buf, in_string, i = [], [], False, 0
    while i < len(text):
        ch = text[i]
        if in_string:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(text):
                buf.append(text[i + 1]); i += 1
            elif ch == '"':
                pieces.append((True, "".join(buf))); buf, in_string = [], False
        else:
            if ch == '"':
                if buf: pieces.append((False, "".join(buf)))
                buf, in_string = [ch], True
            else:
                buf.append(ch)
        i += 1
    if buf: pieces.append((in_string, "".join(buf)))
    return pieces


# The fact fields prompt v5 asks for before the route. Logged only, never scored: every check
# runs on `reply`, and the route check on `route`. A missing or malformed value becomes None.
FACT_FIELDS = ["device", "incident", "enrolled_days_ago", "inside_waiting_period",
               "inside_manufacturer_warranty", "claim_limit_reached"]


def _fact(value):
    """A fact field as written: a stripped string, a number, or None for anything else."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _as_object(candidate):
    """The parsed record if `candidate` is the object the task asks for, else None. `route`
    and `reply` are required strings. `reasoning` (prompt --reasoning) and the FACT_FIELDS
    (prompt v5) are optional: stored for reading, never scored, None when absent or malformed."""
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    route, reply = obj.get("route"), obj.get("reply")
    if not isinstance(route, str) or not isinstance(reply, str):
        return None
    reasoning = obj.get("reasoning")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else None
    facts = {name: _fact(obj.get(name)) for name in FACT_FIELDS}
    return {"route": route.strip(), "reply": reply.strip(), "reasoning": reasoning, "facts": facts}


def _repair(candidate):
    """Fix the three syntax slips seen in practice, and only those. Returns (text, repairs)."""
    repairs = []
    fixed = candidate

    # 1. an unquoted string value, e.g. "route": tech_support or "device": not stated
    def quote(match):
        word = match.group(2)
        if word in ("true", "false", "null"):
            return match.group(0)
        return f'{match.group(1)}"{word}"{match.group(3)}'
    quoted = "".join(seg if is_str else _BARE_VALUE.sub(quote, seg)
                     for is_str, seg in _outside_strings(fixed))
    if quoted != fixed:
        repairs.append("quoted bare value")
        fixed = quoted

    # 2. a stray ']' with no '[' to match it
    if fixed.count("]") > fixed.count("["):
        fixed = fixed.replace("]", "", fixed.count("]") - fixed.count("["))
        repairs.append("removed stray ]")

    # 3. the object never closed
    if fixed.count("{") > fixed.count("}"):
        fixed = fixed + "}" * (fixed.count("{") - fixed.count("}"))
        repairs.append("added closing brace")

    return fixed, repairs


def parse_record(text):
    """A dict {route, reply, reasoning, facts, repairs} from the policy's text. route and reply
    are None when the text is not the object the task asks for.

    The candidate is the span from the first '{' to the last '}' (or to the end of the text if
    the object was never closed), so words around the object do not fail it. If that span is the
    right object, repairs is empty. If it is the right object with one of the three known syntax
    slips, it is repaired and repairs says what was done. Anything else returns (None, None, [])
    and scores 0: the parser does not go looking for a route name in loose text."""
    empty = {"route": None, "reply": None, "reasoning": None,
             "facts": {name: None for name in FACT_FIELDS}, "repairs": []}
    start = text.find("{")
    if start == -1:
        return empty
    end = text.rfind("}")
    candidate = text[start:end + 1] if end > start else text[start:]

    parsed = _as_object(candidate)
    if parsed is not None:
        return {**parsed, "repairs": []}

    fixed, repairs = _repair(candidate)
    parsed = _as_object(fixed) if repairs else None
    if parsed is None:
        return empty
    return {**parsed, "repairs": repairs}


def parse_fields(text):
    """(route, reply, reasoning, repairs): parse_record without the fact fields."""
    r = parse_record(text)
    return r["route"], r["reply"], r["reasoning"], r["repairs"]


def parse_output(text):
    """(route, reply, repairs): for callers that only score. Neither the reasoning nor the fact
    fields is an input to any check, so dropping them here changes nothing."""
    r = parse_record(text)
    return r["route"], r["reply"], r["repairs"]


_TIMING_LOCK = threading.Lock()


def judge_item(item, customer, reply):
    """One judge item, one call. Returns (verdict, reasoning). The counters are shared across
    the threads trl_rewards may use, hence the lock."""
    context = (f"Customer message: \"{customer}\"\n\n"
               f"Support agent reply: \"{reply}\"")
    start = time.time()
    try:
        verdict, reasoning = ask_with_reasoning(item["question"], context)
    finally:
        elapsed = time.time() - start
        with _TIMING_LOCK:
            TIMING["judge_s"] += elapsed
            TIMING["judge_calls"] += 1
            TIMING["judge_each_s"].append(round(elapsed, 2))
    return verdict, reasoning


def consistency_item(route):
    """The scoring-time item: does the reply say something consistent with the chosen route?
    Built here rather than stored on the task because it depends on the route the model
    picked, which is not known until the completion exists."""
    meaning = ROUTE_MEANING.get(route)
    question = (f"The agent chose the route \"{route}\", which means: {meaning}. "
                f"Does the reply tell the customer something consistent with that, rather "
                f"than the opposite?")
    return {"id": "consistency.route_reply", "check": "judge", "question": question,
            "expect": "yes"}


def score(task, text):
    """Reward for one completion, plus the per-item detail behind it."""
    record = parse_record(text)
    route, reply, reasoning_text, repairs = (record["route"], record["reply"],
                                             record["reasoning"], record["repairs"])
    if route is None:
        return 0.0, {"parsed": False, "repaired": False, "route_score": 0.0, "items": []}

    route_score = 1.0 if route == task["route_answer"] else 0.0
    if repairs:
        REPAIRS.append({"task_id": task.get("task_id"), "original": text, "repairs": repairs,
                        "route": route, "route_correct": route_score == 1.0})

    items = []
    for item in task["rubric"] + [consistency_item(route)]:
        item = rubric_wording.reword(item)   # current question wording by id (see rubric_wording.py)
        reasoning = None                # code items and the unknown-route shortcut have none
        if item["check"] == "code":
            answer = rubric_checks.answer(item, reply)
        elif item["id"] == "consistency.route_reply" and route not in ROUTE_MEANING:
            answer = "no"          # an unknown route cannot be consistent with anything
        else:
            answer, reasoning = judge_item(item, task["customer"], reply)
        items.append({"id": item["id"], "answer": answer, "expect": item["expect"],
                      "passed": answer == item["expect"], "reasoning": reasoning})
    rubric_score = sum(i["passed"] for i in items) / len(items)
    reward = ROUTE_WEIGHT * route_score + (1 - ROUTE_WEIGHT) * rubric_score
    # `reasoning` and `facts` below are the policy's own fields, kept for reading only. Every
    # check above ran on `reply` alone (code items and judge items both receive `reply`, never
    # these), and the route check on `route`.
    return reward, {"parsed": True, "repaired": bool(repairs), "repairs": repairs,
                    "route": route, "route_score": route_score,
                    "rubric_score": rubric_score, "items": items, "reasoning": reasoning_text,
                    "facts": record["facts"]}
