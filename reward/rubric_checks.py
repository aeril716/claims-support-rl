"""The rubric items a script can answer without a judge.

Two items on every task are counted rather than judged: the word cap and the question cap.
Each function answers the item's yes/no question the same way the judge would, so reward.py
can treat code items and judge items alike: an item passes when the answer equals `expect`.
"""

import re

WORD_CAP = 60
QUESTION_CAP = 2


def word_count(reply):
    return len(reply.split())


def question_count(reply):
    """Question marks, not sentences: "Which device, and when?" is one question mark and is
    treated as one question, which is the lenient reading."""
    return reply.count("?")


def answer(item, reply):
    """Return "yes" or "no" for a code item, keyed by the item id so the wording of the
    question can change without touching this file."""
    if item["id"] == "common.word_count":
        return "yes" if word_count(reply) <= WORD_CAP else "no"
    if item["id"] == "common.question_cap":
        return "yes" if question_count(reply) <= QUESTION_CAP else "no"
    raise ValueError(f"no code check for rubric item {item['id']!r}")
