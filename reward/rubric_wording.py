"""Versioned rubric question wording, applied by item id at scoring time.

The task files store each rubric item's question and the judge reads it from there. When a
question is reworded the task files are not regenerated; the scorer swaps in the wording of
the active version by item id (reward.score -> reword). Which version is active is chosen by
the scripts' --rubric flag: train/train_grpo.py defaults to v1 (so a run continues under the
wording earlier runs trained with), eval/before_after.py and eval/judge_agreement.py default
to v2. data/generate_tasks.py carries the current (v2) strings for any future generation, and
reward/test_judge.py checks the two agree.

- v1: the wording stored in the task files (runs 1-10 trained with it).
- v2 (2026-09-14): three judgment items rewritten after the Haiku/ollama agreement check
  showed them at 0.00-0.67 agreement.
"""

VERSIONS = {
    "v1": {},   # no override: the question text stored on the task record is used as-is
    "v2": {
        "explain_not_covered.grounded":
            "Does the reply name the specific plan rule that excludes this incident (for example, "
            "that loss and theft coverage applies to phones only)?",
        "explain_not_covered.alternative":
            "Does the reply suggest at least one concrete thing the customer can still do?",
        "refer_to_manufacturer.why":
            "Does the reply say that the manufacturer's warranty is still active and that it covers "
            "this kind of failure (a defect, malfunction, or wear), so the manufacturer handles it "
            "rather than the plan?",
    },
}
CURRENT = "v2"            # what data/generate_tasks.py writes into new task files
QUESTIONS = VERSIONS[CURRENT]

_active = {"version": CURRENT}


def set_version(version):
    if version not in VERSIONS:
        raise ValueError(f"unknown rubric wording version {version!r}; known: {sorted(VERSIONS)}")
    _active["version"] = version


def active_version():
    return _active["version"]


def reword(item):
    """The item with the active version's wording applied, if this id is reworded in it."""
    table = VERSIONS[_active["version"]]
    if item["id"] in table and item.get("question") != table[item["id"]]:
        return {**item, "question": table[item["id"]]}
    return item
