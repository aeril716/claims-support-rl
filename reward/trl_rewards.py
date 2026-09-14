"""The scorer in reward/reward.py, adapted to TRL's GRPOTrainer reward-function interface.

TRL calls each reward function as f(prompts, completions, **kwargs), where every extra dataset
column arrives as a keyword argument of the same name, and expects a list with one number per
completion (or None to exclude that completion from that function).

Two functions are exposed on purpose, so TRL logs the route curve and the rubric curve under
their own names and they can be read independently during training. They share one scoring
pass: route_reward calls score() once per completion and stores each detail dict; rubric_reward
reads those details and never calls score() itself. That is what keeps the judge from being
called twice per step.

Order dependency: TRL iterates reward_funcs in list order (verified in
GRPOTrainer._calculate_rewards), so pass reward_funcs=[route_reward, rubric_reward] and route
always runs first. rubric_reward checks that the stored details match the batch it was given
and raises otherwise, rather than returning values for the wrong completions.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from reward import judge
from reward import reward as R

# Detail dicts from the most recent route_reward call, one per completion, in order.
# Cleared at the start of route_reward only; rubric_reward only reads it.
DETAILS = []

TASK_COLUMN = "task"


def _check_task_column(kwargs):
    """Fail loudly if the task column is not JSON strings. A struct column would otherwise
    surface as "string indices must be integers" deep inside score()."""
    values = kwargs[TASK_COLUMN]
    if not all(isinstance(value, str) for value in values):
        bad = next(type(value).__name__ for value in values if not isinstance(value, str))
        raise TypeError(f"dataset column {TASK_COLUMN!r} must hold each task as a JSON string; "
                        f"got {bad}. Build the dataset with data/build_dataset.py.")
    return values


def _parse_tasks(kwargs):
    """The one place the task column is parsed. It is a JSON string, not an Arrow struct,
    because Arrow unifies a struct's schema across the column and silently turns
    account={} into a dict of Nones; a string round-trips unchanged."""
    return [json.loads(value) for value in _check_task_column(kwargs)]


def _text(completion):
    """Completions arrive as plain text or, for conversational datasets, as a list of
    messages. Either way the scorer wants the text."""
    if isinstance(completion, list):
        return "".join(message.get("content", "") for message in completion)
    return completion


def route_reward(prompts, completions, **kwargs):
    """Score every completion once and return its route score: 1.0 when the chosen route
    matches the answer key, 0.0 otherwise, including when the output did not parse."""
    tasks = _parse_tasks(kwargs)
    DETAILS.clear()
    # The ollama box serves one request at a time, so its path stays sequential and identical
    # to runs 1-7. The anthropic backend allows judge.MAX_PARALLEL calls in flight, so the
    # completions are scored in a thread pool; map() keeps the order.
    workers = judge.MAX_PARALLEL if judge.BACKEND == "anthropic" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda pair: R.score(pair[0], _text(pair[1])), zip(tasks, completions)))
    scores = []
    for _reward, detail in results:
        DETAILS.append(detail)
        scores.append(float(detail["route_score"]))
    return scores


def rubric_reward(prompts, completions, **kwargs):
    """Return the rubric score for each completion from the details stored by route_reward.

    A completion that did not parse gets None, which excludes it from this reward function
    only. Parsing success is a format property and the rubric score is a content property;
    keeping them apart stops the rubric curve from moving when the parse rate moves. The
    decision is made on detail["parsed"]: the failure branch of score() carries no
    "rubric_score" key at all, so key presence is not something to count on."""
    _check_task_column(kwargs)
    if len(DETAILS) != len(completions):
        raise RuntimeError(
            f"rubric_reward saw {len(completions)} completions but route_reward stored "
            f"{len(DETAILS)} details. route_reward must run first on the same batch: pass "
            "reward_funcs=[route_reward, rubric_reward] in that order.")
    return [float(detail["rubric_score"]) if detail["parsed"] else None for detail in DETAILS]
