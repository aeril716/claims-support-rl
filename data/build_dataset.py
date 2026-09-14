"""Build the TRL training dataset from task files.

    python data/build_dataset.py [task file ...] [--save <dir>]

Two columns, and only two:
- "prompt": the fully rendered policy prompt string. Assembled by train_grpo.load_tasks (the
  messages) and train_grpo.render_prompt (the chat template with the generation prompt
  appended), so there is exactly one definition of the training prompt. Stored rendered, as a
  string, TRL takes it as-is; rendering it here rather than letting TRL render a message list
  was verified to give the same bytes.
- "task": the whole task dict as a JSON string, which TRL hands to the reward functions as
  kwargs["task"]; reward/trl_rewards.py reads that name and parses it. It is a string, not a
  dict, on purpose: Arrow unifies a struct column's schema across all rows, so a task with
  `account: {}` came back as `{"enrolled_days_ago": None, ...}` with no error. One string per
  row round-trips byte for byte. Keys are sorted so equal dicts always give the same string.

Nothing here reads data/judge_only/.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "data"))

import train_grpo                                  # noqa: E402  (the one prompt definition)

POLICY_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_FILES = [ROOT / "data" / "v3_kb_definitions" / "tasks_train.jsonl"]


def check_json_round_trip(task):
    """Fail the build if this task would not survive json.dumps/json.loads unchanged. Catches a
    value that is not JSON-serialisable, a tuple that would come back as a list, or an int key
    that would come back as a string. Every task comes from jsonl, so this should always pass;
    the check stays regardless."""
    back = json.loads(json.dumps(task, ensure_ascii=False, sort_keys=True))
    if back != task:
        changed = sorted(k for k in set(task) | set(back) if task.get(k) != back.get(k))
        detail = "; ".join(f"{k}: {task.get(k)!r} -> {back.get(k)!r}" for k in changed)
        raise AssertionError(f"task {task.get('task_id')} does not round-trip through JSON: {detail}")


def build(files, tokenizer, prompt_version="v1", reasoning=False):
    """A datasets.Dataset with columns prompt (str) and task (JSON str), one row per task.

    The prompt version and the reasoning switch are passed in and set on this module's copy of
    train_grpo before anything is rendered. They are arguments, not globals read from the
    caller, because when train/train_grpo.py runs as a script it is the module `__main__` and
    the `import train_grpo` above loads a second copy; a global set in one is invisible to the
    other. run7's first launch (2026-09-13) trained on prompt v1 that way while logging v7."""
    from datasets import Dataset

    train_grpo.PROMPT_VERSION = prompt_version
    train_grpo.REASONING = reasoning
    rows = []
    for path in files:
        tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rendered = train_grpo.load_tasks(path)
        assert len(tasks) == len(rendered)
        for task, row in zip(tasks, rendered):
            assert task["task_id"] == row["task_id"]
            check_json_round_trip(task)
            rows.append({"prompt": train_grpo.render_prompt(row["prompt"], tokenizer),
                         "task": json.dumps(task, ensure_ascii=False, sort_keys=True)})
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--save", type=Path, default=None, help="save_to_disk here")
    parser.add_argument("--prompt-version", default="v1", help="as train/train_grpo.py --prompt-version")
    parser.add_argument("--reasoning", action="store_true", help="as train/train_grpo.py --reasoning")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(POLICY_TOKENIZER)
    dataset = build(args.files, tokenizer, args.prompt_version, args.reasoning)
    print(f"rows: {len(dataset)}   columns: {dataset.column_names}")
    if args.save:
        dataset.save_to_disk(str(args.save))
        print(f"saved to {args.save}")
    return dataset


if __name__ == "__main__":
    main()
