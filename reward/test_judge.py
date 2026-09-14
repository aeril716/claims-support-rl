"""Offline tests for the anthropic judge path: parsing of awkward replies and the guarantee
that a scoring run never dies on a judge reply. Run with:

    python -m unittest reward.test_judge

No network: the anthropic SDK is replaced by a stub that returns scripted replies.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The reply that killed run8 on Colab at step 6: a json fence, a long reasoning, cut by
# max_tokens before the closing brace.
TRUNCATED_FENCED = (
    '```json\n{\n  "reasoning": "The reply provides the deductible amount, explains how to start '
    'the claim, lists the documents that may be requested, and reassures the customer at length '
    'about the repair timeline and the replacement device, and it continues to describe the '
    'return label and the ten-day window and then the'
)


class _Usage:
    input_tokens = 100
    output_tokens = 50


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Message:
    def __init__(self, text, stop_reason):
        self.content, self.usage, self.stop_reason = [_Block(text)], _Usage(), stop_reason


class _Messages:
    def __init__(self, script):
        self.script, self.seen = script, []

    def create(self, **kwargs):
        self.seen.append(kwargs["messages"][0]["content"])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text, stop = item
        return _Message(text, stop)


def install_stub(script):
    fake = types.ModuleType("anthropic")
    messages = _Messages(script)
    fake.Anthropic = lambda **kwargs: types.SimpleNamespace(messages=messages)
    sys.modules["anthropic"] = fake
    os.environ["ANTHROPIC_API_KEY"] = "test"
    os.environ["JUDGE_BACKEND"] = "anthropic"
    from reward import judge
    importlib.reload(judge)
    return judge, messages


CONTEXT = 'Customer message: "x"\n\nSupport agent reply: "y"'


class AnthropicJudgeTests(unittest.TestCase):
    def test_truncated_fenced_reply_never_raises(self):
        judge, messages = install_stub([(TRUNCATED_FENCED, "max_tokens"),
                                        ('{"reasoning": "short", "verdict": "yes"}', "end_turn")])
        verdict, reasoning = judge.ask_with_reasoning("Does the reply say the incident is covered?", CONTEXT)
        self.assertEqual((verdict, reasoning), ("yes", "short"))
        self.assertEqual(len(messages.seen), 2, "one stricter retry after the truncated reply")
        self.assertEqual(judge.judge_failures(), 0)

    def test_truncated_fenced_reply_with_end_turn_is_still_treated_as_truncated(self):
        judge, messages = install_stub([(TRUNCATED_FENCED, "end_turn"),
                                        ('{"reasoning": "short", "verdict": "no"}', "end_turn")])
        self.assertEqual(judge.ask_with_reasoning("q", CONTEXT), ("no", "short"))
        self.assertEqual(len(messages.seen), 2)

    def test_two_bad_replies_become_a_counted_fallback(self):
        judge, _ = install_stub([(TRUNCATED_FENCED, "max_tokens"), ("still no json", "end_turn")])
        self.assertEqual(judge.ask_with_reasoning("q", CONTEXT), ("no", "judge_parse_failure"))
        self.assertEqual(judge.judge_failures(), 1)
        self.assertEqual(judge.judge_info()["judge_failures"], 1)

    def test_api_error_becomes_a_counted_fallback(self):
        judge, _ = install_stub([RuntimeError("connection reset")])
        verdict, reasoning = judge.ask_with_reasoning("q", CONTEXT)
        self.assertEqual(verdict, "no")
        self.assertTrue(reasoning.startswith("judge_api_failure"))
        self.assertEqual(judge.judge_failures(), 1)

    def test_fenced_and_capitalised_reply_parses(self):
        judge, messages = install_stub([('```json\n{"reasoning": "r", "verdict": "No"}\n```', "end_turn")])
        self.assertEqual(judge.ask_with_reasoning("q", CONTEXT), ("no", "r"))
        self.assertEqual(len(messages.seen), 1)

    def test_prose_around_object_parses(self):
        judge, _ = install_stub([('Sure. {"reasoning": "r", "verdict": "yes"} Hope this helps.', "end_turn")])
        self.assertEqual(judge.ask_with_reasoning("q", CONTEXT), ("yes", "r"))

    def test_score_survives_a_failed_item(self):
        import json
        judge, _ = install_stub([(TRUNCATED_FENCED, "max_tokens"), ("nope", "end_turn")]
                                + [('{"reasoning": "ok", "verdict": "yes"}', "end_turn")] * 20)
        from reward import reward
        importlib.reload(reward)
        task_file = Path(__file__).resolve().parent.parent / "data" / "v3_kb_definitions" / "tasks_train_sub35.jsonl"
        task = next(json.loads(l) for l in open(task_file) if '"t047"' in l)
        _reward, detail = reward.score(task, '{"route": "explain_waiting_period", "reply": "Coverage starts on day 31."}')
        failed = [i for i in detail["items"] if i["reasoning"] == "judge_parse_failure"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(judge.judge_failures(), 1)


class RubricWordingTests(unittest.TestCase):
    def test_generator_and_scorer_wording_agree(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
        import generate_tasks
        from reward import rubric_wording
        for route, items in generate_tasks.PER_ROUTE_ITEMS.items():
            for suffix, question, _expect in items:
                iid = f"{route}.{suffix}"
                if iid in rubric_wording.QUESTIONS:
                    self.assertEqual(question, rubric_wording.QUESTIONS[iid], iid)

    def test_reword_replaces_only_listed_ids(self):
        from reward import rubric_wording
        old = {"id": "refer_to_manufacturer.why", "check": "judge", "question": "Does the reply explain why?", "expect": "yes"}
        self.assertEqual(rubric_wording.reword(old)["question"], rubric_wording.QUESTIONS["refer_to_manufacturer.why"])
        other = {"id": "file_claim.covered", "check": "judge", "question": "Does the reply say the incident is covered?", "expect": "yes"}
        self.assertIs(rubric_wording.reword(other), other)


if __name__ == "__main__":
    unittest.main()
