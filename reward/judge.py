"""One yes/no question to the judge model.

Two backends, chosen by the JUDGE_BACKEND environment variable: "ollama" (default; qwen3:30b-a3b
served by Ollama on another machine on the local network) and "anthropic" (Claude through the
Messages API, used from Colab). Both get the same prompt text and must answer with the same
JSON object. The rest of this docstring describes the ollama path. The
judge answers yes/no questions with a one-sentence reason and is never asked for a score. Two
shapes of call: ask_with_reasoning asks one question; ask_batch asks every question about one
piece of context in a single call, so a rubric of seven items costs one round trip instead of
seven. Both are constrained with Ollama's JSON-schema `format`, so `verdict` can only be "yes"
or "no" and there is nothing to parse line by line. This module is shared by the rubric grader
in reward/ and by the validation filters in data/generate_tasks.py.

On this box the model generates inside its thinking span, so the JSON arrives in the reply's
`thinking` field and `response` comes back empty. The reader therefore looks at `thinking`
first and falls back to `response`.

Standard library only: urllib.request, no client package.
"""

import json
import os
import re
import threading
import urllib.request

# Which judge answers. "ollama" is the local box used for runs 1-7; "anthropic" calls the
# Messages API with Claude and is what the Colab notebook uses. Verdicts from the two are not
# comparable with each other (different models), so every run's summary records which one
# judged it (see judge_info()).
BACKEND = os.environ.get("JUDGE_BACKEND", "ollama")

ENDPOINT = "http://192.168.88.59:11434/api/generate"
MODEL = "qwen3:30b-a3b"
KEEP_ALIVE = "30m"          # a cold load costs about 40 seconds; keep the model resident
TIMEOUT_S = 600

ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_MAX_TOKENS = 400
MAX_PARALLEL = 4            # concurrent Anthropic calls; the scorer fans out over completions
_SEMAPHORE = threading.Semaphore(MAX_PARALLEL)
_ANTHROPIC_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def judge_info():
    """Which backend and model are judging, for run summaries."""
    if BACKEND == "anthropic":
        return {"backend": "anthropic", "model": ANTHROPIC_MODEL}
    return {"backend": "ollama", "model": MODEL, "endpoint": ENDPOINT}


def _anthropic_client():
    """One shared client; created on first use so the ollama path never imports the SDK.
    The SDK retries 429 and 5xx with backoff on its own (max_retries)."""
    global _ANTHROPIC_CLIENT
    with _CLIENT_LOCK:
        if _ANTHROPIC_CLIENT is None:
            import anthropic
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("JUDGE_BACKEND=anthropic needs ANTHROPIC_API_KEY in the environment")
            _ANTHROPIC_CLIENT = anthropic.Anthropic(max_retries=5)
        return _ANTHROPIC_CLIENT


def _call_anthropic(prompt, accept):
    """POST one prompt to the Messages API and return the JSON object in the reply text.
    Same prompt text as the ollama path; the schema is asked for in the prompt and checked by
    `accept` after parsing, since the Messages API does not constrain the output."""
    client = _anthropic_client()
    # No sampling parameters: anthropic 1.x removed `temperature` from messages.create (it is
    # rejected as an unexpected keyword). The verdict is constrained to yes/no by the prompt
    # and checked by `accept`, so determinism is not relied on.
    with _SEMAPHORE:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt + ANTHROPIC_SUFFIX}],
        )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    LAST.clear()
    LAST.update({"backend": "anthropic", "model": ANTHROPIC_MODEL,
                 "input_tokens": response.usage.input_tokens,
                 "output_tokens": response.usage.output_tokens,
                 "stop_reason": response.stop_reason})
    obj = _first_object(text, accept)
    if obj is not None:
        return obj
    raise ValueError(f"judge reply carried no schema-shaped JSON: {text[:200]!r}")


ANTHROPIC_SUFFIX = "\n\nReply with only the JSON object, no code fences."


def _strip_fences(text):
    """Remove markdown code fences (``` or ```json) wrapping the reply, if any."""
    lines = [l for l in text.strip().splitlines() if not l.strip().startswith("```")]
    return "\n".join(lines).strip()


def _first_object(text, accept):
    """The first JSON object in `text` that `accept` approves of. Tries the fence-stripped text
    as a whole, then every balanced {...} candidate from each opening brace, so prose or fences
    around the object, or a second object after it, do not fail the verdict."""
    text = _strip_fences(text)
    decoder = json.JSONDecoder()
    candidates = [text] if text.startswith("{") else []
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        candidates.append(obj)
    for cand in candidates:
        if isinstance(cand, str):
            try:
                cand = json.loads(cand)
            except json.JSONDecodeError:
                continue
        cand = _normalise_verdicts(cand)
        if accept(cand):
            return cand
    return None


def _normalise_verdicts(obj):
    """Lower-case and strip any "verdict" string, in an object or a list of objects, so "No"
    or " yes" satisfy the yes/no check the schema enforces on the ollama path."""
    items = obj if isinstance(obj, list) else [obj]
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("verdict"), str):
            item["verdict"] = item["verdict"].strip().lower()
    return obj

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["reasoning", "verdict"],
}

# One object per question, each carrying the question's id so the answers can be matched back.
BATCH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "reasoning": {"type": "string"},
            "verdict": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["id", "reasoning", "verdict"],
    },
}

# Details of the most recent call, for reports and benchmarks. Measurement only.
LAST = {}


def _call(prompt, schema, accept):
    """One judge call on the active backend. ollama: POST constrained by `schema` and return
    the first field ("thinking", then "response") whose text parses as JSON that `accept`
    approves of. anthropic: see _call_anthropic."""
    if BACKEND == "anthropic":
        return _call_anthropic(prompt, accept)
    if BACKEND != "ollama":
        raise ValueError(f"unknown JUDGE_BACKEND {BACKEND!r}; use 'ollama' or 'anthropic'")
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0},
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        data = json.loads(response.read().decode())

    LAST.clear()
    LAST.update({"eval_count": data.get("eval_count"),
                 "prompt_eval_count": data.get("prompt_eval_count"),
                 "total_duration_s": round((data.get("total_duration") or 0) / 1e9, 2),
                 "field": None})

    for field in ("thinking", "response"):
        text = (data.get(field) or "").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if accept(obj):
            LAST["field"] = field
            return obj

    raise ValueError("judge reply carried no schema-shaped JSON: "
                     f"thinking={data.get('thinking', '')[:150]!r} "
                     f"response={data.get('response', '')[:150]!r}")


def _is_verdict(obj):
    return isinstance(obj, dict) and obj.get("verdict") in ("yes", "no")


def _is_verdict_list(obj):
    return isinstance(obj, list) and all(_is_verdict(x) and isinstance(x.get("id"), str)
                                         for x in obj)


def ask_with_reasoning(question, context):
    """Return (verdict, reasoning). `context` is the material the question is about; `question`
    is the one thing being asked."""
    prompt = (f"{context.strip()}\n\n"
              f"Question: {question.strip()}\n\n"
              "Answer as a JSON object with two fields: \"reasoning\", one sentence, and "
              "\"verdict\", yes or no.")
    obj = _call(prompt, SCHEMA, _is_verdict)
    return obj["verdict"], str(obj.get("reasoning", "")).strip()


def ask_batch(items, context):
    """Ask every question in `items` about one `context` in a single call.

    Currently unused. It was measured against one call per item on 24 completions and set
    aside: 23 calls at 112.6s against 176 calls at 155.1s, only 1.4x faster, because the format
    schema had already brought a single call down to about 0.9s and the cost is now the
    generated tokens rather than the round trips. Meanwhile 18% of verdicts changed with the
    same model, temperature and questions, and two open-ended items dropped below 50%
    agreement. Worth revisiting if the judge changes to a slower model where round trips
    dominate again.

    `items` is a list of dicts with at least "id" and "question". Returns a list of
    (item_id, verdict, reasoning) in the order the items were given. The reply must answer
    every requested id exactly once and nothing else; otherwise this raises and names the ids
    that are missing, repeated, or unrequested, rather than filling in anything."""
    wanted = [item["id"] for item in items]
    lines = [f"- {item['id']}: {item['question'].strip()}" for item in items]
    prompt = (f"{context.strip()}\n\n"
              "Questions:\n" + "\n".join(lines) + "\n\n"
              "Answer as a JSON array with one object per question, in the same order, each "
              "with three fields: \"id\" copied exactly from the question, \"reasoning\", one "
              "sentence, and \"verdict\", yes or no.")
    answers = _call(prompt, BATCH_SCHEMA, _is_verdict_list)

    got = [a["id"] for a in answers]
    missing = [i for i in wanted if i not in got]
    extra = sorted(set(got) - set(wanted))
    repeated = sorted({i for i in got if got.count(i) > 1})
    if missing or extra or repeated:
        raise ValueError(f"judge batch reply did not answer the questions asked: "
                         f"missing={missing} extra={extra} repeated={repeated}")

    by_id = {a["id"]: a for a in answers}
    return [(i, by_id[i]["verdict"], str(by_id[i].get("reasoning", "")).strip()) for i in wanted]


def ask_yes_no(question, context):
    """Return "yes" or "no", backed by the same call."""
    return ask_with_reasoning(question, context)[0]


if __name__ == "__main__":
    print(ask_with_reasoning("Does this message say the device got wet?",
                             "Customer message: my phone fell in the sink and now it won't turn on."))
