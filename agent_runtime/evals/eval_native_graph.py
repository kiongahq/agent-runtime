"""Native graph eval: starter templates run through the interpreter against a real LLM.

Paid lane. Runs only when ``MLAIOPS_RUN_EVALS=1`` and a real LLM backend is
configured (``MLAIOPS_LLM_BACKEND`` + provider key). Scoring is deterministic:

- ``structured-extraction``: the reply parses as JSON and each expected field
  matches (case-insensitive); the score is the fraction of matching fields.
- ``customer-support``: the condition routes to the expected branch (exact), and
  the reply mentions every expected keyword; the score averages the two.

Pass threshold: mean score >= 0.7 and no case below 0.4.

Run:
    MLAIOPS_RUN_EVALS=1 MLAIOPS_LLM_BACKEND=openai OPENAI_API_KEY=... \
        python -m pytest agent_runtime/evals/eval_native_graph.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

MEAN_THRESHOLD = 0.7
CASE_FLOOR = 0.4
TEMPLATES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "native_graph" / "templates"
GOLDEN = Path(__file__).with_name("native_graph_golden.jsonl")


@dataclass
class CaseResult:
    case_id: str
    reply: str
    score: float


def golden_cases() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        value = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        value = None
    return value if isinstance(value, dict) else {}


def score_case(case: dict, events: list[dict]) -> tuple[str, float]:
    last = events[-1]
    if last["type"] != "run.completed":
        return f"<{last['type']}: {last.get('error', {}).get('message', '')}>", 0.0
    reply = last["output"]
    if case["template"] == "structured-extraction":
        found = _json_object(reply)
        expected = case["expected"]
        matches = sum(
            str(found.get(key)).strip().lower() == str(value).strip().lower() for key, value in expected.items()
        )
        return reply, matches / len(expected)
    route = next(e.get("port") for e in events if e.get("node_id") == "route" and e["type"] == "node.completed")
    routed = 1.0 if route == case["expected_route"] else 0.0
    keywords = case["expected_keywords"]
    mentioned = sum(keyword.lower() in reply.lower() for keyword in keywords) / len(keywords)
    return reply, (routed + mentioned) / 2


async def _run_case(case: dict) -> list[dict]:
    from mlaiops_sdk.llm import build_chat_model

    from agent_runtime.native_graph import normalize, run_graph

    graph = normalize(json.loads((TEMPLATES / f"{case['template']}.json").read_text())["graph"])
    return [event async for event in run_graph(
        graph, input_text=case["input"], run_id="agr-" + "e" * 24, llm_factory=build_chat_model,
    )]


def run_eval() -> list[CaseResult]:
    results = []
    for case in golden_cases():
        reply, score = score_case(case, asyncio.run(_run_case(case)))
        results.append(CaseResult(case["id"], reply, score))
    return results


def summarize(results: list[CaseResult]) -> tuple[float, bool]:
    mean = sum(result.score for result in results) / len(results)
    return mean, mean >= MEAN_THRESHOLD and all(result.score >= CASE_FLOOR for result in results)


def test_native_graph_eval():
    import pytest

    if os.environ.get("MLAIOPS_RUN_EVALS") != "1":
        pytest.skip("evals are the paid lane; set MLAIOPS_RUN_EVALS=1")
    results = run_eval()
    mean, passed = summarize(results)
    detail = "\n".join(f"  {r.case_id}: {r.score:.2f} — {r.reply[:80]}" for r in results)
    assert passed, f"eval below threshold (mean={mean:.2f}, floor={CASE_FLOOR}):\n{detail}"


if __name__ == "__main__":
    if os.environ.get("MLAIOPS_RUN_EVALS") != "1":
        raise SystemExit("set MLAIOPS_RUN_EVALS=1 to run the paid eval lane")
    eval_results = run_eval()
    eval_mean, eval_passed = summarize(eval_results)
    for result in eval_results:
        print(f"{result.case_id}: {result.score:.2f}")
    print(f"mean={eval_mean:.2f} threshold={MEAN_THRESHOLD} -> {'PASS' if eval_passed else 'FAIL'}")
    raise SystemExit(0 if eval_passed else 1)
