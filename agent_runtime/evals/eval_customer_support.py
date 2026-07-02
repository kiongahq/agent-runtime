"""Customer-support agent eval: golden Q/A scored by an LLM judge.

Paid lane. Runs only when ``MLAIOPS_RUN_EVALS=1`` and a real LLM backend is
configured (``MLAIOPS_LLM_BACKEND`` + provider key). The judge uses the same
provider. Pass threshold: mean score >= 0.7 and no case below 0.4.

Run:
    MLAIOPS_RUN_EVALS=1 OPENAI_API_KEY=... python -m pytest python/agent_runtime/evals -q
or standalone:
    MLAIOPS_RUN_EVALS=1 python python/agent_runtime/evals/eval_customer_support.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

MEAN_THRESHOLD = 0.7
CASE_FLOOR = 0.4

JUDGE_PROMPT = """You are grading a customer support agent's answer.

Question: {question}
Expected facts: {expected}
Agent answer: {answer}

Score how faithfully the agent answer covers the expected facts on a scale
from 0.0 (wrong or missing) to 1.0 (fully correct). Extra correct detail is
fine; contradictions are not. Reply with only the numeric score."""


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    score: float


def golden_cases() -> list[dict]:
    path = Path(__file__).with_name("golden.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_score(text: str) -> float:
    match = re.search(r"(?:0?\.\d+|1\.0|0|1)", text)
    if not match:
        raise ValueError(f"judge returned no score: {text!r}")
    return max(0.0, min(1.0, float(match.group())))


def run_eval() -> list[CaseResult]:
    from agents.customer_support.graph import build
    from mlaiops_sdk.llm import build_chat_model

    agent = build()
    judge = build_chat_model()
    results = []
    for case in golden_cases():
        state = agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"[customer entity_id={case['user_id']}] {case['question']}",
                    )
                ]
            }
        )
        answer = str(state["messages"][-1].content)
        verdict = judge.invoke(
            JUDGE_PROMPT.format(
                question=case["question"], expected=case["expected"], answer=answer
            )
        )
        results.append(
            CaseResult(
                case_id=case["id"],
                question=case["question"],
                answer=answer,
                score=_parse_score(str(verdict.content)),
            )
        )
    return results


def summarize(results: list[CaseResult]) -> tuple[float, bool]:
    mean = sum(result.score for result in results) / len(results)
    passed = mean >= MEAN_THRESHOLD and all(result.score >= CASE_FLOOR for result in results)
    return mean, passed


def test_customer_support_eval():
    import pytest

    if os.environ.get("MLAIOPS_RUN_EVALS") != "1":
        pytest.skip("evals are the paid lane; set MLAIOPS_RUN_EVALS=1")
    results = run_eval()
    mean, passed = summarize(results)
    detail = "\n".join(f"  {r.case_id}: {r.score:.2f} — {r.answer[:80]}" for r in results)
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
