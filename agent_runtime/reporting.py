"""Deterministic usage extraction and gateway session reporting.

Token counts come from LangChain ``usage_metadata`` on AI messages — never
estimated in latent space. Cost is computed from configured per-1k rates
(``MLAIOPS_COST_PER_1K_INPUT`` / ``MLAIOPS_COST_PER_1K_OUTPUT``), defaulting
to zero when unset so numbers are exact, not guessed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from mlaiops_sdk.tracing import record_trace


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        input_rate = float(os.environ.get("MLAIOPS_COST_PER_1K_INPUT", "0"))
        output_rate = float(os.environ.get("MLAIOPS_COST_PER_1K_OUTPUT", "0"))
        return round(
            self.input_tokens / 1000 * input_rate + self.output_tokens / 1000 * output_rate,
            6,
        )


def usage_from_messages(messages: list[Any]) -> Usage:
    """Sum LangChain ``usage_metadata`` across all AI messages in a run."""
    input_tokens = output_tokens = 0
    for message in messages:
        metadata = getattr(message, "usage_metadata", None)
        if metadata:
            input_tokens += int(metadata.get("input_tokens", 0))
            output_tokens += int(metadata.get("output_tokens", 0))
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def report_session(
    *,
    agent_id: str,
    session_id: str,
    user_id: str,
    name: str,
    status: str,
    current_node: str,
    duration_ms: int,
    usage: Usage,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Send one turn summary to the gateway; upserts the live session there."""
    record_trace(
        {
            "agent_id": agent_id,
            "session_id": session_id,
            "user_id": user_id,
            "name": name,
            "status": status,
            "current_node": current_node,
            "duration_ms": duration_ms,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "metadata": metadata or {},
        }
    )
