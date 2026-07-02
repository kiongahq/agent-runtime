"""The three real-time demo processors.

Each processor is a pure-ish function: event in, enrichment via the online
feature store, decision out. Scoring falls back to deterministic rules when
no live model/agent endpoint is configured, so the demos run on a bare stack
and upgrade themselves when the full platform is up.
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

FRAUD_TOPIC_IN = "mlaiops.transactions"
FRAUD_TOPIC_OUT = "mlaiops.fraud.alerts"
CALLCENTER_TOPIC_IN = "mlaiops.callcenter.transcripts"
CALLCENTER_TOPIC_OUT = "mlaiops.callcenter.insights"
RECS_TOPIC_IN = "mlaiops.user.activity"
RECS_TOPIC_OUT = "mlaiops.recs.results"

_CATALOG = [
    {"item": "gpu-quota-upgrade", "audience": "pro"},
    {"item": "pipeline-templates-pack", "audience": "pro"},
    {"item": "starter-onboarding-course", "audience": "free"},
    {"item": "team-seat-bundle", "audience": "free"},
    {"item": "observability-addon", "audience": "any"},
]

_NEGATIVE_TERMS = {"angry", "refund", "cancel", "broken", "terrible", "worst", "frustrated"}
_POSITIVE_TERMS = {"thanks", "great", "love", "resolved", "perfect", "happy"}


def get_features(entity_id: str, feature_view: str, *, client: httpx.Client) -> dict[str, Any]:
    gateway = os.environ.get("MLAIOPS_FEATURE_GATEWAY_URL")
    if not gateway:
        return {}
    response = client.post(
        f"{gateway.rstrip('/')}/get-online-features",
        json={"feature_service": feature_view, "entities": [{"entity_id": entity_id}]},
        timeout=5,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0].get("values", {}) if results else {}


def score_fraud(event: dict[str, Any], features: dict[str, Any], *, client: httpx.Client) -> dict[str, Any]:
    """Score one transaction. Uses the live model endpoint when configured,
    otherwise a deterministic logistic rule over amount, velocity, and
    region mismatch."""
    amount = float(event.get("amount", 0))
    endpoint = os.environ.get("FRAUD_MODEL_ENDPOINT")
    if endpoint:
        response = client.post(
            f"{endpoint.rstrip('/')}/invocations",
            json={"inputs": [[amount, float(features.get("txn_count_5m", 0)), float(features.get("txn_amount_5m", 0))]]},
            timeout=10,
        )
        response.raise_for_status()
        score = float(response.json()["predictions"][0])
    else:
        velocity = float(features.get("txn_count_5m", 0))
        recent = float(features.get("txn_amount_5m", 0))
        region_mismatch = 1.0 if event.get("region") and features.get("home_region") and event["region"] != features["home_region"] else 0.0
        z = -4.0 + 0.002 * amount + 0.35 * velocity + 0.001 * recent + 2.2 * region_mismatch
        score = round(1 / (1 + math.exp(-z)), 4)
    return {
        "entity_id": event.get("entity_id", ""),
        "amount": amount,
        "score": score,
        "flagged": score >= 0.5,
        "source": "model" if endpoint else "rule",
    }


def analyze_transcript(event: dict[str, Any], *, client: httpx.Client) -> dict[str, Any]:
    """Analyze one call transcript. Routes through the deployed support agent
    when configured, otherwise deterministic keyword sentiment."""
    transcript = str(event.get("transcript", ""))
    gateway = os.environ.get("MLAIOPS_URL")
    agent_id = os.environ.get("CALLCENTER_AGENT_ID")
    if gateway and agent_id:
        response = client.post(
            f"{gateway.rstrip('/')}/api/v1/agents/{agent_id}/invoke",
            json={
                "message": "Summarize sentiment and intent in one line: " + transcript,
                "session_id": str(event.get("session_id", "")),
                "user_id": str(event.get("entity_id", "")),
            },
            timeout=60,
        )
        response.raise_for_status()
        return {
            "session_id": event.get("session_id", ""),
            "analysis": response.json().get("reply", ""),
            "source": "agent",
        }
    words = {word.strip(".,!?").lower() for word in transcript.split()}
    negative = len(words & _NEGATIVE_TERMS)
    positive = len(words & _POSITIVE_TERMS)
    sentiment = "negative" if negative > positive else "positive" if positive > negative else "neutral"
    return {
        "session_id": event.get("session_id", ""),
        "analysis": f"sentiment={sentiment} negative_terms={negative} positive_terms={positive}",
        "sentiment": sentiment,
        "source": "rule",
    }


def recommend(event: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Rank catalog items for the user's plan, deterministically."""
    plan = str(features.get("plan", "free"))
    ranked = sorted(
        _CATALOG,
        key=lambda item: (0 if item["audience"] == plan else 1 if item["audience"] == "any" else 2, item["item"]),
    )
    return {
        "entity_id": event.get("entity_id", ""),
        "plan": plan,
        "recommendations": [item["item"] for item in ranked[:3]],
    }
