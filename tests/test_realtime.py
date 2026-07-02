"""Gate tests for the real-time demo processors. No network, no Kafka."""

import httpx
import pytest

from realtime import demos
from realtime.kafka import KafkaRest
from realtime.service import Stats, process_event


def offline_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network call: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


# -- fraud --------------------------------------------------------------------


def test_fraud_rule_flags_velocity_and_region_mismatch(monkeypatch):
    monkeypatch.delenv("FRAUD_MODEL_ENDPOINT", raising=False)
    features = {"txn_count_5m": 9, "txn_amount_5m": 2350.0, "home_region": "us-west"}
    event = {"entity_id": "u789", "amount": 1800.0, "region": "ap-south"}
    result = demos.score_fraud(event, features, client=offline_client())
    assert result["flagged"] is True
    assert result["source"] == "rule"


def test_fraud_rule_passes_normal_transaction(monkeypatch):
    monkeypatch.delenv("FRAUD_MODEL_ENDPOINT", raising=False)
    features = {"txn_count_5m": 0, "txn_amount_5m": 0.0, "home_region": "us-east"}
    event = {"entity_id": "u456", "amount": 12.5, "region": "us-east"}
    result = demos.score_fraud(event, features, client=offline_client())
    assert result["flagged"] is False


def test_fraud_scoring_deterministic(monkeypatch):
    monkeypatch.delenv("FRAUD_MODEL_ENDPOINT", raising=False)
    features = {"txn_count_5m": 3, "txn_amount_5m": 500.0, "home_region": "eu-west"}
    event = {"entity_id": "u123", "amount": 250.0, "region": "eu-west"}
    first = demos.score_fraud(event, features, client=offline_client())
    second = demos.score_fraud(event, features, client=offline_client())
    assert first == second


def test_fraud_uses_live_model_when_configured(monkeypatch):
    monkeypatch.setenv("FRAUD_MODEL_ENDPOINT", "http://model:5001")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/invocations"
        return httpx.Response(200, json={"predictions": [0.91]})

    result = demos.score_fraud(
        {"entity_id": "u1", "amount": 10.0},
        {},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert result["score"] == 0.91
    assert result["source"] == "model"


# -- call center --------------------------------------------------------------


def test_callcenter_rule_sentiment(monkeypatch):
    monkeypatch.delenv("CALLCENTER_AGENT_ID", raising=False)
    negative = demos.analyze_transcript(
        {"session_id": "c1", "transcript": "I am angry, this is broken, refund now."},
        client=offline_client(),
    )
    assert negative["sentiment"] == "negative"
    positive = demos.analyze_transcript(
        {"session_id": "c2", "transcript": "Thanks, resolved, works great!"},
        client=offline_client(),
    )
    assert positive["sentiment"] == "positive"


def test_callcenter_routes_through_agent(monkeypatch):
    monkeypatch.setenv("MLAIOPS_URL", "http://gateway")
    monkeypatch.setenv("CALLCENTER_AGENT_ID", "agt-1")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/agents/agt-1/invoke"
        return httpx.Response(200, json={"reply": "sentiment=negative intent=refund"})

    result = demos.analyze_transcript(
        {"session_id": "c1", "transcript": "refund please"},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert result["source"] == "agent"
    assert "refund" in result["analysis"]


# -- recommendations ----------------------------------------------------------


def test_recommendations_rank_by_plan():
    pro = demos.recommend({"entity_id": "u123"}, {"plan": "pro"})
    assert pro["recommendations"][0] == "gpu-quota-upgrade"
    free = demos.recommend({"entity_id": "u456"}, {"plan": "free"})
    assert free["recommendations"][0] == "starter-onboarding-course"
    assert pro["recommendations"] != free["recommendations"]


# -- service routing ----------------------------------------------------------


def test_process_event_routes_topics(monkeypatch):
    monkeypatch.delenv("MLAIOPS_FEATURE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("CALLCENTER_AGENT_ID", raising=False)
    monkeypatch.delenv("FRAUD_MODEL_ENDPOINT", raising=False)
    demo, out_topic, result = process_event(
        demos.RECS_TOPIC_IN, {"entity_id": "u1"}, client=offline_client()
    )
    assert demo == "recommendations"
    assert out_topic == demos.RECS_TOPIC_OUT
    assert result["recommendations"]


def test_stats_snapshot_measures_latency():
    stats = Stats()
    stats.record("fraud", 10.0, flagged=True)
    stats.record("fraud", 20.0, flagged=False)
    snapshot = stats.snapshot("fraud")
    assert snapshot == {"events": 2, "flagged": 1, "avg_latency_ms": 15.0}


def test_kafka_produce_uses_confluent_envelope():
    recorded = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        recorded["path"] = request.url.path
        recorded["content_type"] = request.headers["content-type"]
        recorded["body"] = json.loads(request.content)
        return httpx.Response(200, json={"offsets": []})

    rest = KafkaRest("http://kafka-rest:8082", client=httpx.Client(transport=httpx.MockTransport(handler)))
    rest.produce("mlaiops.transactions", {"entity_id": "u1"})
    assert recorded["path"] == "/topics/mlaiops.transactions"
    assert recorded["content_type"] == "application/vnd.kafka.json.v2+json"
    assert recorded["body"] == {"records": [{"value": {"entity_id": "u1"}}]}


def test_kafka_produce_fails_closed():
    rest = KafkaRest(
        "http://kafka-rest:8082",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
    )
    with pytest.raises(httpx.HTTPStatusError):
        rest.produce("t", {})
