"""Gate tests for the real-time demo processors. No network, no Kafka."""

import httpx
import pytest

from realtime import demos
from realtime.kafka import Consumer, KafkaRest
from realtime.service import Stats, process_batch, process_event


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
    monkeypatch.setenv("KIONGA_URL", "http://gateway")
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
    monkeypatch.delenv("KIONGA_FEATURE_GATEWAY_URL", raising=False)
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
    rest.produce("kionga.transactions", {"entity_id": "u1"})
    assert recorded["path"] == "/topics/kionga.transactions"
    assert recorded["content_type"] == "application/vnd.kafka.json.v2+json"
    assert recorded["body"] == {"records": [{"value": {"entity_id": "u1"}}]}


def test_kafka_produce_fails_closed():
    rest = KafkaRest(
        "http://kafka-rest:8082",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
    )
    with pytest.raises(httpx.HTTPStatusError):
        rest.produce("t", {})


# -- at-least-once delivery ---------------------------------------------------


class FakeKafkaRest:
    """A Kafka REST proxy over one in-memory partition per topic.

    A new consumer instance resumes from the group's committed offset; the
    proxy commits ``offset + 1`` as Karapace and Confluent v2 do."""

    def __init__(self, log):
        self.log = log  # topic -> list of values
        self.committed = {}  # (topic, partition) -> next offset to read
        self.produced = []
        self.creates = []
        self.commits = []
        self.positions = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        path, body = request.url.path, json.loads(request.content) if request.content else None
        if request.method == "POST" and path.startswith("/consumers/") and path.count("/") == 2:
            self.creates.append(body)
            name = body["name"]
            return httpx.Response(200, json={"instance_id": name, "base_uri": f"http://kafka-rest:8082/consumers/g/instances/{name}"})
        if path.endswith("/subscription"):
            name = path.split("/")[-2]
            self.positions[name] = {(topic, 0): self.committed.get((topic, 0), 0) for topic in body["topics"]}
            return httpx.Response(204)
        if path.endswith("/records"):
            name = path.split("/")[-2]
            records = []
            for (topic, partition), position in self.positions[name].items():
                for offset in range(position, len(self.log.get(topic, []))):
                    records.append({"topic": topic, "partition": partition, "offset": offset, "key": None, "value": self.log[topic][offset]})
                self.positions[name][(topic, partition)] = len(self.log.get(topic, []))
            return httpx.Response(200, json=records)
        if path.endswith("/offsets"):
            self.commits.append(body)
            for entry in body["offsets"]:
                self.committed[(entry["topic"], entry["partition"])] = entry["offset"] + 1
            return httpx.Response(204)
        if path.startswith("/topics/"):
            self.produced.append((path.removeprefix("/topics/"), body["records"][0]["value"]))
            return httpx.Response(200, json={"offsets": []})
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(f"unexpected call {request.method} {path}")


def fake_kafka(log):
    fake = FakeKafkaRest(log)
    return fake, KafkaRest("http://kafka-rest:8082", client=httpx.Client(transport=httpx.MockTransport(fake.handler)))


def test_consumer_disables_auto_commit():
    fake, rest = fake_kafka({})
    Consumer(rest, group="g", topics=["t"], name="c1")
    assert fake.creates[0]["auto.commit.enable"] == "false"


def test_consumer_commits_highest_offset_per_partition():
    fake, rest = fake_kafka({})
    consumer = Consumer(rest, group="g", topics=["t"], name="c1")
    consumer.commit([
        {"topic": "a", "partition": 0, "offset": 3}, {"topic": "a", "partition": 0, "offset": 7},
        {"topic": "a", "partition": 1, "offset": 2}, {"topic": "b", "partition": 0, "offset": 0},
    ])
    assert fake.commits == [{"offsets": [
        {"topic": "a", "partition": 0, "offset": 7}, {"topic": "a", "partition": 1, "offset": 2},
        {"topic": "b", "partition": 0, "offset": 0},
    ]}]
    consumer.commit([])
    assert len(fake.commits) == 1


def test_produce_failure_mid_batch_commits_nothing_and_replays(monkeypatch):
    """A batch of three: the second produce fails. No offset is committed, a
    recreated consumer fetches all three again, and after recovery every
    record has a result carrying its source offset."""
    monkeypatch.delenv("KIONGA_URL", raising=False)
    monkeypatch.delenv("KIONGA_FEATURE_GATEWAY_URL", raising=False)
    fake, rest = fake_kafka({demos.RECS_TOPIC_IN: [{"entity_id": f"u{i}"} for i in range(3)]})
    client, stats = offline_client(), Stats()

    consumer = Consumer(rest, group="g", topics=[demos.RECS_TOPIC_IN], name="c1")
    original = fake.handler
    calls = {"produce": 0}

    def flaky(request):
        if request.url.path.startswith("/topics/"):
            calls["produce"] += 1
            if calls["produce"] == 2:
                return httpx.Response(500)
        return original(request)

    rest.client = httpx.Client(transport=httpx.MockTransport(flaky))
    with pytest.raises(httpx.HTTPStatusError):
        process_batch(consumer.poll(), consumer=consumer, rest=rest, client=client, stats=stats)
    assert fake.commits == []
    assert fake.committed == {}
    consumer.close()

    rest.client = httpx.Client(transport=httpx.MockTransport(original))
    retry = Consumer(rest, group="g", topics=[demos.RECS_TOPIC_IN], name="c2")
    records = retry.poll()
    assert [record["offset"] for record in records] == [0, 1, 2]
    process_batch(records, consumer=retry, rest=rest, client=client, stats=stats)
    assert fake.committed == {(demos.RECS_TOPIC_IN, 0): 3}
    offsets = [value["source_offset"] for _, value in fake.produced]
    # u0 was produced before the failure and again on replay: at least once.
    assert offsets == [
        {"topic": demos.RECS_TOPIC_IN, "partition": 0, "offset": 0},
        {"topic": demos.RECS_TOPIC_IN, "partition": 0, "offset": 0},
        {"topic": demos.RECS_TOPIC_IN, "partition": 0, "offset": 1},
        {"topic": demos.RECS_TOPIC_IN, "partition": 0, "offset": 2},
    ]
    assert retry.poll() == []


def test_commit_happens_after_every_produce():
    """Ordering: the offsets request is the last call of a batch."""
    order = []
    fake, rest = fake_kafka({demos.RECS_TOPIC_IN: [{"entity_id": "u1"}, {"entity_id": "u2"}, "not-a-dict"]})
    original = fake.handler

    def recording(request):
        order.append(request.url.path.rsplit("/", 1)[-1] if "/instances/" in request.url.path else request.url.path)
        return original(request)

    rest.client = httpx.Client(transport=httpx.MockTransport(recording))
    consumer = Consumer(rest, group="g", topics=[demos.RECS_TOPIC_IN], name="c1")
    records = consumer.poll()
    order.clear()
    process_batch(records, consumer=consumer, rest=rest, client=offline_client(), stats=Stats())
    assert order == [f"/topics/{demos.RECS_TOPIC_OUT}", f"/topics/{demos.RECS_TOPIC_OUT}", "offsets"]
    # The skipped malformed record is committed too, so it is not re-read forever.
    assert fake.committed == {(demos.RECS_TOPIC_IN, 0): 3}
