"""Real-time consumer service.

Consumes the three demo topics, enriches events with online features, scores
them, produces results to the output topics, and reports live statistics to
the gateway so the console's Real-Time panel shows actual throughput and
latency. Stats are measured, never estimated.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import httpx

from . import demos
from .kafka import Consumer, KafkaRest

TOPIC_HANDLERS = {
    demos.FRAUD_TOPIC_IN: ("fraud", demos.FRAUD_TOPIC_OUT),
    demos.CALLCENTER_TOPIC_IN: ("callcenter", demos.CALLCENTER_TOPIC_OUT),
    demos.RECS_TOPIC_IN: ("recommendations", demos.RECS_TOPIC_OUT),
}


class Stats:
    def __init__(self) -> None:
        self.counters: dict[str, dict[str, float]] = defaultdict(
            lambda: {"events": 0, "flagged": 0, "total_latency_ms": 0.0}
        )

    def record(self, demo: str, latency_ms: float, flagged: bool = False) -> None:
        entry = self.counters[demo]
        entry["events"] += 1
        entry["total_latency_ms"] += latency_ms
        if flagged:
            entry["flagged"] += 1

    def snapshot(self, demo: str) -> dict[str, Any]:
        entry = self.counters[demo]
        events = int(entry["events"])
        return {
            "events": events,
            "flagged": int(entry["flagged"]),
            "avg_latency_ms": round(entry["total_latency_ms"] / events, 2) if events else 0,
        }


def process_event(topic: str, event: dict[str, Any], *, client: httpx.Client) -> tuple[str, str, dict[str, Any]]:
    """Route one event through its demo processor. Returns (demo, out_topic, result)."""
    demo, out_topic = TOPIC_HANDLERS[topic]
    if demo == "fraud":
        features = demos.get_features(str(event.get("entity_id", "")), "transaction_stats_5m", client=client)
        return demo, out_topic, demos.score_fraud(event, features, client=client)
    if demo == "callcenter":
        return demo, out_topic, demos.analyze_transcript(event, client=client)
    features = demos.get_features(str(event.get("entity_id", "")), "customer_profile", client=client)
    return demo, out_topic, demos.recommend(event, features)


def report_stats(stats: Stats, demo: str, *, client: httpx.Client) -> None:
    gateway = os.environ.get("MLAIOPS_URL")
    if not gateway:
        return
    try:
        client.post(
            f"{gateway.rstrip('/')}/api/v1/realtime/{demo}",
            json=stats.snapshot(demo),
            headers={"X-MLAIOps-Actor": "realtime-processor"},
            timeout=5,
        )
    except httpx.HTTPError:
        pass


def run_forever() -> None:  # pragma: no cover - long-running loop
    rest = KafkaRest()
    client = httpx.Client(timeout=30)
    consumer = Consumer(rest, group="mlaiops-realtime", topics=list(TOPIC_HANDLERS))
    print(f"realtime processor consuming {list(TOPIC_HANDLERS)}")
    try:
        while True:
            for record in consumer.poll(timeout_ms=2000):
                topic, value = record.get("topic", ""), record.get("value", {})
                if topic not in TOPIC_HANDLERS or not isinstance(value, dict):
                    continue
                started = time.perf_counter()
                demo, out_topic, result = process_event(topic, value, client=client)
                latency_ms = (time.perf_counter() - started) * 1000
                rest.produce(out_topic, result)
                _stats.record(demo, latency_ms, flagged=bool(result.get("flagged")))
                report_stats(_stats, demo, client=client)
            time.sleep(0.2)
    finally:
        consumer.close()


_stats = Stats()


if __name__ == "__main__":  # pragma: no cover
    run_forever()
