"""Demo event producer: `python -m realtime.produce --demo fraud --count 5`."""

from __future__ import annotations

import argparse

from . import demos
from .kafka import KafkaRest

_EVENTS = {
    "fraud": (
        demos.FRAUD_TOPIC_IN,
        [
            {"entity_id": "u123", "amount": 42.0, "merchant": "grocer", "region": "eu-west"},
            {"entity_id": "u789", "amount": 1800.0, "merchant": "electronics", "region": "ap-south"},
            {"entity_id": "u456", "amount": 12.5, "merchant": "coffee", "region": "us-east"},
        ],
    ),
    "callcenter": (
        demos.CALLCENTER_TOPIC_IN,
        [
            {"entity_id": "u123", "session_id": "call-1", "transcript": "I am angry, my pipeline is broken and I want a refund."},
            {"entity_id": "u456", "session_id": "call-2", "transcript": "Thanks, the issue is resolved and everything works great."},
        ],
    ),
    "recommendations": (
        demos.RECS_TOPIC_IN,
        [
            {"entity_id": "u123", "context": "pricing-page"},
            {"entity_id": "u456", "context": "docs"},
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=sorted(_EVENTS), required=True)
    parser.add_argument("--count", type=int, default=1)
    arguments = parser.parse_args()
    topic, samples = _EVENTS[arguments.demo]
    rest = KafkaRest()
    sent = 0
    for index in range(arguments.count):
        rest.produce(topic, samples[index % len(samples)])
        sent += 1
    print(f"produced {sent} event(s) to {topic}")


if __name__ == "__main__":
    main()
