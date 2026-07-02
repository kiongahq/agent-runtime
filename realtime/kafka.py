"""Kafka access through the Kafka REST Proxy (Confluent v2 contract).

The platform already runs kafka-rest for the Go services; using the same
door here keeps the Python side dependency-free (httpx only) and identical
in local and deployed environments.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx

_JSON_V2 = "application/vnd.kafka.json.v2+json"


class KafkaRest:
    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or os.environ.get("KAFKA_REST_URL", "http://localhost:8082")).rstrip("/")
        self.client = client or httpx.Client(timeout=15)

    def produce(self, topic: str, value: dict[str, Any]) -> None:
        response = self.client.post(
            f"{self.base_url}/topics/{topic}",
            content=json.dumps({"records": [{"value": value}]}),
            headers={"Content-Type": _JSON_V2},
        )
        response.raise_for_status()


class Consumer:
    """One consumer-group instance subscribed to a set of topics."""

    def __init__(self, rest: KafkaRest, group: str, topics: list[str], name: str | None = None) -> None:
        self.rest = rest
        self.group = group
        self.name = name or f"{group}-{uuid.uuid4().hex[:8]}"
        create = self.rest.client.post(
            f"{self.rest.base_url}/consumers/{self.group}",
            content=json.dumps(
                {"name": self.name, "format": "json", "auto.offset.reset": "earliest"}
            ),
            headers={"Content-Type": "application/vnd.kafka.v2+json"},
        )
        create.raise_for_status()
        self.instance_url = create.json()["base_uri"].replace(
            "http://kafka-rest:8082", self.rest.base_url
        )
        subscribe = self.rest.client.post(
            f"{self.instance_url}/subscription",
            content=json.dumps({"topics": topics}),
            headers={"Content-Type": "application/vnd.kafka.v2+json"},
        )
        subscribe.raise_for_status()

    def poll(self, timeout_ms: int = 1000) -> list[dict[str, Any]]:
        response = self.rest.client.get(
            f"{self.instance_url}/records",
            params={"timeout": timeout_ms},
            headers={"Accept": _JSON_V2},
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        try:
            self.rest.client.delete(
                self.instance_url,
                headers={"Content-Type": "application/vnd.kafka.v2+json"},
            )
        except httpx.HTTPError:
            pass
