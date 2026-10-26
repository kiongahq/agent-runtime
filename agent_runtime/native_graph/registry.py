"""Component registry: node kinds, typed ports and conversions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("components.v1.json")


@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text())


def components() -> dict[str, dict[str, Any]]:
    return {item["kind"]: item for item in registry()["components"]}


def component(kind: str) -> dict[str, Any] | None:
    return components().get(kind)


def conversions() -> set[tuple[str, str]]:
    return {(item["from"], item["to"]) for item in registry()["conversions"]}


def output_port(kind: str, name: str) -> dict[str, Any] | None:
    spec = component(kind)
    return next((p for p in spec["outputs"] if p["name"] == name), None) if spec else None


def input_port(kind: str, name: str) -> dict[str, Any] | None:
    spec = component(kind)
    return next((p for p in spec["inputs"] if p["name"] == name), None) if spec else None


def compatible(output_type: str, accepts: list[str]) -> bool:
    """An output feeds an input when the type is accepted or one registered
    conversion turns it into an accepted type."""
    allowed = conversions()
    return output_type in accepts or any((output_type, target) in allowed for target in accepts)


def config_default(kind: str, key: str) -> str:
    spec = component(kind) or {"config": []}
    field = next((f for f in spec["config"] if f["key"] == key), None)
    return str(field.get("default", "")) if field else ""
