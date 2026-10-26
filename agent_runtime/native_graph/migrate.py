"""Upgrade a stored graph to schema version 2 in memory.

Rules follow contracts ``agent-graph/README.md``. Normalizing is idempotent, so
the runner normalizes whatever it receives: the gateway sends version 2, but a
version 1 draft posted directly still runs the same way.
"""

from __future__ import annotations

import copy
from typing import Any

from . import registry

DEFAULT_SETTINGS = {"max_steps": 50, "timeout_seconds": 60, "max_tokens": 50000}


def default_from_port(kind: str) -> str:
    spec = registry.component(kind)
    if spec and spec["outputs"]:
        return spec["outputs"][0]["name"]
    return "out"


def default_to_port(kind: str) -> str:
    spec = registry.component(kind)
    if spec and spec["inputs"]:
        return spec["inputs"][0]["name"]
    return "in"


def normalize(graph: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(graph)
    result["schema_version"] = 2
    nodes = result.get("nodes") or []
    kinds = {node.get("id"): node.get("kind", "") for node in nodes if isinstance(node, dict)}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        config = node.get("config")
        node["config"] = dict(config) if isinstance(config, dict) else {}
        if node.get("kind") == "model" and not node["config"].get("model"):
            node["config"]["model"] = "llm"
    for edge in result.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if not edge.get("from_port"):
            edge["from_port"] = default_from_port(kinds.get(edge.get("from"), ""))
        if not edge.get("to_port"):
            edge["to_port"] = default_to_port(kinds.get(edge.get("to"), ""))
    settings = result.get("settings") if isinstance(result.get("settings"), dict) else {}
    result["settings"] = {**DEFAULT_SETTINGS, **settings}
    return result
