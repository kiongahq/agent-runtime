"""Structural validation of a normalized graph before it runs.

The gateway is the authority on resources and permissions; the runtime still
refuses anything it cannot execute safely: unknown kinds, bad IDs, missing
ports, incompatible types, cycles and out-of-range budgets.
"""

from __future__ import annotations

import re
from typing import Any

from . import registry

NODE_ID = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
SETTING_LIMITS = {"max_steps": (1, 200), "timeout_seconds": (1, 300), "max_tokens": (1, 1_000_000)}


def topological_order(graph: dict[str, Any]) -> list[str] | None:
    """Kahn's algorithm, stable in node declaration order. ``None`` on a cycle."""
    ids = [node["id"] for node in graph["nodes"]]
    indegree = {node_id: 0 for node_id in ids}
    children: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for edge in graph["edges"]:
        indegree[edge["to"]] += 1
        children[edge["from"]].append(edge["to"])
    position = {node_id: index for index, node_id in enumerate(ids)}
    ready = sorted((node_id for node_id in ids if indegree[node_id] == 0), key=position.get)
    order = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=position.get)
    return order if len(order) == len(ids) else None


def validate_graph(graph: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(graph, dict):
        return ["The graph must be a JSON object."]
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 100:
        return ["A graph needs 2-100 nodes."]
    if not isinstance(edges, list) or len(edges) > 300:
        return ["A graph can have at most 300 connections."]
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not NODE_ID.match(str(node.get("id", ""))):
            issues.append("Every node needs a lowercase ID.")
            continue
        if node["id"] in by_id:
            issues.append(f"Node ID {node['id']!r} is used twice.")
            continue
        if registry.component(str(node.get("kind", ""))) is None:
            issues.append(f"Node {node['id']!r} has an unsupported type.")
        config = node.get("config") or {}
        if not isinstance(config, dict) or len(config) > 30 or any(
            not isinstance(key, str) or not isinstance(value, str) or len(key) > 80 or len(value) > 16000
            for key, value in config.items()
        ):
            issues.append(f"Node {node['id']!r} has invalid settings.")
        by_id[node["id"]] = node
    kinds = [node.get("kind") for node in by_id.values()]
    if kinds.count("input") != 1 or kinds.count("output") != 1:
        issues.append("A graph needs exactly one input and one output node.")
    if issues:
        return issues
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in by_id or edge.get("to") not in by_id \
                or edge["from"] == edge["to"]:
            issues.append("Connections must link two different existing nodes.")
            continue
        source, target = by_id[edge["from"]], by_id[edge["to"]]
        out = registry.output_port(source["kind"], str(edge.get("from_port", "")))
        port = registry.input_port(target["kind"], str(edge.get("to_port", "")))
        if out is None or port is None:
            issues.append(f"Connection {edge['from']} -> {edge['to']} uses a port that does not exist.")
            continue
        if not registry.compatible(out["type"], port["accepts"]):
            issues.append(f"Connection {edge['from']} -> {edge['to']} joins incompatible types.")
        key = (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        if key in seen:
            issues.append(f"Duplicate connection {edge['from']} -> {edge['to']}.")
        seen.add(key)
    if issues:
        return issues
    if topological_order(graph) is None:
        issues.append("Connections contain a cycle.")
    settings = graph.get("settings") or {}
    for key, (low, high) in SETTING_LIMITS.items():
        value = settings.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high):
            issues.append(f"Setting {key} must be an integer from {low} to {high}.")
    return issues
