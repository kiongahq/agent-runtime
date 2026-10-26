"""Canonical behavior document and its SHA-256 (contracts ``agent-graph/README.md``).

Names, positions and groups are excluded, so moving or relabelling a node never
changes the hash of what a published version does.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(graph: dict[str, Any]) -> str:
    nodes = sorted(
        ({"id": node["id"], "kind": node["kind"],
          "config": {key: value for key, value in (node.get("config") or {}).items() if value != ""}}
         for node in graph["nodes"]),
        key=lambda node: node["id"],
    )
    edges = []
    for edge in graph["edges"]:
        item = {"from": edge["from"], "to": edge["to"]}
        for key in ("from_port", "to_port"):
            if edge.get(key):
                item[key] = edge[key]
        edges.append(item)
    edges.sort(key=lambda e: (e["from"], e.get("from_port", ""), e["to"], e.get("to_port", "")))
    document = {"schema_version": 2, "nodes": nodes, "edges": edges,
                "settings": graph.get("settings") or {}}
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # Go's encoder always escapes these two line separators; match it byte for byte.
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def content_sha256(graph: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
