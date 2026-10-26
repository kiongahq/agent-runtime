"""Serve a published native Agent Designer version through the Kionga agent protocol.

The Kubernetes operator mounts the immutable version artifact (contracts
``agent-graph/artifact.v1.schema.json``) at a fixed path. The runtime refuses an
artifact whose behavior hash does not match, so what runs is exactly what was
published. Resource nodes call the adjacent loopback ``kionga-api`` helper, which
holds the agent's project-scoped identity; no token enters this process or the
graph.
"""

from __future__ import annotations

import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent_runtime.graphs import GraphLoadError
from agent_runtime.native_graph import content_sha256, normalize, run_graph, validate_graph
from agent_runtime.native_graph.resources import GatewayResources, Resources

GRAPH_MODULE = "agents.native_graph:build_native"
LOOPBACK_API = "http://127.0.0.1:8890/api/v1"
MAX_SESSIONS = 1000
MAX_TURNS = 50


def load_artifact(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        raise GraphLoadError(f"published native agent version is missing at {path}")
    try:
        artifact = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise GraphLoadError(f"published native agent version is unreadable: {error}") from error
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != 1:
        raise GraphLoadError("unsupported native agent artifact version")
    if (artifact.get("runtime") or {}).get("graph_module") != GRAPH_MODULE:
        raise GraphLoadError("artifact is not a native agent graph")
    graph = artifact.get("graph")
    if not isinstance(graph, dict):
        raise GraphLoadError("artifact has no graph")
    if content_sha256(graph) != artifact.get("content_sha256"):
        raise GraphLoadError("artifact content hash does not match its graph; refusing to run it")
    issues = validate_graph(normalize(graph))
    if issues:
        raise GraphLoadError("artifact graph is invalid: " + "; ".join(issues))
    subflows = artifact.get("subflows") or {}
    if not isinstance(subflows, dict) or len(subflows) > 10:
        raise GraphLoadError("artifact subflows are invalid")
    for version, entry in subflows.items():
        nested = entry.get("graph") if isinstance(entry, dict) else None
        if not isinstance(nested, dict) or content_sha256(nested) != entry.get("content_sha256"):
            raise GraphLoadError(f"subflow {version} content hash does not match its graph; refusing to run it")
        if validate_graph(normalize(nested)):
            raise GraphLoadError(f"subflow {version} graph is invalid")
    return artifact


class NativeGraph:
    framework = "native"
    # Model nodes wired straight to the output stream their tokens.
    streaming_mode = "incremental"

    def __init__(self, artifact_path: Path | None = None, *, resources: Resources | None = None,
                 llm_factory: Any = None):
        path = artifact_path or Path(os.environ.get("KIONGA_NATIVE_GRAPH_PATH", "/app/flows/flow.json"))
        self.artifact = load_artifact(path)
        self.graph = normalize(self.artifact["graph"])
        self.subflows = {version: normalize(entry["graph"]) for version, entry in (self.artifact.get("subflows") or {}).items()}
        kinds = {node["id"]: node["kind"] for node in self.graph["nodes"]}
        # Only replies on their way to the user stream; intermediate model text never does.
        self.streamed = {edge["from"] for edge in self.graph["edges"]
                         if kinds.get(edge["to"]) == "output" and kinds.get(edge["from"]) == "model"}
        self.resources = resources or GatewayResources(LOOPBACK_API)
        self.llm_factory = llm_factory or _platform_llm
        self.sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    def _history(self, session_id: str) -> list[dict[str, str]]:
        if not session_id:
            return []
        history = self.sessions.pop(session_id, [])
        self.sessions[session_id] = history
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)
        return history

    async def _events(self, payload: dict[str, Any], config: dict[str, Any]):
        messages = payload.get("messages") or []
        message = messages[-1] if messages else None
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            raise ValueError("native agents require a text message")
        session_id = str(config.get("configurable", {}).get("thread_id", ""))
        history = self._history(session_id)
        async for event in run_graph(
            self.graph, input_text=message.content, run_id="agr-" + uuid.uuid4().hex[:24], history=history,
            allow_side_effects=True, resources=self.resources, llm_factory=self.llm_factory, subflows=self.subflows,
        ):
            yield event
            if event["type"] == "run.completed" and session_id:
                history.extend([{"role": "user", "text": message.content}, {"role": "assistant", "text": event.get("output", "")}])
                del history[:-MAX_TURNS]

    def _final(self, reply: str, usage: dict[str, Any], timings: list[dict[str, Any]], content: str) -> AIMessage:
        input_tokens, output_tokens = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        return AIMessage(
            content=content,
            usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens},
            response_metadata={"native_events": timings, "native_version": self.artifact.get("version"),
                               "content_sha256": self.artifact["content_sha256"]},
        )

    async def ainvoke(self, payload: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        reply, usage, timings, failure = "", {}, [], None
        async for event in self._events(payload, config):
            if event["type"] in ("node.completed", "node.failed", "node.skipped"):
                timings.append({key: event[key] for key in ("node_id", "kind", "type", "duration_ms") if key in event})
            if event["type"] == "run.completed":
                reply, usage = event.get("output", ""), event.get("usage", {})
            elif event["type"] in ("run.failed", "run.cancelled"):
                failure = event.get("error", {}).get("message", "run failed")
        if failure is not None:
            raise RuntimeError(f"native agent run failed: {failure}")
        return {"messages": [self._final(reply, usage, timings, reply)]}

    async def astream(self, payload: dict[str, Any], *, config: dict[str, Any], stream_mode: str = "messages"):
        """Yield token chunks from the model feeding the output as they arrive,
        then one final chunk carrying usage (and the reply, when nothing
        streamed, e.g. an output fed by a prompt or a tool)."""
        from langchain_core.messages import AIMessageChunk

        streamed_text, timings = [], []
        async for event in self._events(payload, config):
            if event["type"] == "node.delta" and event.get("node_id") in self.streamed:
                streamed_text.append(event["delta"])
                yield AIMessageChunk(content=event["delta"]), {}
            elif event["type"] in ("node.completed", "node.failed", "node.skipped"):
                timings.append({key: event[key] for key in ("node_id", "kind", "type", "duration_ms") if key in event})
            elif event["type"] == "run.completed":
                reply = event.get("output", "")
                rest = "" if streamed_text and "".join(streamed_text) == reply else reply
                yield self._final(reply, event.get("usage", {}), timings, rest), {}
            elif event["type"] in ("run.failed", "run.cancelled"):
                raise RuntimeError(f"native agent run failed: {event.get('error', {}).get('message', 'run failed')}")


def _platform_llm():
    from mlaiops_sdk.llm import build_chat_model

    return build_chat_model()


def build_native() -> NativeGraph:
    return NativeGraph()


build_native.kionga_framework = "native"
