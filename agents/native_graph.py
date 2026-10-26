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
    return artifact


class NativeGraph:
    framework = "native"
    streaming_mode = "complete"

    def __init__(self, artifact_path: Path | None = None, *, resources: Resources | None = None,
                 llm_factory: Any = None):
        path = artifact_path or Path(os.environ.get("KIONGA_NATIVE_GRAPH_PATH", "/app/flows/flow.json"))
        self.artifact = load_artifact(path)
        self.graph = normalize(self.artifact["graph"])
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

    async def ainvoke(self, payload: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        message = messages[-1] if messages else None
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            raise ValueError("native agents require a text message")
        session_id = str(config.get("configurable", {}).get("thread_id", ""))
        history = self._history(session_id)
        reply, usage, timings, failure = "", {}, [], None
        async for event in run_graph(
            self.graph, input_text=message.content, run_id="agr-" + uuid.uuid4().hex[:24],
            history=history, allow_side_effects=True, resources=self.resources, llm_factory=self.llm_factory,
        ):
            if event["type"] in ("node.completed", "node.failed", "node.skipped"):
                timings.append({key: event[key] for key in ("node_id", "kind", "type", "duration_ms") if key in event})
            if event["type"] == "run.completed":
                reply, usage = event.get("output", ""), event.get("usage", {})
            elif event["type"] in ("run.failed", "run.cancelled"):
                failure = event.get("error", {}).get("message", "run failed")
        if failure is not None:
            raise RuntimeError(f"native agent run failed: {failure}")
        if session_id:
            history.extend([{"role": "user", "text": message.content}, {"role": "assistant", "text": reply}])
            del history[:-MAX_TURNS]
        input_tokens, output_tokens = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        return {"messages": [AIMessage(
            content=reply,
            usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens},
            response_metadata={"native_events": timings, "native_version": self.artifact.get("version"),
                               "content_sha256": self.artifact["content_sha256"]},
        )]}

    async def astream(self, payload: dict[str, Any], *, config: dict[str, Any], stream_mode: str = "messages"):
        result = await self.ainvoke(payload, config=config)
        yield result["messages"][-1], {}


def _platform_llm():
    from mlaiops_sdk.llm import build_chat_model

    return build_chat_model()


def build_native() -> NativeGraph:
    return NativeGraph()


build_native.kionga_framework = "native"
