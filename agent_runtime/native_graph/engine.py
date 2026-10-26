"""Async interpreter for native agent graphs.

``run_graph`` executes a validated graph in topological order and yields events
matching contracts ``agent-graph/run-event.v1.schema.json``. Execution rules:

- A node with inbound edges runs when at least one of them carried a value;
  otherwise it is reported as ``node.skipped``. A node with no inbound edges
  (the input, or a source such as a dataset loader) always runs.
- A condition emits a value on exactly one port (``true`` or ``false``).
- Each node gets its component's timeout and retry budget, bounded by the run's
  ``timeout_seconds``; the run also enforces ``max_steps`` and ``max_tokens``.
- ``tool`` and ``api`` nodes fail with ``side_effects_not_approved`` before any
  call unless the caller allowed side effects.
- Previews and error messages are redacted and truncated before they are emitted.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from ..reporting import Usage
from . import conditions, registry
from .resources import ResourceError, Resources
from .validate import topological_order

PREVIEW_LIMIT = 2000
OUTPUT_LIMIT = 65536
ERROR_LIMIT = 500
RETRYABLE = {"timeout", "node_error", "resource_unavailable"}
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9-]{0,62})\}")
SECRET_PATTERNS = [
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|access[_-]?key|client[_-]?secret|secret|password|passwd|token)"
                r"(\"?\s*[=:]\s*\"?)([^\s\"',;}]+)"), r"\1\2[REDACTED]"),
]


def redact(text: str, limit: int = PREVIEW_LIMIT) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class NodeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Value:
    type: str
    data: Any


def to_text(value: Value) -> str:
    if value.type == "text":
        return str(value.data)
    if value.type == "json":
        return value.data if isinstance(value.data, str) else json.dumps(value.data, ensure_ascii=False, separators=(",", ":"))
    if value.type == "documents":
        return "\n\n".join(str(doc.get("text", "")) for doc in value.data)
    if value.type == "chat":
        return str(value.data.get("text", ""))
    return str(value.data)


def convert(value: Value, accepts: list[str]) -> Value:
    if value.type in accepts:
        return value
    for target in accepts:
        if (value.type, target) not in registry.conversions():
            continue
        if target == "text":
            return Value("text", to_text(value))
        if target == "chat":
            return Value("chat", {"text": to_text(value), "history": []})
        if target == "json":
            text = to_text(value)
            try:
                return Value("json", json.loads(text))
            except json.JSONDecodeError:
                return Value("json", {"input": text})
    raise NodeError("invalid_graph", f"cannot convert {value.type} to {' or '.join(accepts)}")


@dataclass
class RunState:
    input_text: str
    history: list[dict[str, str]]
    outputs: dict[str, Value] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def render(template: str, state: RunState, context: str) -> str:
    """Single-pass literal substitution: values are never re-scanned, so text
    from a user or an upstream node cannot inject further placeholders."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "input":
            return state.input_text
        if name == "context":
            return context
        if name in state.outputs:
            return to_text(state.outputs[name])
        return match.group(0)

    return PLACEHOLDER.sub(replace, template)


def _config(node: dict[str, Any], key: str) -> str:
    value = (node.get("config") or {}).get(key, "")
    return value if value != "" else registry.config_default(node["kind"], key)


def _required(node: dict[str, Any], key: str) -> str:
    value = _config(node, key).strip()
    if not value:
        raise NodeError("invalid_graph", f"node {node['id']} needs {key}")
    return value


def _integer(node: dict[str, Any], key: str) -> int:
    spec = next(f for f in registry.component(node["kind"])["config"] if f["key"] == key)
    raw = _config(node, key)
    try:
        value = int(raw)
    except ValueError:
        raise NodeError("invalid_graph", f"node {node['id']} {key} must be an integer") from None
    if not spec["minimum"] <= value <= spec["maximum"]:
        raise NodeError("invalid_graph", f"node {node['id']} {key} must be {spec['minimum']}-{spec['maximum']}")
    return value


def _resource_error(error: ResourceError) -> NodeError:
    code = "resource_denied" if error.status in (401, 403, 404) else "resource_unavailable"
    return NodeError(code, str(error))


async def _call(coroutine):
    try:
        return await coroutine
    except ResourceError as error:
        raise _resource_error(error) from None


class Executor:
    def __init__(self, resources: Resources, llm_factory: Callable[[], Any] | None, allow_side_effects: bool):
        self.resources = resources
        self.llm_factory = llm_factory
        self.allow_side_effects = allow_side_effects
        self._llm = None

    def llm(self):
        if self._llm is None:
            if self.llm_factory is None:
                raise NodeError("resource_unavailable", "no platform chat model is configured")
            self._llm = self.llm_factory()
        return self._llm

    async def execute(self, node: dict[str, Any], value: Value | None, context: str,
                      state: RunState) -> tuple[str, Value, Usage | None]:
        kind = node["kind"]
        spec = registry.component(kind)
        if spec["side_effects"] and not self.allow_side_effects:
            raise NodeError("side_effects_not_approved", f"{spec['label']} can change external systems; approve side effects to run it")
        if spec["inputs"] and value is not None:
            value = convert(value, spec["inputs"][0]["accepts"])
        if kind == "input":
            return "out", Value("text", state.input_text), None
        if kind == "output":
            return "out", value, None
        if kind == "prompt":
            return "out", Value("text", render(_required(node, "template"), state, context)), None
        if kind == "condition":
            try:
                test = conditions.parse(_required(node, "expression"))
            except conditions.ConditionError as error:
                raise NodeError("invalid_graph", f"node {node['id']}: {error}") from None
            text = to_text(value)
            return ("true" if test(text) else "false"), Value("text", text), None
        if kind == "memory":
            strategy = _config(node, "strategy")
            history = [] if strategy == "none" else state.history[-_integer(node, "turns"):]
            return "out", Value("chat", {"text": to_text(value), "history": list(history)}), None
        if kind == "model":
            return await self._model(node, value, state)
        if kind == "tool":
            result = await _call(self.resources.invoke_function(_required(node, "tool"), {"input": to_text(value)}))
            return "out", Value("json", result), None
        if kind == "api":
            template = _config(node, "request")
            rendered = render(template, state, context) if template.strip() else ""
            if not rendered:
                payload: Any = {"input": state.input_text}
            else:
                try:
                    payload = json.loads(rendered)
                except json.JSONDecodeError:
                    payload = {"input": rendered}
            result = await _call(self.resources.invoke_connection(_required(node, "connection"), payload))
            return "out", Value("json", result), None
        if kind == "knowledge":
            from mlaiops_sdk.agents import HashingEmbedder

            base = _required(node, "knowledge_base")
            limit = _integer(node, "limit")
            query = render(_config(node, "query") or "{input}", state, context)
            dimensions = await _call(self.resources.knowledge_dimensions(base))
            vector = HashingEmbedder(dimensions).embed(query)
            documents = await _call(self.resources.search_knowledge(base, vector, limit))
            return "out", Value("documents", documents), None
        if kind == "storage":
            dataset, key = _required(node, "dataset"), _required(node, "key")
            result = await _call(self.resources.dataset_object(dataset, key))
            if result.get("truncated"):
                raise NodeError("node_error", "object exceeds Kionga's text preview limit")
            return "out", Value("documents", [{"text": str(result.get("content", "")),
                                               "metadata": {"dataset": dataset, "key": key}}]), None
        raise NodeError("invalid_graph", f"unsupported node type {kind}")

    async def _model(self, node, value: Value, state: RunState):
        model = _config(node, "model") or "llm"
        if model != "llm":
            result = await _call(self.resources.predict(model, to_text(value)))
            return "out", Value("text", result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)), None
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list[Any] = []
        system = _config(node, "system")
        if system.strip():
            messages.append(SystemMessage(content=render(system, state, "")))
        if value.type == "chat":
            for turn in value.data.get("history", []):
                text = str(turn.get("text", ""))
                messages.append(AIMessage(content=text) if turn.get("role") == "assistant" else HumanMessage(content=text))
            text = str(value.data.get("text", ""))
        else:
            text = to_text(value)
        messages.append(HumanMessage(content=text))
        try:
            reply = await self.llm().ainvoke(messages)
        except NodeError:
            raise
        except Exception as error:
            raise NodeError("node_error", f"model call failed: {type(error).__name__}") from None
        metadata = getattr(reply, "usage_metadata", None) or {}
        usage = Usage(int(metadata.get("input_tokens", 0)), int(metadata.get("output_tokens", 0)))
        return "out", Value("text", str(reply.content)), usage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ancestors(graph: dict[str, Any], target: str) -> set[str]:
    parents: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        parents.setdefault(edge["to"], []).append(edge["from"])
    seen, pending = set(), [target]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        pending.extend(parents.get(node_id, []))
    return seen


async def run_graph(
    graph: dict[str, Any],
    *,
    input_text: str,
    run_id: str,
    history: list[dict[str, str]] | None = None,
    allow_side_effects: bool = False,
    stop_after: str | None = None,
    resources: Resources | None = None,
    llm_factory: Callable[[], Any] | None = None,
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run a normalized, validated graph and yield contract events."""
    seq = 0

    def event(event_type: str, **fields: Any) -> dict[str, Any]:
        nonlocal seq
        item = {"seq": seq, "type": event_type, "run_id": run_id, "ts": _now()}
        item.update({key: value for key, value in fields.items() if value is not None})
        seq += 1
        return item

    def error(code: str, message: str) -> dict[str, str]:
        return {"code": code, "message": redact(message, ERROR_LIMIT)}

    settings = graph.get("settings") or {}
    max_steps = int(settings.get("max_steps", 50))
    max_tokens = int(settings.get("max_tokens", 50000))
    deadline = time.monotonic() + int(settings.get("timeout_seconds", 60))
    cancel = cancel or asyncio.Event()
    executor = Executor(resources or Resources(), llm_factory, allow_side_effects)
    state = RunState(input_text=input_text, history=list(history or []))
    nodes = {node["id"]: node for node in graph["nodes"]}
    inbound: dict[str, list[dict[str, Any]]] = {}
    for edge in graph["edges"]:
        inbound.setdefault(edge["to"], []).append(edge)
    delivered: dict[tuple[str, str], Value] = {}
    order = topological_order(graph) or []
    scope = _ancestors(graph, stop_after) if stop_after else set(nodes)
    if stop_after and stop_after not in nodes:
        yield event("run.started")
        yield event("run.failed", error=error("invalid_graph", f"node {stop_after} does not exist"))
        return
    started = time.monotonic()
    steps = 0
    yield event("run.started")

    def totals() -> dict[str, Any]:
        usage = Usage(state.input_tokens, state.output_tokens)
        return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd}

    for node_id in order:
        if node_id not in scope:
            continue
        node = nodes[node_id]
        spec = registry.component(node["kind"])
        if cancel.is_set():
            yield event("run.cancelled", error=error("cancelled", "run cancelled"))
            return
        edges = inbound.get(node_id, [])
        values = [delivered[(edge["from"], edge["from_port"])] for edge in edges
                  if (edge["from"], edge["from_port"]) in delivered]
        if edges and not values:
            yield event("node.skipped", node_id=node_id, kind=node["kind"])
            continue
        steps += 1
        if steps > max_steps:
            yield event("run.failed", error=error("budget_exceeded", f"run exceeded {max_steps} steps"))
            return
        context = "\n\n".join(to_text(value) for value in values)
        value = values[0] if len(values) == 1 else (Value("text", context) if values else None)
        yield event("node.started", node_id=node_id, kind=node["kind"],
                    input_preview=redact(to_text(value)) if value is not None else None)
        attempts = 1 + int(spec.get("retries", 0))
        failure: NodeError | None = None
        result = None
        node_started = time.monotonic()
        for attempt in range(1, attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = NodeError("timeout", f"run exceeded {settings.get('timeout_seconds', 60)} seconds")
                break
            budget = min(float(spec["timeout_seconds"]), remaining)
            task = asyncio.ensure_future(executor.execute(node, value, context, state))
            waiter = asyncio.ensure_future(cancel.wait())
            done, _ = await asyncio.wait({task, waiter}, timeout=budget, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if task not in done:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                if cancel.is_set():
                    yield event("node.failed", node_id=node_id, kind=node["kind"], attempt=attempt,
                                duration_ms=int((time.monotonic() - node_started) * 1000),
                                error=error("cancelled", "run cancelled"))
                    yield event("run.cancelled", error=error("cancelled", "run cancelled"))
                    return
                failure = NodeError("timeout", f"node exceeded {budget:.0f} seconds")
            else:
                try:
                    result = task.result()
                    failure = None
                except NodeError as caught:
                    failure = caught
                except Exception as caught:  # a handler bug must not crash the stream
                    failure = NodeError("node_error", f"{type(caught).__name__}: {caught}")
            if failure is None:
                break
            if failure.code not in RETRYABLE or attempt == attempts or deadline <= time.monotonic():
                break
            yield event("node.retrying", node_id=node_id, kind=node["kind"], attempt=attempt + 1,
                        error=error(failure.code, failure.message))
        duration_ms = int((time.monotonic() - node_started) * 1000)
        if failure is not None:
            yield event("node.failed", node_id=node_id, kind=node["kind"], duration_ms=duration_ms,
                        error=error(failure.code, failure.message))
            yield event("run.failed", error=error(failure.code, failure.message))
            return
        port, output, usage = result
        usage_fields = None
        if usage is not None:
            state.input_tokens += usage.input_tokens
            state.output_tokens += usage.output_tokens
            usage_fields = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                            "cost_usd": usage.cost_usd}
            if state.input_tokens + state.output_tokens > max_tokens:
                message = f"run exceeded {max_tokens} tokens"
                yield event("node.failed", node_id=node_id, kind=node["kind"], duration_ms=duration_ms,
                            usage=usage_fields, error=error("budget_exceeded", message))
                yield event("run.failed", error=error("budget_exceeded", message))
                return
        state.outputs[node_id] = output
        delivered[(node_id, port)] = output
        yield event("node.completed", node_id=node_id, kind=node["kind"], duration_ms=duration_ms,
                    port=port if node["kind"] == "condition" else None,
                    output_preview=redact(to_text(output)), usage=usage_fields)
        if node_id == stop_after or node["kind"] == "output":
            yield event("run.completed", output=to_text(output)[:OUTPUT_LIMIT], usage=totals(),
                        duration_ms=int((time.monotonic() - started) * 1000))
            return
    yield event("run.failed", error=error("node_error", "the graph produced no output; no branch reached the output node"))
