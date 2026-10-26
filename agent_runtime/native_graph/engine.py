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
# Failures a connected ``error`` port may absorb (contracts README, "Error ports").
ROUTABLE = {"timeout", "node_error", "resource_unavailable", "resource_denied"}
PATH = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$")
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


def _extract(node: dict[str, Any], value: Value) -> Any:
    path = _required(node, "path")
    if len(path) > 200 or not PATH.match(path):
        raise NodeError("invalid_graph", f"node {node['id']} has an invalid field path")
    current = value.data
    if value.type == "text":
        try:
            current = json.loads(current)
        except json.JSONDecodeError:
            raise NodeError("node_error", f"node {node['id']} input is not JSON") from None
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            if _config(node, "missing") == "empty":
                return None
            raise NodeError("node_error", f"field {path} is missing")
    return current


def _resource_error(error: ResourceError) -> NodeError:
    code = "resource_denied" if error.status in (401, 403, 404) else "resource_unavailable"
    return NodeError(code, str(error))


async def _call(coroutine):
    try:
        return await coroutine
    except ResourceError as error:
        raise _resource_error(error) from None


@dataclass
class NodeResult:
    port: str
    value: Value
    usage: Usage | None = None
    iterations: int | None = None


@dataclass
class Budget:
    """Step, token and time budgets shared by a run and every nested flow."""
    max_steps: int
    max_tokens: int
    deadline: float
    timeout_seconds: int
    cancel: asyncio.Event
    steps: int = 0
    tokens: int = 0

    def take_step(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise NodeError("budget_exceeded", f"run exceeded {self.max_steps} steps")

    def add_tokens(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self.tokens += usage.input_tokens + usage.output_tokens
        if self.tokens > self.max_tokens:
            raise NodeError("budget_exceeded", f"run exceeded {self.max_tokens} tokens")


MAX_DEPTH = 3
APPROVAL_POLL_SECONDS = 1.0
# Failures that end the whole run even inside a node (never retried or routed).
FATAL = {"budget_exceeded", "cancelled", "side_effects_not_approved", "invalid_graph"}


class Executor:
    def __init__(self, resources: Resources, llm_factory: Callable[[], Any] | None, allow_side_effects: bool,
                 subflows: dict[str, dict[str, Any]], budget: Budget, run_id: str, depth: int):
        self.resources = resources
        self.llm_factory = llm_factory
        self.allow_side_effects = allow_side_effects
        self.subflows = subflows
        self.budget = budget
        self.run_id = run_id
        self.depth = depth
        self._llm = None

    def llm(self):
        if self._llm is None:
            if self.llm_factory is None:
                raise NodeError("resource_unavailable", "no platform chat model is configured")
            self._llm = self.llm_factory()
        return self._llm

    async def execute(self, node: dict[str, Any], value: Value | None, context: str, state: RunState,
                      emit: Callable[..., None], upstream: list[tuple[str, Value | None]]) -> NodeResult:
        kind = node["kind"]
        spec = registry.component(kind)
        if spec["side_effects"] and not self.allow_side_effects:
            raise NodeError("side_effects_not_approved", f"{spec['label']} can change external systems; approve side effects to run it")
        if spec["inputs"] and value is not None and kind != "join":
            value = convert(value, spec["inputs"][0]["accepts"])
        if kind == "input":
            return NodeResult("out", Value("text", state.input_text))
        if kind == "output":
            return NodeResult("out", value)
        if kind == "prompt":
            return NodeResult("out", Value("text", render(_required(node, "template"), state, context)))
        if kind == "condition":
            test = _condition(node, "expression")
            text = to_text(value)
            return NodeResult("true" if test(text) else "false", Value("text", text))
        if kind == "router":
            text = to_text(value)
            for index in range(1, 5):
                expression = _config(node, f"route_{index}").strip()
                if expression and _condition(node, f"route_{index}")(text):
                    return NodeResult(f"route_{index}", Value("text", text))
            return NodeResult("default", Value("text", text))
        if kind == "join":
            if _config(node, "require") == "all" and any(item is None for _, item in upstream):
                missing = ", ".join(source for source, item in upstream if item is None)
                raise NodeError("node_error", f"join requires every branch; {missing} produced nothing")
            return NodeResult("out", Value("json", {source: (item.data if item.type == "json" else to_text(item))
                                                    for source, item in upstream if item is not None}))
        if kind == "extract":
            return NodeResult("out", Value("json", _extract(node, value)))
        if kind == "memory":
            strategy = _config(node, "strategy")
            history = [] if strategy == "none" else state.history[-_integer(node, "turns"):]
            return NodeResult("out", Value("chat", {"text": to_text(value), "history": list(history)}))
        if kind == "model":
            return await self._model(node, value, state, emit)
        if kind == "tool":
            result = await _call(self.resources.invoke_function(_required(node, "tool"), {"input": to_text(value)}))
            return NodeResult("out", Value("json", result))
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
            return NodeResult("out", Value("json", result))
        if kind == "knowledge":
            from mlaiops_sdk.agents import HashingEmbedder

            base = _required(node, "knowledge_base")
            limit = _integer(node, "limit")
            query = render(_config(node, "query") or "{input}", state, context)
            dimensions = await _call(self.resources.knowledge_dimensions(base))
            vector = HashingEmbedder(dimensions).embed(query)
            documents = await _call(self.resources.search_knowledge(base, vector, limit))
            return NodeResult("out", Value("documents", documents))
        if kind == "storage":
            dataset, key = _required(node, "dataset"), _required(node, "key")
            result = await _call(self.resources.dataset_object(dataset, key))
            if result.get("truncated"):
                raise NodeError("node_error", "object exceeds Kionga's text preview limit")
            return NodeResult("out", Value("documents", [{"text": str(result.get("content", "")),
                                                          "metadata": {"dataset": dataset, "key": key}}]))
        if kind == "subflow":
            output, usage = await self._nested(node, to_text(value))
            return NodeResult("out", Value("text", output), usage)
        if kind == "map":
            return await self._map(node, value)
        if kind == "loop":
            return await self._loop(node, value)
        if kind == "approval":
            return await self._approval(node, value, state, context, emit)
        raise NodeError("invalid_graph", f"unsupported node type {kind}")

    async def _model(self, node, value: Value, state: RunState, emit: Callable[..., None]) -> NodeResult:
        model = _config(node, "model") or "llm"
        if model != "llm":
            result = await _call(self.resources.predict(model, to_text(value)))
            return NodeResult("out", Value("text", result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)))
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
        reply = None
        llm = self.llm()
        try:
            if hasattr(llm, "astream"):
                # Each streamed chunk becomes a node.delta; the sum is the reply.
                async for chunk in llm.astream(messages):
                    reply = chunk if reply is None else reply + chunk
                    delta = chunk.content if isinstance(chunk.content, str) else ""
                    if delta:
                        emit("node.delta", delta=redact(delta))
            else:
                reply = await llm.ainvoke(messages)
        except NodeError:
            raise
        except Exception as error:
            raise NodeError("node_error", f"model call failed: {type(error).__name__}") from None
        if reply is None:
            raise NodeError("node_error", "model returned no output")
        metadata = getattr(reply, "usage_metadata", None) or {}
        usage = Usage(int(metadata.get("input_tokens", 0)), int(metadata.get("output_tokens", 0)))
        content = reply.content if isinstance(reply.content, str) else json.dumps(reply.content, ensure_ascii=False)
        return NodeResult("out", Value("text", content), usage)

    async def _nested(self, node: dict[str, Any], text: str) -> tuple[str, Usage]:
        version = _required(node, "subflow")
        graph = self.subflows.get(version)
        if graph is None:
            raise NodeError("invalid_graph", f"published version {version} was not provided with the run")
        if self.depth + 1 > MAX_DEPTH:
            raise NodeError("invalid_graph", f"flows can nest at most {MAX_DEPTH} levels")
        nested = GraphRun(graph, input_text=text, run_id=self.run_id, history=[], allow_side_effects=self.allow_side_effects,
                          stop_after=None, resources=self.resources, llm_factory=self.llm_factory, subflows=self.subflows,
                          budget=self.budget, depth=self.depth + 1, emit=lambda *args, **kwargs: None)
        outcome = await nested.drive()
        if outcome.error is not None:
            code, message = outcome.error
            if code in FATAL:
                raise NodeError(code, message)
            raise NodeError("node_error", f"{version}: {message}")
        return outcome.output, outcome.usage

    async def _map(self, node: dict[str, Any], value: Value) -> NodeResult:
        if value.type == "text" and value.data.lstrip().startswith("["):
            try:  # text holding a JSON list (a typed message or a model reply) is mapped as that list
                parsed = json.loads(value.data)
                if isinstance(parsed, list):
                    value = Value("json", parsed)
            except json.JSONDecodeError:
                pass
        if value.type == "json" and isinstance(value.data, list):
            items = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in value.data]
        elif value.type == "json":
            items = [to_text(value)]
        else:
            items = [line.strip() for line in to_text(value).splitlines() if line.strip()]
        limit = _integer(node, "max_items")
        if len(items) > limit:
            raise NodeError("node_error", f"the list has {len(items)} items; this map allows {limit}")
        semaphore = asyncio.Semaphore(_integer(node, "concurrency"))

        async def one(item: str) -> tuple[str, Usage]:
            async with semaphore:
                return await self._nested(node, item)

        results = await asyncio.gather(*(one(item) for item in items))
        usage = Usage(sum(u.input_tokens for _, u in results), sum(u.output_tokens for _, u in results))
        return NodeResult("out", Value("json", [output for output, _ in results]), usage, iterations=len(items))

    async def _loop(self, node: dict[str, Any], value: Value) -> NodeResult:
        until = _condition(node, "until")
        current, input_tokens, output_tokens = to_text(value), 0, 0
        limit = _integer(node, "max_iterations")
        for iteration in range(1, limit + 1):
            current, usage = await self._nested(node, current)
            input_tokens, output_tokens = input_tokens + usage.input_tokens, output_tokens + usage.output_tokens
            if until(current):
                return NodeResult("out", Value("text", current), Usage(input_tokens, output_tokens), iterations=iteration)
        return NodeResult("exhausted", Value("text", current), Usage(input_tokens, output_tokens), iterations=limit)

    async def _approval(self, node, value: Value, state: RunState, context: str, emit) -> NodeResult:
        message = render(_config(node, "message"), state, context)[:2000]
        seconds = _integer(node, "timeout_seconds")
        seconds = max(1, min(seconds, int(self.budget.deadline - time.monotonic()) - 1))
        approval = await _call(self.resources.create_approval(self.run_id, node["id"], redact(message), seconds))
        approval_id = str(approval.get("id", ""))
        emit("approval.requested", approval_id=approval_id, output_preview=redact(message))
        expires = time.monotonic() + seconds
        status = str(approval.get("status", "pending"))
        while status == "pending" and time.monotonic() < expires:
            await asyncio.sleep(APPROVAL_POLL_SECONDS)
            status = str((await _call(self.resources.get_approval(approval_id))).get("status", "pending"))
        port = "approved" if status == "approved" else "rejected"
        emit("approval.resolved", approval_id=approval_id, port=port)
        return NodeResult(port, Value("text", to_text(value)))


def _condition(node: dict[str, Any], key: str):
    try:
        return conditions.parse(_required(node, key))
    except conditions.ConditionError as error:
        raise NodeError("invalid_graph", f"node {node['id']}: {error}") from None


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


@dataclass
class Outcome:
    output: str = ""
    usage: Usage = field(default_factory=Usage)
    error: tuple[str, str] | None = None
    cancelled: bool = False


class GraphRun:
    """Schedules one graph: every node whose upstream nodes have finished runs
    concurrently. Events go to ``emit``; nested flows pass a no-op."""

    def __init__(self, graph, *, input_text, run_id, history, allow_side_effects, stop_after, resources,
                 llm_factory, subflows, budget: Budget, depth: int, emit: Callable[..., None]):
        self.graph, self.stop_after, self.budget, self.emit = graph, stop_after, budget, emit
        self.executor = Executor(resources, llm_factory, allow_side_effects, subflows, budget, run_id, depth)
        self.state = RunState(input_text=input_text, history=list(history))
        self.nodes = {node["id"]: node for node in graph["nodes"]}
        self.inbound: dict[str, list[dict[str, Any]]] = {}
        for edge in graph["edges"]:
            self.inbound.setdefault(edge["to"], []).append(edge)
        self.has_error_edge = {edge["from"] for edge in graph["edges"] if edge["from_port"] == "error"}
        self.scope = _ancestors(graph, stop_after) if stop_after else set(self.nodes)
        self.order = [node_id for node_id in (topological_order(graph) or []) if node_id in self.scope]
        self.delivered: dict[tuple[str, str], Value] = {}
        self.resolved: set[str] = set()

    def _error(self, code: str, message: str) -> dict[str, str]:
        return {"code": code, "message": redact(message, ERROR_LIMIT)}

    async def drive(self) -> Outcome:
        outcome = Outcome()
        running: dict[str, asyncio.Task] = {}
        terminal_value: Value | None = None
        try:
            while True:
                if self.budget.cancel.is_set():
                    outcome.cancelled = True
                    outcome.error = ("cancelled", "run cancelled")
                    return outcome
                for node_id in self.order:
                    if node_id in self.resolved or node_id in running:
                        continue
                    if all(edge["from"] in self.resolved for edge in self.inbound.get(node_id, [])):
                        running[node_id] = asyncio.ensure_future(self._node(node_id))
                if not running:
                    break
                waiter = asyncio.ensure_future(self.budget.cancel.wait())
                done, _ = await asyncio.wait({*running.values(), waiter}, return_when=asyncio.FIRST_COMPLETED)
                waiter.cancel()
                for node_id, task in list(running.items()):
                    if task not in done:
                        continue
                    del running[node_id]
                    self.resolved.add(node_id)
                    result = task.result()
                    if isinstance(result, NodeError):
                        outcome.error = (result.code, result.message)
                        outcome.cancelled = result.code == "cancelled"
                        return outcome
                    if result is not None and (node_id == self.stop_after or self.nodes[node_id]["kind"] == "output"):
                        terminal_value = result
            if terminal_value is None:
                outcome.error = ("node_error", "the graph produced no output; no branch reached the output node")
                return outcome
            outcome.output = to_text(terminal_value)[:OUTPUT_LIMIT]
            outcome.usage = Usage(self.state.input_tokens, self.state.output_tokens)
            return outcome
        finally:
            for task in running.values():
                task.cancel()
            for task in running.values():
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _node(self, node_id: str) -> Value | NodeError | None:
        """Run one node. Returns its terminal value, None when it produced no
        terminal value (skipped, or routed to its error port), or the NodeError
        that ends the run."""
        node = self.nodes[node_id]
        spec = registry.component(node["kind"])
        edges = self.inbound.get(node_id, [])
        upstream = [(edge["from"], self.delivered.get((edge["from"], edge["from_port"]))) for edge in edges]
        values = [item for _, item in upstream if item is not None]
        if edges and not values:
            self.emit("node.skipped", node_id=node_id, kind=node["kind"])
            return None
        try:
            self.budget.take_step()
        except NodeError as failure:
            return failure
        context = "\n\n".join(to_text(value) for value in values)
        value = values[0] if len(values) == 1 else (Value("text", context) if values else None)
        self.emit("node.started", node_id=node_id, kind=node["kind"],
                  input_preview=redact(to_text(value)) if value is not None else None)

        def node_emit(event_type: str, **fields: Any) -> None:
            self.emit(event_type, node_id=node_id, kind=node["kind"], **fields)

        attempts = 1 + int(spec.get("retries", 0))
        failure: NodeError | None = None
        result: NodeResult | None = None
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            remaining = self.budget.deadline - time.monotonic()
            if remaining <= 0:
                failure = NodeError("timeout", f"run exceeded {self.budget.timeout_seconds} seconds")
                break
            limit = min(float(spec["timeout_seconds"]), remaining)
            task = asyncio.ensure_future(self.executor.execute(node, value, context, self.state, node_emit, upstream))
            waiter = asyncio.ensure_future(self.budget.cancel.wait())
            done, _ = await asyncio.wait({task, waiter}, timeout=limit, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if task not in done:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                if self.budget.cancel.is_set():
                    node_emit("node.failed", attempt=attempt, duration_ms=int((time.monotonic() - started) * 1000),
                              error=self._error("cancelled", "run cancelled"))
                    return NodeError("cancelled", "run cancelled")
                failure = NodeError("timeout", f"node exceeded {limit:.0f} seconds")
            else:
                try:
                    result, failure = task.result(), None
                except NodeError as caught:
                    failure = caught
                except Exception as caught:  # a handler bug must not crash the stream
                    failure = NodeError("node_error", f"{type(caught).__name__}: {caught}")
            if failure is None:
                break
            if failure.code not in RETRYABLE or attempt == attempts or self.budget.deadline <= time.monotonic():
                break
            node_emit("node.retrying", attempt=attempt + 1, error=self._error(failure.code, failure.message))
        duration_ms = int((time.monotonic() - started) * 1000)
        if failure is not None:
            node_emit("node.failed", duration_ms=duration_ms, error=self._error(failure.code, failure.message))
            routed = (failure.code in ROUTABLE and time.monotonic() < self.budget.deadline
                      and node_id in self.has_error_edge and any(port["name"] == "error" for port in spec["outputs"]))
            if not routed:
                return failure
            # The failure becomes the value on the error branch; out-port consumers are skipped.
            message = Value("text", redact(f"{failure.code}: {failure.message}", ERROR_LIMIT))
            self.state.outputs[node_id] = message
            self.delivered[(node_id, "error")] = message
            return None
        usage_fields = None
        if result.usage is not None:
            self.state.input_tokens += result.usage.input_tokens
            self.state.output_tokens += result.usage.output_tokens
            usage_fields = {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
                            "cost_usd": result.usage.cost_usd}
            try:
                self.budget.add_tokens(result.usage)
            except NodeError as exceeded:
                node_emit("node.failed", duration_ms=duration_ms, usage=usage_fields,
                          error=self._error(exceeded.code, exceeded.message))
                return exceeded
        self.state.outputs[node_id] = result.value
        self.delivered[(node_id, result.port)] = result.value
        node_emit("node.completed", duration_ms=duration_ms,
                  port=result.port if len(spec["outputs"]) > 1 else None, iterations=result.iterations,
                  output_preview=redact(to_text(result.value)), usage=usage_fields)
        return result.value


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
    subflows: dict[str, dict[str, Any]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run a normalized, validated graph and yield contract events.

    ``subflows`` maps published version IDs to their normalized graphs for map,
    loop and subflow nodes."""
    seq = 0
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def emit(event_type: str, **fields: Any) -> None:
        queue.put_nowait({"type": event_type, **{key: value for key, value in fields.items() if value is not None}})

    def error(code: str, message: str) -> dict[str, str]:
        return {"code": code, "message": redact(message, ERROR_LIMIT)}

    settings = graph.get("settings") or {}
    timeout_seconds = int(settings.get("timeout_seconds", 60))
    budget = Budget(max_steps=int(settings.get("max_steps", 50)), max_tokens=int(settings.get("max_tokens", 50000)),
                    deadline=time.monotonic() + timeout_seconds, timeout_seconds=timeout_seconds,
                    cancel=cancel or asyncio.Event())
    started = time.monotonic()
    nodes = {node["id"] for node in graph["nodes"]}

    async def driver() -> None:
        try:
            emit("run.started")
            if stop_after and stop_after not in nodes:
                emit("run.failed", error=error("invalid_graph", f"node {stop_after} does not exist"))
                return
            run = GraphRun(graph, input_text=input_text, run_id=run_id, history=history or [],
                           allow_side_effects=allow_side_effects, stop_after=stop_after, resources=resources or Resources(),
                           llm_factory=llm_factory, subflows=subflows or {}, budget=budget, depth=1, emit=emit)
            outcome = await run.drive()
            if outcome.cancelled:
                emit("run.cancelled", error=error("cancelled", "run cancelled"))
            elif outcome.error is not None:
                emit("run.failed", error=error(*outcome.error))
            else:
                usage = outcome.usage
                emit("run.completed", output=outcome.output,
                     usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd},
                     duration_ms=int((time.monotonic() - started) * 1000))
        finally:
            queue.put_nowait(done)

    task = asyncio.ensure_future(driver())
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            yield {"seq": seq, "run_id": run_id, "ts": _now(), **item}
            seq += 1
        await task
    finally:
        if not task.done():
            budget.cancel.set()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
