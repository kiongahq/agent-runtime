"""``POST /agent-turns``: exactly one model call for an Agent Teams worker.

The gateway owns the agent loop, durable state and tool dispatch. This module only
converts a transcript to LangChain messages, binds the tools, makes one call and
reports the outcome. It stores nothing between requests and executes nothing the
model returns: tool calls go back to the gateway as data.

Every expected failure is a 200 with ``{"error": {...}}``, so the gateway handles one
shape:

================== ==============================================================
``rate_limited``    Provider 429, or this runner at capacity. Carries
                    ``retry_after_seconds`` when the provider sent ``Retry-After``.
``provider_unavailable`` Provider 5xx, 408, 409, connection failure, timeout, or an
                    exhausted quota (a 429 that waiting briefly will not clear).
``invalid_request`` Malformed request or transcript, or a provider 4xx that rejects
                    the request itself (for example an oversized context).
``model_error``     Anything a retry will not fix: bad provider credentials, unknown
                    model, missing configuration, a malformed tool call.
================== ==============================================================

Message content never reaches a log line or an error message. Exceptions are reduced
to their class name, HTTP status and a short provider code, because a provider's error
text and any traceback can quote the prompt. Credentials must not be sent here at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from email.utils import parsedate_to_datetime
from typing import Annotated, Any, Callable, Iterator, Literal

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .native_graph.engine import redact

logger = logging.getLogger("kionga.agent_turns")

ERROR_CODES = ("rate_limited", "provider_unavailable", "invalid_request", "model_error")
PLATFORM_MODEL = "llm"
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_CONTENT = 1024 * 1024
MAX_RETRY_AFTER_SECONDS = 86400
CAPACITY_RETRY_SECONDS = 5
DISCONNECT_POLL_SECONDS = 0.5
DEFAULT_MAX_ACTIVE = 32
DEFAULT_TIMEOUT_SECONDS = 600.0
TOOL_NAME = r"^[A-Za-z0-9_-]{1,64}$"
TOOL_NAME_MATCH = re.compile(TOOL_NAME)
# Ids this runner returns. Stricter than the contract's request limit, because Anthropic
# refuses anything else when the transcript comes back.
TOOL_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
PROVIDER_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
# Matched by class name across the MRO so this works for the openai and anthropic SDKs
# and for whichever HTTP client they use, without importing any of them.
CONNECTION_ERRORS = {
    "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ConnectionError",
    "ReadError", "ReadTimeout", "RemoteProtocolError", "TimeoutError", "TimeoutException", "TransportError",
}


class TurnError(Exception):
    """An outcome for the gateway, safe to send as is. Never holds message content."""

    def __init__(self, code: str, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = redact(message, 500)
        self.retry_after_seconds = retry_after_seconds

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.retry_after_seconds is not None:
            error["retry_after_seconds"] = self.retry_after_seconds
        return {"error": error}


class _Closed(BaseModel):
    """The contract is closed: an unlisted field is an error, so none can carry a credential."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ToolCall(_Closed):
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(pattern=TOOL_NAME)
    arguments: dict[str, Any]


class UserMessage(_Closed):
    role: Literal["user"]
    content: str = Field(max_length=MAX_CONTENT)


class AssistantMessage(_Closed):
    role: Literal["assistant"]
    content: str = Field(max_length=MAX_CONTENT)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=256)


class ToolResultMessage(_Closed):
    role: Literal["tool"]
    tool_call_id: str = Field(min_length=1, max_length=256)
    content: str = Field(max_length=MAX_CONTENT)


TurnMessage = Annotated[UserMessage | AssistantMessage | ToolResultMessage, Field(discriminator="role")]


class ToolSpec(_Closed):
    name: str = Field(pattern=TOOL_NAME)
    description: str = Field(max_length=8192)
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def _object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type", "object") != "object":
            raise ValueError("input_schema must describe an object")
        return value


class AgentTurnRequest(_Closed):
    run_id: str = Field(pattern=r"^atr-[a-f0-9]{24}$")
    worker_id: str = Field(pattern=r"^atw-[a-f0-9]{24}$")
    turn: int = Field(ge=1)
    model: str = Field(min_length=1, max_length=256)
    system: str = Field(max_length=256 * 1024)
    messages: list[TurnMessage] = Field(min_length=1, max_length=8192)
    tools: list[ToolSpec] = Field(max_length=128)
    max_output_tokens: int = Field(ge=1, le=200000)

    @field_validator("model")
    @classmethod
    def _platform_model(cls, value: str) -> str:
        if value != PLATFORM_MODEL:
            raise ValueError(f"only the platform model {PLATFORM_MODEL!r} supports agent turns")
        return value

    @model_validator(mode="after")
    def _well_formed(self) -> "AgentTurnRequest":
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")
        check_transcript(self.messages)
        return self


def check_transcript(messages: list[Any]) -> None:
    """Refuse a transcript providers would treat differently.

    OpenAI and Anthropic both need every tool call answered immediately, but disagree
    on a transcript that ends with the assistant (a new reply against prefill) and on
    empty text. Rejecting here gives one deterministic result on every backend,
    including the mock.
    """
    if messages[0].role != "user":
        raise ValueError("the first message must have role user")
    pending: list[str] = []
    seen: set[str] = set()
    for index, message in enumerate(messages):
        if message.role == "tool":
            if message.tool_call_id not in pending:
                raise ValueError(f"messages[{index}]: tool message does not answer a pending tool call")
            pending.remove(message.tool_call_id)
            continue
        if pending:
            raise ValueError(f"messages[{index}]: {len(pending)} tool call(s) have no result")
        if message.role == "user":
            if not message.content:
                raise ValueError(f"messages[{index}]: user content is empty")
            continue
        if not message.content and not message.tool_calls:
            raise ValueError(f"messages[{index}]: assistant message has no content or tool calls")
        for call in message.tool_calls:
            if call.id in seen:
                raise ValueError(f"messages[{index}]: duplicate tool call id")
            seen.add(call.id)
            pending.append(call.id)
    if pending:
        raise ValueError(f"{len(pending)} tool call(s) have no result")
    if messages[-1].role == "assistant":
        raise ValueError("the last message must have role user or tool")


def to_langchain_messages(system: str, messages: list[Any]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    converted: list[Any] = [SystemMessage(content=system)] if system else []
    for message in messages:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "tool":
            converted.append(ToolMessage(content=message.content, tool_call_id=message.tool_call_id))
        else:
            converted.append(AIMessage(content=message.content, tool_calls=[
                {"id": call.id, "name": call.name, "args": call.arguments, "type": "tool_call"}
                for call in message.tool_calls
            ]))
    return converted


def to_bound_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """OpenAI function format, which ``bind_tools`` accepts on every LangChain provider."""
    return [{"type": "function", "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": {"type": "object", "properties": {}, **tool.input_schema},
    }} for tool in tools]


def _text(content: Any) -> str:
    """Model text, without reasoning or tool blocks that some providers put in ``content``."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _tool_calls(reply: Any, turn: int) -> list[dict[str, Any]]:
    if getattr(reply, "invalid_tool_calls", None):
        raise TurnError("model_error", "the model returned a malformed or truncated tool call")
    calls = []
    seen: set[str] = set()
    for index, call in enumerate(getattr(reply, "tool_calls", None) or []):
        call_id = call.get("id") or ""
        # Some OpenAI-compatible servers omit ids or use ones Anthropic refuses when the
        # transcript comes back. A stable replacement keeps the round trip valid.
        if not TOOL_CALL_ID.match(call_id) or call_id in seen:
            call_id = f"call-{turn}-{index}"
        seen.add(call_id)
        if not TOOL_NAME_MATCH.match(str(call.get("name", ""))):
            raise TurnError("model_error", "the model returned a tool call with an invalid name")
        calls.append({"id": call_id, "name": call["name"], "arguments": call.get("args") or {}})
    return calls


def _model_name(reply: Any, llm: Any) -> str:
    metadata = getattr(reply, "response_metadata", None) or {}
    for value in (metadata.get("model_name"), metadata.get("model"), getattr(llm, "model_name", None),
                  getattr(llm, "model", None)):
        if isinstance(value, str) and value:
            return value[:256]
    return "unknown"


def _chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status(error: BaseException) -> int | None:
    for value in (getattr(error, "status_code", None), getattr(getattr(error, "response", None), "status_code", None)):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def parse_retry_after(headers: Any, now: float | None = None) -> int | None:
    """Whole seconds to wait, or None so the gateway applies its own backoff.

    Accepts ``retry-after-ms``, delta seconds and the HTTP date form. A value of zero or
    less, or anything unparseable, is None: honoring it would mean retrying at once.
    """
    if headers is None:
        return None
    try:
        millis, value = headers.get("retry-after-ms"), headers.get("retry-after")
    except AttributeError:
        return None
    seconds: float | None = None
    try:
        if millis is not None:
            seconds = float(millis) / 1000
        elif value is not None:
            seconds = float(value)
    except (TypeError, ValueError):
        seconds = None
    if seconds is None and isinstance(value, str):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError):
            return None
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return None
    return min(math.ceil(seconds), MAX_RETRY_AFTER_SECONDS)


def classify(error: BaseException) -> TurnError:
    """Reduce any provider failure to a contract error, reading only safe attributes."""
    for item in _chain(error):
        if isinstance(item, TurnError):
            return item
    for item in _chain(error):
        status = _status(item)
        if status is not None:
            return _classify_status(item, status)
    names = {cls.__name__ for item in _chain(error) for cls in type(item).__mro__}
    if names & CONNECTION_ERRORS:
        return TurnError("provider_unavailable", f"could not reach the model provider ({type(error).__name__})")
    if "MissingLLMConfiguration" in names:
        # The SDK's message names environment variables only.
        return TurnError("model_error", f"model backend is not configured: {error}")
    return TurnError("model_error", f"model call failed ({type(error).__name__})")


def _classify_status(error: BaseException, status: int) -> TurnError:
    kind = type(error).__name__
    code = getattr(error, "code", None)
    detail = f"HTTP {status}, {kind}"
    if isinstance(code, str) and PROVIDER_CODE.match(code):
        detail += f", {code}"
    retry_after = parse_retry_after(getattr(getattr(error, "response", None), "headers", None))
    if status == 429:
        if code == "insufficient_quota":
            return TurnError("provider_unavailable", f"model provider quota is exhausted ({detail})", retry_after)
        return TurnError("rate_limited", f"model provider rate limit ({detail})", retry_after)
    if status >= 500 or status in (408, 409):
        return TurnError("provider_unavailable", f"model provider is unavailable ({detail})", retry_after)
    if status in (401, 403):
        return TurnError("model_error", f"model provider refused the configured credentials ({detail})")
    if status == 404:
        return TurnError("model_error", f"model provider does not know the configured model ({detail})")
    return TurnError("invalid_request", f"model provider rejected the request ({detail})")


async def run_turn(llm: Any, payload: AgentTurnRequest, timeout: float | None = None) -> dict[str, Any]:
    """Make the one model call. Raises ``TurnError`` for every expected failure."""
    try:
        bound = llm.bind(max_tokens=payload.max_output_tokens) if not payload.tools else llm.bind_tools(
            to_bound_tools(payload.tools), max_tokens=payload.max_output_tokens)
        reply = await asyncio.wait_for(bound.ainvoke(to_langchain_messages(payload.system, payload.messages)), timeout)
    except Exception as error:  # every provider failure becomes a contract error
        raise classify(error) from None
    metadata = getattr(reply, "usage_metadata", None) or {}
    return {
        "content": _text(reply.content),
        "tool_calls": _tool_calls(reply, payload.turn),
        "usage": {"input_tokens": int(metadata.get("input_tokens", 0)),
                  "output_tokens": int(metadata.get("output_tokens", 0))},
        "model": _model_name(reply, llm),
    }


def parse_request(body: bytes) -> AgentTurnRequest:
    """Validate a request. Errors name fields, never their values."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise TurnError("invalid_request", "request body is not valid JSON") from None
    try:
        return AgentTurnRequest.model_validate(data)
    except ValidationError as error:
        issues = [f"{'.'.join(str(part) for part in issue['loc']) or 'request'}: {issue['msg']}"
                  for issue in error.errors(include_input=False, include_url=False, include_context=False)]
        raise TurnError("invalid_request", "; ".join(issues[:5])) from None


async def read_body(request: Request) -> bytes:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise TurnError("invalid_request", f"request body exceeds {MAX_BODY_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def default_model_factory(timeout: float) -> Callable[[], Any]:
    """The platform model, or the scripted one under ``MLAIOPS_LLM_BACKEND=mock``.

    Retries are the gateway's job: it persists ``waiting_retry`` and survives restarts,
    whereas a retry inside the SDK holds this request and hides the rate limit. The
    explicit timeout also stops the Anthropic SDK refusing large non-streaming calls.
    """
    def build() -> Any:
        if os.environ.get("MLAIOPS_LLM_BACKEND", "").lower() == "mock":
            from .agent_turn_mock import ScriptedTurnModel

            return ScriptedTurnModel()
        from mlaiops_sdk.llm import build_chat_model

        model = build_chat_model(max_retries=0, timeout=timeout)
        # The provider SDKs log full request bodies at DEBUG (OPENAI_LOG=debug, or a
        # DEBUG root logger). Message content must not reach logs on this path.
        for name in ("openai", "anthropic"):
            logging.getLogger(name).setLevel(logging.INFO)
        return model

    return build


class AgentTurns:
    """The request handler, with the capacity counter and the one shared model client."""

    def __init__(self, model_factory: Callable[[], Any] | None = None, *, max_active: int | None = None,
                 timeout: float | None = None):
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("KIONGA_AGENT_TURN_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        self.max_active = max_active if max_active is not None else int(
            os.environ.get("KIONGA_DESIGNER_MAX_AGENT_TURNS", DEFAULT_MAX_ACTIVE))
        self.factory = model_factory or default_model_factory(self.timeout)
        self.active = 0
        self._model: Any = None

    def _llm(self) -> Any:
        if self._model is None:
            self._model = self.factory()
        return self._model

    async def handle(self, request: Request) -> dict[str, Any]:
        started = time.monotonic()
        run_id = worker_id = "-"
        turn = -1
        try:
            payload = parse_request(await read_body(request))
            run_id, worker_id, turn = payload.run_id, payload.worker_id, payload.turn
            if self.active >= self.max_active:
                raise TurnError("rate_limited", "the runner is at capacity", CAPACITY_RETRY_SECONDS)
            self.active += 1
            try:
                result = await self._race_disconnect(request, payload)
            finally:
                self.active -= 1
        except TurnError as error:
            logger.warning("agent_turn run_id=%s worker_id=%s turn=%d outcome=%s retry_after=%s duration_ms=%d detail=%s",
                           run_id, worker_id, turn, error.code, error.retry_after_seconds,
                           (time.monotonic() - started) * 1000, error.message)
            return error.body()
        logger.info("agent_turn run_id=%s worker_id=%s turn=%d outcome=ok tool_calls=%d input_tokens=%d output_tokens=%d duration_ms=%d",
                    run_id, worker_id, turn, len(result["tool_calls"]), result["usage"]["input_tokens"],
                    result["usage"]["output_tokens"], (time.monotonic() - started) * 1000)
        return result

    async def _race_disconnect(self, request: Request, payload: AgentTurnRequest) -> dict[str, Any]:
        """Cancel the provider call when the gateway hangs up, so a cancelled run stops spending."""
        async def call() -> dict[str, Any]:
            try:
                llm = self._llm()
            except Exception as error:
                raise classify(error) from None
            return await run_turn(llm, payload, self.timeout)

        # Polled here, not in a watcher task: is_disconnected runs in an anyio cancel scope
        # that swallows task.cancel() when the two coincide, which would leave the watcher
        # running and any await on it hanging.
        task = asyncio.create_task(call())
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL_SECONDS)
                if done:
                    return task.result()
                if await request.is_disconnected():
                    raise TurnError("invalid_request", "the caller disconnected")
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
