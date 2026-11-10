"""FastAPI service wrapping LangGraph or a framework-neutral Kionga adapter.

Endpoints:

- ``GET  /healthz``  — liveness.
- ``POST /invoke``   — one agent turn; returns the reply and available usage.
- ``POST /stream``   — the same turn as Server-Sent Events token stream. Only
  model (AI message) text is streamed; tool results never reach the client,
  only a redacted ``{"tool": name, "status": ...}`` progress event.

Configuration (environment):

- ``KIONGA_GRAPH_MODULE``   — ``package.module:attribute`` (required).
- ``KIONGA_AGENT_ID``       — control-plane agent id for session reporting.
- ``KIONGA_URL``            — gateway base URL for session reporting.
- ``DATABASE_URL`` / ``KIONGA_CHECKPOINT_DSN`` — enables the PostgreSQL
  checkpointer; falls back to in-memory checkpoints when absent.
- ``KIONGA_LLM_BACKEND`` / ``KIONGA_LLM_MODEL`` / ``KIONGA_LLM_BASE_URL`` —
  chat model selection (see ``kionga.llm``).
- ``KIONGA_RUNTIME_AUTH`` — ``required`` or ``disabled``. Defaults to
  ``required`` when ``KIONGA_ENVIRONMENT=production`` and ``disabled``
  otherwise; ``disabled`` is refused in production.
- ``KIONGA_INTERNAL_TOKEN`` — the shared gateway credential. With auth
  required, ``/invoke`` and ``/stream`` need ``Authorization: Bearer <token>``
  and the startup fails unless it is at least 16 characters.

Caller identity: with auth required, the agent id and name come from the
``X-Kionga-Agent-ID`` / ``X-Kionga-Agent-Name`` headers and the user from
``X-Kionga-User-ID``, and they are trusted only because the request carries
the gateway credential. With auth disabled (local development) the same
headers and the body ``user_id`` are taken as given.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote
from dataclasses import dataclass
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kionga.llm import build_chat_model
from kionga.tracing import langfuse_handler

from .graphs import GraphLoadError, build_graph, resolve_attribute
from .reporting import Usage, report_session, usage_from_messages


# Caller-chosen session ids are opaque labels, never storage keys on their
# own: see thread_key.
SESSION_ID_PATTERN = r"^(|[A-Za-z0-9][A-Za-z0-9._:@-]{0,199})$"


class InvokeRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(default="", pattern=SESSION_ID_PATTERN)
    user_id: str = Field(default="", max_length=256)


class InvokeResponse(BaseModel):
    usage_available: bool = False
    reply: str
    session_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int


logger = logging.getLogger(__name__)

MIN_TOKEN_LENGTH = 16


@dataclass(frozen=True)
class AuthConfig:
    """Whether ``/invoke`` and ``/stream`` require the gateway credential."""

    required: bool
    token: str = ""

    @classmethod
    def from_env(cls) -> "AuthConfig":
        production = os.environ.get("KIONGA_ENVIRONMENT", "").strip().lower() == "production"
        mode = os.environ.get("KIONGA_RUNTIME_AUTH", "").strip().lower() or ("required" if production else "disabled")
        if mode not in {"required", "disabled"}:
            raise RuntimeError("KIONGA_RUNTIME_AUTH must be 'required' or 'disabled'")
        if mode == "disabled":
            if production:
                raise RuntimeError("KIONGA_RUNTIME_AUTH=disabled is not allowed when KIONGA_ENVIRONMENT=production")
            logger.warning("agent runtime authentication is disabled; any caller can invoke the agent (local development only)")
            return cls(required=False)
        token = os.environ.get("KIONGA_INTERNAL_TOKEN", "")
        if len(token) < MIN_TOKEN_LENGTH:
            raise RuntimeError(
                f"KIONGA_RUNTIME_AUTH=required needs KIONGA_INTERNAL_TOKEN of at least {MIN_TOKEN_LENGTH} characters"
            )
        return cls(required=True, token=token)


@dataclass(frozen=True)
class Caller:
    """Who a turn runs for: the control-plane agent and the end user."""

    agent_id: str
    agent_name: str
    user_id: str
    authenticated: bool


def _check_graph_module(request: Request) -> None:
    expected = request.headers.get("X-Kionga-Graph-Module", "")
    loaded = os.environ.get("KIONGA_GRAPH_MODULE", "")
    if expected and loaded and expected != loaded:
        raise HTTPException(status_code=409, detail="This runtime serves a different entrypoint; deploy a dedicated runtime for this agent")


def _authenticate(request: Request, auth: AuthConfig) -> bool:
    """Verify the gateway credential in constant time. Returns whether the
    request is authenticated; raises 401 when auth is required and it is not."""
    if not auth.required:
        return False
    supplied = request.headers.get("Authorization", "")
    if not hmac.compare_digest(supplied.encode(), f"Bearer {auth.token}".encode()):
        raise HTTPException(status_code=401, detail="gateway credential required",
                            headers={"WWW-Authenticate": "Bearer"})
    return True


def _resolve_caller(request: Request, authenticated: bool, body_user_id: str) -> Caller:
    _check_graph_module(request)
    agent_id, agent_name = _agent_identity(request)
    user_id = body_user_id
    if authenticated:
        # The gateway asserts the end user. A body user_id is caller-chosen
        # and is ignored, so it can never select another user's thread.
        user_id = request.headers.get("X-Kionga-User-ID", "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="X-Kionga-User-ID is required on gateway requests")
    return Caller(agent_id=agent_id, agent_name=agent_name, user_id=user_id, authenticated=authenticated)


def _agent_identity(request: Request | None = None) -> tuple[str, str]:
    """Per-request identity wins: the gateway proxy sends the control-plane
    agent id/name as headers, so one shared runtime can serve any deployed
    agent and still report sessions against the right one. Environment
    variables (set by the operator per-pod) are the fallback. Call only after
    the request is authenticated, or with auth disabled."""
    header_id = header_name = ""
    if request is not None:
        header_id = request.headers.get("X-Kionga-Agent-ID", "")
        header_name = request.headers.get("X-Kionga-Agent-Name", "")
    return (
        header_id or os.environ.get("KIONGA_AGENT_ID", "unassigned"),
        header_name or os.environ.get("KIONGA_AGENT_NAME", "agent"),
    )


def _message_text(content: Any) -> str:
    """Text of a message's content: a string, or the text blocks of a list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block)
                       for block in content if isinstance(block, (str, dict)))
    return ""


def thread_key(agent_id: str, user_id: str, session_id: str) -> str:
    """The checkpoint thread for one conversation, namespaced by agent and
    user so knowing another user's session id never loads their history.
    Agent and user ids are percent-encoded, so ``:`` separates the parts
    unambiguously."""
    return f"{quote(agent_id, safe='')}:{quote(user_id, safe='')}:{session_id}"


def _run_config(session_id: str, caller: Caller) -> dict[str, Any]:
    user_id = caller.user_id
    config: dict[str, Any] = {
        # Tools that touch per-user data read the caller from
        # ``configurable["user_id"]``, never from model-chosen arguments.
        "configurable": {"thread_id": thread_key(caller.agent_id, user_id, session_id), "user_id": user_id},
        "metadata": {"langfuse_session_id": session_id, "langfuse_user_id": user_id},
    }
    handler = langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def create_app(graph: Any = None, *, auth: AuthConfig | None = None) -> FastAPI:
    """Build the runtime app. Pass ``graph`` directly in tests; production
    resolves ``KIONGA_GRAPH_MODULE`` during startup. ``auth`` defaults to
    :meth:`AuthConfig.from_env`, which raises on an unsafe configuration so
    the process fails before it listens."""
    auth = auth if auth is not None else AuthConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checkpointer_cm = None
        if graph is not None:
            app.state.graph = graph
        else:
            spec = os.environ.get("KIONGA_GRAPH_MODULE", "")
            if not spec:
                raise GraphLoadError("KIONGA_GRAPH_MODULE is not set")
            framework_managed = hasattr(resolve_attribute(spec), "kionga_framework")
            model = None if framework_managed else build_chat_model()
            checkpointer = None
            if framework_managed:
                pass  # Adapter owns provider configuration and session persistence.
            elif os.environ.get("KIONGA_CHECKPOINT_DSN") or os.environ.get("DATABASE_URL"):
                from kionga.agents import get_langgraph_checkpointer

                checkpointer_cm = get_langgraph_checkpointer()
                checkpointer = await checkpointer_cm.__aenter__()
                await checkpointer.setup()
            else:
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
            app.state.graph = build_graph(spec, model=model, checkpointer=checkpointer)
        try:
            yield
        finally:
            if checkpointer_cm is not None:
                await checkpointer_cm.__aexit__(None, None, None)

    app = FastAPI(title="kionga-agent-runtime", lifespan=lifespan)

    def require_gateway(http_request: Request) -> None:
        # A route dependency without a body, so it runs before body
        # validation: an unauthenticated caller gets 401, never a 422 that
        # describes the request schema.
        http_request.state.authenticated = _authenticate(http_request, auth)

    def caller(payload: InvokeRequest, http_request: Request) -> Caller:
        return _resolve_caller(http_request, http_request.state.authenticated, payload.user_id)

    @app.get("/healthz")
    async def healthz(http_request: Request) -> dict[str, str]:
        # Unauthenticated liveness: no identity in the response.
        _check_graph_module(http_request)
        return {"status": "ok", "auth": "required" if auth.required else "disabled",
                "framework": getattr(app.state.graph, "framework", "langgraph"),
                "streaming_mode": getattr(app.state.graph, "streaming_mode", "incremental")}

    @app.post("/invoke", response_model=InvokeResponse, dependencies=[Depends(require_gateway)])
    async def invoke(
        payload: InvokeRequest, background: BackgroundTasks, who: Caller = Depends(caller)
    ) -> InvokeResponse:
        from langchain_core.messages import HumanMessage

        session_id = payload.session_id or str(uuid.uuid4())
        agent_id, agent_name, user_id = who.agent_id, who.agent_name, who.user_id
        started = time.perf_counter()
        status, reply, usage = "succeeded", "", Usage()
        usage_available = False
        timing = None
        try:
            result = await app.state.graph.ainvoke(
                {"messages": [HumanMessage(content=payload.message)]},
                config=_run_config(session_id, who),
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            reply = str(messages[-1].content) if messages else ""
            usage = usage_from_messages(messages)
            usage_available = any(getattr(message, "usage_metadata", None) is not None for message in messages)
            timing = getattr(messages[-1], "response_metadata", {}).get("lfx_timing") if messages else None
        except Exception as error:
            status = "failed"
            raise HTTPException(status_code=502, detail=f"agent execution failed: {error}")
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            background.add_task(
                report_session,
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                name=agent_name,
                status=status,
                current_node="respond",
                duration_ms=duration_ms,
                usage=usage,
                metadata={"framework": getattr(app.state.graph, "framework", "langgraph"),
                          "usage_available": usage_available,
                          **({"lfx_timing": timing} if timing else {})},
            )
        return InvokeResponse(
            usage_available=usage_available,
            reply=reply,
            session_id=session_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            duration_ms=duration_ms,
        )

    @app.post("/stream", dependencies=[Depends(require_gateway)])
    async def stream(payload: InvokeRequest, who: Caller = Depends(caller)) -> StreamingResponse:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        session_id = payload.session_id or str(uuid.uuid4())
        agent_id, agent_name, user_id = who.agent_id, who.agent_name, who.user_id

        async def event_stream():
            yield f"data: {json.dumps({'streaming_mode': getattr(app.state.graph, 'streaming_mode', 'incremental')})}\n\n"
            started = time.perf_counter()
            status, collected = "succeeded", []
            try:
                async for chunk, _ in app.state.graph.astream(
                    {"messages": [HumanMessage(content=payload.message)]},
                    config=_run_config(session_id, who),
                    stream_mode="messages",
                ):
                    # LangGraph's messages mode also emits messages returned
                    # by nodes, such as ToolNode results holding raw tool
                    # output. Only model text goes to the client, matching
                    # /invoke, which returns the final AI message alone.
                    if isinstance(chunk, ToolMessage):
                        yield f"data: {json.dumps({'tool': chunk.name or 'tool', 'status': chunk.status or 'success'})}\n\n"
                        continue
                    if not isinstance(chunk, AIMessage):
                        continue
                    collected.append(chunk)  # usage counts every model chunk
                    text = _message_text(chunk.content)
                    if text:
                        yield f"data: {json.dumps({'delta': text})}\n\n"
            except Exception as error:
                status = "failed"
                yield f"data: {json.dumps({'error': str(error)})}\n\n"
            usage = usage_from_messages(collected)
            timing = getattr(collected[-1], "response_metadata", {}).get("lfx_timing") if collected else None
            duration_ms = int((time.perf_counter() - started) * 1000)
            yield (
                "data: "
                + json.dumps(
                    {
                        "done": True,
                        "usage_available": any(getattr(item, "usage_metadata", None) is not None for item in collected),
                        "session_id": session_id,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cost_usd": usage.cost_usd,
                        "duration_ms": duration_ms,
                    }
                )
                + "\n\n"
            )
            report_session(
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                name=agent_name,
                status=status,
                current_node="respond",
                duration_ms=duration_ms,
                usage=usage,
                metadata={"framework": getattr(app.state.graph, "framework", "langgraph"),
                          "usage_available": any(getattr(item, "usage_metadata", None) is not None for item in collected),
                          **({"lfx_timing": timing} if timing else {})},
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
