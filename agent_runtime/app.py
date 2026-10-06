"""FastAPI service wrapping LangGraph or a framework-neutral Kionga adapter.

Endpoints:

- ``GET  /healthz``  — liveness.
- ``POST /invoke``   — one agent turn; returns the reply and available usage.
- ``POST /stream``   — the same turn as Server-Sent Events token stream.

Configuration (environment):

- ``MLAIOPS_GRAPH_MODULE``   — ``package.module:attribute`` (required).
- ``MLAIOPS_AGENT_ID``       — control-plane agent id for session reporting.
- ``MLAIOPS_URL``            — gateway base URL for session reporting.
- ``DATABASE_URL`` / ``MLAIOPS_CHECKPOINT_DSN`` — enables the PostgreSQL
  checkpointer; falls back to in-memory checkpoints when absent.
- ``MLAIOPS_LLM_BACKEND`` / ``MLAIOPS_LLM_MODEL`` / ``MLAIOPS_LLM_BASE_URL`` —
  chat model selection (see ``mlaiops_sdk.llm``).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlaiops_sdk.llm import build_chat_model
from mlaiops_sdk.tracing import langfuse_handler

from .graphs import GraphLoadError, build_graph, resolve_attribute
from .reporting import Usage, report_session, usage_from_messages


class InvokeRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = ""
    user_id: str = ""


class InvokeResponse(BaseModel):
    usage_available: bool = False
    reply: str
    session_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int


def _agent_identity(request: Request | None = None) -> tuple[str, str]:
    """Per-request identity wins: the gateway proxy sends the control-plane
    agent id/name as headers, so one shared runtime can serve any deployed
    agent and still report sessions against the right one. Environment
    variables (set by the operator per-pod) are the fallback."""
    header_id = header_name = ""
    if request is not None:
        expected = request.headers.get("X-MLAIOps-Graph-Module", "")
        loaded = os.environ.get("MLAIOPS_GRAPH_MODULE", "")
        if expected and loaded and expected != loaded:
            raise HTTPException(status_code=409, detail="This runtime serves a different entrypoint; deploy a dedicated runtime for this agent")
        header_id = request.headers.get("X-MLAIOps-Agent-ID", "")
        header_name = request.headers.get("X-MLAIOps-Agent-Name", "")
    return (
        header_id or os.environ.get("MLAIOPS_AGENT_ID", "unassigned"),
        header_name or os.environ.get("MLAIOPS_AGENT_NAME", "agent"),
    )


def _run_config(session_id: str, user_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "metadata": {"langfuse_session_id": session_id, "langfuse_user_id": user_id},
    }
    handler = langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def create_app(graph: Any = None) -> FastAPI:
    """Build the runtime app. Pass ``graph`` directly in tests; production
    resolves ``MLAIOPS_GRAPH_MODULE`` during startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checkpointer_cm = None
        if graph is not None:
            app.state.graph = graph
        else:
            spec = os.environ.get("MLAIOPS_GRAPH_MODULE", "")
            if not spec:
                raise GraphLoadError("MLAIOPS_GRAPH_MODULE is not set")
            framework_managed = hasattr(resolve_attribute(spec), "kionga_framework")
            model = None if framework_managed else build_chat_model()
            checkpointer = None
            if framework_managed:
                pass  # Adapter owns provider configuration and session persistence.
            elif os.environ.get("MLAIOPS_CHECKPOINT_DSN") or os.environ.get("DATABASE_URL"):
                from mlaiops_sdk.agents import get_langgraph_checkpointer

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

    app = FastAPI(title="mlaiops-agent-runtime", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz(http_request: Request) -> dict[str, str]:
        agent_id, agent_name = _agent_identity(http_request)
        return {"status": "ok", "agent_id": agent_id, "agent_name": agent_name,
                "framework": getattr(app.state.graph, "framework", "langgraph"),
                "streaming_mode": getattr(app.state.graph, "streaming_mode", "incremental")}

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(
        payload: InvokeRequest, http_request: Request, background: BackgroundTasks
    ) -> InvokeResponse:
        from langchain_core.messages import HumanMessage

        session_id = payload.session_id or str(uuid.uuid4())
        agent_id, agent_name = _agent_identity(http_request)
        started = time.perf_counter()
        status, reply, usage = "succeeded", "", Usage()
        usage_available = False
        try:
            result = await app.state.graph.ainvoke(
                {"messages": [HumanMessage(content=payload.message)]},
                config=_run_config(session_id, payload.user_id),
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            reply = str(messages[-1].content) if messages else ""
            usage = usage_from_messages(messages)
            usage_available = any(getattr(message, "usage_metadata", None) is not None for message in messages)
        except Exception as error:
            status = "failed"
            raise HTTPException(status_code=502, detail=f"agent execution failed: {error}")
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            background.add_task(
                report_session,
                agent_id=agent_id,
                session_id=session_id,
                user_id=payload.user_id,
                name=agent_name,
                status=status,
                current_node="respond",
                duration_ms=duration_ms,
                usage=usage,
                metadata={"framework": getattr(app.state.graph, "framework", "langgraph"),
                          "usage_available": usage_available},
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

    @app.post("/stream")
    async def stream(payload: InvokeRequest, http_request: Request) -> StreamingResponse:
        from langchain_core.messages import HumanMessage

        session_id = payload.session_id or str(uuid.uuid4())
        agent_id, agent_name = _agent_identity(http_request)

        async def event_stream():
            yield f"data: {json.dumps({'streaming_mode': getattr(app.state.graph, 'streaming_mode', 'incremental')})}\n\n"
            started = time.perf_counter()
            status, collected = "succeeded", []
            try:
                async for chunk, _ in app.state.graph.astream(
                    {"messages": [HumanMessage(content=payload.message)]},
                    config=_run_config(session_id, payload.user_id),
                    stream_mode="messages",
                ):
                    text = getattr(chunk, "content", "")
                    if text:
                        collected.append(chunk)
                        yield f"data: {json.dumps({'delta': text})}\n\n"
            except Exception as error:
                status = "failed"
                yield f"data: {json.dumps({'error': str(error)})}\n\n"
            usage = usage_from_messages(collected)
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
                user_id=payload.user_id,
                name=agent_name,
                status=status,
                current_node="respond",
                duration_ms=duration_ms,
                usage=usage,
                metadata={"framework": getattr(app.state.graph, "framework", "langgraph"),
                          "usage_available": any(getattr(item, "usage_metadata", None) is not None for item in collected)},
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
