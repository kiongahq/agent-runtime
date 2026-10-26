"""Playground runner for native Agent Designer drafts.

``python -m agent_runtime.designer`` serves:

- ``GET  /healthz`` — liveness and capacity.
- ``POST /runs``    — run one draft turn; the response is newline-delimited JSON
  run events (contracts ``agent-graph/run-event.v1.schema.json``).

Only the gateway calls this service. It authenticates with
``Authorization: Bearer $KIONGA_DESIGNER_RUNNER_TOKEN`` and passes a run-scoped
gateway credential; every resource call goes back through the gateway with
``X-Kionga-Run-ID`` and ``X-Kionga-Run-Token``, so permissions are checked on
each call and a cancelled run's credential stops working. The runner never sees
provider or resource secrets, and graphs are data interpreted by fixed handlers.

Configuration (environment):

- ``KIONGA_DESIGNER_RUNNER_TOKEN`` (or ``KIONGA_DESIGNER_RUNNER_TOKEN_FILE``, a
  mounted secret) — required; the service refuses to start without it.
- ``KIONGA_DESIGNER_PORT`` — listen port (default 9100).
- ``KIONGA_DESIGNER_MAX_RUNS`` — concurrent runs before 429 (default 8).
- ``MLAIOPS_LLM_BACKEND`` / ``MLAIOPS_LLM_MODEL`` / ``MLAIOPS_LLM_BASE_URL`` — the
  platform chat model used by ``model`` nodes set to ``llm``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .native_graph import normalize, run_graph, validate_graph
from .native_graph.judge import RUBRIC_VERSION, JudgeError, build_judge_model, judge, judge_backend
from .native_graph.resources import GatewayResources, Resources


VERSION_ID = re.compile(r"^agv-[a-f0-9]{24}$")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=32000)


class GatewayCredential(BaseModel):
    url: str = Field(pattern=r"^https?://[^\s/?#@]+(/[^\s?#@]*)?$", max_length=500)
    run_token: str = Field(min_length=16, max_length=256)


class RunRequest(BaseModel):
    run_id: str = Field(pattern=r"^agr-[a-f0-9]{24}$")
    graph: dict[str, Any]
    input: str = Field(min_length=1, max_length=32000)
    session_id: str = Field(default="", max_length=200)
    history: list[Turn] = Field(default_factory=list, max_length=50)
    allow_side_effects: bool = False
    stop_after: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,62}$")
    gateway: GatewayCredential
    # Published versions used by map, loop and subflow nodes, resolved and
    # authorized by the gateway.
    subflows: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=10)


def _runner_token() -> str:
    """The token from the environment, or from a mounted secret file
    (``KIONGA_DESIGNER_RUNNER_TOKEN_FILE``, as Kubernetes provides it)."""
    value = os.environ.get("KIONGA_DESIGNER_RUNNER_TOKEN", "")
    path = os.environ.get("KIONGA_DESIGNER_RUNNER_TOKEN_FILE", "")
    if value and path:
        raise RuntimeError("set only one of KIONGA_DESIGNER_RUNNER_TOKEN and KIONGA_DESIGNER_RUNNER_TOKEN_FILE")
    if path:
        with open(path, encoding="utf-8") as handle:
            value = handle.read(4097).rstrip("\r\n")
        if len(value) > 4096:
            raise RuntimeError("KIONGA_DESIGNER_RUNNER_TOKEN_FILE exceeds 4 KiB")
    return value


def _default_llm_factory():
    from mlaiops_sdk.llm import build_chat_model

    return build_chat_model()


def _default_resources(request: RunRequest) -> Resources:
    return GatewayResources(
        request.gateway.url,
        {"X-Kionga-Run-ID": request.run_id, "X-Kionga-Run-Token": request.gateway.run_token},
    )


class JudgeRequest(BaseModel):
    criteria: list[str] = Field(min_length=1, max_length=4)
    input: str = Field(max_length=32000)
    output: str = Field(max_length=65536)
    reference: str = Field(default="", max_length=8000)
    context: str = Field(default="", max_length=32000)


def create_app(
    *,
    token: str | None = None,
    max_runs: int | None = None,
    llm_factory: Callable[[], Any] | None = None,
    resources_factory: Callable[[RunRequest], Resources] | None = None,
    judge_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    token = token if token is not None else _runner_token()
    if len(token) < 16:
        raise RuntimeError("KIONGA_DESIGNER_RUNNER_TOKEN must be set to at least 16 characters")
    max_runs = max_runs if max_runs is not None else int(os.environ.get("KIONGA_DESIGNER_MAX_RUNS", "8"))
    llm_factory = llm_factory or _default_llm_factory
    judge_factory = judge_factory or build_judge_model
    resources_factory = resources_factory or _default_resources
    app = FastAPI(title="kionga-agent-designer-runner")
    app.state.active = 0

    def authorize(request: Request) -> None:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="runner credential required")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "active_runs": app.state.active, "max_runs": max_runs}

    @app.post("/judge", dependencies=[Depends(authorize)])
    async def judge_case(payload: JudgeRequest) -> dict[str, Any]:
        """Score one evaluated case with the LLM rubric judge."""
        try:
            model = judge_factory()
        except JudgeError as error:
            raise HTTPException(status_code=503, detail={"error": "judge_unavailable", "message": str(error)}) from None
        try:
            result = await judge(model, criteria=payload.criteria, request=payload.input, reply=payload.output,
                                 reference=payload.reference, context=payload.context)
        except ValueError as error:
            raise HTTPException(status_code=422, detail={"error": "invalid_criteria", "message": str(error)}) from None
        except JudgeError as error:
            raise HTTPException(status_code=502, detail={"error": "judge_failed", "message": str(error)}) from None
        except Exception as error:  # provider outage, quota, network
            raise HTTPException(status_code=502, detail={"error": "judge_failed", "message": f"judge model call failed: {type(error).__name__}"}) from None
        backend, model_name = judge_backend()
        return {**result, "judge": {"backend": backend, "model": model_name or "default", "rubric_version": RUBRIC_VERSION}}

    # Authorization is a dependency so an unauthenticated caller learns nothing
    # from body validation errors.
    @app.post("/runs", dependencies=[Depends(authorize)])
    async def runs(payload: RunRequest, http_request: Request) -> StreamingResponse:
        graph = normalize(payload.graph)
        issues = validate_graph(graph)
        if issues:
            raise HTTPException(status_code=422, detail={"error": "invalid_graph", "issues": issues})
        subflows = {}
        for version, nested in payload.subflows.items():
            nested = normalize(nested)
            nested_issues = validate_graph(nested)
            if not VERSION_ID.match(version) or nested_issues:
                raise HTTPException(status_code=422, detail={"error": "invalid_graph", "issues": [f"subflow {version}: {issue}" for issue in nested_issues] or [f"invalid version id {version}"]})
            subflows[version] = nested
        if payload.stop_after and payload.stop_after not in {node["id"] for node in graph["nodes"]}:
            raise HTTPException(status_code=422, detail={"error": "invalid_graph", "issues": ["stop_after node does not exist"]})
        if app.state.active >= max_runs:
            raise HTTPException(status_code=429, detail="runner is at capacity; retry shortly")
        app.state.active += 1
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                app.state.active -= 1

        cancel = asyncio.Event()

        async def watch_disconnect() -> None:
            while not cancel.is_set():
                if await http_request.is_disconnected():
                    cancel.set()
                    return
                await asyncio.sleep(0.5)

        async def stream():
            watcher = asyncio.create_task(watch_disconnect())
            events = run_graph(
                graph, input_text=payload.input, run_id=payload.run_id,
                history=[turn.model_dump() for turn in payload.history],
                allow_side_effects=payload.allow_side_effects, stop_after=payload.stop_after,
                resources=resources_factory(payload), llm_factory=llm_factory, cancel=cancel, subflows=subflows,
            )
            try:
                async for item in events:
                    yield json.dumps(item, ensure_ascii=False) + "\n"
            finally:
                cancel.set()
                watcher.cancel()
                await events.aclose()
                release()

        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store"}, background=BackgroundTask(release))

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("KIONGA_DESIGNER_HOST", "0.0.0.0"),  # nosec B104 -- container listener
        port=int(os.environ.get("KIONGA_DESIGNER_PORT", "9100")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
