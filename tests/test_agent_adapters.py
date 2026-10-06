import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from agent_runtime.app import create_app
from mlaiops_sdk.agent_adapters import AgentResult, AgnoAdapter, CallableAdapter, NooaAdapter


def state(text="hello"):
    return {"messages": [HumanMessage(content=text)]}


def config(session="session-1", user="alice"):
    return {"configurable": {"thread_id": session}, "metadata": {"langfuse_user_id": user}}


@pytest.mark.asyncio
async def test_agno_routes_context_and_run_metrics_without_sharing_instances():
    instances, calls = [], []

    class FakeAgno:
        async def arun(self, message, **kwargs):
            calls.append((message, kwargs))
            return SimpleNamespace(content={"answer": message},
                                   metrics=SimpleNamespace(input_tokens=10, output_tokens=3))

    def factory(request):
        agent = FakeAgno()
        instances.append(agent)
        return agent

    adapter = AgnoAdapter(factory)
    results = await asyncio.gather(adapter.ainvoke(state(), config()),
                                   adapter.ainvoke(state("world"), config("session-2", "bob")))
    assert instances[0] is not instances[1]
    assert calls[0][1] == dict(session_id="session-1", user_id="alice", stream=False)
    assert calls[1][1]["user_id"] == "bob"
    assert results[0]["messages"][0].usage_metadata["total_tokens"] == 13
    assert results[0]["messages"][0].content == '{"answer": "hello"}'


@pytest.mark.asyncio
async def test_nooa_invokes_typed_method_and_reports_unknown_usage():
    class Analyst:
        async def analyze(self, text):
            return {"summary": text}

    adapter = NooaAdapter(lambda request: Analyst(), "analyze")
    result = await adapter.ainvoke(state(), config())
    assert not result["usage_available"]
    assert result["messages"][0].usage_metadata is None
    assert result["messages"][0].content == '{"summary": "hello"}'
    with pytest.raises(ValueError, match="public"):
        NooaAdapter(lambda request: Analyst(), "__dict__")


@pytest.mark.asyncio
async def test_custom_sync_and_async_contracts():
    def run(request):
        return AgentResult(request.message)

    result = await CallableAdapter(run).ainvoke(state(), config())
    assert result["messages"][0].content == "hello"
    with pytest.raises(TypeError, match="AgentResult"):
        await CallableAdapter(lambda request: "wrong result").ainvoke(state(), config())
    with pytest.raises(ValueError, match="non-negative"):
        await CallableAdapter(lambda request: AgentResult("bad", -1, 2)).ainvoke(state(), config())


def test_framework_managed_startup_does_not_construct_langgraph_provider(monkeypatch):
    monkeypatch.setenv("MLAIOPS_GRAPH_MODULE", "agents.framework_examples:build_custom")
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "not-a-langchain-provider")
    monkeypatch.setenv("DATABASE_URL", "must-not-be-used-by-custom-adapter")
    with TestClient(create_app()) as client:
        assert client.get("/healthz").json()["framework"] == "custom"
        response = client.post("/invoke", json={"message": "hello"})
        assert response.status_code == 200
        assert response.json()["usage_available"] is False
        with client.stream("POST", "/stream", json={"message": "hello"}) as stream:
            text = "".join(stream.iter_text())
        assert '"streaming_mode": "buffered"' in text
        assert text.count('"delta"') == 1
        assert '"usage_available": false' in text


def test_runtime_rejects_different_deployed_entrypoint(monkeypatch):
    monkeypatch.setenv("MLAIOPS_GRAPH_MODULE", "agents.framework_examples:build_custom")
    with TestClient(create_app()) as client:
        for route in ("/healthz", "/invoke", "/stream"):
            headers = {"X-MLAIOps-Graph-Module": "other.agent:build"}
            response = client.get(route, headers=headers) if route == "/healthz" else client.post(
                route, headers=headers, json={"message": "hello"})
            assert response.status_code == 409


def test_adapter_failure_is_not_reported_as_success():
    async def broken(request):
        raise RuntimeError("provider failed")

    with TestClient(create_app(graph=CallableAdapter(broken))) as client:
        assert client.post("/invoke", json={"message": "hello"}).status_code == 502
