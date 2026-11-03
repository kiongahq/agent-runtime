"""Gate tests for the agent runtime and the demo agent graph.

Everything runs with the mock chat model: no network, no keys, <2s.
"""

import secrets

import pytest
from fastapi.testclient import TestClient

from agent_runtime.app import AuthConfig, create_app
from agent_runtime.graphs import GraphLoadError, build_graph
from agent_runtime.reporting import Usage, usage_from_messages
from mlaiops_sdk.llm import MockChatModel


def _demo_graph(responses=None):
    from agents.customer_support.graph import build

    return build(model=MockChatModel(responses=responses), checkpointer=None)


# -- graph loading contract ---------------------------------------------------


def test_build_graph_rejects_malformed_spec():
    with pytest.raises(GraphLoadError, match="package.module:attribute"):
        build_graph("no-colon-here")


def test_build_graph_rejects_unknown_module():
    with pytest.raises(GraphLoadError, match="cannot import"):
        build_graph("definitely.not.a.module:thing")


def test_build_graph_resolves_factory_with_model():
    graph = build_graph("agents.customer_support.graph:build", model=MockChatModel())
    assert hasattr(graph, "ainvoke")


# -- demo agent behavior ------------------------------------------------------


def test_demo_agent_answers_directly():
    graph = _demo_graph(responses=["Your invoice arrives on the 1st."])
    result = graph.invoke({"messages": [("user", "When do invoices arrive?")]})
    assert result["messages"][-1].content == "Your invoice arrives on the 1st."


def test_demo_agent_executes_tool_loop():
    """reason -> tools -> reason: the model requests a feature lookup, the
    ToolNode runs the real registered tool, and the model answers from it."""
    from langchain_core.messages import ToolMessage

    graph = _demo_graph(
        responses=[
            {"tool_call": {"name": "feature_store_lookup", "args": {"entity_id": "u123"}}},
            "You are on the pro plan.",
        ]
    )
    result = graph.invoke({"messages": [("user", "What plan am I on? I'm u123.")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert '"plan": "pro"' in tool_messages[0].content
    assert result["messages"][-1].content == "You are on the pro plan."


def test_demo_tools_are_deterministic():
    from agents.customer_support.graph import feature_store_lookup, kb_search

    profile = feature_store_lookup("u123")
    assert profile["features"]["plan"] == "pro"
    assert profile["source"] == "demo"

    hits = kb_search("how long are pipeline logs retained")
    assert hits and hits[0]["id"] == "kb-2"


# -- usage extraction ---------------------------------------------------------


def test_usage_from_messages_sums_ai_usage():
    model = MockChatModel(responses=["one two three"])
    message = model.invoke("count for me please")
    usage = usage_from_messages([message, "not-a-message"])
    assert usage.output_tokens == 3
    assert usage.input_tokens > 0


def test_cost_is_deterministic_from_rates(monkeypatch):
    monkeypatch.setenv("MLAIOPS_COST_PER_1K_INPUT", "3.0")
    monkeypatch.setenv("MLAIOPS_COST_PER_1K_OUTPUT", "15.0")
    usage = Usage(input_tokens=1000, output_tokens=2000)
    assert usage.cost_usd == pytest.approx(3.0 + 30.0)


def test_cost_defaults_to_zero():
    assert Usage(input_tokens=500, output_tokens=500).cost_usd == 0.0


# -- HTTP surface -------------------------------------------------------------


@pytest.fixture()
def client():
    app = create_app(graph=_demo_graph(responses=["Hello from the mock agent."]))
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"


def test_healthz_reveals_no_identity(client, monkeypatch):
    monkeypatch.setenv("MLAIOPS_AGENT_ID", "agt-secret-id")
    body = client.get("/healthz", headers={"X-MLAIOps-Agent-ID": "agt-1"}).json()
    assert "agent_id" not in body and "agent_name" not in body
    assert "agt-" not in str(body)


def test_invoke_returns_reply_and_usage(client):
    response = client.post("/invoke", json={"message": "hi", "user_id": "u123"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Hello from the mock agent."
    assert body["session_id"]
    assert body["output_tokens"] > 0


def test_invoke_reports_session_to_gateway(client, monkeypatch, respx_or_none=None):
    recorded = {}

    def fake_record(payload):
        recorded.update(payload)

    monkeypatch.setattr("agent_runtime.reporting.record_trace", fake_record)
    response = client.post("/invoke", json={"message": "hi", "session_id": "sess-1"})
    assert response.status_code == 200
    assert recorded["session_id"] == "sess-1"
    assert recorded["status"] == "succeeded"
    assert recorded["output_tokens"] > 0


def test_langflow_timings_reach_trace_metadata(monkeypatch):
    from langchain_core.messages import AIMessage

    class TimedGraph:
        framework = "langflow"

        async def ainvoke(self, payload, *, config):
            return {"messages": [AIMessage(content="done", response_metadata={
                "lfx_timing": {"execution_time": 0.02, "component_timings": [
                    {"component": "Chat Output", "duration": 0.01}
                ]},
            })]}

    recorded = {}
    monkeypatch.setattr("agent_runtime.reporting.record_trace", lambda payload: recorded.update(payload))
    with TestClient(create_app(graph=TimedGraph())) as timed_client:
        response = timed_client.post("/invoke", json={"message": "hi"})
    assert response.status_code == 200
    assert recorded["metadata"]["lfx_timing"]["component_timings"][0]["component"] == "Chat Output"
    assert recorded["metadata"]["usage_available"] is False


def test_invoke_uses_gateway_identity_headers(client, monkeypatch):
    """Local development (auth disabled): one shared runtime serves many
    agents, and the identity headers win over the runtime's own environment.
    With auth required they are honored only with the gateway credential."""
    recorded = {}
    monkeypatch.setattr(
        "agent_runtime.reporting.record_trace", lambda payload: recorded.update(payload)
    )
    response = client.post(
        "/invoke",
        json={"message": "hi"},
        headers={"X-MLAIOps-Agent-ID": "agt-42", "X-MLAIOps-Agent-Name": "billing-bot"},
    )
    assert response.status_code == 200
    assert recorded["agent_id"] == "agt-42"
    assert recorded["name"] == "billing-bot"


def test_invoke_validates_empty_message(client):
    assert client.post("/invoke", json={"message": ""}).status_code == 422


def test_stream_emits_final_summary(client):
    with client.stream("POST", "/stream", json={"message": "hi"}) as response:
        payload = "".join(response.iter_text())
    assert '"done": true' in payload


# -- gateway authentication ---------------------------------------------------

# Assembled at runtime so no credential-shaped literal is committed.
GATEWAY_TOKEN = "-".join(["runtime", "test", secrets.token_hex(12)])
GATEWAY = {"Authorization": f"Bearer {GATEWAY_TOKEN}", "X-MLAIOps-User-ID": "alice"}


@pytest.fixture()
def secured(monkeypatch):
    recorded = []
    monkeypatch.setattr("agent_runtime.reporting.record_trace", recorded.append)
    app = create_app(graph=_demo_graph(responses=["Hello."] * 20),
                     auth=AuthConfig(required=True, token=GATEWAY_TOKEN))
    with TestClient(app) as test_client:
        test_client.recorded = recorded
        yield test_client


def _call(client, path, headers=None, body=None):
    body = body if body is not None else {"message": "hi"}
    if path == "/stream":
        with client.stream("POST", path, json=body, headers=headers or {}) as response:
            return response.status_code, "".join(response.iter_text())
    response = client.post(path, json=body, headers=headers or {})
    return response.status_code, response.text


@pytest.mark.parametrize("path", ["/invoke", "/stream"])
@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong-token-of-enough-length"},
    {"Authorization": GATEWAY_TOKEN},
    {"Authorization": f"Bearer {GATEWAY_TOKEN}x"},
    {"X-MLAIOps-Agent-ID": "agt-42", "X-MLAIOps-User-ID": "alice"},
], ids=["missing", "wrong", "no-scheme", "suffix", "spoofed-headers-only"])
def test_unauthenticated_requests_are_rejected(secured, path, headers):
    status, _ = _call(secured, path, headers)
    assert status == 401
    assert secured.recorded == []


@pytest.mark.parametrize("path", ["/invoke", "/stream"])
def test_unauthenticated_invalid_body_gets_401_not_422(secured, path):
    status, _ = _call(secured, path, {}, body={"message": ""})
    assert status == 401


@pytest.mark.parametrize("path", ["/invoke", "/stream"])
def test_gateway_requests_succeed_and_carry_gateway_identity(secured, path):
    headers = {**GATEWAY, "X-MLAIOps-Agent-ID": "agt-42", "X-MLAIOps-Agent-Name": "billing-bot"}
    status, _ = _call(secured, path, headers)
    assert status == 200
    assert secured.recorded[-1]["agent_id"] == "agt-42"
    assert secured.recorded[-1]["name"] == "billing-bot"
    assert secured.recorded[-1]["user_id"] == "alice"


@pytest.mark.parametrize("path", ["/invoke", "/stream"])
def test_gateway_requests_must_name_the_user(secured, path):
    status, _ = _call(secured, path, {"Authorization": f"Bearer {GATEWAY_TOKEN}"})
    assert status == 401


@pytest.mark.parametrize("path", ["/invoke", "/stream"])
def test_body_user_id_cannot_override_the_authenticated_user(secured, path):
    status, _ = _call(secured, path, GATEWAY, body={"message": "hi", "user_id": "mallory"})
    assert status == 403
    status, _ = _call(secured, path, GATEWAY, body={"message": "hi", "user_id": "alice"})
    assert status == 200


def test_healthz_stays_open_with_auth_required(secured):
    assert secured.get("/healthz").json()["auth"] == "required"


def _clear_auth_env(monkeypatch):
    for name in ("KIONGA_ENVIRONMENT", "MLAIOPS_RUNTIME_AUTH", "MLAIOPS_INTERNAL_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_auth_defaults_to_disabled_for_local_development(monkeypatch):
    _clear_auth_env(monkeypatch)
    assert AuthConfig.from_env().required is False


def test_auth_is_required_in_production(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("KIONGA_ENVIRONMENT", "production")
    monkeypatch.setenv("MLAIOPS_INTERNAL_TOKEN", GATEWAY_TOKEN)
    config = AuthConfig.from_env()
    assert config.required is True and config.token == GATEWAY_TOKEN


@pytest.mark.parametrize("env, message", [
    ({"KIONGA_ENVIRONMENT": "production"}, "MLAIOPS_INTERNAL_TOKEN"),
    ({"KIONGA_ENVIRONMENT": "production", "MLAIOPS_INTERNAL_TOKEN": "short"}, "at least 16"),
    ({"KIONGA_ENVIRONMENT": "production", "MLAIOPS_RUNTIME_AUTH": "disabled"}, "not allowed"),
    ({"MLAIOPS_RUNTIME_AUTH": "required"}, "MLAIOPS_INTERNAL_TOKEN"),
    ({"MLAIOPS_RUNTIME_AUTH": "sometimes"}, "must be"),
])
def test_unsafe_auth_configuration_fails_startup(monkeypatch, env, message):
    _clear_auth_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
        create_app(graph=_demo_graph())
