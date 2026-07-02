"""Gate tests for the agent runtime and the demo agent graph.

Everything runs with the mock chat model: no network, no keys, <2s.
"""

import pytest
from fastapi.testclient import TestClient

from agent_runtime.app import create_app
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


def test_invoke_uses_gateway_identity_headers(client, monkeypatch):
    """One shared runtime serves many agents: the gateway's identity headers
    must win over the runtime's own environment."""
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
