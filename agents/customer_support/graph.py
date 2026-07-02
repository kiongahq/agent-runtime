"""Customer support agent: a real LangGraph StateGraph.

reason -> tools -> reason ... -> respond. The model decides tool use; tools
come from the platform tool registry and are converted to LangChain
``StructuredTool`` at build time. Feature lookups hit the online feature
gateway when configured and fall back to a deterministic local profile so the
agent works out of the box with no infrastructure.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from mlaiops_sdk.tools import langchain_tools, register_tool

SYSTEM_PROMPT = (
    "You are Nexus customer support. Use feature_store_lookup to fetch the "
    "customer's live profile before answering account questions, and kb_search "
    "to ground answers in the knowledge base. Answer concisely."
)

_DEMO_PROFILES = {
    "u123": {"plan": "pro", "region": "eu-west", "open_tickets": 1, "csat_90d": 4.6},
    "u456": {"plan": "free", "region": "us-east", "open_tickets": 0, "csat_90d": 4.1},
}

_KNOWLEDGE_BASE = [
    {
        "id": "kb-1",
        "topic": "billing",
        "text": "Invoices are issued on the 1st of each month; pro plans bill per seat.",
    },
    {
        "id": "kb-2",
        "topic": "data-retention",
        "text": "Pipeline logs are retained 30 days on the free plan and 365 days on pro.",
    },
    {
        "id": "kb-3",
        "topic": "support-sla",
        "text": "Pro plan support responds within 4 business hours; free within 2 days.",
    },
]


@register_tool(
    name="feature_store_lookup",
    version="1.0",
    description="Retrieve real-time customer profile features from the platform feature store",
    tags=["data", "features", "feast"],
)
def feature_store_lookup(entity_id: str) -> dict:
    """Return the online feature vector for a customer entity."""
    gateway = os.environ.get("MLAIOPS_FEATURE_GATEWAY_URL")
    if gateway:
        response = httpx.post(
            f"{gateway.rstrip('/')}/get-online-features",
            json={
                "feature_service": "customer_profile",
                "entities": [{"entity_id": entity_id}],
            },
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    return {"entity_id": entity_id, "features": _DEMO_PROFILES.get(entity_id, {}), "source": "demo"}


@register_tool(
    name="kb_search",
    version="1.0",
    description="Search the support knowledge base for grounded answers",
    tags=["retrieval", "kb"],
)
def kb_search(query: str) -> list[dict]:
    """Deterministic keyword search over the knowledge base."""
    terms = {term for term in query.lower().split() if len(term) > 2}
    scored = []
    for entry in _KNOWLEDGE_BASE:
        haystack = (entry["topic"] + " " + entry["text"]).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [entry for _, entry in scored[:3]]


def build(model: Any = None, checkpointer: Any = None):
    """Compile the customer support graph.

    ``model`` defaults to the platform-configured chat model; ``checkpointer``
    to in-memory checkpoints (the runtime injects PostgreSQL in production).
    """
    if model is None:
        from mlaiops_sdk.llm import build_chat_model

        model = build_chat_model()

    tools = langchain_tools(["feature_store_lookup", "kb_search"])
    bound = model.bind_tools(tools)

    def reason(state: MessagesState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        return {"messages": [bound.invoke(messages)]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("reason", reason)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "reason")
    return graph.compile(checkpointer=checkpointer)
