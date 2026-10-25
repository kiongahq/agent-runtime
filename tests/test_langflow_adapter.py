import json
import os

import pytest
from langchain_core.messages import HumanMessage

from agents.langflow_adapter import LangflowGraph, _result_text, build_langflow
from agent_runtime.graphs import GraphLoadError, build_graph


def test_result_text_finds_nested_chat_output():
    payload = {"outputs": [{"results": {"message": {"text": "hello"}}}]}
    assert _result_text(payload) == "hello"
    assert _result_text({"result": [{"message": {"text": "hello"}}]}) == "hello"


def test_missing_published_flow_fails_before_server_starts(tmp_path):
    with pytest.raises(GraphLoadError, match="missing"):
        LangflowGraph(tmp_path / "missing.json")


@pytest.mark.asyncio
async def test_headless_langflow_invocation_uses_mounted_flow_and_session(tmp_path, monkeypatch):
    flow = tmp_path / "flow.json"
    flow.write_text(json.dumps({"data": {"nodes": [], "edges": []}}))
    executable = tmp_path / "lfx"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "args=sys.argv[1:]\n"
        "print(json.dumps({'result': {'text': args[args.index('--input-value')+1] + ':' + args[args.index('--session-id')+1]}}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    graph = LangflowGraph(flow)
    result = await graph.ainvoke({"messages": [HumanMessage(content="hello")]},
                                 config={"configurable": {"thread_id": "session-1"}})
    assert result["messages"][-1].content == "hello:session-1"
    streamed = [item async for item in graph.astream(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"thread_id": "session-2"}}, stream_mode="messages")]
    assert streamed[0][0].content == "hello:session-2"
    assert build_langflow.kionga_framework == "langflow"


def test_framework_adapter_does_not_require_chat_model(tmp_path, monkeypatch):
    flow = tmp_path / "flow.json"
    flow.write_text("{}")
    monkeypatch.setenv("KIONGA_LANGFLOW_FLOW_PATH", str(flow))
    graph = build_graph("agents.langflow_adapter:build_langflow")
    assert graph.framework == "langflow"
