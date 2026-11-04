"""``POST /agent-turns``: the gate lane. Real LangChain providers against a loopback server."""

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_runtime import agent_turns as module
from agent_runtime.agent_turn_mock import MARKER, SAFE_COMMAND, ScriptedTurnModel
from agent_runtime.agent_turns import (
    AgentTurnRequest,
    AgentTurns,
    UserMessage,
    classify,
    parse_retry_after,
    to_langchain_messages,
)
from agent_runtime.designer import create_app

# Copied from kiongahq/contracts agent-team (PR 42); update together.
CONTRACT = Path(__file__).parent / "fixtures" / "agent_team"
RESPONSE_SCHEMA = json.loads((CONTRACT / "runner-turn-response.v1.schema.json").read_text())
REQUEST_SCHEMA = json.loads((CONTRACT / "runner-turn-request.v1.schema.json").read_text())
TOKEN = "runner-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CANARY = "CANARY-do-not-log-4417"
TRANSCRIPT = [
    {"role": "user", "content": f"task {CANARY}"},
    {"role": "assistant", "content": "two calls", "tool_calls": [
        {"id": "c1", "name": "exec", "arguments": {"command": "ls"}},
        {"id": "c2", "name": "read_file", "arguments": {"path": "a.txt"}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": f"exec result {CANARY}"},
    {"role": "tool", "tool_call_id": "c2", "content": "file result"},
]
TOOLS = [
    {"name": "exec", "description": "Run a command", "input_schema": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a file", "input_schema": {}},
]


def turn(**overrides):
    return {"run_id": "atr-" + "a" * 24, "worker_id": "atw-" + "b" * 24, "turn": 3, "model": "llm",
            "system": "You are a worker.", "messages": TRANSCRIPT, "tools": TOOLS, "max_output_tokens": 500, **overrides}


class Provider:
    """A loopback model provider: canned responses in order, every request recorded."""

    def __init__(self):
        self.responses, self.requests = [], []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                provider.requests.append({"path": self.path, "json": json.loads(body)})
                status, headers, payload = provider.responses.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider():
    server = Provider()
    yield server
    server.close()


def openai_reply(content="done", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "served-gpt",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


def anthropic_reply(blocks):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "served-claude", "content": blocks,
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 7}}


def openai_function(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def use_backend(monkeypatch, backend, url):
    """Configure the environment as an operator would; the default factory does the rest."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL",
                 "MLAIOPS_LLM_MODEL", "MLAIOPS_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", backend)
    monkeypatch.setenv("MLAIOPS_LLM_MODEL", "test-model")
    if backend == "anthropic":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-key")
        monkeypatch.setenv("ANTHROPIC_API_URL", url)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
        monkeypatch.setenv("MLAIOPS_LLM_BASE_URL", url)


def client(**kwargs):
    return TestClient(create_app(token=TOKEN, agent_turns=AgentTurns(**kwargs) if kwargs else None))


def post(app, payload):
    response = app.post("/agent-turns", json=payload, headers=AUTH)
    assert response.status_code == 200, response.text
    # Every response any test produces must satisfy the closed contract.
    jsonschema.validate(response.json(), RESPONSE_SCHEMA)
    return response.json()


# --- auth ---------------------------------------------------------------------------------


def test_the_runner_credential_is_checked_before_the_body(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    app = client()
    assert app.post("/agent-turns", json={}).status_code == 401
    assert app.post("/agent-turns", content=b"not json").status_code == 401
    assert app.post("/agent-turns", json=turn(), headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert app.post("/agent-turns", json=turn(), headers=AUTH).status_code == 200


# --- contract -----------------------------------------------------------------------------


def test_the_test_request_is_itself_a_contract_request():
    jsonschema.validate(turn(), REQUEST_SCHEMA)


def test_contract_valid_requests_are_accepted(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    examples = sorted((CONTRACT / "valid").glob("runner-turn-request--*.json"))
    assert examples
    for path in examples:
        assert "error" not in post(client(), json.loads(path.read_text())), path.name


def test_contract_invalid_requests_are_refused(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    examples = sorted((CONTRACT / "invalid").glob("runner-turn-request--*.json"))
    assert len(examples) >= 3
    for path in examples:
        document = json.loads(path.read_text())["document"]
        assert post(client(), document)["error"]["code"] == "invalid_request", path.name


def test_contract_response_examples_validate():
    examples = list((CONTRACT / "valid").glob("runner-turn-response--*.json"))
    assert examples
    for path in examples:
        jsonschema.validate(json.loads(path.read_text()), RESPONSE_SCHEMA)


def test_empty_content_and_tool_calls_are_always_present(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply(None, [openai_function("c9", "exec", "{}")])))
    provider.responses.append((200, {}, openai_reply("text only")))
    app = client()
    assert post(app, turn())["content"] == ""
    assert post(app, turn())["tool_calls"] == []


def test_the_response_model_is_never_empty():
    class Nameless:
        def bind_tools(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="x")

    assert post(client(model_factory=Nameless), turn())["model"] == "unknown"


def test_a_tool_name_outside_the_contract_is_model_error(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply("", [openai_function("c9", "not a name!", "{}")])))
    assert post(client(), turn())["error"]["code"] == "model_error"


def test_a_long_error_message_stays_inside_the_contract():
    assert len(module.TurnError("model_error", "x" * 5000).message) <= 500


# --- conversion ---------------------------------------------------------------------------


def test_transcript_converts_to_real_langchain_messages():
    system, user, assistant, first, second = to_langchain_messages(
        "rules", AgentTurnRequest.model_validate(turn()).messages)
    assert isinstance(system, SystemMessage) and system.content == "rules"
    assert isinstance(user, HumanMessage)
    assert isinstance(assistant, AIMessage)
    assert [(call["id"], call["name"], call["args"]) for call in assistant.tool_calls] == [
        ("c1", "exec", {"command": "ls"}), ("c2", "read_file", {"path": "a.txt"})]
    assert isinstance(first, ToolMessage) and first.tool_call_id == "c1"
    assert isinstance(second, ToolMessage) and second.tool_call_id == "c2" and second.content == "file result"


def test_no_system_message_when_system_is_empty():
    messages = to_langchain_messages("", [UserMessage(role="user", content="hi")])
    assert [type(message) for message in messages] == [HumanMessage]


def test_openai_wire_format_and_response(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply("thinking", [
        openai_function("call_1", "exec", '{"command": "pwd"}'),
        openai_function("call_2", "read_file", '{"path": "b"}'),
    ])))
    result = post(client(), turn())
    assert result == {
        "content": "thinking",
        "tool_calls": [{"id": "call_1", "name": "exec", "arguments": {"command": "pwd"}},
                       {"id": "call_2", "name": "read_file", "arguments": {"path": "b"}}],
        "usage": {"input_tokens": 11, "output_tokens": 7},
        "model": "served-gpt",
    }
    request = provider.requests[0]
    assert request["path"].endswith("/chat/completions")
    wire = request["json"]
    assert [message["role"] for message in wire["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    assert [call["id"] for call in wire["messages"][2]["tool_calls"]] == ["c1", "c2"]
    assert json.loads(wire["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
    assert [(message["tool_call_id"], message["content"]) for message in wire["messages"][3:]] == [
        ("c1", f"exec result {CANARY}"), ("c2", "file result")]
    assert wire["tools"][0]["function"]["parameters"]["required"] == ["command"]
    # An empty input_schema must still be a valid object schema.
    assert wire["tools"][1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert (wire.get("max_completion_tokens") or wire.get("max_tokens")) == 500


def test_anthropic_wire_format_and_response(monkeypatch, provider):
    use_backend(monkeypatch, "anthropic", provider.url)
    provider.responses.append((200, {}, anthropic_reply([
        {"type": "text", "text": "thinking"},
        {"type": "tool_use", "id": "toolu_1", "name": "exec", "input": {"command": "pwd"}},
        {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "b"}},
    ])))
    result = post(client(), turn())
    assert result == {
        "content": "thinking",
        "tool_calls": [{"id": "toolu_1", "name": "exec", "arguments": {"command": "pwd"}},
                       {"id": "toolu_2", "name": "read_file", "arguments": {"path": "b"}}],
        "usage": {"input_tokens": 11, "output_tokens": 7},
        "model": "served-claude",
    }
    wire = provider.requests[0]["json"]
    assert provider.requests[0]["path"].endswith("/v1/messages")
    assert wire["max_tokens"] == 500
    assert wire["system"] in ("You are a worker.", [{"type": "text", "text": "You are a worker."}])
    assert [message["role"] for message in wire["messages"]] == ["user", "assistant", "user"]
    uses = [block for block in wire["messages"][1]["content"] if block["type"] == "tool_use"]
    assert [(block["id"], block["input"]) for block in uses] == [("c1", {"command": "ls"}), ("c2", {"path": "a.txt"})]
    results = [block for block in wire["messages"][2]["content"] if block["type"] == "tool_result"]
    assert [block["tool_use_id"] for block in results] == ["c1", "c2"], "both results answer in one user message"
    assert {tool["name"] for tool in wire["tools"]} == {"exec", "read_file"}
    assert wire["tools"][1]["input_schema"]["type"] == "object"


def test_anthropic_accepts_a_large_output_limit_without_streaming(monkeypatch, provider):
    """The SDK refuses long non-streaming calls unless the client has an explicit timeout."""
    use_backend(monkeypatch, "anthropic", provider.url)
    provider.responses.append((200, {}, anthropic_reply([{"type": "text", "text": "ok"}])))
    assert post(client(), turn(max_output_tokens=64000))["content"] == "ok"


def test_no_tools_are_bound_when_none_are_offered(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply("hello")))
    result = post(client(), turn(messages=[{"role": "user", "content": "hi"}], tools=[]))
    assert result["content"] == "hello" and result["tool_calls"] == []
    assert "tools" not in provider.requests[0]["json"], "OpenAI rejects an empty tools list"


def test_nulls_are_refused_as_the_closed_contract_requires(monkeypatch):
    """A Go nil slice marshals to null, so the gateway must send [] or omit the field."""
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    app = client()
    assert post(app, turn(tools=None))["error"]["code"] == "invalid_request"
    assert post(app, turn(messages=[USER, {"role": "assistant", "content": "x", "tool_calls": None}, USER]))["error"]["code"] == "invalid_request"
    assert post(app, turn(messages=[{"role": "user", "content": None}]))["error"]["code"] == "invalid_request"


def test_unusable_tool_call_ids_are_replaced_so_the_transcript_round_trips(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply("", [
        openai_function("functions.exec:0", "exec", "{}"),
        openai_function("dup", "exec", "{}"),
        openai_function("dup", "exec", "{}"),
    ])))
    assert [call["id"] for call in post(client(), turn())["tool_calls"]] == ["call-3-0", "dup", "call-3-2"]


def test_block_content_keeps_only_text():
    assert module._text([{"type": "reasoning", "summary": "x"}, {"type": "text", "text": "a"}, "b"]) == "ab"
    assert module._text(None) == ""


# --- error mapping ------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["openai-compatible", "anthropic"])
@pytest.mark.parametrize("status,headers,code,retry_after", [
    (429, {"Retry-After": "30"}, "rate_limited", 30),
    (429, {"Retry-After": "1.5"}, "rate_limited", 2),
    (429, {"retry-after-ms": "2500"}, "rate_limited", 3),
    (429, {}, "rate_limited", None),
    (429, {"Retry-After": "soon"}, "rate_limited", None),
    (429, {"Retry-After": "0"}, "rate_limited", None),
    (429, {"Retry-After": "-5"}, "rate_limited", None),
    (429, {"Retry-After": "999999999"}, "rate_limited", 86400),
    (500, {}, "provider_unavailable", None),
    (503, {"Retry-After": "12"}, "provider_unavailable", 12),
    (529, {}, "provider_unavailable", None),
    (408, {}, "provider_unavailable", None),
    (400, {}, "invalid_request", None),
    (413, {}, "invalid_request", None),
    (401, {}, "model_error", None),
    (403, {}, "model_error", None),
    (404, {}, "model_error", None),
])
def test_provider_status_maps_to_contract_errors(monkeypatch, provider, caplog, backend, status, headers, code, retry_after):
    use_backend(monkeypatch, backend, provider.url)
    # Providers can quote the prompt in an error; the text must go no further.
    provider.responses.append((status, headers, {"error": {"type": "x", "message": f"echo {CANARY}"}}))
    with caplog.at_level(logging.DEBUG):
        result = post(client(), turn())
    assert set(result) == {"error"}
    assert result["error"]["code"] == code
    assert result["error"].get("retry_after_seconds") == retry_after
    assert CANARY not in json.dumps(result) and CANARY not in caplog.text
    assert len(provider.requests) == 1, "the SDK must not retry; the gateway owns retries"


def test_retry_after_http_date():
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:40 GMT"}, now=1445412480.0) == 40
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:40 GMT"}, now=1445412600.0) is None
    assert parse_retry_after(None) is None


def test_exhausted_quota_is_not_a_short_rate_limit(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((429, {}, {"error": {"message": "no", "type": "insufficient_quota", "code": "insufficient_quota"}}))
    error = post(client(), turn())["error"]
    assert error["code"] == "provider_unavailable" and "insufficient_quota" in error["message"]


@pytest.mark.parametrize("backend", ["openai-compatible", "anthropic"])
def test_connection_failure_is_provider_unavailable(monkeypatch, backend):
    use_backend(monkeypatch, backend, "http://127.0.0.1:9")
    assert post(client(), turn())["error"]["code"] == "provider_unavailable"


def test_a_wrapped_provider_error_is_still_classified():
    class RateLimitError(Exception):
        status_code = 429
        response = type("R", (), {"headers": {"retry-after": "7"}})()

    try:
        try:
            raise RateLimitError("secret text")
        except RateLimitError as inner:
            raise ValueError("wrapper") from inner
    except ValueError as outer:
        mapped = classify(outer)
    assert (mapped.code, mapped.retry_after_seconds) == ("rate_limited", 7)
    assert "secret text" not in mapped.message


def test_an_unknown_exception_is_model_error_with_the_class_name_only():
    mapped = classify(KeyError(f"prompt {CANARY}"))
    assert mapped.code == "model_error" and "KeyError" in mapped.message and CANARY not in mapped.message


def test_a_malformed_tool_call_is_model_error(monkeypatch, provider):
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply("", [openai_function("c9", "exec", '{"command": "tru')])))
    assert post(client(), turn())["error"]["code"] == "model_error"


def test_missing_provider_configuration_is_model_error_and_recovers(monkeypatch, provider):
    use_backend(monkeypatch, "openai", provider.url)
    monkeypatch.delenv("OPENAI_API_KEY")
    app = client()
    error = post(app, turn())["error"]
    assert error["code"] == "model_error" and "OPENAI_API_KEY" in error["message"]
    use_backend(monkeypatch, "openai-compatible", provider.url)
    provider.responses.append((200, {}, openai_reply("back")))
    assert post(app, turn())["content"] == "back", "a failed build is not cached"


class SlowModel(ScriptedTurnModel):
    async def _agenerate(self, *args, **kwargs):
        await asyncio.sleep(30)


def test_a_slow_provider_is_provider_unavailable():
    error = post(client(model_factory=SlowModel, timeout=0.01), turn())["error"]
    assert error["code"] == "provider_unavailable"


def test_capacity_is_a_retryable_rate_limit(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    turns = AgentTurns(max_active=1)
    turns.active = 1
    app = TestClient(create_app(token=TOKEN, agent_turns=turns))
    assert post(app, turn())["error"] == {"code": "rate_limited", "message": "the runner is at capacity",
                                          "retry_after_seconds": module.CAPACITY_RETRY_SECONDS}
    turns.active = 0
    assert "error" not in post(app, turn())
    assert turns.active == 0
    assert app.get("/healthz").json()["max_agent_turns"] == 1


def test_a_caller_disconnect_cancels_the_model_call(monkeypatch):
    monkeypatch.setattr(module, "DISCONNECT_POLL_SECONDS", 0.01)
    cancelled = []

    class Hanging(ScriptedTurnModel):
        async def _agenerate(self, *args, **kwargs):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    turns = AgentTurns(model_factory=Hanging)
    app = create_app(token=TOKEN, agent_turns=turns)
    body = json.dumps(turn()).encode()
    sent = []

    async def drive():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/agent-turns", "raw_path": b"/agent-turns", "query_string": b"", "root_path": "",
                 "headers": [(b"authorization", AUTH["Authorization"].encode()), (b"content-type", b"application/json")],
                 "client": ("t", 1), "server": ("t", 80)}
        await asyncio.wait_for(app(scope, receive, send), 5)

    asyncio.run(drive())
    assert cancelled == [True]
    assert turns.active == 0


# --- request validation -------------------------------------------------------------------


USER = {"role": "user", "content": "hi"}
CALL = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "exec", "arguments": {}}]}


@pytest.mark.parametrize("overrides,fragment", [
    ({"api_key": CANARY}, "api_key"),
    ({"run_id": "agr-" + "a" * 24}, "run_id"),
    ({"worker_id": "nope"}, "worker_id"),
    ({"turn": 0}, "turn"),
    ({"model": "mdl-1"}, "platform model"),
    ({"messages": []}, "messages"),
    ({"max_output_tokens": 0}, "max_output_tokens"),
    ({"tools": [{"name": "bad name", "description": "", "input_schema": {}}]}, "tools"),
    ({"tools": [{"name": "a", "description": "", "input_schema": {}}, {"name": "a", "description": "", "input_schema": {}}]}, "unique"),
    ({"tools": [{"name": "a", "description": "", "input_schema": {"type": "array"}}]}, "object"),
    ({"messages": [{"role": "system", "content": CANARY}]}, "role"),
    ({"messages": [{"role": "assistant", "content": CANARY}, USER]}, "first message"),
    ({"messages": [USER, {"role": "assistant", "content": CANARY}]}, "last message"),
    ({"messages": [USER, CALL]}, "no result"),
    ({"messages": [USER, CALL, USER]}, "no result"),
    ({"messages": [USER, {"role": "tool", "tool_call_id": "c1", "content": CANARY}]}, "pending"),
    ({"messages": [USER, CALL, {"role": "tool", "tool_call_id": "c2", "content": CANARY}]}, "pending"),
    ({"messages": [USER, CALL, {"role": "tool", "content": CANARY}]}, "tool_call_id"),
    ({"messages": [{"role": "user", "content": ""}]}, "empty"),
    ({"messages": [{"role": "user", "content": CANARY, "tool_calls": []}]}, "tool_calls"),
    ({"messages": [USER, {"role": "assistant", "content": ""}, USER]}, "no content"),
    ({"messages": [USER, {"role": "assistant", "content": "x", "tool_calls": [
        {"id": "c1", "name": "a", "arguments": {}}, {"id": "c1", "name": "a", "arguments": {}}]}, USER]}, "duplicate"),
])
def test_bad_requests_are_invalid_request_and_echo_no_content(monkeypatch, overrides, fragment):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    error = post(client(), turn(**overrides))["error"]
    assert error["code"] == "invalid_request"
    assert fragment in error["message"]
    assert CANARY not in error["message"]


def test_non_json_and_oversized_bodies_are_invalid_request(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    app = client()
    response = app.post("/agent-turns", content=b"{not json", headers=AUTH)
    assert response.status_code == 200 and response.json()["error"]["code"] == "invalid_request"
    assert post(app, ["a list"])["error"]["code"] == "invalid_request"
    monkeypatch.setattr(module, "MAX_BODY_BYTES", 16)
    assert "exceeds" in post(app, turn())["error"]["message"]


# --- logging ------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["openai-compatible", "anthropic"])
def test_a_successful_turn_logs_identifiers_and_counts_but_no_content(monkeypatch, provider, caplog, capfd, backend):
    use_backend(monkeypatch, backend, provider.url)
    reply = openai_reply(f"reply {CANARY}") if backend != "anthropic" else anthropic_reply([{"type": "text", "text": f"reply {CANARY}"}])
    provider.responses.append((200, {}, reply))
    payload = turn(system=f"system {CANARY}")
    with caplog.at_level(logging.DEBUG):
        result = post(client(), payload)
    assert CANARY in result["content"], "the reply itself still reaches the gateway"
    assert CANARY not in caplog.text
    captured = capfd.readouterr()
    assert CANARY not in captured.out and CANARY not in captured.err
    line = next(record.getMessage() for record in caplog.records if record.name == "kionga.agent_turns")
    assert "run_id=atr-" in line and "outcome=ok" in line and "input_tokens=11" in line


# --- mock backend -------------------------------------------------------------------------


COORDINATOR_SYSTEM = "Base rules.\n\nTeam members you can spawn:\n- bug-fixer: Fixes bugs.\n- reviewer: Reviews.\n\nReport format."
SPAWN = {"name": "spawn_worker", "description": "Start a member", "input_schema": {
    "type": "object", "properties": {"agent": {"type": "string"}, "task": {"type": "string"}}}}
FINISH = {"name": "finish", "description": "Hand back", "input_schema": {
    "type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]}}
EXEC = {"name": "exec", "description": "Run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}


@pytest.fixture
def mock_app(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    return client()


def assistant(reply):
    return {"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]}


def hand_back(name, kind="hand-back"):
    return {"role": "user", "content": f'<kionga-data source="worker:{name}" kind="{kind}" nonce="n">done</kionga-data>'}


def test_mock_worker_runs_one_safe_exec_then_finishes(mock_app):
    messages = [{"role": "user", "content": "task"}]
    first = post(mock_app, turn(messages=messages, tools=[EXEC, FINISH]))
    assert [(call["name"], call["arguments"]) for call in first["tool_calls"]] == [("exec", {"command": SAFE_COMMAND})]
    assert first["usage"]["input_tokens"] > 0 and first["usage"]["output_tokens"] > 0 and first["model"] == "mock"
    messages += [assistant(first), {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": "ok"}]
    second = post(mock_app, turn(messages=messages, tools=[EXEC, FINISH]))
    assert [call["name"] for call in second["tool_calls"]] == ["finish"]
    report = second["tool_calls"][0]["arguments"]["report"]
    for section in ("Result:", "Evidence:", "Not verified:", "Follow-ups:"):
        assert section in report


def test_mock_worker_without_exec_finishes_at_once(mock_app):
    result = post(mock_app, turn(messages=[{"role": "user", "content": "task"}], tools=[FINISH]))
    assert [call["name"] for call in result["tool_calls"]] == ["finish"]


def test_mock_worker_without_finish_replies_with_text(mock_app):
    result = post(mock_app, turn(messages=[{"role": "user", "content": "task"}], tools=[]))
    assert result["tool_calls"] == [] and "Result:" in result["content"]


def test_mock_coordinator_spawns_each_member_once_waits_then_finishes(mock_app):
    tools = [SPAWN, FINISH]
    messages = [{"role": "user", "content": "fix the bug"}]
    spawn = post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=messages, tools=tools))
    assert [call["arguments"]["agent"] for call in spawn["tool_calls"]] == ["bug-fixer", "reviewer"]
    assert len({call["id"] for call in spawn["tool_calls"]}) == 2
    messages += [assistant(spawn)] + [
        {"role": "tool", "tool_call_id": call["id"], "content": '{"worker_id": "atw-1"}'} for call in spawn["tool_calls"]]
    waiting = post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=messages, tools=tools))
    assert waiting["tool_calls"] == [] and "bug-fixer" in waiting["content"] and "reviewer" in waiting["content"]
    messages += [assistant(waiting), hand_back("bug-fixer")]
    still = post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=messages, tools=tools))
    assert still["tool_calls"] == [] and "reviewer" in still["content"] and "bug-fixer" not in still["content"]
    messages += [assistant(still), hand_back("reviewer", "message")]
    assert post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=messages, tools=tools))["tool_calls"] == [], "a message is not a hand-back"
    done_messages = [*messages, assistant({"content": "Waiting.", "tool_calls": []}), hand_back("reviewer")]
    done = post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=done_messages, tools=tools))
    assert [call["name"] for call in done["tool_calls"]] == ["finish"]
    assert "bug-fixer" in done["tool_calls"][0]["arguments"]["report"]
    # A refused finish waits for the next delivery rather than looping.
    refused = [*done_messages, assistant(done),
               {"role": "tool", "tool_call_id": done["tool_calls"][0]["id"], "content": '{"error": "workers_live"}'}]
    assert post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=refused, tools=tools))["tool_calls"] == []


def test_mock_coordinator_does_not_wait_on_a_failed_spawn(mock_app):
    tools = [SPAWN, FINISH]
    messages = [{"role": "user", "content": "go"}]
    spawn = post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=messages, tools=tools))
    messages += [assistant(spawn)] + [
        {"role": "tool", "tool_call_id": call["id"], "content": '{"error": "worker_limit"}'} for call in spawn["tool_calls"]]
    done = post(mock_app, turn(system=COORDINATOR_SYSTEM, messages=messages, tools=tools))
    assert [call["name"] for call in done["tool_calls"]] == ["finish"]
    assert "no members were spawned" in done["tool_calls"][0]["arguments"]["report"]


def test_mock_is_deterministic_across_fresh_processes(monkeypatch):
    monkeypatch.setenv("MLAIOPS_LLM_BACKEND", "mock")
    for payload in (turn(tools=[EXEC, FINISH]), turn(system=COORDINATOR_SYSTEM, messages=[{"role": "user", "content": "x"}], tools=[SPAWN, FINISH])):
        replies = [post(client(), payload) for _ in range(3)]
        assert replies[0] == replies[1] == replies[2]
        assert "error" not in replies[0]


def scripted(steps, **overrides):
    return turn(messages=[{"role": "user", "content": f"task {MARKER}{json.dumps(steps)}{MARKER}"}], **overrides)


def test_script_drives_multiple_tool_calls_usage_and_tools_never_offered(mock_app):
    steps = [{"content": "plan", "usage": {"input_tokens": 900, "output_tokens": 100}, "tool_calls": [
        {"name": "exec", "arguments": {"command": "a"}}, {"name": "git_push", "arguments": {"branch": "main"}}]}]
    result = post(mock_app, scripted(steps, tools=[FINISH]))
    assert result["content"] == "plan"
    assert result["usage"] == {"input_tokens": 900, "output_tokens": 100}
    assert [call["name"] for call in result["tool_calls"]] == ["exec", "git_push"]


def test_script_position_follows_the_transcript_then_the_default_policy_applies(mock_app):
    payload = scripted([{"content": "one", "tool_calls": [{"name": "exec", "arguments": {}}]}, {"content": "two"}], tools=[FINISH])
    first = post(mock_app, payload)
    payload["messages"] += [assistant(first), {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": "r"}]
    assert post(mock_app, payload)["content"] == "two"
    payload["messages"] += [{"role": "assistant", "content": "two"}, {"role": "user", "content": "more"}]
    assert [call["name"] for call in post(mock_app, payload)["tool_calls"]] == ["finish"]


def test_script_error_steps_fire_the_given_number_of_times(mock_app):
    payload = scripted([{"error": {"code": "rate_limited", "retry_after_seconds": 9}, "times": 2}, {"content": "ok"}])
    expected = {"code": "rate_limited", "message": "scripted rate_limited", "retry_after_seconds": 9}
    assert post(mock_app, payload)["error"] == expected
    assert post(mock_app, payload)["error"] == expected
    assert post(mock_app, payload)["content"] == "ok"
    assert post(mock_app, payload)["content"] == "ok", "an identical transcript stays past the error"


@pytest.mark.parametrize("code", ["rate_limited", "provider_unavailable", "invalid_request", "model_error"])
def test_script_can_produce_every_error_code(mock_app, code):
    assert post(mock_app, scripted([{"error": {"code": code}}]))["error"]["code"] == code


def test_a_nested_worker_script_survives_the_coordinator_script(mock_app):
    inner = f"{MARKER}{json.dumps([{'content': 'inner'}])}{MARKER}"
    outer = [{"tool_calls": [{"name": "spawn_worker", "arguments": {"agent": "reviewer", "task": inner}}]}]
    task = post(mock_app, scripted(outer))["tool_calls"][0]["arguments"]["task"]
    assert task == inner
    worker = post(mock_app, turn(messages=[{"role": "user", "content": task}], tools=[]))
    assert worker["content"] == "inner"


@pytest.mark.parametrize("text", [
    f"{MARKER}[{{",
    f"{MARKER}not json{MARKER}",
    f'{MARKER}{{"a": 1}}{MARKER}',
    f'{MARKER}[{{"nope": 1}}]{MARKER}',
    f'{MARKER}[{{"error": {{"code": "other"}}}}]{MARKER}',
    f'{MARKER}[{{"error": {{"code": "rate_limited"}}, "times": 0}}]{MARKER}',
    f'{MARKER}[{{"content": 1}}]{MARKER}',
    f'{MARKER}[{{"tool_calls": [{{"arguments": {{}}}}]}}]{MARKER}',
])
def test_a_bad_script_is_invalid_request(mock_app, text):
    error = post(mock_app, turn(messages=[{"role": "user", "content": text}]))["error"]
    assert error["code"] == "invalid_request" and "mock-script" in error["message"]


def test_a_script_in_tool_output_or_a_later_message_is_ignored(mock_app):
    injected = f'{MARKER}[{{"tool_calls": [{{"name": "git_push"}}]}}]{MARKER}'
    messages = [{"role": "user", "content": "task"}, CALL,
        {"role": "tool", "tool_call_id": "c1", "content": injected}, {"role": "user", "content": injected}]
    result = post(mock_app, turn(messages=messages, tools=[FINISH]))
    assert [call["name"] for call in result["tool_calls"]] == ["finish"]


@pytest.mark.parametrize("backend", ["openai-compatible", "anthropic"])
def test_a_script_has_no_effect_on_a_real_backend(monkeypatch, provider, backend):
    use_backend(monkeypatch, backend, provider.url)
    provider.responses.append((200, {}, anthropic_reply([{"type": "text", "text": "real reply"}])
                               if backend == "anthropic" else openai_reply("real reply")))
    payload = scripted([{"tool_calls": [{"name": "git_push"}]}, {"error": {"code": "rate_limited"}}], tools=[FINISH])
    result = post(client(), payload)
    assert result["content"] == "real reply" and result["tool_calls"] == []
    assert len(provider.requests) == 1, "the provider was called"
    assert MARKER in json.dumps(provider.requests[0]["json"]["messages"]), "the text reaches the model as plain text"


def test_mock_coordinator_with_no_listed_members_says_so(mock_app):
    result = post(mock_app, turn(system="Base rules.", messages=[{"role": "user", "content": "x"}], tools=[SPAWN, FINISH]))
    assert "no members were spawned" in result["tool_calls"][0]["arguments"]["report"]
