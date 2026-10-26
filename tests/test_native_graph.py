"""Native graph contract, migration, conditions and engine semantics."""

import asyncio
import json
from pathlib import Path

import jsonschema
import pytest
from mlaiops_sdk.llm import MockChatModel

from agent_runtime.native_graph import canonical_json, content_sha256, normalize, run_graph, validate_graph
from agent_runtime.native_graph import conditions
from agent_runtime.native_graph.engine import redact
from agent_runtime.native_graph.registry import REGISTRY_PATH
from agent_runtime.native_graph.resources import ResourceError, Resources

FIXTURES = Path(__file__).parent / "fixtures" / "native_graph"
CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "agent-graph"
EVENT_SCHEMA = json.loads((FIXTURES / "run-event.v1.schema.json").read_text())
RUN_ID = "agr-" + "0" * 24


class FakeResources(Resources):
    def __init__(self, **overrides):
        self.calls = []
        self.overrides = overrides

    async def _result(self, name, default, *args):
        self.calls.append((name, *args))
        value = self.overrides.get(name, default)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return await value(*args)
        return value

    async def predict(self, model_id, text):
        return await self._result("predict", {"predictions": [1]}, model_id, text)

    async def invoke_function(self, name, payload):
        return await self._result("invoke_function", {"ok": True, "echo": payload}, name, payload)

    async def invoke_connection(self, connection_id, payload):
        return await self._result("invoke_connection", {"status": "shipped"}, connection_id, payload)

    async def knowledge_dimensions(self, knowledge_base_id):
        return await self._result("knowledge_dimensions", 16, knowledge_base_id)

    async def search_knowledge(self, knowledge_base_id, vector, limit):
        return await self._result("search_knowledge", [{"text": "Refunds take 5 days.", "score": 0.9}],
                                  knowledge_base_id, len(vector), limit)

    async def dataset_object(self, dataset_id, key):
        return await self._result("dataset_object", {"content": "Plan: Gold", "truncated": False}, dataset_id, key)


def graph(nodes, edges, **settings):
    document = {"name": "test", "nodes": [{"id": n[0], "kind": n[1], "name": n[0], "config": n[2] if len(n) > 2 else {}}
                                          for n in nodes],
                "edges": [{"from": e[0], "to": e[1], **({"from_port": e[2]} if len(e) > 2 else {})} for e in edges]}
    if settings:
        document["settings"] = settings
    normalized = normalize(document)
    assert validate_graph(normalized) == []
    return normalized


async def collect(document, text="hello", **kwargs):
    kwargs.setdefault("llm_factory", lambda: MockChatModel(["mock reply"]))
    events = [event async for event in run_graph(document, input_text=text, run_id=RUN_ID, **kwargs)]
    validator = jsonschema.Draft202012Validator(EVENT_SCHEMA)
    for index, event in enumerate(events):
        assert list(validator.iter_errors(event)) == [], event
        assert event["seq"] == index
    return events


def final(events):
    return events[-1]


def by_node(events, node_id):
    return [event for event in events if event.get("node_id") == node_id]


LINEAR = [("input", "input"), ("prompt", "prompt", {"template": "Say hi to {input}"}), ("model", "model"), ("output", "output")]
LINEAR_EDGES = [("input", "prompt"), ("prompt", "model"), ("model", "output")]


# Contract copies --------------------------------------------------------------

@pytest.mark.skipif(not CONTRACTS.is_dir(), reason="sibling contracts checkout not present")
def test_registry_and_fixtures_match_contracts_byte_for_byte():
    assert REGISTRY_PATH.read_bytes() == (CONTRACTS / "components.v1.json").read_bytes()
    for name in ("hash-fixture.json", "v1-draft.json"):
        assert (FIXTURES / name).read_bytes() == (CONTRACTS / "examples" / name).read_bytes()
    for name in ("run-event.v1.schema.json", "artifact.v1.schema.json"):
        assert (FIXTURES / name).read_bytes() == (CONTRACTS / name).read_bytes()
    for template in (CONTRACTS / "templates").glob("*.json"):
        assert (FIXTURES / "templates" / template.name).read_bytes() == template.read_bytes()


def test_canonical_hash_matches_cross_language_fixture():
    fixture = json.loads((FIXTURES / "hash-fixture.json").read_text())
    assert canonical_json(fixture["graph"]) == fixture["canonical"]
    assert content_sha256(fixture["graph"]) == fixture["expected_sha256"]


# Migration --------------------------------------------------------------------

def test_version_one_draft_normalizes_to_version_two_without_loss():
    legacy = json.loads((FIXTURES / "v1-draft.json").read_text())
    upgraded = normalize(legacy)
    assert "schema_version" not in legacy and upgraded["schema_version"] == 2
    assert upgraded["settings"] == {"max_steps": 50, "timeout_seconds": 60, "max_tokens": 50000}
    ports = {(e["from"], e["to"]): (e["from_port"], e["to_port"]) for e in upgraded["edges"]}
    assert ports[("condition-1", "prompt-1")] == ("true", "in")
    assert ports[("input", "memory-1")] == ("out", "in")
    model = next(n for n in upgraded["nodes"] if n["id"] == "model-1")
    assert model["config"]["model"] == "llm"
    assert [n["id"] for n in upgraded["nodes"]] == [n["id"] for n in legacy["nodes"]]
    assert normalize(upgraded) == upgraded, "normalize must be idempotent"
    assert validate_graph(upgraded) == []


@pytest.mark.asyncio
async def test_version_one_free_text_condition_fails_as_invalid_graph():
    events = await collect(normalize(json.loads((FIXTURES / "v1-draft.json").read_text())))
    assert final(events)["type"] == "run.failed"
    assert final(events)["error"]["code"] == "invalid_graph"
    assert by_node(events, "condition-1")[-1]["type"] == "node.failed"


def test_validation_rejects_cycles_unknown_kinds_ports_and_types():
    base = {"name": "x", "nodes": [{"id": "input", "kind": "input", "name": "i"}, {"id": "a", "kind": "prompt", "name": "a"},
                                   {"id": "b", "kind": "prompt", "name": "b"}, {"id": "output", "kind": "output", "name": "o"}]}
    cyclic = normalize({**base, "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}]})
    assert "Connections contain a cycle." in validate_graph(cyclic)
    unknown = normalize({**base, "nodes": base["nodes"] + [{"id": "z", "kind": "shell", "name": "z"}], "edges": []})
    assert any("unsupported type" in issue for issue in validate_graph(unknown))
    bad_port = normalize({**base, "edges": [{"from": "a", "to": "b", "from_port": "nope"}]})
    assert any("port that does not exist" in issue for issue in validate_graph(bad_port))
    # Every version 1 port type converts to text, so type mismatches only arise
    # between non-text types; the rule itself is checked directly.
    from agent_runtime.native_graph import registry

    assert registry.compatible("documents", ["text"]) and registry.compatible("text", ["json"])
    assert not registry.compatible("documents", ["json"])
    assert not registry.compatible("chat", ["documents"])
    assert validate_graph(normalize({**base, "edges": [], "settings": {"max_steps": 999}}))


# Conditions -------------------------------------------------------------------

@pytest.mark.parametrize("expression, value, expected", [
    ("contains:Refund", "I want a refund", True),
    ("contains:refund", "hello", False),
    ("not_contains:refund", "hello", True),
    ("equals:yes", "  YES ", True),
    ("equals:yes", "yes please", False),
    ("matches:^order-[0-9]+$", "order-42", True),
    ("matches:^order-[0-9]+$", "order-x", False),
    ("empty", "   ", True),
    ("not_empty", "x", True),
    ("longer_than:3", "abcd", True),
    ("longer_than:3", "abc", False),
])
def test_condition_operators(expression, value, expected):
    assert conditions.parse(expression)(value) is expected


@pytest.mark.parametrize("expression", [
    "", "route urgent tickets", "contains", "contains:", "empty:x", "longer_than:abc", "longer_than:-1",
    "matches:(", "matches:" + "a" * 201, "eval:__import__('os')",
])
def test_condition_parse_errors(expression):
    with pytest.raises(conditions.ConditionError):
        conditions.parse(expression)


# Engine -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_linear_run_streams_node_events_usage_and_output():
    events = await collect(graph(LINEAR, LINEAR_EDGES), text="Ada")
    assert [e["type"] for e in events][:2] == ["run.started", "node.started"]
    assert by_node(events, "prompt")[-1]["output_preview"] == "Say hi to Ada"
    model_done = by_node(events, "model")[-1]
    assert model_done["type"] == "node.completed" and model_done["usage"]["output_tokens"] == 2
    assert final(events)["type"] == "run.completed"
    assert final(events)["output"] == "mock reply"
    assert final(events)["usage"]["input_tokens"] == 4


@pytest.mark.asyncio
async def test_condition_runs_only_selected_branch_and_skips_the_other():
    document = graph(
        [("input", "input"), ("route", "condition", {"expression": "contains:refund"}),
         ("yes", "prompt", {"template": "REFUND {input}"}), ("no", "prompt", {"template": "OTHER {input}"}),
         ("output", "output")],
        [("input", "route"), ("route", "yes", "true"), ("route", "no", "false"), ("yes", "output"), ("no", "output")],
    )
    events = await collect(document, text="refund please")
    assert by_node(events, "route")[-1]["port"] == "true"
    assert by_node(events, "no") == [{**by_node(events, "no")[0], "type": "node.skipped"}]
    assert final(events)["output"] == "REFUND refund please"
    events = await collect(document, text="hello")
    assert by_node(events, "yes")[0]["type"] == "node.skipped"
    assert final(events)["output"] == "OTHER hello"


@pytest.mark.asyncio
async def test_multiple_inbound_values_feed_context_and_node_placeholders():
    document = graph(
        [("input", "input"), ("a", "prompt", {"template": "A:{input}"}), ("b", "prompt", {"template": "B:{input}"}),
         ("join", "prompt", {"template": "[{context}] {a} {missing} {input}"}), ("output", "output")],
        [("input", "a"), ("input", "b"), ("a", "join"), ("b", "join"), ("join", "output")],
    )
    events = await collect(document, text="x")
    assert final(events)["output"] == "[A:x\n\nB:x] A:x {missing} x"


@pytest.mark.asyncio
async def test_placeholders_in_values_are_not_reexpanded():
    document = graph([("input", "input"), ("p", "prompt", {"template": "{input}|{p}"}), ("output", "output")],
                     [("input", "p"), ("p", "output")])
    events = await collect(document, text="{context}{input}")
    assert final(events)["output"] == "{context}{input}|{p}"


@pytest.mark.asyncio
async def test_documents_and_json_convert_to_text():
    resources = FakeResources()
    document = graph(
        [("input", "input"), ("kb", "knowledge", {"knowledge_base": "kb-1", "limit": "2"}),
         ("tool", "tool", {"tool": "lookup"}), ("p", "prompt", {"template": "{kb} / {tool}"}), ("output", "output")],
        [("input", "kb"), ("input", "tool"), ("kb", "p"), ("tool", "p"), ("p", "output")],
    )
    events = await collect(document, text="refund", resources=resources, allow_side_effects=True)
    assert final(events)["output"] == 'Refunds take 5 days. / {"ok":true,"echo":{"input":"refund"}}'
    assert ("search_knowledge", "kb-1", 16, 2) in resources.calls


@pytest.mark.asyncio
async def test_side_effect_nodes_make_no_call_without_approval():
    resources = FakeResources()
    document = graph([("input", "input"), ("tool", "tool", {"tool": "refund"}), ("output", "output")],
                     [("input", "tool"), ("tool", "output")])
    events = await collect(document, resources=resources)
    assert resources.calls == []
    assert final(events)["error"]["code"] == "side_effects_not_approved"
    api = graph([("input", "input"), ("api", "api", {"connection": "api-1"}), ("output", "output")],
                [("input", "api"), ("api", "output")])
    events = await collect(api, resources=resources)
    assert resources.calls == [] and final(events)["error"]["code"] == "side_effects_not_approved"


@pytest.mark.asyncio
async def test_api_request_template_renders_json_or_wraps_text():
    resources = FakeResources()
    for template, expected in [('{"q": "{input}"}', {"q": "hi"}), ("look up {input}", {"input": "look up hi"}),
                               ("", {"input": "hi"})]:
        resources.calls.clear()
        document = graph([("input", "input"), ("api", "api", {"connection": "api-1", "request": template}),
                          ("output", "output")], [("input", "api"), ("api", "output")])
        events = await collect(document, text="hi", resources=resources, allow_side_effects=True)
        assert final(events)["type"] == "run.completed"
        assert resources.calls == [("invoke_connection", "api-1", expected)]


@pytest.mark.parametrize("status, code", [(403, "resource_denied"), (404, "resource_denied"),
                                          (503, "resource_unavailable"), (0, "resource_unavailable")])
@pytest.mark.asyncio
async def test_resource_errors_map_to_contract_codes(status, code):
    resources = FakeResources(predict=ResourceError(status, "Kionga rejected this resource request"))
    document = graph([("input", "input"), ("m", "model", {"model": "mdl-123"}), ("output", "output")],
                     [("input", "m"), ("m", "output")])
    events = await collect(document, resources=resources)
    assert final(events)["error"]["code"] == code
    retries = [e for e in events if e["type"] == "node.retrying"]
    assert len(retries) == (1 if code == "resource_unavailable" else 0)


@pytest.mark.asyncio
async def test_retry_then_success_and_per_node_timeout(monkeypatch):
    attempts = []

    async def flaky(model_id, text):
        attempts.append(1)
        if len(attempts) == 1:
            raise ResourceError(503, "busy")
        return "ok"

    document = graph([("input", "input"), ("m", "model", {"model": "mdl-1"}), ("output", "output")],
                     [("input", "m"), ("m", "output")])
    events = await collect(document, resources=FakeResources(predict=flaky))
    assert [e["type"] for e in by_node(events, "m")] == ["node.started", "node.retrying", "node.completed"]
    assert final(events)["output"] == "ok"

    async def hang(model_id, text):
        await asyncio.sleep(10)

    from agent_runtime.native_graph import registry

    model = registry.component("model")
    monkeypatch.setitem(model, "timeout_seconds", 0.05)
    events = await collect(document, resources=FakeResources(predict=hang))
    assert final(events)["error"]["code"] == "timeout"
    assert sum(e["type"] == "node.retrying" for e in events) == 1


@pytest.mark.asyncio
async def test_run_timeout_bounds_the_whole_run():
    async def hang(model_id, text):
        await asyncio.sleep(10)

    document = graph([("input", "input"), ("m", "model", {"model": "mdl-1"}), ("output", "output")],
                     [("input", "m"), ("m", "output")], timeout_seconds=1)
    events = await collect(document, resources=FakeResources(predict=hang))
    assert final(events)["error"]["code"] == "timeout"


@pytest.mark.asyncio
async def test_step_and_token_budgets():
    events = await collect(graph(LINEAR, LINEAR_EDGES, max_steps=2))
    assert final(events)["error"]["code"] == "budget_exceeded"
    events = await collect(graph(LINEAR, LINEAR_EDGES, max_tokens=3))
    assert by_node(events, "model")[-1]["type"] == "node.failed"
    assert final(events)["error"]["code"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_cancellation_stops_a_running_node():
    cancel = asyncio.Event()

    async def hang(model_id, text):
        cancel.set()
        await asyncio.sleep(10)

    document = graph([("input", "input"), ("m", "model", {"model": "mdl-1"}), ("output", "output")],
                     [("input", "m"), ("m", "output")])
    events = await collect(document, resources=FakeResources(predict=hang), cancel=cancel)
    assert final(events)["type"] == "run.cancelled"
    assert final(events)["error"]["code"] == "cancelled"


@pytest.mark.asyncio
async def test_stop_after_runs_only_ancestors_and_returns_that_node():
    resources = FakeResources()
    document = graph([("input", "input"), ("p", "prompt", {"template": "P:{input}"}), ("m", "model", {"model": "mdl-9"}),
                      ("output", "output")], [("input", "p"), ("p", "m"), ("m", "output")])
    events = await collect(document, text="x", stop_after="p", resources=resources)
    assert final(events)["type"] == "run.completed" and final(events)["output"] == "P:x"
    assert resources.calls == [] and by_node(events, "m") == []


@pytest.mark.asyncio
async def test_memory_window_and_chat_history_reach_the_model():
    seen = []

    class Recorder:
        async def ainvoke(self, messages):
            seen.append([(type(m).__name__, m.content) for m in messages])
            from langchain_core.messages import AIMessage

            return AIMessage(content="done", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    history = [{"role": "user", "text": f"u{i}"} if i % 2 == 0 else {"role": "assistant", "text": f"a{i}"} for i in range(6)]
    document = graph([("input", "input"), ("mem", "memory", {"turns": "2"}),
                      ("m", "model", {"system": "Be brief about {input}"}), ("output", "output")],
                     [("input", "mem"), ("mem", "m"), ("m", "output")])
    events = await collect(document, text="now", history=history, llm_factory=Recorder)
    assert final(events)["output"] == "done"
    assert seen == [[("SystemMessage", "Be brief about now"), ("HumanMessage", "u4"), ("AIMessage", "a5"),
                     ("HumanMessage", "now")]]


@pytest.mark.asyncio
async def test_source_nodes_without_inbound_edges_always_run():
    template = json.loads((FIXTURES / "templates" / "dataset-grounded.json").read_text())
    document = normalize(template["graph"])
    next(n for n in document["nodes"] if n["id"] == "load")["config"].update(dataset="ds-1", key="plans.txt")
    resources = FakeResources()
    events = await collect(document, text="Which plan?", resources=resources,
                           llm_factory=lambda: MockChatModel(["Gold"]))
    assert ("dataset_object", "ds-1", "plans.txt") in resources.calls
    assert by_node(events, "load")[-1]["type"] == "node.completed"
    assert "Plan: Gold" in by_node(events, "prompt")[-1]["output_preview"]
    assert final(events)["output"] == "Gold"


@pytest.mark.asyncio
async def test_truncated_dataset_objects_fail_instead_of_answering_from_a_fragment():
    document = graph([("input", "input"), ("load", "storage", {"dataset": "ds", "key": "big"}),
                      ("output", "output")], [("load", "output")])
    events = await collect(document, resources=FakeResources(dataset_object={"content": "x", "truncated": True}))
    assert final(events)["type"] == "run.failed"


def test_redaction_removes_common_secret_shapes_and_truncates():
    text = ('Authorization: Bearer abcdefghijklmnop sk-live1234567890 AKIAABCDEFGHIJKLMNOP '
            'password=hunter2 {"api_key": "zzz-secret"}')
    clean = redact(text)
    for secret in ("abcdefghijklmnop", "sk-live1234567890", "AKIAABCDEFGHIJKLMNOP", "hunter2", "zzz-secret"):
        assert secret not in clean
    assert len(redact("x" * 5000)) == 2000


@pytest.mark.asyncio
async def test_previews_are_redacted_in_events():
    document = graph([("input", "input"), ("p", "prompt", {"template": "token={input}"}), ("output", "output")],
                     [("input", "p"), ("p", "output")])
    events = await collect(document, text="supersecretvalue")
    assert by_node(events, "p")[-1]["output_preview"] == "token=[REDACTED]"
    assert "supersecretvalue" not in json.dumps(by_node(events, "output"))


TEMPLATES = sorted((FIXTURES / "templates").glob("*.json"))


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.stem)
@pytest.mark.asyncio
async def test_every_starter_template_runs_once_resources_are_chosen(path):
    document = normalize(json.loads(path.read_text())["graph"])
    assert validate_graph(document) == []
    picks = {"knowledge": {"knowledge_base": "kb-1"}, "tool": {"tool": "lookup"},
             "api": {"connection": "api-1"}, "storage": {"dataset": "ds-1", "key": "doc.txt"}}
    for node in document["nodes"]:
        node["config"].update(picks.get(node["kind"], {}))
    events = await collect(document, text="I need a refund", resources=FakeResources(), allow_side_effects=True)
    assert final(events)["type"] == "run.completed", final(events)
    assert final(events)["output"]
