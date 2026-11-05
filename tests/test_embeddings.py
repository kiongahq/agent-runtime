"""``POST /embeddings``: the same vector a designer knowledge node searches with."""

import asyncio
import json
import logging
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from agent_runtime import agent_turns as module
from agent_runtime.designer import create_app
from agent_runtime.native_graph import normalize, run_graph, validate_graph
from agent_runtime.native_graph.resources import Resources

CONTRACT = Path(__file__).parent / "fixtures" / "agent_team"
RESPONSE_SCHEMA = json.loads((CONTRACT / "runner-embedding-response.v1.schema.json").read_text())
TOKEN = "runner-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CANARY = "CANARY-do-not-log-8821"


@pytest.fixture
def app():
    return TestClient(create_app(token=TOKEN))


class Capturing(Resources):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.vector = None

    async def knowledge_dimensions(self, knowledge_base_id):
        return self.dimensions

    async def search_knowledge(self, knowledge_base_id, vector, limit):
        self.vector = vector
        return [{"text": "found", "score": 1.0}]


def designer_vector(text, dimensions):
    """Run a real knowledge node and capture the vector it searched with."""
    document = normalize({"name": "t", "nodes": [
        {"id": "input", "kind": "input", "name": "input", "config": {}},
        {"id": "kb", "kind": "knowledge", "name": "kb", "config": {"knowledge_base": "kb-1", "limit": "2"}},
        {"id": "output", "kind": "output", "name": "output", "config": {}},
    ], "edges": [{"from": "input", "to": "kb"}, {"from": "kb", "to": "output"}]})
    assert validate_graph(document) == []
    resources = Capturing(dimensions)

    async def run():
        return [event async for event in run_graph(document, input_text=text, run_id="agr-" + "a" * 24, resources=resources)]

    assert asyncio.run(run())[-1]["type"] == "run.completed"
    return resources.vector


@pytest.mark.parametrize("text,dimensions", [
    ("How do I rotate an API key?", 384),
    ("Refunds, RETURNS and refunds", 16),
    ("one", 1),
    ("unicode café 中文 text", 4096),
])
def test_the_vector_equals_the_designer_knowledge_path(app, text, dimensions):
    response = app.post("/embeddings", json={"text": text, "dimensions": dimensions}, headers=AUTH)
    assert response.status_code == 200
    jsonschema.validate(response.json(), RESPONSE_SCHEMA)
    vector = response.json()["vector"]
    assert len(vector) == dimensions
    assert vector == designer_vector(text, dimensions), "exact, including after the JSON round trip"


def test_text_with_no_tokens_is_a_zero_vector(app):
    assert app.post("/embeddings", json={"text": "!!!", "dimensions": 4}, headers=AUTH).json() == {"vector": [0.0] * 4}


def test_the_runner_credential_is_checked_before_the_body(app):
    assert app.post("/embeddings", content=b"not json").status_code == 401
    wrong = {"Authorization": "Bearer wrong"}
    assert app.post("/embeddings", json={"text": "x", "dimensions": 4}, headers=wrong).status_code == 401


@pytest.mark.parametrize("body", [
    {"text": CANARY, "dimensions": 0},
    {"text": CANARY, "dimensions": 4097},
    {"text": CANARY, "dimensions": True},
    {"text": CANARY, "dimensions": "8"},
    {"text": CANARY, "dimensions": 1.5},
    {"text": CANARY},
    {"dimensions": 4},
    {"text": "", "dimensions": 4},
    {"text": "x" * 32001, "dimensions": 4},
    {"text": CANARY, "dimensions": 4, "model": "other"},
    [CANARY],
])
def test_bad_requests_are_422_and_echo_no_text(app, body):
    response = app.post("/embeddings", json=body, headers=AUTH)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert CANARY not in response.text


def test_bounds_are_inclusive(app):
    assert len(app.post("/embeddings", json={"text": "x" * 32000, "dimensions": 4096}, headers=AUTH).json()["vector"]) == 4096
    assert len(app.post("/embeddings", json={"text": "x", "dimensions": 1}, headers=AUTH).json()["vector"]) == 1


def test_non_json_and_oversized_bodies_are_422(app):
    assert app.post("/embeddings", content=b"{nope", headers=AUTH).status_code == 422
    huge = app.post("/embeddings", content=b" " * (module.MAX_EMBEDDING_BODY_BYTES + 1), headers=AUTH)
    assert huge.status_code == 422 and "exceeds" in huge.json()["error"]["message"]


def test_text_is_not_logged(app, caplog):
    with caplog.at_level(logging.DEBUG):
        app.post("/embeddings", json={"text": CANARY, "dimensions": 8}, headers=AUTH)
        app.post("/embeddings", json={"text": CANARY, "dimensions": 0}, headers=AUTH)
    assert CANARY not in caplog.text


def test_contract_examples():
    valid = json.loads((CONTRACT / "valid" / "runner-embedding-request--query.json").read_text())
    app = TestClient(create_app(token=TOKEN))
    assert app.post("/embeddings", json=valid, headers=AUTH).status_code == 200
    examples = sorted((CONTRACT / "invalid").glob("runner-embedding-request--*.json"))
    assert len(examples) >= 5
    for path in examples:
        document = json.loads(path.read_text())["document"]
        assert app.post("/embeddings", json=document, headers=AUTH).status_code == 422, path.name


def test_the_docs_warn_about_langsmith():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "LANGSMITH_TRACING" in readme and "external service" in readme
