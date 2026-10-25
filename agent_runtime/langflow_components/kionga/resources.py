"""Kionga resource nodes for the Langflow visual canvas.

The adjacent loopback helper supplies a revocable, project-scoped identity.
Neither a Kionga API token nor a model/provider secret enters a flow definition.
"""

import json
import math
import urllib.error
import urllib.parse
import urllib.request

from lfx.custom import Component
from lfx.io import BoolInput, DropdownInput, MessageTextInput, Output, StrInput
from lfx.schema import Data
from lfx.schema.message import Message

BASE = "http://127.0.0.1:8890/api/v1"


def request(path: str, payload: object = None) -> object:
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Invalid Kionga API path")
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310 -- fixed loopback base and validated relative path
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Kionga rejected this resource request ({error.code})") from error
    except (OSError, ValueError) as error:
        raise ValueError("Kionga resource service is unavailable") from error


class KiongaModels(Component):
    display_name = "Kionga Models"
    description = "List models visible to this workspace's project. Use a model ID with Kionga Model Inference."
    name = "KiongaModels"
    icon = "Database"
    inputs = []
    outputs = [Output(name="models", display_name="Models", method="list_models")]

    def list_models(self) -> Data:
        items = request("/models").get("items", [])
        return Data(data={"models": [
            {key: item.get(key) for key in ("id", "name", "version", "status", "endpoint_url")}
            for item in items
        ]})


class KiongaModelInference(Component):
    display_name = "Kionga Model Inference"
    description = "Invoke an authorized, deployed Kionga model without exposing endpoint credentials."
    name = "KiongaModelInference"
    icon = "Bot"
    inputs = [
        BoolInput(name="refresh_models", display_name="Refresh Kionga models", value=False, real_time_refresh=True),
        DropdownInput(name="model_id", display_name="Deployed model ID", options=[], required=True, combobox=True),
        MessageTextInput(name="input_text", display_name="Input", required=True),
    ]
    outputs = [Output(name="prediction", display_name="Prediction", method="predict")]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "refresh_models":
            try:
                models = request("/models").get("items", [])
                build_config["model_id"]["options"] = [
                    item["id"] for item in models if item.get("endpoint_url")
                ]
            except ValueError:
                build_config["model_id"]["options"] = []
        return build_config

    def predict(self) -> Message:
        model_id = urllib.parse.quote(self.model_id, safe="")
        result = request(f"/models/{model_id}/predict", {"inputs": [self.input_text]})
        return Message(text=json.dumps(result, ensure_ascii=False))


class KiongaFunctions(Component):
    display_name = "Kionga Function"
    description = "Invoke an authorized Kionga function as an agent tool."
    name = "KiongaFunction"
    icon = "Wrench"
    inputs = [
        StrInput(name="function_name", display_name="Function name", required=True),
        MessageTextInput(name="input_text", display_name="Input", tool_mode=True),
    ]
    outputs = [Output(name="result", display_name="Result", method="invoke")]

    def invoke(self) -> Message:
        name = urllib.parse.quote(self.function_name, safe="")
        result = request(f"/functions/{name}/invoke", {"input": self.input_text})
        return Message(text=json.dumps(result, ensure_ascii=False))


class KiongaStorage(Component):
    display_name = "Kionga Storage"
    description = "List project-authorized object storage buckets."
    name = "KiongaStorage"
    icon = "Database"
    inputs = []
    outputs = [Output(name="buckets", display_name="Buckets", method="list_buckets")]

    def list_buckets(self) -> Data:
        return Data(data={"buckets": request("/storage/buckets")})


class KiongaObjects(Component):
    display_name = "Kionga Objects"
    description = "Browse dataset and artifact objects in an authorized project bucket."
    name = "KiongaObjects"
    icon = "Database"
    inputs = [
        StrInput(name="bucket", display_name="Bucket", required=True),
        StrInput(name="prefix", display_name="Prefix", value=""),
    ]
    outputs = [Output(name="objects", display_name="Objects", method="list_objects")]

    def list_objects(self) -> Data:
        query = urllib.parse.urlencode({"bucket": self.bucket, "prefix": self.prefix})
        return Data(data=request("/storage/objects?" + query))


class KiongaObjectText(Component):
    display_name = "Kionga Object Text"
    description = "Read a bounded text preview of an authorized dataset or artifact object."
    name = "KiongaObjectText"
    icon = "FileText"
    inputs = [
        StrInput(name="bucket", display_name="Bucket", required=True),
        StrInput(name="key", display_name="Object key", required=True),
    ]
    outputs = [Output(name="content", display_name="Content", method="read_object")]

    def read_object(self) -> Message:
        query = urllib.parse.urlencode({"bucket": self.bucket, "key": self.key})
        preview = request("/storage/object?" + query)
        if preview.get("truncated"):
            raise ValueError("Object exceeds Kionga's text preview limit")
        return Message(text=preview.get("content", ""))


class KiongaFeatureViews(Component):
    display_name = "Kionga Feature Views"
    description = "Discover authorized feature views and their metadata."
    name = "KiongaFeatureViews"
    icon = "Database"
    inputs = []
    outputs = [Output(name="features", display_name="Features", method="list_features")]

    def list_features(self) -> Data:
        return Data(data={"features": request("/features")})


class KiongaTools(Component):
    display_name = "Kionga Tools"
    description = "Discover registered tool schemas. Invoke an executable function with Kionga Function."
    name = "KiongaTools"
    icon = "Wrench"
    inputs = []
    outputs = [Output(name="tools", display_name="Tools", method="list_tools")]

    def list_tools(self) -> Data:
        items = request("/tools").get("items", [])
        return Data(data={"tools": [
            {key: item.get(key) for key in ("name", "version", "description", "input_schema", "status")}
            for item in items
        ]})


def embedding_vector(raw: str) -> list[float]:
    try:
        vector = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Provide an embedding as a JSON array of numbers") from error
    if not isinstance(vector, list) or not vector or len(vector) > 4096:
        raise ValueError("Provide an embedding as a JSON array of 1–4096 numbers")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in vector):
        raise ValueError("Every embedding value must be a number")
    return vector


class KiongaKnowledgeBases(Component):
    display_name = "Kionga Knowledge Bases"
    description = "Discover project-authorized Qdrant and pgvector knowledge bases without connection secrets."
    name = "KiongaKnowledgeBases"
    icon = "Database"
    inputs = []
    outputs = [Output(name="knowledge_bases", display_name="Knowledge Bases", method="list_bases")]

    def list_bases(self) -> Data:
        return Data(data={"knowledge_bases": request("/knowledge-bases").get("items", [])})


class KiongaKnowledgeSearch(Component):
    display_name = "Kionga Knowledge Search"
    description = "Search a Kionga knowledge base with a JSON embedding from the matching model."
    name = "KiongaKnowledgeSearch"
    icon = "Search"
    inputs = [
        BoolInput(name="refresh_bases", display_name="Refresh knowledge bases", value=False, real_time_refresh=True),
        DropdownInput(name="knowledge_base_id", display_name="Knowledge base", options=[], required=True, combobox=True),
        MessageTextInput(name="embedding_json", display_name="Embedding JSON", required=True),
        StrInput(name="limit", display_name="Maximum matches", value="5"),
    ]
    outputs = [Output(name="matches", display_name="Matches", method="search")]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "refresh_bases":
            try:
                build_config["knowledge_base_id"]["options"] = [
                    item["id"] for item in request("/knowledge-bases").get("items", [])
                ]
            except ValueError:
                build_config["knowledge_base_id"]["options"] = []
        return build_config

    def search(self) -> Data:
        base_id = urllib.parse.quote(self.knowledge_base_id, safe="")
        try:
            limit = int(self.limit)
        except (TypeError, ValueError) as error:
            raise ValueError("Maximum matches must be a number from 1 to 20") from error
        if not 1 <= limit <= 20:
            raise ValueError("Maximum matches must be a number from 1 to 20")
        result = request(
            f"/knowledge-bases/{base_id}/search",
            {"vector": embedding_vector(self.embedding_json), "limit": limit},
        )
        return Data(data={"matches": result.get("items", [])})


class KiongaKnowledgeChunk(Component):
    display_name = "Kionga Knowledge Chunk"
    description = "Add text and a matching JSON embedding to an authorized Qdrant or pgvector knowledge base."
    name = "KiongaKnowledgeChunk"
    icon = "FileText"
    inputs = [
        StrInput(name="knowledge_base_id", display_name="Knowledge base ID", required=True),
        MessageTextInput(name="text", display_name="Text", required=True),
        MessageTextInput(name="embedding_json", display_name="Embedding JSON", required=True),
    ]
    outputs = [Output(name="chunk", display_name="Chunk", method="add_chunk")]

    def add_chunk(self) -> Data:
        base_id = urllib.parse.quote(self.knowledge_base_id, safe="")
        result = request(
            f"/knowledge-bases/{base_id}/chunks",
            {"text": self.text, "vector": embedding_vector(self.embedding_json)},
        )
        return Data(data=result)
