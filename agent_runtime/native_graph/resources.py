"""Resource access for graph nodes, always through the Kionga gateway.

Graphs hold Kionga resource IDs only. The gateway resolves endpoints and
credentials, and re-checks the caller's project access on every call, so a
revoked grant stops the next node that needs it. The playground runner sends a
run-scoped credential; a deployed agent uses its loopback ``kionga-api`` helper.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlencode

import httpx

MAX_RESPONSE_BYTES = 1024 * 1024


class ResourceError(RuntimeError):
    """A gateway call failed. ``status`` is the HTTP status, or 0 when unreachable."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Resources:
    """Interface the engine calls. Tests substitute fakes."""

    async def predict(self, model_id: str, text: str) -> Any:
        raise ResourceError(503, "model inference is unavailable")

    async def invoke_function(self, name: str, payload: dict[str, Any]) -> Any:
        raise ResourceError(503, "tools are unavailable")

    async def invoke_connection(self, connection_id: str, payload: Any) -> Any:
        raise ResourceError(503, "API connections are unavailable")

    async def knowledge_dimensions(self, knowledge_base_id: str) -> int:
        raise ResourceError(503, "knowledge bases are unavailable")

    async def search_knowledge(self, knowledge_base_id: str, vector: list[float], limit: int) -> list[dict[str, Any]]:
        raise ResourceError(503, "knowledge bases are unavailable")

    async def dataset_object(self, dataset_id: str, key: str) -> dict[str, Any]:
        raise ResourceError(503, "datasets are unavailable")


def _segment(value: str) -> str:
    return quote(str(value), safe="")


class GatewayResources(Resources):
    def __init__(self, base_url: str, headers: dict[str, str] | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Accept": "application/json", **(headers or {})}
        self.timeout = timeout

    async def _request(self, method: str, path: str, payload: Any = None) -> Any:
        if not path.startswith("/") or path.startswith("//"):
            raise ResourceError(400, "invalid resource path")
        kwargs: dict[str, Any] = {"headers": self.headers}
        if payload is not None:
            kwargs["content"] = json.dumps(payload).encode()
            kwargs["headers"] = {**self.headers, "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                async with client.stream(method, self.base_url + path, **kwargs) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ResourceError(502, "resource response exceeds 1 MiB")
                    status = response.status_code
        except httpx.HTTPError:
            raise ResourceError(0, "Kionga resource service is unavailable") from None
        if status < 200 or status >= 300:
            raise ResourceError(status, f"Kionga rejected this resource request ({status})")
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError:
            raise ResourceError(502, "Kionga returned invalid JSON") from None

    async def predict(self, model_id: str, text: str) -> Any:
        return await self._request("POST", f"/models/{_segment(model_id)}/predict", {"inputs": [text]})

    async def invoke_function(self, name: str, payload: dict[str, Any]) -> Any:
        return await self._request("POST", f"/functions/{_segment(name)}/invoke", payload)

    async def invoke_connection(self, connection_id: str, payload: Any) -> Any:
        return await self._request("POST", f"/agent-api-connections/{_segment(connection_id)}/invoke", payload)

    async def knowledge_dimensions(self, knowledge_base_id: str) -> int:
        listing = await self._request("GET", "/knowledge-bases")
        for item in listing.get("items", []) if isinstance(listing, dict) else []:
            if isinstance(item, dict) and item.get("id") == knowledge_base_id:
                dimensions = item.get("dimensions")
                if isinstance(dimensions, int) and 1 <= dimensions <= 65536:
                    return dimensions
        raise ResourceError(404, "knowledge base not found")

    async def search_knowledge(self, knowledge_base_id: str, vector: list[float], limit: int) -> list[dict[str, Any]]:
        result = await self._request(
            "POST", f"/knowledge-bases/{_segment(knowledge_base_id)}/search", {"vector": vector, "limit": limit}
        )
        items = result.get("items", []) if isinstance(result, dict) else []
        return [
            {"text": str(item.get("text", "")), "score": item.get("score"), "metadata": item.get("metadata") or {}}
            for item in items if isinstance(item, dict)
        ]

    async def dataset_object(self, dataset_id: str, key: str) -> dict[str, Any]:
        query = urlencode({"key": key})
        result = await self._request("GET", f"/datasets/{_segment(dataset_id)}/object?{query}")
        return result if isinstance(result, dict) else {}
