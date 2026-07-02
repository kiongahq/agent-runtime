"""Gate tests for feature definitions and the materializer. No network."""

import httpx
import pytest

from features.definitions import FEATURE_VIEWS, source_rows
from features.materialize import Materializer


def test_definitions_are_well_formed():
    names = set()
    for view in FEATURE_VIEWS:
        assert view["name"] not in names, "duplicate view name"
        names.add(view["name"])
        assert view["entity"]
        assert view["fields"], f"{view['name']} has no fields"
        field_names = {field["name"] for field in view["fields"]}
        for row in source_rows(view["name"]):
            assert view["entity"] in row
            assert field_names == set(row) - {view["entity"]}, (
                f"{view['name']} rows do not match declared fields"
            )


def test_source_rows_deterministic():
    assert source_rows("customer_profile") == source_rows("customer_profile")


def test_unknown_view_raises():
    with pytest.raises(KeyError):
        source_rows("nope")


def _fake_platform(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/presign":
            return httpx.Response(200, json={"url": "http://storage/upload-target"})
        if request.url.host == "storage":
            return httpx.Response(200)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


def test_materializer_full_flow():
    calls: list[tuple[str, str]] = []
    materializer = Materializer(
        gateway_url="http://gateway",
        feature_gateway_url="http://features",
        storage_proxy_url="http://proxy",
        client=httpx.Client(transport=_fake_platform(calls)),
    )
    counts = materializer.run()
    assert counts == {"customer_profile": 3, "transaction_stats_5m": 3}
    applies = [path for method, path in calls if path == "/api/v1/features" and method == "POST"]
    assert len(applies) == len(FEATURE_VIEWS)
    online_writes = [path for method, path in calls if method == "PUT" and "/internal/v1/features/" in path]
    assert len(online_writes) == 6
    assert ("PUT", "/internal/v1/features/customer_profile/entity_id=u123") in calls
    reports = [path for method, path in calls if path.endswith("/materialized")]
    assert len(reports) == len(FEATURE_VIEWS)
    snapshots = [path for method, path in calls if path == "/presign"]
    assert len(snapshots) == len(FEATURE_VIEWS)


def test_materializer_skips_offline_without_storage_proxy():
    calls: list[tuple[str, str]] = []
    materializer = Materializer(
        gateway_url="http://gateway",
        feature_gateway_url="http://features",
        storage_proxy_url="",
        client=httpx.Client(transport=_fake_platform(calls)),
    )
    materializer.run()
    assert all(path != "/presign" for _, path in calls)


def test_materializer_fails_closed_on_gateway_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    materializer = Materializer(
        gateway_url="http://gateway",
        feature_gateway_url="http://features",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        materializer.run()
