"""Gate tests: point-in-time joins (no future leakage), TTL sweep, lineage."""

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from features.materialize import Materializer
from features.point_in_time import as_utc, point_in_time_join
from features.retention import sweep

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


# A leakage fixture: each user's plan changes over time, and the label rows
# are observed between changes. A correct join must never see a value
# written after the label timestamp.
FEATURES = [
    {"user_id": "u1", "event_timestamp": at(0), "plan": "free", "spend": 0},
    {"user_id": "u1", "event_timestamp": at(10), "plan": "pro", "spend": 50},
    {"user_id": "u1", "event_timestamp": at(20), "plan": "churned", "spend": 0},  # the future for most labels
    {"user_id": "u2", "event_timestamp": at(5), "plan": "pro", "spend": 30},
]
LABELS = [
    {"user_id": "u1", "event_timestamp": at(9), "label": 0},
    {"user_id": "u1", "event_timestamp": at(10), "label": 0},
    {"user_id": "u1", "event_timestamp": at(19), "label": 1},
    {"user_id": "u2", "event_timestamp": at(4), "label": 0},
    {"user_id": "u3", "event_timestamp": at(30), "label": 1},
]


def test_point_in_time_join_never_uses_future_values():
    joined = point_in_time_join(LABELS, FEATURES, entity_key="user_id", features=["plan", "spend"])
    assert [row["plan"] for row in joined] == ["free", "pro", "pro", None, None]
    assert [row["spend"] for row in joined] == [0, 50, 50, None, None]
    for label, row in zip(LABELS, joined):
        assert row["label"] == label["label"] and row["event_timestamp"] == label["event_timestamp"]
        assert "churned" not in json.dumps(row, default=str)


def test_point_in_time_join_brute_force_equivalence():
    """Every output equals the latest feature row at or before the label."""
    joined = point_in_time_join(LABELS, FEATURES, entity_key="user_id", features=["plan"])
    for label, row in zip(LABELS, joined):
        eligible = [f for f in FEATURES if f["user_id"] == label["user_id"] and f["event_timestamp"] <= label["event_timestamp"]]
        expected = max(eligible, key=lambda f: f["event_timestamp"])["plan"] if eligible else None
        assert row["plan"] == expected


def test_point_in_time_join_respects_ttl():
    joined = point_in_time_join(LABELS[:3], FEATURES, entity_key="user_id", features=["plan"], ttl_seconds=300)
    # at(9): latest is at(0), 9 minutes old > 5 minute TTL -> missing.
    assert [row["plan"] for row in joined] == [None, "pro", None]


def test_point_in_time_join_breaks_ties_by_created_timestamp_and_accepts_strings():
    features = [
        {"user_id": "u1", "event_timestamp": "2026-10-01T12:00:00Z", "created_timestamp": "2026-10-01T12:05:00Z", "plan": "corrected"},
        {"user_id": "u1", "event_timestamp": "2026-10-01T12:00:00Z", "created_timestamp": "2026-10-01T12:01:00Z", "plan": "original"},
    ]
    labels = [{"user_id": "u1", "event_timestamp": "2026-10-01T13:00:00+01:00"}]  # same instant as 12:00Z
    joined = point_in_time_join(labels, features, entity_key="user_id", features=["plan"], prefix="f_")
    assert joined[0]["f_plan"] == "corrected"


def test_as_utc_rejects_non_timestamps():
    assert as_utc(datetime(2026, 1, 1)) == datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(TypeError):
        as_utc(1234)


class FakeRedis:
    def __init__(self, keys):
        self.store = dict(keys)  # key -> ttl (-1 = no expiry)

    def scan_iter(self, match=None, count=None):
        prefix = match.rstrip("*").encode()
        return [key for key in list(self.store) if key.startswith(prefix)]

    def ttl(self, name):
        return self.store[name]

    def expire(self, name, time):
        self.store[name] = time

    def delete(self, *names):
        for name in names:
            self.store.pop(name, None)
        return len(names)


VIEWS = [{"name": "customer_profile", "ttl_seconds": 3600}, {"name": "static_view", "ttl_seconds": 0}]


def fake_keys():
    return FakeRedis({
        b"mlaiops:features:customer_profile:entity_id=u1": -1,
        b"mlaiops:features:customer_profile:entity_id=u2": 120,
        b"mlaiops:features:static_view:entity_id=u1": -1,
        b"mlaiops:features:removed_view:entity_id=u1": -1,
        b"other:key": -1,
    })


def test_ttl_sweep_sets_expiry_and_removes_orphans():
    client = fake_keys()
    report = sweep(client, VIEWS)
    assert client.store[b"mlaiops:features:customer_profile:entity_id=u1"] == 3600
    assert client.store[b"mlaiops:features:customer_profile:entity_id=u2"] == 120  # existing expiry kept
    assert client.store[b"mlaiops:features:static_view:entity_id=u1"] == -1  # no TTL, untouched
    assert b"mlaiops:features:removed_view:entity_id=u1" not in client.store
    assert b"other:key" in client.store
    assert (report.scanned, report.expiry_set, report.deleted, report.untouched) == (4, 1, 1, 2)
    assert report.by_view["removed_view"]["deleted"] == 1


def test_ttl_sweep_dry_run_and_keep_orphans():
    client = fake_keys()
    before = dict(client.store)
    report = sweep(client, VIEWS, dry_run=True)
    assert client.store == before
    assert report.expiry_set == 1 and report.deleted == 1
    kept = fake_keys()
    sweep(kept, VIEWS, delete_orphans=False)
    assert b"mlaiops:features:removed_view:entity_id=u1" in kept.store


def lineage_platform(calls, fail_online=False, legacy=False):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if fail_online and request.method == "PUT":
            return httpx.Response(503)
        if legacy and request.url.path.endswith("/materializations"):
            return httpx.Response(404)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


def materializer(transport):
    return Materializer(gateway_url="http://gateway", feature_gateway_url="http://features", storage_proxy_url="",
                        client=httpx.Client(transport=transport))


def test_materializer_reports_lineage_per_view():
    calls = []
    job = materializer(lineage_platform(calls))
    job.run()
    reports = [body for method, path, body in calls if path.endswith("/materializations")]
    assert len(reports) == 2
    assert {report["source_dataset"] for report in reports} == {"s3://mlaiops-features/customer_profile", "s3://mlaiops-features/transaction_stats_5m"}
    assert all(report["run_id"] == job.run_id and report["status"] == "succeeded" and report["entity_count"] == 3 for report in reports)


def test_materializer_reports_failure_then_raises():
    calls = []
    with pytest.raises(httpx.HTTPStatusError):
        materializer(lineage_platform(calls, fail_online=True)).run()
    failures = [body for method, path, body in calls if path.endswith("/materializations")]
    assert failures and failures[0]["status"] == "failed" and "503" in failures[0]["error"]


def test_materializer_falls_back_to_legacy_count_report():
    calls = []
    materializer(lineage_platform(calls, legacy=True)).run()
    assert sum(1 for _, path, _ in calls if path.endswith("/materialized")) == 2
