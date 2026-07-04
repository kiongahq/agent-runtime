"""Version-controlled feature view definitions.

Definitions are declarative data: the materializer applies them to the
control plane and computes rows deterministically. Real deployments replace
``source_rows`` with reads from the offline store or a warehouse connection.
"""

from __future__ import annotations

from typing import Any

FEATURE_VIEWS: list[dict[str, Any]] = [
    {
        "name": "customer_profile",
        "entity": "entity_id",
        "fields": [
            {"name": "plan", "type": "string"},
            {"name": "region", "type": "string"},
            {"name": "open_tickets", "type": "int64"},
            {"name": "csat_90d", "type": "float64"},
        ],
        "tags": ["customer", "support"],
        "source": "s3://mlaiops-features/customer_profile",
        "ttl_seconds": 3600,
    },
    {
        "name": "transaction_stats_5m",
        "entity": "entity_id",
        "fields": [
            {"name": "txn_count_5m", "type": "int64"},
            {"name": "txn_amount_5m", "type": "float64"},
            {"name": "distinct_merchants_5m", "type": "int64"},
            {"name": "home_region", "type": "string"},
        ],
        "tags": ["fraud", "realtime"],
        "source": "s3://mlaiops-features/transaction_stats_5m",
        "ttl_seconds": 300,
    },
]


def source_rows(view_name: str) -> list[dict[str, Any]]:
    """Deterministic demo source data, keyed by entity id."""
    if view_name == "customer_profile":
        return [
            {"entity_id": "u123", "plan": "pro", "region": "eu-west", "open_tickets": 1, "csat_90d": 4.6},
            {"entity_id": "u456", "plan": "free", "region": "us-east", "open_tickets": 0, "csat_90d": 4.1},
            {"entity_id": "u789", "plan": "pro", "region": "us-west", "open_tickets": 3, "csat_90d": 3.9},
        ]
    if view_name == "transaction_stats_5m":
        return [
            {"entity_id": "u123", "txn_count_5m": 2, "txn_amount_5m": 84.5, "distinct_merchants_5m": 2, "home_region": "eu-west"},
            {"entity_id": "u456", "txn_count_5m": 0, "txn_amount_5m": 0.0, "distinct_merchants_5m": 0, "home_region": "us-east"},
            {"entity_id": "u789", "txn_count_5m": 9, "txn_amount_5m": 2350.0, "distinct_merchants_5m": 7, "home_region": "us-west"},
        ]
    raise KeyError(f"unknown feature view {view_name!r}")
