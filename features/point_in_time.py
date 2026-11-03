"""Point-in-time correct feature joins for training data.

For every entity row (an entity id plus the ``event_timestamp`` at which a
label was observed) the join picks, per entity, the latest feature row that
was available at that moment. A feature row is eligible only when both hold:

- its ``event_timestamp`` is at or before the entity row's timestamp, and
- its ``created_timestamp`` (when the row has one) is at or before the entity
  row's timestamp too.

The second condition matters for late-arriving and backfilled rows: a row
whose event time precedes the label but which was written after it did not
exist when the prediction would have been made, so it is skipped and the next
older eligible row is used instead. Feature rows from the future, by either
clock, are never used. A feature row older than ``ttl_seconds`` (measured on
its event time) is treated as missing, matching online serving.

This is the same "as of" semantics as Feast's ``get_historical_features``
and ``pandas.merge_asof(direction="backward", by=entity)``, implemented over
plain rows so it has no dependency and is deterministic.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def as_utc(value: Any) -> datetime:
    """Normalize a datetime or ISO-8601 string to an aware UTC datetime.

    Naive datetimes are interpreted as UTC (the platform stores UTC)."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError(f"timestamp must be a datetime or ISO-8601 string, got {type(value).__name__}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def point_in_time_join(
    entity_rows: Iterable[Mapping[str, Any]],
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    entity_key: str,
    features: list[str],
    timestamp_key: str = "event_timestamp",
    feature_timestamp_key: str = "event_timestamp",
    created_key: str = "created_timestamp",
    ttl_seconds: int | None = None,
    prefix: str = "",
    respect_created_timestamp: bool = True,
) -> list[dict[str, Any]]:
    """Return one output row per entity row, in input order.

    A feature row is eligible when its ``feature_timestamp_key`` and, if
    present and ``respect_created_timestamp`` is true, its ``created_key`` are
    both at or before the entity row's timestamp. The latest eligible row by
    event time wins; ties are broken by ``created_key`` (latest write wins),
    then by input order. Entity rows without an eligible feature row get
    ``None`` for every feature.
    """
    history: dict[Any, list[tuple[datetime, datetime, int, Mapping[str, Any], datetime | None]]] = defaultdict(list)
    for index, row in enumerate(feature_rows):
        created = row.get(created_key)
        created_at = as_utc(created) if created is not None else None
        history[row[entity_key]].append(
            (as_utc(row[feature_timestamp_key]), created_at or datetime.min.replace(tzinfo=timezone.utc), index, row, created_at)
        )
    timelines: dict[Any, tuple[list[datetime], list[datetime | None], list[Mapping[str, Any]]]] = {}
    for entity, items in history.items():
        items.sort(key=lambda item: (item[0], item[1], item[2]))
        timelines[entity] = ([item[0] for item in items], [item[4] for item in items], [item[3] for item in items])

    joined: list[dict[str, Any]] = []
    for row in entity_rows:
        at = as_utc(row[timestamp_key])
        output = dict(row)
        match: Mapping[str, Any] | None = None
        timeline = timelines.get(row[entity_key])
        if timeline:
            times, created_times, rows = timeline
            position = bisect_right(times, at) - 1
            # Walk back past rows that were written after the label time; the
            # first remaining row is the latest one that existed at ``at``.
            while respect_created_timestamp and position >= 0 and created_times[position] is not None and created_times[position] > at:
                position -= 1
            if position >= 0:
                candidate_time = times[position]
                if ttl_seconds is None or ttl_seconds <= 0 or (at - candidate_time).total_seconds() <= ttl_seconds:
                    match = rows[position]
        for name in features:
            output[prefix + name] = match.get(name) if match is not None else None
        joined.append(output)
    return joined
