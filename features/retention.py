"""TTL and retention sweep for online feature keys in Redis.

Online rows live at ``mlaiops:features:<view>:<entity>``. The feature
gateway writes them without an expiry, so this sweep enforces each view's
``ttl_seconds``:

* a key of a view with a TTL and no expiry gets ``EXPIRE ttl_seconds``, so an
  entity that stops being materialized disappears within one TTL of the next
  sweep (re-materialization resets it and the next sweep re-arms it);
* a key whose view is no longer defined is deleted when ``delete_orphans``;
* views without a TTL are left untouched.

Run it on a schedule after materialization, e.g.
``python -m features.retention`` with ``REDIS_URL`` set. ``dry_run`` reports
what would change without changing anything.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

KEY_PREFIX = "mlaiops:features:"


class RedisLike(Protocol):
    def scan_iter(self, match: str | None = None, count: int | None = None) -> Iterable[Any]: ...
    def ttl(self, name: Any) -> int: ...
    def expire(self, name: Any, time: int) -> Any: ...
    def delete(self, *names: Any) -> int: ...


@dataclass
class SweepReport:
    scanned: int = 0
    expiry_set: int = 0
    deleted: int = 0
    untouched: int = 0
    by_view: dict[str, dict[str, int]] = field(default_factory=dict)

    def count(self, view: str, outcome: str) -> None:
        self.by_view.setdefault(view, {"expiry_set": 0, "deleted": 0, "untouched": 0})[outcome] += 1


def _text(key: Any) -> str:
    return key.decode() if isinstance(key, bytes) else str(key)


def sweep(
    client: RedisLike,
    views: Iterable[dict[str, Any]],
    *,
    delete_orphans: bool = True,
    dry_run: bool = False,
    batch: int = 500,
) -> SweepReport:
    ttl_by_view = {view["name"]: int(view.get("ttl_seconds") or 0) for view in views}
    report = SweepReport()
    orphans: list[Any] = []
    for key in client.scan_iter(match=KEY_PREFIX + "*", count=batch):
        report.scanned += 1
        rest = _text(key)[len(KEY_PREFIX):]
        view, separator, _ = rest.partition(":")
        if not separator:
            report.untouched += 1
            continue
        if view not in ttl_by_view:
            if delete_orphans:
                orphans.append(key)
                report.deleted += 1
                report.count(view, "deleted")
            else:
                report.untouched += 1
                report.count(view, "untouched")
            continue
        ttl = ttl_by_view[view]
        if ttl > 0 and client.ttl(key) == -1:
            if not dry_run:
                client.expire(key, ttl)
            report.expiry_set += 1
            report.count(view, "expiry_set")
        else:
            report.untouched += 1
            report.count(view, "untouched")
        if len(orphans) >= batch and not dry_run:
            client.delete(*orphans)
            orphans.clear()
    if orphans and not dry_run:
        client.delete(*orphans)
    return report


def main() -> int:
    import redis

    from .definitions import FEATURE_VIEWS

    client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    report = sweep(client, FEATURE_VIEWS, dry_run="--dry-run" in sys.argv)
    print(
        f"feature retention sweep: scanned={report.scanned} expiry_set={report.expiry_set} "
        f"deleted={report.deleted} untouched={report.untouched}"
    )
    for view, counts in sorted(report.by_view.items()):
        print(f"  {view}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
