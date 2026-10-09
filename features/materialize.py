"""Feature materialization job.

For every feature view definition it:

1. applies the definition to the control plane (``POST /api/v1/features``),
2. writes each entity row into the online store through the feature-gateway
   (``PUT /internal/v1/features/{view}/{entity}``),
3. uploads an offline snapshot to the object store via a storage-proxy
   presigned URL (Parquet when pyarrow is available, JSONL otherwise),
4. reports lineage to the control plane (``POST /api/v1/features/{view}/
   materializations``): run id, source dataset, offline snapshot URI and the
   entity count, or the failure that stopped the run. Older gateways without
   that endpoint get the plain ``/materialized`` count report instead.

Runs as a container, a cron job, or ``python -m features.materialize``.
Everything here is deterministic: same definitions, same rows, same result.
"""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
from typing import Any

import httpx

from .definitions import FEATURE_VIEWS, source_rows


class Materializer:
    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        feature_gateway_url: str | None = None,
        storage_proxy_url: str | None = None,
        internal_token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.gateway_url = (gateway_url or os.environ.get("MLAIOPS_URL", "http://localhost:8080")).rstrip("/")
        self.feature_gateway_url = (
            feature_gateway_url
            or os.environ.get("MLAIOPS_FEATURE_GATEWAY_URL", "http://localhost:8083")
        ).rstrip("/")
        self.storage_proxy_url = (
            storage_proxy_url or os.environ.get("MLAIOPS_STORAGE_PROXY_URL", "")
        ).rstrip("/")
        self.internal_token = internal_token or os.environ.get("MLAIOPS_INTERNAL_TOKEN", "")
        self.client = client or httpx.Client(timeout=10)
        self.run_id = f"mat-{uuid.uuid4().hex[:12]}"

    def _headers(self) -> dict[str, str]:
        headers = {"X-MLAIOps-Actor": "materializer"}
        if self.internal_token:
            headers["Authorization"] = f"Bearer {self.internal_token}"
        return headers

    def apply_definitions(self) -> None:
        for view in FEATURE_VIEWS:
            response = self.client.post(
                f"{self.gateway_url}/api/v1/features", json=view, headers=self._headers()
            )
            response.raise_for_status()

    def materialize_view(self, view: dict[str, Any]) -> int:
        try:
            rows = source_rows(view["name"])
            entity_key = view["entity"]
            for row in rows:
                entity_value = row[entity_key]
                values = {name: value for name, value in row.items() if name != entity_key}
                response = self.client.put(
                    f"{self.feature_gateway_url}/internal/v1/features/{view['name']}"
                    f"/{entity_key}={entity_value}",
                    json=values,
                    headers=self._headers(),
                )
                response.raise_for_status()
            offline_uri = self._write_offline_snapshot(view["name"], rows)
        except Exception as error:
            self._report(view, status="failed", entity_count=0, error=str(error)[:2000])
            raise
        self._report(view, status="succeeded", entity_count=len(rows), offline_uri=offline_uri)
        return len(rows)

    def _report(self, view: dict[str, Any], *, status: str, entity_count: int, offline_uri: str = "", error: str = "") -> None:
        """Record lineage. A failure to report a failure never hides the
        original error, so failed reports are best effort."""
        payload = {
            "run_id": self.run_id,
            "source_dataset": view.get("source") or f"definitions:{view['name']}",
            "entity_count": entity_count,
            "status": status,
        }
        if offline_uri:
            payload["offline_uri"] = offline_uri
        if error:
            payload["error"] = error
        try:
            response = self.client.post(
                f"{self.gateway_url}/api/v1/features/{view['name']}/materializations",
                json=payload,
                headers=self._headers(),
            )
            if response.status_code in (404, 405) and status == "succeeded":
                response = self.client.post(
                    f"{self.gateway_url}/api/v1/features/{view['name']}/materialized",
                    json={"entity_count": entity_count},
                    headers=self._headers(),
                )
            response.raise_for_status()
        except httpx.HTTPError:
            if status == "succeeded":
                raise

    def _snapshot_bytes(self, rows: list[dict[str, Any]]) -> tuple[bytes, str]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows)
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            return buffer.getvalue(), "parquet"
        except ImportError:
            lines = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
            return lines.encode(), "jsonl"

    def _write_offline_snapshot(self, view_name: str, rows: list[dict[str, Any]]) -> str:
        """Upload the offline snapshot through a presigned URL. Skipped (not
        failed) when no storage proxy is configured, so online materialization
        still works on minimal stacks."""
        if not self.storage_proxy_url:
            return ""
        content, extension = self._snapshot_bytes(rows)
        presign = self.client.post(
            f"{self.storage_proxy_url}/presign",
            json={
                "bucket": "mlaiops-features",
                "key": f"{view_name}/snapshot.{extension}",
                "operation": "PUT",
                "ttl_seconds": 300,
            },
            headers=self._headers(),
        )
        presign.raise_for_status()
        upload = self.client.put(presign.json()["url"], content=content)
        upload.raise_for_status()
        return f"s3://mlaiops-features/{view_name}/snapshot.{extension}"

    def run(self) -> dict[str, int]:
        self.apply_definitions()
        return {view["name"]: self.materialize_view(view) for view in FEATURE_VIEWS}


def main() -> int:
    counts = Materializer().run()
    for name, count in counts.items():
        print(f"materialized {name}: {count} entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
