"""Persist each capture run's per-URL outcomes to ``reports/<run_id>.json``.

The coverage report's job is to surface sources we should be collecting. A cited
source we have not archived falls into two very different buckets:

- **never fetched** — it was harvested into a manifest but no capture run ever
  attempted it. This is the real "we should go collect this" signal.
- **fetched but failed** — we tried and the fetch/quality gate rejected it
  (Cloudflare, PDF, 404…). A capture problem, not a collection problem.

Without persisted run outcomes the two are indistinguishable. Writing each run's
outcomes here lets coverage tell them apart.
"""

from __future__ import annotations

import json

from forecasting_tools.agents_and_tools.source_archive.canonicalize import (
    canonicalize_url,
)
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.storage.blob_store import (
    BlobStore,
)

CAPTURED_STATUSES = {"stored", "deduped", "cache_hit"}
FAILED_STATUSES = {"quality_failed", "error"}


def report_key(run_id: str, config: ArchiveConfig) -> str:
    return f"{config.s3_prefix.rstrip('/')}/reports/{run_id}.json"


def write_run_report(
    store: BlobStore, run_id: str, summary, config: ArchiveConfig
) -> str:
    """Persist a run's per-URL outcomes; ``summary`` is a ``PipelineSummary``.

    ``backend`` is the fetcher that produced the capture (from the stored
    capture, so it is set for stored/deduped/cache_hit outcomes and ``""``
    when unknown, e.g. errors) — it enables per-domain cost attribution.
    """
    rows = [
        {
            "url": o.url,
            "status": o.status,
            "reason": getattr(o, "reason", ""),
            "backend": o.stored.fetcher if o.stored is not None else "",
        }
        for o in summary.outcomes
    ]
    key = report_key(run_id, config)
    store.put(
        key, json.dumps(rows, indent=2).encode("utf-8"), content_type="application/json"
    )
    return key


def read_outcomes(store: BlobStore, config: ArchiveConfig) -> dict[str, str]:
    """Map canonical URL -> last known capture status across all run reports.

    A captured status wins over a failed one (if we ever succeeded, that's the
    truth). Returns ``{}`` if no reports exist yet.
    """
    prefix = config.s3_prefix.rstrip("/")
    out: dict[str, str] = {}
    for key in store.list_keys(f"{prefix}/reports/"):
        if not key.endswith(".json"):
            continue
        try:
            rows = json.loads(store.get(key).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        for r in rows:
            url = canonicalize_url(r.get("url", ""))
            status = r.get("status", "")
            if not url:
                continue
            if url not in out or status in CAPTURED_STATUSES:
                out[url] = status
    return out
