from __future__ import annotations

import json

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.cost import (
    RunCost,
    write_cost_report,
)
from forecasting_tools.agents_and_tools.source_archive.models import (
    StoredCapture,
    url_hash,
)
from forecasting_tools.agents_and_tools.source_archive.pipeline import (
    CaptureOutcome,
    PipelineSummary,
)
from forecasting_tools.agents_and_tools.source_archive.reports import (
    read_outcomes,
    write_run_report,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


def _stored(url: str, fetcher: str) -> StoredCapture:
    return StoredCapture(
        url=url, url_hash=url_hash(url), content_hash="c1", fetcher=fetcher
    )


def test_run_report_roundtrip_canonicalizes(tmp_path):
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    summary = PipelineSummary(
        outcomes=[
            CaptureOutcome(url="https://a.test/p?utm_source=x", status="stored"),
            CaptureOutcome(url="https://b.test/q", status="error", reason="cloudflare"),
        ]
    )
    write_run_report(store, "r1", summary, config)

    out = read_outcomes(store, config)
    # keys are canonicalized (tracking param stripped)
    assert out["https://a.test/p"] == "stored"
    assert out["https://b.test/q"] == "error"


def test_run_report_records_backend_per_url(tmp_path):
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    summary = PipelineSummary(
        outcomes=[
            CaptureOutcome(
                url="https://a.test/p",
                status="stored",
                stored=_stored("https://a.test/p", "firecrawl"),
            ),
            CaptureOutcome(
                url="https://c.test/r",
                status="cache_hit",
                stored=_stored("https://c.test/r", "playwright"),
            ),
            CaptureOutcome(url="https://b.test/q", status="error", reason="cloudflare"),
        ]
    )
    key = write_run_report(store, "r1", summary, config)

    rows = {r["url"]: r for r in json.loads(store.get(key).decode("utf-8"))}
    assert rows["https://a.test/p"]["backend"] == "firecrawl"
    assert rows["https://c.test/r"]["backend"] == "playwright"
    assert rows["https://b.test/q"]["backend"] == ""  # nothing fetched


def test_captured_status_wins_over_failure(tmp_path):
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    write_run_report(
        store,
        "early",
        PipelineSummary(
            outcomes=[CaptureOutcome(url="https://a.test", status="error")]
        ),
        config,
    )
    write_run_report(
        store,
        "later",
        PipelineSummary(
            outcomes=[CaptureOutcome(url="https://a.test", status="stored")]
        ),
        config,
    )
    assert read_outcomes(store, config)["https://a.test"] == "stored"


def test_read_outcomes_ignores_cost_reports(tmp_path):
    """Cost reports live under reports/ too (``<run_id>_cost.json``, a JSON
    dict, not a list of rows) — read_outcomes must skip them, not crash."""
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    write_run_report(
        store,
        "r1",
        PipelineSummary(
            outcomes=[CaptureOutcome(url="https://a.test", status="stored")]
        ),
        config,
    )
    write_cost_report(store, "r1", RunCost(run_id="r1", archived=1), config)

    out = read_outcomes(store, config)
    assert out == {"https://a.test": "stored"}
