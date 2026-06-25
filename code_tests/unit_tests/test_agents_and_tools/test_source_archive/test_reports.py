from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.pipeline import (
    CaptureOutcome,
    PipelineSummary,
)
from forecasting_tools.agents_and_tools.source_archive.reports import (
    read_outcomes,
    write_run_report,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


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
