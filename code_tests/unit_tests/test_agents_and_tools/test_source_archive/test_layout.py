from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive import layout
from forecasting_tools.agents_and_tools.source_archive import manifest as manifest_io
from forecasting_tools.agents_and_tools.source_archive.catalog import _load_all_records
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord
from forecasting_tools.agents_and_tools.source_archive.pipeline import (
    CaptureOutcome,
    PipelineSummary,
)
from forecasting_tools.agents_and_tools.source_archive.reports import (
    read_outcomes,
    write_run_report,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


# --- key helpers --------------------------------------------------------------
def test_manifest_key_nests_by_group_daily_or_adhoc():
    assert (
        layout.manifest_key("r1", group="sprints/myrun")
        == "manifests/sprints/myrun/r1.jsonl"
    )
    assert (
        layout.manifest_key("daily-2026-07-01")
        == "manifests/daily/2026-07/daily-2026-07-01.jsonl"
    )
    assert layout.manifest_key("r1") == "manifests/adhoc/r1.jsonl"


def test_report_key_nests_and_keeps_suffix():
    assert (
        layout.report_key("daily-2026-07-01", ".json")
        == "reports/daily/2026-07/daily-2026-07-01.json"
    )
    assert layout.report_key("r1", "_cost.json") == "reports/adhoc/r1_cost.json"
    assert (
        layout.report_key("r1", ".json", group="sprints/myrun")
        == "reports/sprints/myrun/r1.json"
    )


def test_group_slashes_are_normalized():
    assert layout.manifest_key("r1", group="/sprints/myrun/") == (
        "manifests/sprints/myrun/r1.jsonl"
    )


def test_candidates_prefer_nested_then_legacy_flat():
    assert layout.manifest_key_candidates("r1") == [
        "manifests/adhoc/r1.jsonl",
        "manifests/r1.jsonl",
    ]
    assert layout.report_key_candidates("r1", ".json") == [
        "reports/adhoc/r1.json",
        "reports/r1.json",
    ]


# --- readers ------------------------------------------------------------------
def test_read_blob_falls_back_to_legacy_flat_key(tmp_path):
    store = LocalBlobStore(tmp_path)
    cfg = ArchiveConfig(s3_prefix="t")
    legacy = manifest_io.dumps([CitationRecord(url="https://old.test", run_id="r1")])
    store.put("t/manifests/r1.jsonl", legacy.encode("utf-8"))

    assert manifest_io.read_blob(store, "r1", cfg)[0].url == "https://old.test"


def test_read_blob_prefers_nested_over_legacy(tmp_path):
    store = LocalBlobStore(tmp_path)
    cfg = ArchiveConfig(s3_prefix="t")
    legacy = manifest_io.dumps([CitationRecord(url="https://old.test", run_id="r1")])
    store.put("t/manifests/r1.jsonl", legacy.encode("utf-8"))
    manifest_io.write_blob(
        store, "r1", [CitationRecord(url="https://new.test", run_id="r1")], cfg
    )

    assert manifest_io.read_blob(store, "r1", cfg)[0].url == "https://new.test"


def test_catalog_loads_nested_and_flat_manifests(tmp_path):
    store = LocalBlobStore(tmp_path)
    cfg = ArchiveConfig(s3_prefix="t")
    store.put(
        "t/manifests/old.jsonl",
        manifest_io.dumps([CitationRecord(url="https://old.test")]).encode("utf-8"),
    )
    manifest_io.write_blob(
        store, "daily-2026-07-01", [CitationRecord(url="https://daily.test")], cfg
    )
    manifest_io.write_blob(
        store,
        "sprint-run",
        [CitationRecord(url="https://sprint.test")],
        cfg,
        group="sprints/myrun",
    )

    urls = {r.url for r in _load_all_records(store, "t")}
    assert urls == {"https://old.test", "https://daily.test", "https://sprint.test"}


def test_read_outcomes_sees_nested_and_flat_reports(tmp_path):
    store = LocalBlobStore(tmp_path)
    cfg = ArchiveConfig(s3_prefix="t")
    store.put(
        "t/reports/old.json",
        b'[{"url": "https://old.test", "status": "stored", "reason": ""}]',
    )
    write_run_report(
        store,
        "daily-2026-07-01",
        PipelineSummary(
            outcomes=[CaptureOutcome(url="https://daily.test", status="error")]
        ),
        cfg,
    )

    out = read_outcomes(store, cfg)
    assert out["https://old.test"] == "stored"
    assert out["https://daily.test"] == "error"
