from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive import manifest
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord
from forecasting_tools.agents_and_tools.source_archive.pipeline import CapturePipeline
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


def _pipeline(tmp_path, fetcher) -> CapturePipeline:
    store = ContentStore(
        LocalBlobStore(tmp_path), ArchiveConfig(s3_prefix="t", ttl_days=14)
    )
    return CapturePipeline(fetcher, store)


def test_manifest_roundtrip_and_unique_urls():
    records = [
        CitationRecord(url="https://a.test", run_id="r1", bot="b", tool_name="search"),
        CitationRecord(url="https://a.test", run_id="r1", bot="b", tool_name="fetch"),
        CitationRecord(url="https://b.test", run_id="r1", bot="b"),
    ]
    back = manifest.loads(manifest.dumps(records))
    assert [r.url for r in back] == [r.url for r in records]
    assert list(manifest.unique_urls(back)) == ["https://a.test", "https://b.test"]


def test_manifest_blob_roundtrip(tmp_path):
    store = LocalBlobStore(tmp_path)
    cfg = ArchiveConfig(s3_prefix="t")
    records = [CitationRecord(url="https://a.test", run_id="r1")]
    manifest.write_blob(store, "r1", records, cfg)
    assert store.exists("t/manifests/r1.jsonl")
    assert manifest.read_blob(store, "r1", cfg)[0].url == "https://a.test"


def test_pipeline_stores_then_cache_hits(tmp_path, make_fetcher):
    fetcher = make_fetcher()
    fetcher.add("https://a.test")
    pipeline = _pipeline(tmp_path, fetcher)

    first = pipeline.run(["https://a.test"])
    assert first.count("stored") == 1
    assert fetcher.calls == ["https://a.test"]

    second = pipeline.run(["https://a.test"])
    assert second.count("cache_hit") == 1
    assert fetcher.calls == ["https://a.test"]  # not refetched


def test_pipeline_quality_failed_not_stored(tmp_path, make_fetcher):
    fetcher = make_fetcher()
    fetcher.add("https://bad.test", status_code=404)
    pipeline = _pipeline(tmp_path, fetcher)

    summary = pipeline.run(["https://bad.test"])
    assert summary.count("quality_failed") == 1
    assert summary.captures == {}


def test_pipeline_error_when_no_backend_succeeds(tmp_path, make_fetcher):
    fetcher = make_fetcher()  # no canned responses -> FetchError
    pipeline = _pipeline(tmp_path, fetcher)
    summary = pipeline.run(["https://missing.test"])
    assert summary.count("error") == 1


def test_pipeline_run_manifest_dedups_urls(tmp_path, make_fetcher):
    fetcher = make_fetcher()
    fetcher.add("https://a.test")
    pipeline = _pipeline(tmp_path, fetcher)
    records = [
        CitationRecord(url="https://a.test", tool_name="search"),
        CitationRecord(url="https://a.test", tool_name="fetch"),
    ]
    summary = pipeline.run_manifest(records)
    assert len(summary.outcomes) == 1
    assert fetcher.calls == ["https://a.test"]
