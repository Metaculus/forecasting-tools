from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive import manifest
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord
from forecasting_tools.agents_and_tools.source_archive.pipeline import (
    CapturePipeline,
    capture_urls_concurrent,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


def _pipeline(tmp_path, fetcher) -> CapturePipeline:
    store = ContentStore(
        LocalBlobStore(tmp_path), ArchiveConfig(s3_prefix="t", ttl_days=14)
    )
    return CapturePipeline(fetcher, store)


def test_capture_urls_concurrent_captures_all(tmp_path, make_fetcher):
    from contextlib import contextmanager

    config = ArchiveConfig(s3_prefix="t", concurrency=4)
    store = ContentStore(LocalBlobStore(tmp_path), config)
    urls = [f"https://s{i}.test/p" for i in range(12)]

    @contextmanager
    def factory(_cfg):
        f = make_fetcher()
        for u in urls:
            f.add(u)
        yield f

    summary = capture_urls_concurrent(urls, store, config, factory)

    assert len(summary.outcomes) == 12
    assert summary.count("stored") == 12
    # every URL is resolvable afterwards (proves the shared store got all writes)
    assert all(store.lookup(u) is not None for u in urls)


def test_concurrent_supervisor_recovers_a_stuck_worker(tmp_path, make_fetcher):
    import threading
    from contextlib import contextmanager

    config = ArchiveConfig(s3_prefix="t", concurrency=1)
    store = ContentStore(LocalBlobStore(tmp_path), config)
    urls = ["https://stuck.test/x"]
    reaped = threading.Event()
    builds = {"n": 0}

    class _Wedges:
        name = "wedge"

        def fetch(self, url):
            # Block until the supervisor's reaper "kills the browser", then surface
            # the dead-browser error a killed Chromium would raise.
            reaped.wait(5)
            raise RuntimeError("Target page, context or browser has been closed")

    @contextmanager
    def factory(_cfg):
        builds["n"] += 1
        if builds["n"] == 1:
            yield _Wedges()  # first browser wedges
        else:
            fetcher = make_fetcher()
            fetcher.add(urls[0])
            yield fetcher  # rebuilt browser works

    # Inject a fake reaper so the test drives the supervisor without real Chromium.
    summary = capture_urls_concurrent(
        urls, store, config, factory, per_url_timeout=0.3, reaper=reaped.set
    )

    assert builds["n"] == 2  # stalled -> reaped -> death -> rebuild -> retry
    assert summary.count("stored") == 1  # recovered and captured on a fresh browser


def test_concurrent_restarts_browser_after_death(tmp_path, make_fetcher):
    from contextlib import contextmanager

    config = ArchiveConfig(s3_prefix="t", concurrency=1)
    store = ContentStore(LocalBlobStore(tmp_path), config)
    urls = ["https://a.test/x"]
    builds = {"n": 0}

    class _DeadBrowser:
        name = "dead"

        def fetch(self, url):
            raise RuntimeError("Target page, context or browser has been closed")

    @contextmanager
    def factory(_cfg):
        builds["n"] += 1
        if builds["n"] == 1:
            yield _DeadBrowser()  # first browser is dead
        else:
            fetcher = make_fetcher()
            fetcher.add(urls[0])
            yield fetcher  # rebuilt browser works

    summary = capture_urls_concurrent(urls, store, config, factory)

    assert builds["n"] == 2  # detected death, rebuilt once
    assert summary.count("stored") == 1  # retry on the fresh browser succeeded


class _BoomFetcher:
    """Raises an unexpected (non-FetchError) exception, like a bad screenshot."""

    name = "boom"

    def fetch(self, url):
        raise ValueError("kaboom")


def test_pipeline_isolates_unexpected_fetcher_errors(tmp_path):
    # One pathological URL must not abort the whole run.
    pipe = _pipeline(tmp_path, _BoomFetcher())
    summary = pipe.run(["https://a.test", "https://b.test"])
    assert summary.count("error") == 2
    assert len(summary.outcomes) == 2
    assert all(o.reason.startswith("unexpected:") for o in summary.outcomes)


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
    assert store.exists("t/manifests/adhoc/r1.jsonl")
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
