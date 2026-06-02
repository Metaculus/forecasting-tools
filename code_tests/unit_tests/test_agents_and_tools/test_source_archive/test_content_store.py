from __future__ import annotations

from datetime import datetime, timedelta, timezone

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.models import (
    CaptureResult,
    url_hash,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


def _store(tmp_path, **cfg) -> ContentStore:
    return ContentStore(LocalBlobStore(tmp_path), ArchiveConfig(s3_prefix="t", **cfg))


def _result(url: str, html: str) -> CaptureResult:
    return CaptureResult(
        url=url,
        final_url=url,
        status_code=200,
        html=html,
        markdown="md " * 50,
        screenshot=b"img",
        screenshot_content_type="image/png",
        fetcher="fake",
    )


def test_store_writes_blobs_and_index(tmp_path):
    store = _store(tmp_path)
    res = store.store(_result("https://a.test", "<p>one</p>"))
    assert res.created is True
    cap = res.capture
    assert store.blobs.exists(cap.html_key)
    assert store.blobs.exists(cap.markdown_key)
    assert store.blobs.exists(cap.screenshot_key)


def test_lookup_within_ttl_is_cache_hit(tmp_path):
    store = _store(tmp_path, ttl_days=14)
    store.store(_result("https://a.test", "<p>one</p>"))
    assert store.lookup("https://a.test") is not None


def test_lookup_after_ttl_expires_returns_none(tmp_path):
    store = _store(tmp_path, ttl_days=14)
    store.store(_result("https://a.test", "<p>one</p>"))

    uh = url_hash("https://a.test")
    index = store._read_index(uh)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    for cap in index["captures"].values():
        cap["last_seen"] = old
    store._write_index(uh, index)

    assert store.lookup("https://a.test") is None


def test_identical_content_is_deduped(tmp_path):
    store = _store(tmp_path)
    first = store.store(_result("https://a.test", "<p>same</p>"))
    second = store.store(_result("https://a.test", "<p>same</p>"))
    assert first.created is True
    assert second.created is False
    assert first.capture.content_hash == second.capture.content_hash


def test_changed_content_creates_new_capture(tmp_path):
    store = _store(tmp_path)
    first = store.store(_result("https://a.test", "<p>v1</p>"))
    second = store.store(_result("https://a.test", "<p>v2 changed</p>"))
    assert second.created is True
    assert first.capture.content_hash != second.capture.content_hash
