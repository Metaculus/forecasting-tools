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


def _result(url: str, html: str, final_url: str | None = None) -> CaptureResult:
    return CaptureResult(
        url=url,
        final_url=final_url if final_url is not None else url,
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


# --- Phase B: redirect aliasing -------------------------------------------
def test_redirect_keys_capture_by_final_url(tmp_path):
    store = _store(tmp_path)
    res = store.store(
        _result("https://bit.ly/x", "<p>dest</p>", final_url="https://dest.test/page")
    )
    # Capture is stored under the FINAL url's hash, not the shortener's.
    assert res.capture.url == "https://dest.test/page"
    assert res.capture.url_hash == url_hash("https://dest.test/page")
    # The canonical index records the cited shortener as an alias.
    canonical = store._read_index(url_hash("https://dest.test/page"))
    assert "https://bit.ly/x" in canonical["aliases"]


def test_lookup_via_shortener_and_final_both_hit(tmp_path):
    store = _store(tmp_path)
    store.store(
        _result("https://bit.ly/x", "<p>dest</p>", final_url="https://dest.test/page")
    )
    via_alias = store.lookup("https://bit.ly/x")
    via_final = store.lookup("https://dest.test/page")
    assert via_alias is not None and via_final is not None
    assert via_alias.content_hash == via_final.content_hash
    assert via_alias.url == "https://dest.test/page"


def test_two_shorteners_to_same_page_store_once(tmp_path):
    store = _store(tmp_path)
    first = store.store(
        _result("https://bit.ly/x", "<p>same</p>", final_url="https://dest.test/page")
    )
    second = store.store(
        _result("https://t.co/y", "<p>same</p>", final_url="https://dest.test/page")
    )
    assert first.created is True
    assert second.created is False  # identical content deduped, not re-stored
    canonical = store._read_index(url_hash("https://dest.test/page"))
    assert set(canonical["aliases"]) == {"https://bit.ly/x", "https://t.co/y"}
    assert len(canonical["captures"]) == 1


# --- Phase C: cross-URL content dedup -------------------------------------
def test_identical_content_across_distinct_urls_reuses_blobs(tmp_path):
    store = _store(tmp_path)
    a = store.store(_result("https://a.test/x", "<p>same</p>"))
    b = store.store(_result("https://b.test/y", "<p>same</p>"))

    # Both are real captures (each URL has its own index entry)...
    assert a.created is True and b.created is True
    # ...but B reuses A's blobs instead of writing its own.
    assert a.capture.content_alias_of is None
    assert b.capture.content_alias_of == url_hash("https://a.test/x")
    assert b.capture.html_key == a.capture.html_key

    # No duplicate blob was written under B's url hash.
    b_own_key = (
        f"t/content/{url_hash('https://b.test/y')}/{b.capture.content_hash}.html"
    )
    assert not store.blobs.exists(b_own_key)
    assert store.blobs.exists(a.capture.html_key)


def test_content_reverse_index_tracks_members(tmp_path):
    store = _store(tmp_path)
    store.store(_result("https://a.test/x", "<p>same</p>"))
    store.store(_result("https://b.test/y", "<p>same</p>"))

    ch = store.store(_result("https://c.test/z", "<p>same</p>")).capture.content_hash
    reverse = store._read_content_index(ch)
    assert reverse["canonical_url_hash"] == url_hash("https://a.test/x")
    member_hashes = {m["url_hash"] for m in reverse["members"]}
    assert member_hashes == {
        url_hash("https://a.test/x"),
        url_hash("https://b.test/y"),
        url_hash("https://c.test/z"),
    }


def test_different_content_not_aliased(tmp_path):
    store = _store(tmp_path)
    a = store.store(_result("https://a.test/x", "<p>one</p>"))
    b = store.store(_result("https://b.test/y", "<p>two different</p>"))
    assert b.capture.content_alias_of is None
    assert b.capture.html_key != a.capture.html_key
