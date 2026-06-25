from __future__ import annotations

import json

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.reindex import (
    analyze,
    rebuild_content_index,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


def _put_index(store, key: str, body: dict) -> None:
    store.put(f"t/index/{key}.json", json.dumps(body).encode("utf-8"))


def _canonical(url: str, content_hash: str) -> dict:
    return {
        "url": url,
        "url_hash": f"hash_of_{url}",
        "latest_content_hash": content_hash,
        "captures": {
            content_hash: {
                "url": url,
                "url_hash": f"hash_of_{url}",
                "content_hash": content_hash,
                "html_key": f"t/content/hash_of_{url}/{content_hash}.html",
            }
        },
    }


def _seed(tmp_path) -> tuple[LocalBlobStore, ArchiveConfig]:
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    # Legacy rows stored under raw hashing: two URLs that now canonicalize equal.
    _put_index(store, "h1", _canonical("https://x.test/p?utm_source=news", "c1"))
    _put_index(store, "h2", _canonical("https://x.test/p", "c2"))
    # Two distinct URLs with byte-identical content (same latest hash).
    _put_index(store, "h3", _canonical("https://a.test/1", "cX"))
    _put_index(store, "h4", _canonical("https://b.test/2", "cX"))
    # Same host+path, meaningful query differs -> Phase D candidate.
    _put_index(store, "h5", _canonical("https://q.test/item?id=1", "n1"))
    _put_index(store, "h6", _canonical("https://q.test/item?id=2", "n2"))
    # An alias (redirect) index -> counted but not a capture.
    _put_index(store, "h7", {"url": "https://bit.ly/z", "alias_of": "hash_of_x"})
    return store, config


def test_analyze_reports_all_three_lenses(tmp_path):
    store, config = _seed(tmp_path)
    report = analyze(store, config)

    assert report.total_url_indexes == 7
    assert report.alias_indexes == 1
    assert report.canonical_captures == 6

    canon_keys = {c.key for c in report.canonicalization_clusters}
    assert "https://x.test/p" in canon_keys

    content_urls = {tuple(c.urls) for c in report.content_clusters}
    assert ("https://a.test/1", "https://b.test/2") in content_urls

    near_keys = {c.key for c in report.near_dup_clusters}
    assert "https://q.test/item" in near_keys


def test_analyze_ignores_reverse_content_index(tmp_path):
    store, config = _seed(tmp_path)
    # A by-content reverse index must not be mistaken for a URL index.
    store.put(
        "t/index/by-content/cX.json",
        json.dumps({"content_hash": "cX", "canonical_url_hash": "x"}).encode("utf-8"),
    )
    report = analyze(store, config)
    assert report.total_url_indexes == 7  # unchanged


def test_rebuild_content_index_is_dry_by_default(tmp_path):
    store, config = _seed(tmp_path)
    groups = rebuild_content_index(store, config, apply=False)
    assert groups >= 1
    # Dry run wrote nothing under by-content/.
    assert not list(store.list_keys("t/index/by-content/"))

    rebuild_content_index(store, config, apply=True)
    assert list(store.list_keys("t/index/by-content/"))
