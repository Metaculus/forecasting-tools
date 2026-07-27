from __future__ import annotations

import pytest

from forecasting_tools.agents_and_tools.source_archive.canonicalize import (
    canonicalize_url,
)
from forecasting_tools.agents_and_tools.source_archive.models import url_hash

# (raw, expected canonical) — each pair documents one normalization rule.
CASES = [
    # fragment dropped
    ("https://a.test/x#section", "https://a.test/x"),
    # trailing slash dropped (non-root)
    ("https://a.test/x/", "https://a.test/x"),
    # root path collapses (with or without slash) to host only
    ("https://a.test/", "https://a.test"),
    ("https://a.test", "https://a.test"),
    # scheme + host lowercased, path case preserved
    ("HTTPS://A.TEST/Path", "https://a.test/Path"),
    # default ports stripped, non-default kept
    ("http://a.test:80/x", "http://a.test/x"),
    ("https://a.test:443/x", "https://a.test/x"),
    ("https://a.test:8443/x", "https://a.test:8443/x"),
    # tracking params removed, meaningful params kept
    ("https://a.test/x?utm_source=z&utm_medium=email", "https://a.test/x"),
    ("https://a.test/x?id=7&fbclid=abc", "https://a.test/x?id=7"),
    ("https://a.test/x?gclid=abc&igshid=q", "https://a.test/x"),
    # remaining params sorted (order-independent)
    ("https://a.test/x?b=2&a=1", "https://a.test/x?a=1&b=2"),
    # bare "ref"/"source" are intentionally preserved
    ("https://a.test/x?ref=home", "https://a.test/x?ref=home"),
    # combination
    (
        "HTTPS://A.TEST:443/Path/?b=2&utm_campaign=spring&a=1#frag",
        "https://a.test/Path?a=1&b=2",
    ),
    # non-http(s) left alone
    ("mailto:someone@a.test", "mailto:someone@a.test"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_canonicalize_cases(raw: str, expected: str):
    assert canonicalize_url(raw) == expected


@pytest.mark.parametrize("raw,_expected", CASES)
def test_canonicalize_is_idempotent(raw: str, _expected: str):
    once = canonicalize_url(raw)
    assert canonicalize_url(once) == once


def test_near_duplicates_share_a_url_hash():
    variants = [
        "https://a.test/article",
        "https://a.test/article/",
        "https://a.test/article#intro",
        "https://a.test/article?utm_source=newsletter",
        "HTTPS://A.test/article",
    ]
    hashes = {url_hash(v) for v in variants}
    assert len(hashes) == 1


def test_distinct_pages_keep_distinct_hashes():
    assert url_hash("https://a.test/x?id=1") != url_hash("https://a.test/x?id=2")
    assert url_hash("https://a.test/x") != url_hash("https://a.test/y")


def test_empty_and_none_safe():
    assert canonicalize_url("") == ""
