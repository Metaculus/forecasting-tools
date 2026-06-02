from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.fetchers.tiered import (
    TieredFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult
from forecasting_tools.agents_and_tools.source_archive.quality import evaluate


def _cap(**kw) -> CaptureResult:
    base = dict(url="u", final_url="u", status_code=200, html=None, markdown="x " * 200)
    base.update(kw)
    return CaptureResult(**base)


def test_quality_passes_real_page():
    assert evaluate(_cap()).passed


def test_quality_fails_404():
    assert not evaluate(_cap(status_code=404)).passed


def test_quality_fails_thin_content():
    assert not evaluate(_cap(markdown="short")).passed


def test_quality_fails_block_page():
    v = evaluate(_cap(markdown="Attention Required! | Cloudflare " * 20))
    assert not v.passed
    assert "block_signature" in v.reason


def test_tiered_falls_back_to_secondary_on_quality_fail(make_fetcher):
    primary = make_fetcher("primary")
    primary.add("https://blocked.test", markdown="please enable javascript " * 20)
    secondary = make_fetcher("secondary")
    secondary.add("https://blocked.test")

    result = TieredFetcher(primary, secondary).fetch("https://blocked.test")
    assert result.fetcher == "secondary"
    assert result.metadata["quality_passed"] is True


def test_tiered_falls_back_on_fetch_error(make_fetcher):
    primary = make_fetcher("primary")  # no canned response -> FetchError
    secondary = make_fetcher("secondary")
    secondary.add("https://x.test")

    result = TieredFetcher(primary, secondary).fetch("https://x.test")
    assert result.fetcher == "secondary"


def test_tiered_returns_failed_capture_when_all_fail(make_fetcher):
    primary = make_fetcher("primary")
    primary.add("https://x.test", status_code=404)
    secondary = make_fetcher("secondary")
    secondary.add("https://x.test", status_code=500)

    result = TieredFetcher(primary, secondary).fetch("https://x.test")
    assert result.metadata["quality_passed"] is False
