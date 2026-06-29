from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.cost import (
    estimate_run_cost,
    price_per_capture,
)
from forecasting_tools.agents_and_tools.source_archive.models import StoredCapture
from forecasting_tools.agents_and_tools.source_archive.pipeline import (
    CaptureOutcome,
    PipelineSummary,
)


def _cap(url: str, fetcher: str) -> StoredCapture:
    return StoredCapture(url=url, url_hash="h", content_hash="c", fetcher=fetcher)


def _stored(url: str, fetcher: str) -> CaptureOutcome:
    return CaptureOutcome(url=url, status="stored", stored=_cap(url, fetcher))


def test_free_backends_cost_nothing():
    cfg = ArchiveConfig()
    for f in ("cloakbrowser", "playwright", "pdf", ""):
        assert price_per_capture(f, cfg) == 0.0


def test_paid_backends_priced_by_config():
    cfg = ArchiveConfig(hyperbrowser_use_proxy=True, firecrawl_proxy="basic")
    assert price_per_capture("hyperbrowser", cfg) == 10 * 0.001
    assert price_per_capture("firecrawl", cfg) == 1 * 0.00083

    cheap = ArchiveConfig(hyperbrowser_use_proxy=False, firecrawl_proxy="auto")
    assert price_per_capture("hyperbrowser", cheap) == 1 * 0.001
    assert price_per_capture("firecrawl", cheap) == 5 * 0.00083


def test_estimate_run_cost_breakdown():
    cfg = ArchiveConfig(hyperbrowser_use_proxy=True, firecrawl_proxy="basic")
    summary = PipelineSummary(
        outcomes=[
            _stored("u1", "cloakbrowser"),
            _stored("u2", "cloakbrowser"),
            _stored("u3", "hyperbrowser"),
            _stored("u4", "firecrawl"),
            CaptureOutcome(
                url="u5", status="cache_hit", stored=_cap("u5", "cloakbrowser")
            ),
            CaptureOutcome(url="u6", status="error", reason="boom"),
        ]
    )
    rc = estimate_run_cost(summary, cfg, run_id="r1")

    assert rc.archived == 5  # 4 stored + 1 cache_hit; the error doesn't count
    assert rc.paid_captures == 2  # hyperbrowser + firecrawl
    assert rc.total_usd == round(0.01 + 0.00083, 4)  # 0.0108 (4-dp rounding)
    by = {b.backend: b for b in rc.by_backend}
    assert by["cloakbrowser"].captures == 2 and by["cloakbrowser"].total_usd == 0.0
    assert by["hyperbrowser"].captures == 1 and by["hyperbrowser"].unit_usd == 0.01
