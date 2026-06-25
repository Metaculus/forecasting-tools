"""Unit tests for the backup capture backends and the bake-off pricing model.

These mock the vendor SDKs so they run without API keys, network, browsers, or
the optional pymupdf/playwright/cloakbrowser packages installed.
"""

from __future__ import annotations

import base64

import pytest

from forecasting_tools.agents_and_tools.source_archive import benchmark as B
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers import (
    build_default_fetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import FetchError
from forecasting_tools.agents_and_tools.source_archive.fetchers.cloakbrowser_fetcher import (
    CloakBrowserFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.firecrawl_fetcher import (
    FirecrawlFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.hyperbrowser_fetcher import (
    HyperbrowserFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.pdf_fetcher import (
    PdfFetcher,
    looks_like_pdf,
)
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult


# --- Firecrawl proxy/stealth wiring ------------------------------------------
def test_firecrawl_basic_sends_no_proxy_key():
    f = FirecrawlFetcher(ArchiveConfig(firecrawl_proxy="basic"))
    assert "proxy" not in f._scrape_kwargs(["markdown"])


@pytest.mark.parametrize("mode", ["auto", "stealth", "enhanced"])
def test_firecrawl_stealth_sends_proxy_key(mode):
    f = FirecrawlFetcher(ArchiveConfig(firecrawl_proxy=mode))
    assert f._scrape_kwargs(["markdown"])["proxy"] == mode


def test_firecrawl_fetch_pdf_markdown():
    class FakeClient:
        def scrape(self, url, **kwargs):
            assert kwargs["formats"] == ["markdown"]
            return {"markdown": "# PDF body " + "x " * 200}

    f = FirecrawlFetcher(ArchiveConfig(firecrawl_api_key="k"), client=FakeClient())
    assert f.fetch_pdf_markdown("https://x/y.pdf").startswith("# PDF body")


# --- Hyperbrowser screenshot coercion + result mapping -----------------------
def test_hyperbrowser_coerce_screenshot_data_uri():
    raw = b"\x89PNG fake"
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    shot, ctype = HyperbrowserFetcher._coerce_screenshot(uri)
    assert shot == raw and ctype == "image/png"


def test_hyperbrowser_coerce_screenshot_bare_base64():
    raw = b"\x89PNG fake"
    shot, ctype = HyperbrowserFetcher._coerce_screenshot(base64.b64encode(raw).decode())
    assert shot == raw and ctype == "image/png"


def test_hyperbrowser_coerce_screenshot_none():
    assert HyperbrowserFetcher._coerce_screenshot(None) == (None, None)


def test_hyperbrowser_fetch_maps_result(monkeypatch):
    class Data:
        metadata = {"statusCode": 200, "title": "T", "sourceURL": "https://final"}
        html = "<html>ok</html>"
        markdown = "ok " * 100
        screenshot = None

    class Resp:
        status = "completed"
        error = None
        data = Data()

    class FakeClient:
        class scrape:
            @staticmethod
            def start_and_wait(params):
                return Resp()

    f = HyperbrowserFetcher(
        ArchiveConfig(hyperbrowser_api_key="k"), client=FakeClient()
    )
    # Avoid constructing real SDK request models in the unit test.
    monkeypatch.setattr(f, "_params", lambda url: None)
    result = f.fetch("https://x")
    assert result.fetcher == "hyperbrowser"
    assert result.final_url == "https://final"
    assert result.status_code == 200
    assert result.metadata["used_proxy"] is True


def test_hyperbrowser_failed_job_raises(monkeypatch):
    class Resp:
        status = "failed"
        error = "blocked"
        data = None

    class FakeClient:
        class scrape:
            @staticmethod
            def start_and_wait(params):
                return Resp()

    f = HyperbrowserFetcher(
        ArchiveConfig(hyperbrowser_api_key="k"), client=FakeClient()
    )
    monkeypatch.setattr(f, "_params", lambda url: None)
    with pytest.raises(FetchError):
        f.fetch("https://x")


def test_hyperbrowser_requires_key():
    with pytest.raises(FetchError):
        HyperbrowserFetcher(ArchiveConfig(hyperbrowser_api_key=None)).fetch("https://x")


# --- PDF fetcher -------------------------------------------------------------
def test_looks_like_pdf():
    assert looks_like_pdf("https://x/report.pdf")
    assert looks_like_pdf("https://x/report.PDF?v=2")
    assert not looks_like_pdf("https://x/report.html")


def test_pdf_rejects_non_pdf_bytes():
    f = PdfFetcher(
        ArchiveConfig(),
        downloader=lambda url, t: (b"<html>not a pdf</html>", url, 200),
    )
    with pytest.raises(FetchError):
        f.fetch("https://x/fake.pdf")


def test_pdf_falls_back_to_firecrawl_when_local_thin(monkeypatch):
    class FakeFirecrawl:
        def fetch_pdf_markdown(self, url):
            return "# Scanned doc recovered by OCR " + "y " * 200

    f = PdfFetcher(
        ArchiveConfig(),
        firecrawl=FakeFirecrawl(),
        downloader=lambda url, t: (b"%PDF- minimal", url, 200),
    )
    # Force the local parser to look thin regardless of whether pymupdf is present.
    monkeypatch.setattr(f, "_parse_local", lambda data: (None, None, None, 3, "none"))
    result = f.fetch("https://x/scan.pdf")
    assert result.metadata["pdf_engine"] == "firecrawl"
    assert "OCR" in result.markdown


def test_pdf_uses_local_when_text_is_rich(monkeypatch):
    f = PdfFetcher(
        ArchiveConfig(),
        downloader=lambda url, t: (b"%PDF- minimal", url, 200),
    )
    rich = "# Title\n" + "real body text " * 100
    monkeypatch.setattr(
        f, "_parse_local", lambda data: (rich, b"png", "image/png", 5, "pymupdf4llm")
    )
    result = f.fetch("https://x/clean.pdf")
    assert result.metadata["pdf_engine"] == "pymupdf4llm"
    assert result.metadata["pdf_pages"] == 5
    assert result.screenshot == b"png"


# --- CloakBrowser ------------------------------------------------------------
def test_cloakbrowser_missing_package_gives_clear_error(monkeypatch):
    # Force every import to fail so this passes whether or not cloakbrowser is
    # actually installed in the test environment.
    import forecasting_tools.agents_and_tools.source_archive.fetchers.cloakbrowser_fetcher as cb

    def _boom(name):
        raise ImportError(name)

    monkeypatch.setattr(cb.importlib, "import_module", _boom)
    f = CloakBrowserFetcher(ArchiveConfig())
    with pytest.raises(FetchError) as exc:
        f._launch_browser()
    assert "cloakbrowser" in str(exc.value).lower()


# --- Pricing model -----------------------------------------------------------
def test_pricing_self_host_is_floor():
    r = CaptureResult(url="u", final_url="u")
    assert B.estimate_cost("playwright", r, 1_000_000, B.Pricing()) == 0.00001
    assert B.estimate_cost("cloakbrowser", r, 1_000_000, B.Pricing()) == 0.00001


def test_pricing_firecrawl_basic_vs_stealth():
    basic = CaptureResult(url="u", final_url="u", metadata={"firecrawl_proxy": "basic"})
    stealth = CaptureResult(
        url="u", final_url="u", metadata={"firecrawl_proxy": "auto"}
    )
    assert B.estimate_cost("firecrawl", basic, 0, B.Pricing()) == pytest.approx(0.00083)
    assert B.estimate_cost(
        "firecrawl-stealth", stealth, 0, B.Pricing()
    ) == pytest.approx(0.00415)


def test_pricing_hyperbrowser_proxy_includes_bandwidth():
    r = CaptureResult(url="u", final_url="u", metadata={"used_proxy": True})
    # 10 credits ($0.01) + 1MB * $10/GB ($0.01) = $0.02
    assert B.estimate_cost("hyperbrowser", r, 1_000_000, B.Pricing()) == pytest.approx(
        0.02
    )


def test_pricing_pdf_local_is_free_firecrawl_is_per_page():
    local = CaptureResult(
        url="u", final_url="u", metadata={"pdf_engine": "pymupdf4llm"}
    )
    ocr = CaptureResult(
        url="u", final_url="u", metadata={"pdf_engine": "firecrawl", "pdf_pages": 10}
    )
    assert B.estimate_cost("pdf", local, 0, B.Pricing()) == 0.0
    assert B.estimate_cost("pdf", ocr, 0, B.Pricing()) == pytest.approx(0.0083)


# --- Default tiered chain composition ----------------------------------------
def _fake_browser():
    from unittest.mock import MagicMock

    return None, MagicMock()  # (playwright_handle, browser) — browser.close() ok


def test_default_chain_cloakbrowser_is_primary(monkeypatch):
    # CloakBrowser available -> it is the single self-hosted browser tier.
    monkeypatch.setattr(
        CloakBrowserFetcher, "_launch_browser", lambda self: _fake_browser()
    )
    config = ArchiveConfig(hyperbrowser_api_key="h", firecrawl_api_key="f")
    with build_default_fetcher(config) as fetcher:
        names = [b.name for b in fetcher._tiered.backends]
    # Note: exactly one browser tier (cloakbrowser), not vanilla + cloak.
    assert names == ["cloakbrowser", "pdf", "hyperbrowser", "firecrawl"]


def test_default_chain_falls_back_to_playwright_and_skips_unkeyed(monkeypatch):
    from forecasting_tools.agents_and_tools.source_archive.fetchers.playwright_fetcher import (
        PlaywrightFetcher,
    )

    # CloakBrowser not installed -> vanilla Playwright is the browser tier.
    def raise_unavailable(self):
        raise FetchError("cloakbrowser not installed")

    monkeypatch.setattr(CloakBrowserFetcher, "_launch_browser", raise_unavailable)
    monkeypatch.setattr(
        PlaywrightFetcher, "_launch_browser", lambda self: _fake_browser()
    )
    config = ArchiveConfig(hyperbrowser_api_key=None, firecrawl_api_key=None)
    with build_default_fetcher(config) as fetcher:
        names = [b.name for b in fetcher._tiered.backends]
    assert names == ["playwright", "pdf"]
