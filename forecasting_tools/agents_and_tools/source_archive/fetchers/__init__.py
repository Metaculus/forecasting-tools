"""Fetchers turn a URL into a CaptureResult (HTML + screenshot + markdown).

Most callers want :func:`build_default_fetcher`, which wires the recommended
cost-ordered tiered setup: self-hosted Playwright primary, then CloakBrowser,
PDF, Firecrawl, and Hyperbrowser backups.
"""

from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import (
    Fetcher,
    FetchError,
)
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
from forecasting_tools.agents_and_tools.source_archive.fetchers.playwright_fetcher import (
    PlaywrightFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.tiered import (
    TieredFetcher,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Fetcher",
    "FetchError",
    "CloakBrowserFetcher",
    "FirecrawlFetcher",
    "HyperbrowserFetcher",
    "PdfFetcher",
    "PlaywrightFetcher",
    "TieredFetcher",
    "looks_like_pdf",
    "build_default_fetcher",
]


def build_default_fetcher(config: ArchiveConfig | None = None) -> PlaywrightFetcher:
    """Return the recommended fetcher as a context manager.

    Use it like::

        with build_default_fetcher(config) as fetcher:
            fetcher.fetch(url)

    Backends are tried in **cost order** — the first capture that passes the
    quality gate wins, so the cheap tiers absorb most of the tail and the paid
    ones only fire on what survives them:

    1. **Self-hosted browser** (~free) — the primary; ~70% of URLs. Uses
       **CloakBrowser** (patched Chromium; matches-or-beats vanilla Playwright on
       anti-bot) when installed, else falls back to vanilla **Playwright**. Only
       one browser is used: two live ``sync_playwright`` instances conflict in a
       single process, so cloak *replaces* vanilla rather than stacking with it.
    2. **PdfFetcher** (local, free; Firecrawl OCR fallback) — captures PDFs,
       which the browsers can't render.
    3. **Firecrawl** (managed) — cheapest stealth + native-PDF safety net
       (~$0.0042/page stealth). Added when ``FIRECRAWL_API_KEY`` is set.
    4. **Hyperbrowser** (managed) — anti-bot fallback of last resort (~$0.01/page
       with proxy, plus bandwidth). Added when ``HYPERBROWSER_API_KEY`` is set.

    The returned object is a :class:`PlaywrightFetcher` subclass so the single
    browser's lifecycle is managed by ``with``.
    """
    config = config or ArchiveConfig()
    return _ManagedTieredFetcher(config)


class _ManagedTieredFetcher(PlaywrightFetcher):
    """PlaywrightFetcher whose ``fetch`` is delegated to a cost-ordered tiered
    pipeline. The single self-hosted browser is CloakBrowser when available
    (overriding ``_launch_browser``), else vanilla Playwright; the extra backends
    are composed once it is live.
    """

    _primary_name = "playwright"

    def _launch_browser(self):
        # Prefer CloakBrowser (patched Chromium, beats vanilla on anti-bot) as
        # the one self-hosted browser. Two live sync_playwright instances in a
        # process conflict, so cloak REPLACES vanilla here rather than stacking.
        try:
            browser = CloakBrowserFetcher(self.config)._launch_browser()
            self._primary_name = "cloakbrowser"
            return browser
        except FetchError as e:
            logger.info("cloakbrowser unavailable, using vanilla Playwright: %s", e)
            self._primary_name = "playwright"
            return super()._launch_browser()

    def __enter__(self) -> "_ManagedTieredFetcher":
        super().__enter__()  # launches the chosen browser via _launch_browser
        backends: list[Fetcher] = [_PrimaryBrowser(self, self._primary_name)]

        # PDFs: free local parse (Firecrawl OCR fallback wired internally when a
        # key is present). Cheap to keep in the chain unconditionally.
        backends.append(PdfFetcher(self.config))

        if self.config.firecrawl_api_key:
            backends.append(FirecrawlFetcher(self.config))
        if self.config.hyperbrowser_api_key:
            backends.append(HyperbrowserFetcher(self.config))

        self._tiered = TieredFetcher(*backends)
        return self

    def fetch(self, url: str):  # type: ignore[override]
        return self._tiered.fetch(url)


class _PrimaryBrowser:
    """Adapts the live primary browser to the Fetcher protocol for tiering,
    calling the un-overridden ``fetch`` so we don't recurse, and labelling the
    capture with the actual browser used (cloakbrowser/playwright)."""

    def __init__(self, owner: PlaywrightFetcher, name: str):
        self._owner = owner
        self.name = name

    def fetch(self, url: str):
        result = PlaywrightFetcher.fetch(self._owner, url)
        result.fetcher = self.name
        return result
