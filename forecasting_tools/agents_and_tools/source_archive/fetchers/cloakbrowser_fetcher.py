"""CloakBrowser fetcher — a self-hosted anti-bot upgrade to Playwright.

CloakBrowser (``CloakHQ/CloakBrowser``) is an open-source, patched-Chromium fork
whose ``cloakbrowser.launch()`` returns a standard Playwright ``Browser`` — so
this fetcher reuses *all* of ``PlaywrightFetcher``'s capture logic (settle,
autoscroll, full-page screenshot, trafilatura→markdown) and only overrides how
the browser is launched. The fork applies source-level fingerprint patches that
get past Cloudflare Turnstile and similar challenges that plain headless Chromium
trips; in the one rigorous 2026 anti-detect benchmark it cleared more Cloudflare
targets than vanilla Playwright.

It runs on your own compute, so the marginal service cost is ~$0/page. The
patched Chromium binary (~200MB) is downloaded automatically on first launch.

Install separately (it is not in the ``source-archive`` extra because of the
binary): ``pip install cloakbrowser``. The package module is configurable via
``config.cloakbrowser_import`` (default ``cloakbrowser``) in case it is renamed.
"""

from __future__ import annotations

import importlib
import logging

from forecasting_tools.agents_and_tools.source_archive.fetchers.base import FetchError
from forecasting_tools.agents_and_tools.source_archive.fetchers.playwright_fetcher import (
    PlaywrightFetcher,
)

logger = logging.getLogger(__name__)


class CloakBrowserFetcher(PlaywrightFetcher):
    name = "cloakbrowser"

    def _launch_browser(self):
        module = self._import_module()
        launch = getattr(module, "launch", None)
        if launch is None:
            raise FetchError(
                f"{module.__name__} has no launch(); the CloakBrowser API may "
                "have changed. Expected `cloakbrowser.launch() -> Browser`."
            )
        # stealth_args=True applies the fingerprint patches; the returned object
        # is a Playwright Browser, so the inherited fetch() drives it unchanged.
        # No separate playwright handle to stop — CloakBrowser owns its driver.
        browser = launch(headless=True, stealth_args=True)
        return None, browser

    def _import_module(self):
        candidates = [self.config.cloakbrowser_import, "cloakbrowser"]
        tried: list[str] = []
        for mod_name in dict.fromkeys(c for c in candidates if c):
            try:
                return importlib.import_module(mod_name)
            except ImportError:
                tried.append(mod_name)
        raise FetchError(
            "cloakbrowser is not installed. Install it with "
            "`pip install cloakbrowser` (or set WEB_ARCHIVE_CLOAKBROWSER_IMPORT "
            f"to the right module). Tried: {', '.join(tried)}."
        )
