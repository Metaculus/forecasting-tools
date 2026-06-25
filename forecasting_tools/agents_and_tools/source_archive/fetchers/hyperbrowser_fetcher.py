"""Hyperbrowser fetcher — a managed FALLBACK backend.

Hyperbrowser exposes a Firecrawl-style ``scrape`` endpoint that returns
markdown + HTML + a screenshot in one call, with optional stealth, residential
proxy, and CAPTCHA-solving session options for getting past Cloudflare and other
anti-bot filters.

Why it's here even though Firecrawl already is: forecasting-tools already uses
Hyperbrowser elsewhere (``research/computer_use.py``), so routing the anti-bot
tail through it consolidates spend onto one vendor/bill.

Cost note: a basic scrape is 1 credit ($0.001); enabling ``use_proxy`` makes it
10 credits ($0.01) plus proxy bandwidth ($10/GB). So the proxy/stealth session
is opt-in and meant for the small hardened-Cloudflare residual, not every URL.
Hyperbrowser has no documented PDF→markdown path, so PDFs go to the dedicated
``PdfFetcher`` instead of here.

The SDK is optional and imported lazily; a screenshot may come back as a hosted
URL (downloaded to bytes) or inline base64.
"""

from __future__ import annotations

import base64
import binascii
import logging
import urllib.request

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import FetchError
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult

logger = logging.getLogger(__name__)


def _attr(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class HyperbrowserFetcher:
    name = "hyperbrowser"

    def __init__(self, config: ArchiveConfig | None = None, client=None):
        self.config = config or ArchiveConfig()
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.config.hyperbrowser_api_key:
            raise FetchError("HYPERBROWSER_API_KEY is not set")
        try:
            from hyperbrowser import Hyperbrowser
        except ImportError as e:
            raise FetchError(
                "hyperbrowser is not installed. Install it with "
                "`pip install forecasting-tools[source-archive]`."
            ) from e
        self._client = Hyperbrowser(api_key=self.config.hyperbrowser_api_key)
        return self._client

    def _params(self, url: str):
        """Build the SDK request objects. Imported here (not at module top) so
        importing this module never requires the SDK."""
        from hyperbrowser.models import (
            CreateSessionParams,
            ScrapeOptions,
            StartScrapeJobParams,
        )

        return StartScrapeJobParams(
            url=url,
            scrape_options=ScrapeOptions(
                formats=["markdown", "html", "screenshot"],
                only_main_content=False,
            ),
            session_options=CreateSessionParams(
                use_proxy=self.config.hyperbrowser_use_proxy,
                use_stealth=self.config.hyperbrowser_use_stealth,
                solve_captchas=self.config.hyperbrowser_solve_captchas,
            ),
        )

    def fetch(self, url: str) -> CaptureResult:
        client = self._get_client()
        try:
            resp = client.scrape.start_and_wait(self._params(url))
        except Exception as e:
            raise FetchError(f"hyperbrowser scrape failed for {url}: {e}") from e

        # The job wrapper carries status/error; the payload is on ``.data``.
        if _attr(resp, "status") == "failed":
            raise FetchError(
                f"hyperbrowser scrape failed for {url}: {_attr(resp, 'error')}"
            )
        data = _attr(resp, "data", resp)

        metadata = _attr(data, "metadata", {}) or {}
        status = _attr(metadata, "statusCode") or _attr(metadata, "status_code")
        final_url = _attr(metadata, "sourceURL") or _attr(metadata, "url") or url

        screenshot, content_type = self._coerce_screenshot(_attr(data, "screenshot"))

        return CaptureResult(
            url=url,
            final_url=final_url,
            status_code=int(status) if status is not None else None,
            html=_attr(data, "html"),
            markdown=_attr(data, "markdown"),
            screenshot=screenshot,
            screenshot_content_type=content_type,
            fetcher=self.name,
            metadata={
                "title": _attr(metadata, "title"),
                "used_proxy": self.config.hyperbrowser_use_proxy,
            },
        )

    @classmethod
    def _coerce_screenshot(cls, value) -> tuple[bytes | None, str | None]:
        """A screenshot may arrive as a hosted URL, a data: URI, or raw base64."""
        if not value or not isinstance(value, str):
            return None, None
        if value.startswith("http://") or value.startswith("https://"):
            return cls._download(value)
        if value.startswith("data:"):
            try:
                header, b64 = value.split(",", 1)
                ctype = header[5:].split(";", 1)[0] or "image/png"
                return base64.b64decode(b64), ctype
            except (ValueError, binascii.Error):
                return None, None
        try:
            return base64.b64decode(value, validate=True), "image/png"
        except (binascii.Error, ValueError):
            return None, None

    @staticmethod
    def _download(src_url: str) -> tuple[bytes | None, str | None]:
        try:
            with urllib.request.urlopen(src_url, timeout=30) as resp:
                return resp.read(), resp.headers.get("Content-Type", "image/png")
        except Exception as e:
            logger.warning("failed to download hyperbrowser screenshot: %s", e)
            return None, None
