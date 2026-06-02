"""Firecrawl fetcher — the FALLBACK backend.

Reserved for sites that block headless Chromium. It costs ~1 credit/page even
with a screenshot, so it only runs when the primary backend fails or its capture
fails the quality gate.

The Firecrawl SDK is optional and imported lazily. The screenshot comes back as
a hosted URL, which we download to bytes.
"""

from __future__ import annotations

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


class FirecrawlFetcher:
    name = "firecrawl"

    def __init__(self, config: ArchiveConfig | None = None, client=None):
        self.config = config or ArchiveConfig()
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.config.firecrawl_api_key:
            raise FetchError("FIRECRAWL_API_KEY is not set")
        try:
            from firecrawl import Firecrawl
        except ImportError as e:
            raise FetchError(
                "firecrawl-py is not installed. Install it with "
                "`pip install forecasting-tools[source-archive]`."
            ) from e
        self._client = Firecrawl(api_key=self.config.firecrawl_api_key)
        return self._client

    def fetch(self, url: str) -> CaptureResult:
        client = self._get_client()
        try:
            doc = client.scrape(url, formats=["markdown", "html", "screenshot"])
        except Exception as e:
            raise FetchError(f"firecrawl scrape failed for {url}: {e}") from e

        metadata = _attr(doc, "metadata", {}) or {}
        status = _attr(metadata, "statusCode") or _attr(metadata, "status_code")
        final_url = _attr(metadata, "sourceURL") or _attr(metadata, "url") or url

        screenshot_url = _attr(doc, "screenshot")
        screenshot, content_type = None, None
        if screenshot_url:
            screenshot, content_type = self._download(screenshot_url)

        return CaptureResult(
            url=url,
            final_url=final_url,
            status_code=int(status) if status is not None else None,
            html=_attr(doc, "html"),
            markdown=_attr(doc, "markdown"),
            screenshot=screenshot,
            screenshot_content_type=content_type,
            fetcher=self.name,
            metadata={"title": _attr(metadata, "title")},
        )

    @staticmethod
    def _download(src_url: str) -> tuple[bytes | None, str | None]:
        try:
            with urllib.request.urlopen(src_url, timeout=30) as resp:
                return resp.read(), resp.headers.get("Content-Type", "image/png")
        except Exception as e:
            logger.warning("failed to download firecrawl screenshot: %s", e)
            return None, None
