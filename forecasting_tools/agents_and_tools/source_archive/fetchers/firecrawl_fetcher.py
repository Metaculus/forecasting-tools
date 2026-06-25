"""Firecrawl fetcher — a managed FALLBACK backend.

Reserved for sites that block headless Chromium. A basic scrape costs 1 credit/
page even with a screenshot, so it only runs when the primary backend fails or
its capture fails the quality gate.

For hardened anti-bot sites, set ``config.firecrawl_proxy`` to ``"auto"`` or
``"stealth"`` (a.k.a. "enhanced") — this routes through residential proxies and
is billed at ~5 credits/page, so it is opt-in and reserved for the Cloudflare
tier. Firecrawl also natively parses PDFs to markdown (1 credit per PDF page),
which is why it is the fallback for the tiered ``PdfFetcher``.

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

    def _scrape_kwargs(self, formats: list[str]) -> dict:
        kwargs: dict = {"formats": formats}
        # Firecrawl 4.x renamed "stealth" to the "enhanced" proxy mode but still
        # accepts the legacy spelling; pass whatever the operator configured and
        # let the SDK normalize. "basic" is the implicit default, so only send
        # the param when something stronger is requested (keeps the call 1-credit
        # unless the operator explicitly opts into the 5-credit proxy).
        proxy = (self.config.firecrawl_proxy or "basic").strip().lower()
        if proxy and proxy != "basic":
            kwargs["proxy"] = proxy
        return kwargs

    def fetch(self, url: str) -> CaptureResult:
        client = self._get_client()
        try:
            doc = client.scrape(
                url, **self._scrape_kwargs(["markdown", "html", "screenshot"])
            )
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
            metadata={
                "title": _attr(metadata, "title"),
                "firecrawl_proxy": self.config.firecrawl_proxy,
            },
        )

    def fetch_pdf_markdown(self, url: str) -> str | None:
        """Scrape just the markdown for a PDF URL via Firecrawl's native PDF
        parser. Used as the fallback inside :class:`PdfFetcher` when local
        extraction yields thin text (e.g. a scanned PDF needing OCR)."""
        client = self._get_client()
        try:
            doc = client.scrape(url, **self._scrape_kwargs(["markdown"]))
        except Exception as e:
            raise FetchError(f"firecrawl pdf scrape failed for {url}: {e}") from e
        return _attr(doc, "markdown")

    @staticmethod
    def _download(src_url: str) -> tuple[bytes | None, str | None]:
        try:
            with urllib.request.urlopen(src_url, timeout=30) as resp:
                return resp.read(), resp.headers.get("Content-Type", "image/png")
        except Exception as e:
            logger.warning("failed to download firecrawl screenshot: %s", e)
            return None, None
