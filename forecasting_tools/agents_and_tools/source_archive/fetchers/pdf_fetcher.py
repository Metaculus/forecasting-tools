"""PDF fetcher — closes the gap Playwright can't.

Headless Chromium *downloads* a PDF instead of rendering it (``page.goto`` raises
"Download is starting"), and trafilatura doesn't parse PDFs, so a cited ``.pdf``
URL produces nothing today. This fetcher fills that hole with a two-tier strategy:

  1. Download the PDF bytes and parse locally with **PyMuPDF4LLM** — free, fast,
     CPU-only, and excellent on text-layer PDFs (most government/legal reports).
     The first page is rendered to an image so the viewer still has a screenshot.
  2. If local extraction yields thin text (a scanned PDF that needs OCR), fall
     back to **Firecrawl's** native PDF parser (~1 credit/page, OCR included).

Both parsers are optional and imported lazily. Use :func:`looks_like_pdf` /
:meth:`PdfFetcher.is_pdf` to decide whether a URL should be routed here.
"""

from __future__ import annotations

import logging
import urllib.request

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import FetchError
from forecasting_tools.agents_and_tools.source_archive.fetchers.firecrawl_fetcher import (
    FirecrawlFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult
from forecasting_tools.agents_and_tools.source_archive.quality import MIN_TEXT_LEN

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"


def looks_like_pdf(url: str) -> bool:
    """Cheap URL-shape heuristic: does this look like a PDF link? (The fetcher
    still confirms by sniffing the magic bytes before parsing.)"""
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(".pdf")


class PdfFetcher:
    name = "pdf"

    def __init__(
        self,
        config: ArchiveConfig | None = None,
        *,
        firecrawl: FirecrawlFetcher | None = None,
        downloader=None,
    ):
        self.config = config or ArchiveConfig()
        # Reuse the configured Firecrawl client for the OCR fallback when a key
        # is present; otherwise the fallback is simply skipped.
        if firecrawl is not None:
            self._firecrawl = firecrawl
        elif self.config.firecrawl_api_key:
            self._firecrawl = FirecrawlFetcher(self.config)
        else:
            self._firecrawl = None
        self._download = downloader or _download_bytes

    def is_pdf(self, url: str, data: bytes | None = None) -> bool:
        if data is not None:
            return data[:5] == _PDF_MAGIC
        return looks_like_pdf(url)

    def fetch(self, url: str) -> CaptureResult:
        data, final_url, status = self._download(url, self.config.nav_timeout_ms)
        if not data or data[:5] != _PDF_MAGIC:
            raise FetchError(f"not a PDF (no %PDF- magic) for {url}")

        markdown, screenshot, ctype, pages, engine = self._parse_local(data)

        thin = not markdown or len(markdown.strip()) < MIN_TEXT_LEN
        if thin and self._firecrawl is not None:
            logger.info("local PDF parse thin for %s; trying Firecrawl OCR", url)
            try:
                fc_md = self._firecrawl.fetch_pdf_markdown(url)
            except FetchError as e:
                logger.info("firecrawl PDF fallback failed for %s: %s", url, e)
            else:
                if fc_md and len(fc_md.strip()) >= MIN_TEXT_LEN:
                    markdown, engine = fc_md, "firecrawl"

        return CaptureResult(
            url=url,
            final_url=final_url or url,
            status_code=status,
            html=None,
            markdown=markdown,
            screenshot=screenshot,
            screenshot_content_type=ctype,
            fetcher=self.name,
            metadata={"pdf_engine": engine, "pdf_pages": pages},
        )

    def _parse_local(
        self, data: bytes
    ) -> tuple[str | None, bytes | None, str | None, int, str]:
        """Return (markdown, screenshot_png, content_type, pages, engine)."""
        try:
            import pymupdf  # PyMuPDF (a.k.a. fitz)
            import pymupdf4llm
        except ImportError:
            logger.warning(
                "pymupdf4llm not installed; local PDF parsing unavailable. "
                "Install with `pip install forecasting-tools[source-archive]`."
            )
            return None, None, None, 0, "none"

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as e:
            raise FetchError(f"could not open PDF: {e}") from e

        try:
            total = doc.page_count
            limit = min(total, self.config.pdf_max_pages) or total
            markdown = pymupdf4llm.to_markdown(doc, pages=list(range(limit)))
            screenshot, ctype = self._render_first_page(doc)
            return markdown or None, screenshot, ctype, total, "pymupdf4llm"
        finally:
            doc.close()

    @staticmethod
    def _render_first_page(doc) -> tuple[bytes | None, str | None]:
        try:
            pix = doc[0].get_pixmap(dpi=110)
            return pix.tobytes("png"), "image/png"
        except Exception as e:
            logger.info("could not render PDF first page: %s", e)
            return None, None


def _download_bytes(
    url: str, timeout_ms: int
) -> tuple[bytes | None, str | None, int | None]:
    # A browser-ish UA avoids the cheapest 403s; the content store needs the
    # bytes, not a render, so plain HTTP is fine and free.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=max(timeout_ms / 1000, 1)) as resp:
            return resp.read(), resp.geturl(), getattr(resp, "status", 200)
    except Exception as e:
        raise FetchError(f"could not download PDF for {url}: {e}") from e
