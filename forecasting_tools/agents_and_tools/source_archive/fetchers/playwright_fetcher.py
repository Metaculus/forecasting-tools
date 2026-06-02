"""Self-hosted Playwright fetcher — the PRIMARY backend.

A single page load yields all three artifacts:

  - HTML       via ``page.content()``
  - screenshot via a full-page capture (height-capped, then compressed)
  - markdown   via trafilatura over the rendered HTML

Self-hosted compute is far cheaper than any managed scraping API, so this is the
default; Firecrawl is reserved for sites that block headless Chromium (see
``TieredFetcher``).

Playwright and trafilatura are optional and imported lazily, so importing this
module never requires a browser. Install everything with
``pip install forecasting-tools[source-archive]`` and then run
``playwright install chromium`` once to download the browser.
"""

from __future__ import annotations

import io
import logging

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import FetchError
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult

logger = logging.getLogger(__name__)


def _to_markdown(html: str, url: str) -> str | None:
    try:
        import trafilatura
    except ImportError:
        logger.warning("trafilatura not installed; markdown will be omitted")
        return None
    return trafilatura.extract(
        html, url=url, output_format="markdown", include_links=True
    )


def _encode_screenshot(png_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Re-encode a PNG screenshot to the requested format using Pillow.

    Pillow is already a forecasting-tools dependency, so true WebP is available
    here (Playwright itself only emits PNG/JPEG).
    """
    fmt = fmt.lower()
    if fmt == "png":
        return png_bytes, "image/png"
    try:
        from PIL import Image
    except ImportError:
        return png_bytes, "image/png"

    image = Image.open(io.BytesIO(png_bytes))
    out = io.BytesIO()
    if fmt == "webp":
        image.save(out, format="WEBP", quality=80, method=6)
        return out.getvalue(), "image/webp"
    if fmt in ("jpeg", "jpg"):
        image.convert("RGB").save(out, format="JPEG", quality=80, optimize=True)
        return out.getvalue(), "image/jpeg"
    return png_bytes, "image/png"


class PlaywrightFetcher:
    """Renders pages with a persistent headless Chromium.

    Use it as a context manager so the browser launches once and is reused
    across many URLs (throughput is thousands of pages/hour single-process)::

        with PlaywrightFetcher(config) as fetcher:
            for url in urls:
                fetcher.fetch(url)
    """

    name = "playwright"

    def __init__(self, config: ArchiveConfig | None = None):
        self.config = config or ArchiveConfig()
        self._playwright = None
        self._browser = None

    def __enter__(self) -> "PlaywrightFetcher":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise FetchError(
                "playwright is not installed. Install it with "
                "`pip install forecasting-tools[source-archive]` and then run "
                "`playwright install chromium`."
            ) from e
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        return self

    def __exit__(self, *exc) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def fetch(self, url: str) -> CaptureResult:
        if self._browser is None:
            raise FetchError("PlaywrightFetcher must be used as a context manager")

        context = self._browser.new_context()
        page = context.new_page()
        try:
            try:
                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.config.nav_timeout_ms,
                )
            except Exception as e:
                raise FetchError(f"navigation failed for {url}: {e}") from e

            status = response.status if response is not None else None
            html = page.content()

            shot_kwargs: dict = {"type": "png"}
            cap = self.config.screenshot_max_height
            dims = page.evaluate(
                "() => ({w: document.documentElement.scrollWidth,"
                " h: document.documentElement.scrollHeight})"
            )
            width = max(int(dims.get("w") or 0), 1)
            height = int(dims.get("h") or 0)
            if cap and height > cap:
                shot_kwargs["clip"] = {"x": 0, "y": 0, "width": width, "height": cap}
            else:
                shot_kwargs["full_page"] = True

            png = page.screenshot(**shot_kwargs)
            screenshot, content_type = _encode_screenshot(
                png, self.config.screenshot_format
            )

            return CaptureResult(
                url=url,
                final_url=page.url,
                status_code=status,
                html=html,
                markdown=_to_markdown(html, page.url),
                screenshot=screenshot,
                screenshot_content_type=content_type,
                fetcher=self.name,
                metadata={"title": page.title()},
            )
        finally:
            context.close()
