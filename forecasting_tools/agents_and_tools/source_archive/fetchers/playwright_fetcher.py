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

# WebP's hard per-side pixel limit; taller captures must be cropped before encode.
_WEBP_MAX_DIM = 16383
# Above this total pixel count, skip the screenshot rather than decode it: a
# pathological full-page render (very tall × wide) costs minutes of CPU in Pillow
# for a screenshot that's nice-to-have, not essential.
_MAX_SCREENSHOT_PIXELS = 200_000_000


def _to_markdown(html: str, url: str) -> str | None:
    try:
        import trafilatura
    except ImportError:
        logger.warning("trafilatura not installed; markdown will be omitted")
        return None
    return trafilatura.extract(
        html, url=url, output_format="markdown", include_links=True
    )


# Scroll the document top-to-bottom (triggering lazy-loaded content) then back
# up, so a subsequent full-page screenshot captures the fully-rendered page.
_AUTOSCROLL_JS = """
async () => {
  await new Promise((resolve) => {
    let y = 0;
    const step = () => {
      window.scrollTo(0, y);
      y += 1000;
      if (y < document.body.scrollHeight) setTimeout(step, 40);
      else resolve();
    };
    step();
  });
  window.scrollTo(0, 0);
}
"""


def _encode_screenshot(
    png_bytes: bytes, fmt: str, max_height: int = 0
) -> tuple[bytes, str]:
    """Crop (to ``max_height``) and re-encode a PNG screenshot using Pillow.

    Pillow is already a forecasting-tools dependency, so true WebP is available
    here (Playwright itself only emits PNG/JPEG). The height cap is enforced by
    cropping the *full-page* render to its top ``max_height`` pixels — never via
    Playwright's ``clip`` (which, without ``full_page``, is bounded by the
    viewport and silently truncates tall pages to a single screen).
    """
    fmt = fmt.lower()
    try:
        from PIL import Image
    except ImportError:
        # No Pillow: can't crop or transcode; hand back the raw full-page PNG.
        return png_bytes, "image/png"

    image = Image.open(io.BytesIO(png_bytes))  # lazy: reads size, doesn't decode
    if image.width * image.height > _MAX_SCREENSHOT_PIXELS:
        raise ValueError(
            f"screenshot too large to encode ({image.width}x{image.height}px)"
        )
    # WebP cannot encode beyond 16383px on a side. Clamp the effective cap for
    # webp so an over-tall page degrades to a top-crop instead of crashing the
    # encoder mid-run (which would propagate out of fetch() and abort the URL).
    limit = max_height or 0
    if fmt == "webp":
        limit = min(limit or _WEBP_MAX_DIM, _WEBP_MAX_DIM)
    if limit and image.height > limit:
        image = image.crop((0, 0, image.width, limit))

    out = io.BytesIO()
    if fmt == "webp":
        image.save(out, format="WEBP", quality=80, method=6)
        return out.getvalue(), "image/webp"
    if fmt in ("jpeg", "jpg"):
        image.convert("RGB").save(out, format="JPEG", quality=80, optimize=True)
        return out.getvalue(), "image/jpeg"
    image.save(out, format="PNG", optimize=True)
    return out.getvalue(), "image/png"


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

    def _launch_browser(self):
        """Start the browser. Returns ``(playwright_or_none, browser)`` where
        ``browser`` is a Playwright ``Browser``. Subclasses override this to swap
        in a different stealth browser (see ``CloakBrowserFetcher``) while reusing
        all of the capture logic. A backend that manages its own driver returns
        ``None`` for the first element."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise FetchError(
                "playwright is not installed. Install it with "
                "`pip install forecasting-tools[source-archive]` and then run "
                "`playwright install chromium`."
            ) from e
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        return playwright, browser

    def __enter__(self) -> "PlaywrightFetcher":
        self._playwright, self._browser = self._launch_browser()
        return self

    def __exit__(self, *exc) -> None:
        # close() raises when the browser process is already gone (crashed or
        # killed by the pipeline's reaper). Still attempt stop(): it tears down
        # the driver and un-registers the sync API's event loop from this
        # thread — without that, the next sync_playwright().start() here fails
        # with "Sync API inside the asyncio loop".
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception as e:
            logger.info("browser close failed (already dead?): %s", e)
        finally:
            self._browser = None
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception as e:
            logger.info("playwright stop failed: %s", e)
        finally:
            self._playwright = None

    def _settle(self, page) -> None:
        """Best-effort: let the page finish rendering before the screenshot.

        ``page.goto`` only waits for ``domcontentloaded``, which fires before
        CSS/images/lazy content have laid out — capturing then yields a short,
        half-built page. Wait for the load/network to quiesce and scroll the
        document to force lazy content in, so the full-page capture is complete.
        Each step is bounded and swallows timeouts: rendering aids are
        nice-to-have, never fatal to the capture.
        """
        try:
            page.wait_for_load_state("load", timeout=self.config.nav_timeout_ms)
        except Exception:
            pass
        try:
            page.wait_for_load_state(
                "networkidle", timeout=min(self.config.nav_timeout_ms, 10_000)
            )
        except Exception:
            pass
        try:
            page.evaluate(_AUTOSCROLL_JS)
            page.wait_for_timeout(500)
        except Exception:
            pass

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

            self._settle(page)

            status = response.status if response is not None else None
            html = page.content()

            # Always capture the entire scrollable page in one shot — Playwright
            # stitches it internally. The height cap is applied afterward by
            # cropping in Pillow (see ``_encode_screenshot``). Fall back to a
            # viewport capture only if a full-page shot fails (e.g. a page taller
            # than Chromium's screenshot limit).
            try:
                png = page.screenshot(full_page=True)
            except Exception as e:
                logger.info("full-page screenshot failed for %s: %s", url, e)
                png = page.screenshot()
            # Encoding can fail on pathological pages (e.g. a 400M-pixel full-page
            # render trips Pillow's decompression-bomb guard). A screenshot is
            # nice-to-have — never lose the whole capture over it.
            try:
                screenshot, content_type = _encode_screenshot(
                    png,
                    self.config.screenshot_format,
                    self.config.screenshot_max_height,
                )
            except Exception as e:
                logger.info("screenshot encode failed for %s: %s", url, e)
                screenshot, content_type = None, None

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
