"""Fetchers turn a URL into a CaptureResult (HTML + screenshot + markdown).

Most callers want :func:`build_default_fetcher`, which wires the recommended
tiered setup: self-hosted Playwright primary, Firecrawl fallback.
"""

from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import (
    Fetcher,
    FetchError,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.firecrawl_fetcher import (
    FirecrawlFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.playwright_fetcher import (
    PlaywrightFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.tiered import (
    TieredFetcher,
)

__all__ = [
    "Fetcher",
    "FetchError",
    "FirecrawlFetcher",
    "PlaywrightFetcher",
    "TieredFetcher",
    "build_default_fetcher",
]


def build_default_fetcher(config: ArchiveConfig | None = None) -> PlaywrightFetcher:
    """Return the recommended fetcher as a context manager.

    Use it like::

        with build_default_fetcher(config) as fetcher:
            fetcher.fetch(url)

    Playwright runs first; if a page fails to render or trips the quality gate
    and a Firecrawl API key is configured, Firecrawl is tried as a fallback.

    The returned object is a :class:`PlaywrightFetcher` so the browser lifecycle
    is managed by ``with``. On ``__enter__`` it transparently composes itself
    with Firecrawl (when available) behind a :class:`TieredFetcher`.
    """
    config = config or ArchiveConfig()
    return _ManagedTieredFetcher(config)


class _ManagedTieredFetcher(PlaywrightFetcher):
    """PlaywrightFetcher whose ``fetch`` is delegated to a tiered pipeline.

    Subclassing PlaywrightFetcher keeps the browser context-manager lifecycle
    while letting us add the Firecrawl fallback once the browser is live.
    """

    def __enter__(self) -> "_ManagedTieredFetcher":
        super().__enter__()
        backends: list[Fetcher] = [_PlaywrightOnly(self)]
        if self.config.firecrawl_api_key:
            backends.append(FirecrawlFetcher(self.config))
        self._tiered = TieredFetcher(*backends)
        return self

    def fetch(self, url: str):  # type: ignore[override]
        return self._tiered.fetch(url)


class _PlaywrightOnly:
    """Adapts a live PlaywrightFetcher to the Fetcher protocol for tiering,
    calling the un-overridden ``fetch`` so we don't recurse."""

    name = "playwright"

    def __init__(self, owner: PlaywrightFetcher):
        self._owner = owner

    def fetch(self, url: str):
        return PlaywrightFetcher.fetch(self._owner, url)
