from __future__ import annotations

import pytest

from forecasting_tools.agents_and_tools.source_archive.fetchers.base import FetchError
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult


class FakeFetcher:
    """Returns canned CaptureResults by URL; raises FetchError for missing ones."""

    name = "fake"

    def __init__(self) -> None:
        self.responses: dict[str, CaptureResult] = {}
        self.calls: list[str] = []

    def add(
        self,
        url: str,
        *,
        html: str | None = None,
        markdown: str | None = None,
        status_code: int = 200,
        screenshot: bytes | None = b"\x89PNG fake",
    ) -> None:
        body = (
            html
            if html is not None
            else "<html><body>" + "content " * 80 + "</body></html>"
        )
        self.responses[url] = CaptureResult(
            url=url,
            final_url=url,
            status_code=status_code,
            html=body,
            markdown=markdown if markdown is not None else "content " * 80,
            screenshot=screenshot,
            screenshot_content_type="image/png",
            fetcher=self.name,
        )

    def fetch(self, url: str) -> CaptureResult:
        self.calls.append(url)
        if url not in self.responses:
            raise FetchError(f"no canned response for {url}")
        return self.responses[url]


@pytest.fixture
def make_fetcher():
    """Factory so a test can spin up one or several independent fake fetchers."""

    def _factory(name: str = "fake") -> FakeFetcher:
        f = FakeFetcher()
        f.name = name
        return f

    return _factory
