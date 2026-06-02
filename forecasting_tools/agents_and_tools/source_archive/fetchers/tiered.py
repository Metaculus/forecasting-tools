"""Tiered fetcher: self-hosted Playwright first, Firecrawl on failure.

A backend "fails" if it raises ``FetchError`` (couldn't render) OR its capture
fails the quality gate (404 / block page / thin content). The first capture that
passes the gate wins. If none pass, the last attempted capture is returned with
``quality_passed=False`` in its metadata so the pipeline can still record the
miss.
"""

from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.source_archive.fetchers.base import (
    Fetcher,
    FetchError,
)
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult
from forecasting_tools.agents_and_tools.source_archive.quality import evaluate

logger = logging.getLogger(__name__)


class TieredFetcher:
    name = "tiered"

    def __init__(self, *backends: Fetcher):
        if not backends:
            raise ValueError("TieredFetcher requires at least one backend")
        self.backends = backends

    def fetch(self, url: str) -> CaptureResult:
        last_result: CaptureResult | None = None
        errors: list[str] = []

        for backend in self.backends:
            try:
                result = backend.fetch(url)
            except FetchError as e:
                errors.append(f"{backend.name}: {e}")
                continue

            verdict = evaluate(result)
            result.metadata["quality_passed"] = verdict.passed
            result.metadata["quality_reason"] = verdict.reason
            if verdict.passed:
                return result
            last_result = result
            errors.append(f"{backend.name}: quality {verdict.reason}")

        if last_result is not None:
            logger.info(
                "all backends failed quality for %s: %s", url, "; ".join(errors)
            )
            return last_result
        raise FetchError(f"all backends failed for {url}: {'; '.join(errors)}")
