"""Capture pipeline: turn a list of cited URLs into archived captures.

For each unique URL:

  1. :meth:`ContentStore.lookup` — within the TTL? cache hit, skip the fetch.
  2. ``fetcher.fetch``           — tiered Playwright -> Firecrawl, quality-gated.
  3. quality gate                — junk (404 / block / thin) is not archived.
  4. :meth:`ContentStore.store`  — write blobs (deduped by content hash).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import (
    Fetcher,
    FetchError,
)
from forecasting_tools.agents_and_tools.source_archive.manifest import unique_urls
from forecasting_tools.agents_and_tools.source_archive.models import (
    CitationRecord,
    StoredCapture,
)
from forecasting_tools.agents_and_tools.source_archive.quality import evaluate

logger = logging.getLogger(__name__)

# "cache_hit" | "stored" | "deduped" | "quality_failed" | "error"
Status = str
_STATUSES = ("cache_hit", "stored", "deduped", "quality_failed", "error")


class CaptureOutcome(BaseModel):
    url: str
    status: Status
    stored: StoredCapture | None = None
    reason: str = ""


class PipelineSummary(BaseModel):
    outcomes: list[CaptureOutcome] = []

    def count(self, status: Status) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def captures(self) -> dict[str, StoredCapture]:
        return {o.url: o.stored for o in self.outcomes if o.stored is not None}

    def __str__(self) -> str:
        body = ", ".join(f"{s}={self.count(s)}" for s in _STATUSES)
        return f"PipelineSummary(total={len(self.outcomes)}, {body})"


class CapturePipeline:
    def __init__(self, fetcher: Fetcher, content_store: ContentStore):
        self.fetcher = fetcher
        self.content_store = content_store

    def capture_url(self, url: str) -> CaptureOutcome:
        cached = self.content_store.lookup(url)
        if cached is not None:
            return CaptureOutcome(url=url, status="cache_hit", stored=cached)

        try:
            result = self.fetcher.fetch(url)
        except FetchError as e:
            logger.info("fetch error for %s: %s", url, e)
            return CaptureOutcome(url=url, status="error", reason=str(e))

        # Gate here so any fetcher is covered; the tiered fetcher also gates
        # internally to decide fallback, but this is the authoritative check.
        verdict = evaluate(result)
        if not verdict.passed:
            return CaptureOutcome(
                url=url, status="quality_failed", reason=verdict.reason
            )

        store_result = self.content_store.store(result)
        status = "stored" if store_result.created else "deduped"
        return CaptureOutcome(url=url, status=status, stored=store_result.capture)

    def run(self, urls: Iterable[str]) -> PipelineSummary:
        summary = PipelineSummary()
        for url in urls:
            summary.outcomes.append(self.capture_url(url))
        return summary

    def run_manifest(self, records: Iterable[CitationRecord]) -> PipelineSummary:
        return self.run(unique_urls(records))
