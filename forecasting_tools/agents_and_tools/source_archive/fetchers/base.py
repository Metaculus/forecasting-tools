"""Fetcher interface.

A fetcher turns a URL into a ``CaptureResult`` (HTML + markdown + screenshot in
one pass). Implementations: self-hosted Playwright (primary) and Firecrawl
(fallback).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult


class FetchError(Exception):
    """Raised when a fetcher cannot produce a capture at all (network/render
    failure). Quality problems with an otherwise-successful fetch are not errors
    — those are handled by the quality gate."""


@runtime_checkable
class Fetcher(Protocol):
    name: str

    def fetch(self, url: str) -> CaptureResult: ...
