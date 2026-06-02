"""Core data structures shared across the source-archive pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def url_hash(url: str) -> str:
    """Stable key for a URL — groups every capture of that URL together."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def content_hash(html: str | bytes) -> str:
    """Hash of page content — dedups identical re-fetches of the same URL."""
    data = html.encode("utf-8") if isinstance(html, str) else html
    return hashlib.sha256(data).hexdigest()


class CaptureResult(BaseModel):
    """What a fetcher returns for a single URL, before it is stored."""

    url: str
    final_url: str
    status_code: int | None = None
    html: str | None = None
    markdown: str | None = None
    screenshot: bytes | None = None
    screenshot_content_type: str | None = None
    fetcher: str = ""
    fetched_at: str = Field(default_factory=utcnow_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        basis = self.html if self.html else (self.markdown or self.final_url)
        return content_hash(basis)


class StoredCapture(BaseModel):
    """Pointer to a stored capture in the object store."""

    url: str
    url_hash: str
    content_hash: str
    status_code: int | None = None
    fetcher: str = ""
    captured_at: str = Field(default_factory=utcnow_iso)
    html_key: str | None = None
    screenshot_key: str | None = None
    markdown_key: str | None = None
    first_seen: str = Field(default_factory=utcnow_iso)
    last_seen: str = Field(default_factory=utcnow_iso)


class CitationRecord(BaseModel):
    """One provenance record per (URL, citation) a bot emitted in a run.

    This is the manifest schema: a run produces a JSONL file of these, which is
    the input to the capture pipeline. Fields are deliberately generic so any
    bot's trace/comment format can be mapped onto them.
    """

    url: str
    run_id: str | None = None
    bot: str | None = None
    question_id: str | None = None
    metaculus_id: str | None = None
    question_url: str | None = None
    trace: str | None = None
    tool_name: str | None = None
    origin: str | None = None
    first_seen: str = Field(default_factory=utcnow_iso)
