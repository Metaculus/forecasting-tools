"""URL content store, keyed by URL + content hash, with a TTL cache.

The big cost lever is **not re-fetching** a URL captured recently: a bot
re-forecasts the same open question every 20-30 minutes for weeks, citing the
same pages over and over, so temporal overlap is near-total.

  - :meth:`ContentStore.lookup` — if a URL was captured within the TTL, return
    the pointer and skip the fetch entirely (the cheap path that makes re-runs
    nearly free).
  - :meth:`ContentStore.store` — write blobs under
    ``content/<url_hash>/<content_hash>.*``; if that exact content hash is
    already stored, skip the write (dedup identical re-fetches) and just refresh
    timestamps.

Object layout (under ``config.s3_prefix``)::

    index/<url_hash>.json                       per-URL index + capture history
    content/<url_hash>/<content_hash>.html
    content/<url_hash>/<content_hash>.<img_ext>
    content/<url_hash>/<content_hash>.md
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.models import (
    CaptureResult,
    StoredCapture,
    url_hash,
    utcnow_iso,
)
from forecasting_tools.agents_and_tools.source_archive.storage.blob_store import (
    BlobStore,
)

_IMG_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


class StoreResult(BaseModel):
    capture: StoredCapture
    created: bool  # False when the content hash was already stored (deduped)


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ContentStore:
    def __init__(self, blob_store: BlobStore, config: ArchiveConfig | None = None):
        self.blobs = blob_store
        self.config = config or ArchiveConfig()
        self.prefix = self.config.s3_prefix.rstrip("/")

    # --- key helpers -------------------------------------------------------
    def _index_key(self, uh: str) -> str:
        return f"{self.prefix}/index/{uh}.json"

    def _content_key(self, uh: str, ch: str, ext: str) -> str:
        return f"{self.prefix}/content/{uh}/{ch}.{ext}"

    # --- index io ----------------------------------------------------------
    def _read_index(self, uh: str) -> dict | None:
        key = self._index_key(uh)
        if not self.blobs.exists(key):
            return None
        return json.loads(self.blobs.get(key).decode("utf-8"))

    def _write_index(self, uh: str, index: dict) -> None:
        data = json.dumps(index, indent=2, sort_keys=True).encode("utf-8")
        self.blobs.put(self._index_key(uh), data, content_type="application/json")

    # --- public api --------------------------------------------------------
    def lookup(self, url: str) -> StoredCapture | None:
        """Return the latest stored capture if within the TTL, else ``None``.

        A non-``None`` return means callers can skip fetching this URL.
        """
        uh = url_hash(url)
        index = self._read_index(uh)
        if not index:
            return None
        latest_ch = index.get("latest_content_hash")
        captures = index.get("captures", {})
        latest = captures.get(latest_ch)
        if not latest:
            return None

        last_seen = _parse_iso(latest["last_seen"])
        age = datetime.now(timezone.utc) - last_seen
        if age > timedelta(days=self.config.ttl_days):
            return None
        return StoredCapture.model_validate(latest)

    def store(self, result: CaptureResult) -> StoreResult:
        """Persist a capture, deduping by content hash. Always updates the index."""
        uh = url_hash(result.url)
        ch = result.content_hash
        now = utcnow_iso()

        index = self._read_index(uh) or {
            "url": result.url,
            "url_hash": uh,
            "first_seen": now,
            "captures": {},
        }
        captures = index.setdefault("captures", {})
        existing = captures.get(ch)

        created = existing is None
        if existing is not None:
            # Identical content already stored — skip blob writes, refresh time.
            existing["last_seen"] = now
            stored = StoredCapture.model_validate(existing)
        else:
            html_key = screenshot_key = markdown_key = None
            if result.html is not None:
                html_key = self._content_key(uh, ch, "html")
                self.blobs.put(
                    html_key, result.html.encode("utf-8"), content_type="text/html"
                )
            if result.markdown is not None:
                markdown_key = self._content_key(uh, ch, "md")
                self.blobs.put(
                    markdown_key,
                    result.markdown.encode("utf-8"),
                    content_type="text/markdown",
                )
            if result.screenshot is not None:
                ext = _IMG_EXT.get(result.screenshot_content_type or "", "png")
                screenshot_key = self._content_key(uh, ch, ext)
                self.blobs.put(
                    screenshot_key,
                    result.screenshot,
                    content_type=result.screenshot_content_type,
                )
            stored = StoredCapture(
                url=result.url,
                url_hash=uh,
                content_hash=ch,
                status_code=result.status_code,
                fetcher=result.fetcher,
                captured_at=result.fetched_at,
                html_key=html_key,
                screenshot_key=screenshot_key,
                markdown_key=markdown_key,
                first_seen=now,
                last_seen=now,
            )
            captures[ch] = stored.model_dump()

        index["latest_content_hash"] = ch
        index["last_checked"] = now
        self._write_index(uh, index)
        return StoreResult(capture=stored, created=created)
