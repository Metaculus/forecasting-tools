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

**Redirect aliasing.** A capture is keyed by its *final* URL (after redirects),
so a link shortener (``bit.ly/x``) and the page it resolves to collapse onto one
capture instead of two. The original cited URL gets a tiny **alias index** that
points at the final URL's index, and the final URL's index lists its aliases for
provenance. So ``lookup(bit.ly/x)`` and ``lookup(final)`` both hit the same
stored page, and we never store the destination twice.

**Cross-URL content dedup.** Different URLs that return byte-identical content
share the blobs rather than storing them three times each. The first URL to
store a given content owns the blobs; later URLs get a capture whose blob keys
point back at them and whose ``content_alias_of`` names the owner. A reverse
``index/by-content/<content_hash>.json`` tracks the owner and every member URL.

Object layout (under ``config.s3_prefix``)::

    index/<final_url_hash>.json        canonical: capture history + "aliases"
    index/<orig_url_hash>.json         alias: {"alias_of": <final_url_hash>}
    index/by-content/<content_hash>.json   reverse: owner + member urls
    content/<final_url_hash>/<content_hash>.html
    content/<final_url_hash>/<content_hash>.<img_ext>
    content/<final_url_hash>/<content_hash>.md
"""

from __future__ import annotations

import json
import threading
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


def _capture_is_complete(cap: dict) -> bool:
    """Whether a stored capture has every format we expect for its type.

    A browser capture is complete only with html + markdown + screenshot; a PDF
    (which has no screenshot) only needs its markdown. Used by :meth:`lookup` so
    an incomplete capture is re-fetched rather than treated as already done.
    """
    if (cap.get("fetcher") or "").lower() == "pdf":
        return bool(cap.get("markdown_key"))
    return bool(
        cap.get("html_key") and cap.get("markdown_key") and cap.get("screenshot_key")
    )


class ContentStore:
    def __init__(self, blob_store: BlobStore, config: ArchiveConfig | None = None):
        self.blobs = blob_store
        self.config = config or ArchiveConfig()
        self.prefix = self.config.s3_prefix.rstrip("/")
        # Serializes the shared by-content reverse index across capture threads
        # (concurrent runs). Per-URL index files are written by a single thread
        # each, so they don't need it; the by-content index can be contended when
        # different URLs return identical content.
        self._content_lock = threading.Lock()

    # --- key helpers -------------------------------------------------------
    def _index_key(self, uh: str) -> str:
        return f"{self.prefix}/index/{uh}.json"

    def _content_key(self, uh: str, ch: str, ext: str) -> str:
        return f"{self.prefix}/content/{uh}/{ch}.{ext}"

    def _content_index_key(self, ch: str) -> str:
        return f"{self.prefix}/index/by-content/{ch}.json"

    # --- index io ----------------------------------------------------------
    def _read_index(self, uh: str) -> dict | None:
        key = self._index_key(uh)
        if not self.blobs.exists(key):
            return None
        return json.loads(self.blobs.get(key).decode("utf-8"))

    def _write_index(self, uh: str, index: dict) -> None:
        data = json.dumps(index, indent=2, sort_keys=True).encode("utf-8")
        self.blobs.put(self._index_key(uh), data, content_type="application/json")

    def _read_content_index(self, ch: str) -> dict | None:
        key = self._content_index_key(ch)
        if not self.blobs.exists(key):
            return None
        try:
            return json.loads(self.blobs.get(key).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A concurrent writer may have left a partial local file mid-write;
            # treat as absent rather than crash. The locked path below is authoritative.
            return None

    def _register_content(
        self, ch: str, uh: str, url: str, blob_keys: dict | None
    ) -> None:
        """Record that ``uh`` produced content ``ch`` in the reverse index.

        The first URL to store a given content becomes its ``canonical_url_hash``
        and owns the blob keys; later URLs with identical content are added as
        ``members`` and reuse those blobs (see :meth:`store`). Locked so concurrent
        capture threads with identical content don't clobber each other's members.
        """
        with self._content_lock:
            reverse = self._read_content_index(ch)
            if reverse is None:
                reverse = {
                    "content_hash": ch,
                    "canonical_url_hash": uh,
                    "blob_keys": blob_keys or {},
                    "members": [],
                }
            members = reverse.setdefault("members", [])
            if not any(m.get("url_hash") == uh for m in members):
                members.append({"url_hash": uh, "url": url})
            data = json.dumps(reverse, indent=2, sort_keys=True).encode("utf-8")
            self.blobs.put(
                self._content_index_key(ch), data, content_type="application/json"
            )

    # --- public api --------------------------------------------------------
    def lookup(self, url: str) -> StoredCapture | None:
        """Return the latest stored capture if within the TTL, else ``None``.

        A non-``None`` return means callers can skip fetching this URL. If ``url``
        is an alias of a previously-redirected target, the alias is followed to
        the canonical capture.
        """
        uh = url_hash(url)
        index = self._read_index(uh)
        if not index:
            return None
        alias_of = index.get("alias_of")
        if alias_of:  # follow the alias to the canonical (final-URL) index
            index = self._read_index(alias_of)
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
        # Skip only a COMPLETE capture. A partial one (e.g. a failed screenshot
        # encode left screenshot_key=None) is treated as a miss so the next run
        # retries the missing format instead of skipping it forever. PDFs have no
        # screenshot by nature, so they only need their markdown.
        if not _capture_is_complete(latest):
            return None
        return StoredCapture.model_validate(latest)

    def store(self, result: CaptureResult) -> StoreResult:
        """Persist a capture, deduping by content hash. Always updates the index.

        The capture is keyed by the *final* URL (after redirects). If the cited
        URL differs from the final one, an alias index is written so the cited
        URL still resolves here, and the cited URL is recorded under the
        canonical index's ``aliases``.
        """
        final_url = result.final_url or result.url
        uh = url_hash(final_url)
        ch = result.content_hash
        now = utcnow_iso()

        index = self._read_index(uh) or {
            "url": final_url,
            "url_hash": uh,
            "first_seen": now,
            "captures": {},
        }
        captures = index.setdefault("captures", {})
        existing = captures.get(ch)

        created = existing is None
        if existing is not None:
            # Identical content already stored for THIS url — skip writes, touch.
            existing["last_seen"] = now
            stored = StoredCapture.model_validate(existing)
        else:
            reverse = self._read_content_index(ch)
            reuse = bool(
                reverse and reverse.get("canonical_url_hash") not in (None, uh)
            )
            if reuse:
                # Byte-identical content already stored under a DIFFERENT url —
                # point at its blobs instead of writing three more (cross-URL
                # content dedup); each url still keeps its own index history.
                bk = reverse.get("blob_keys", {})
                html_key = bk.get("html")
                markdown_key = bk.get("markdown")
                screenshot_key = bk.get("screenshot")
                content_alias_of = reverse["canonical_url_hash"]
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
                content_alias_of = None
            stored = StoredCapture(
                url=final_url,
                url_hash=uh,
                content_hash=ch,
                status_code=result.status_code,
                fetcher=result.fetcher,
                captured_at=result.fetched_at,
                html_key=html_key,
                screenshot_key=screenshot_key,
                markdown_key=markdown_key,
                content_alias_of=content_alias_of,
                first_seen=now,
                last_seen=now,
            )
            captures[ch] = stored.model_dump()
            self._register_content(
                ch,
                uh,
                final_url,
                blob_keys=(
                    None
                    if reuse
                    else {
                        "html": html_key,
                        "markdown": markdown_key,
                        "screenshot": screenshot_key,
                    }
                ),
            )

        index["latest_content_hash"] = ch
        index["last_checked"] = now

        # If the cited URL redirected to a different final URL, record the alias.
        orig_uh = url_hash(result.url)
        if orig_uh != uh:
            aliases = index.setdefault("aliases", [])
            if result.url not in aliases:
                aliases.append(result.url)

        self._write_index(uh, index)

        if orig_uh != uh:
            self._write_alias(orig_uh, result.url, uh, now)

        return StoreResult(capture=stored, created=created)

    def _write_alias(
        self, orig_uh: str, orig_url: str, final_uh: str, now: str
    ) -> None:
        """Write/refresh a pointer from a cited URL's hash to its final capture.

        Never clobbers a canonical index (one that already holds captures), so a
        URL fetched directly in the past keeps its own history.
        """
        existing = self._read_index(orig_uh)
        if existing and existing.get("captures"):
            return
        first_seen = existing.get("first_seen", now) if existing else now
        self._write_index(
            orig_uh,
            {
                "url": orig_url,
                "url_hash": orig_uh,
                "alias_of": final_uh,
                "first_seen": first_seen,
                "last_checked": now,
            },
        )
