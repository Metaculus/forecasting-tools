"""One-off reindex / dedup audit for an existing archive.

This walks the canonical per-URL indexes already in a store and reports how much
the smarter-dedup work (see ``ROADMAP.md`` Plan 1) would collapse, **without
mutating anything by default**. It answers the practical question: *after exact
canonicalization and content dedup, are there still many URLs that look like the
same page?* — i.e. whether the fuzzy near-dup phase (D) is worth building.

Three lenses:

  - **Canonicalization (Phase A):** group stored URLs by :func:`canonicalize_url`.
    Any group with >1 distinct raw URL is a set that *now* shares one key.
  - **Content (Phase C):** group distinct canonical URLs by their latest content
    hash. A group with >1 URL is byte-identical pages reachable at different URLs.
  - **Near-dup signal (Phase D candidate):** of the URLs surviving both dedups,
    group by ``scheme://host/path`` ignoring the query string. Big groups mean
    "same path, differing query" pages that exact dedup leaves separate — the
    cases fuzzy matching would target.

Run it::

    # against the configured S3 bucket (read-only audit)
    WEB_ARCHIVE_S3_BUCKET=metaculus-web-archive WEB_ARCHIVE_AWS_PROFILE=default \\
        python -m forecasting_tools.agents_and_tools.source_archive.reindex

    # against a local capture dir
    python -m forecasting_tools.agents_and_tools.source_archive.reindex --local ./archive

    # additionally (re)build the content reverse index for existing captures
    python -m forecasting_tools.agents_and_tools.source_archive.reindex --apply

``--apply`` only writes the additive ``index/by-content/`` reverse index (safe,
idempotent). It does **not** move blobs or re-key the per-URL indexes; that
heavier migration is intentionally deferred (the archive is young).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from urllib.parse import urlsplit

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive.canonicalize import (
    canonicalize_url,
)
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.storage.blob_store import (
    BlobStore,
)


class Cluster(BaseModel):
    key: str
    urls: list[str]


class AnalysisReport(BaseModel):
    total_url_indexes: int = 0
    alias_indexes: int = 0  # already-collapsed redirects (Phase B)
    canonical_captures: int = 0  # distinct stored URLs with content
    distinct_after_canonicalization: int = 0
    distinct_after_content_dedup: int = 0
    canonicalization_clusters: list[Cluster] = []  # raw URLs that now share a key
    content_clusters: list[Cluster] = []  # different URLs, identical content
    near_dup_clusters: list[Cluster] = []  # same host+path, differing query

    def __str__(self) -> str:
        merged_a = sum(len(c.urls) - 1 for c in self.canonicalization_clusters)
        merged_c = sum(len(c.urls) - 1 for c in self.content_clusters)
        lines = [
            "Source-archive dedup audit",
            "=" * 40,
            f"URL indexes scanned          : {self.total_url_indexes}",
            f"  of which alias (redirect)  : {self.alias_indexes}",
            f"  of which canonical capture : {self.canonical_captures}",
            "",
            f"Distinct URLs (raw)          : {self.canonical_captures}",
            f"After canonicalization (A)   : {self.distinct_after_canonicalization}"
            f"   (−{merged_a} merged)",
            f"After content dedup (C)      : {self.distinct_after_content_dedup}"
            f"   (−{merged_c} byte-identical)",
            "",
            f"Canonicalization clusters    : {len(self.canonicalization_clusters)}",
            f"Identical-content clusters   : {len(self.content_clusters)}",
            f"Near-dup candidates (D)      : {len(self.near_dup_clusters)}"
            "  (same host+path, differing query)",
        ]

        def _show(title: str, clusters: list[Cluster], limit: int = 5) -> None:
            if not clusters:
                return
            lines.append("")
            lines.append(f"--- top {title} ---")
            for c in sorted(clusters, key=lambda x: len(x.urls), reverse=True)[:limit]:
                lines.append(f"  [{len(c.urls)}] {c.key}")
                for u in c.urls[:4]:
                    lines.append(f"        {u}")
                if len(c.urls) > 4:
                    lines.append(f"        … +{len(c.urls) - 4} more")

        _show("canonicalization clusters", self.canonicalization_clusters)
        _show("identical-content clusters", self.content_clusters)
        _show("near-dup candidates (Phase D signal)", self.near_dup_clusters)
        return "\n".join(lines)


def _host_path(url: str) -> str:
    parts = urlsplit(canonicalize_url(url))
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def iter_url_indexes(store: BlobStore, prefix: str):
    """Yield ``(key, index_dict)`` for each per-URL index, skipping the reverse
    content index under ``index/by-content/``."""
    index_prefix = f"{prefix.rstrip('/')}/index/"
    content_sub = f"{index_prefix}by-content/"
    for key in store.list_keys(index_prefix):
        if not key.endswith(".json") or key.startswith(content_sub):
            continue
        try:
            yield key, json.loads(store.get(key).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def analyze(store: BlobStore, config: ArchiveConfig) -> AnalysisReport:
    report = AnalysisReport()
    by_canonical: dict[str, list[str]] = defaultdict(list)
    by_content: dict[str, list[str]] = defaultdict(list)

    for _key, index in iter_url_indexes(store, config.s3_prefix):
        report.total_url_indexes += 1
        if index.get("alias_of"):
            report.alias_indexes += 1
            continue
        url = index.get("url")
        if not url or not index.get("captures"):
            continue
        report.canonical_captures += 1
        by_canonical[canonicalize_url(url)].append(url)
        ch = index.get("latest_content_hash")
        if ch:
            by_content[ch].append(url)

    report.distinct_after_canonicalization = len(by_canonical)
    report.canonicalization_clusters = [
        Cluster(key=k, urls=sorted(set(v)))
        for k, v in by_canonical.items()
        if len(set(v)) > 1
    ]

    # Content dedup operates on the canonicalized URL set.
    content_groups = {k: sorted(set(v)) for k, v in by_content.items()}
    report.content_clusters = [
        Cluster(key=k, urls=v) for k, v in content_groups.items() if len(v) > 1
    ]
    # distinct pages after content dedup = canonical URLs minus those merged away
    merged_by_content = sum(len(v) - 1 for v in content_groups.values() if len(v) > 1)
    report.distinct_after_content_dedup = max(
        0, report.distinct_after_canonicalization - merged_by_content
    )

    # Phase D signal: among canonical URLs, same host+path but differing query.
    survivors = {canonicalize_url(u) for grp in by_canonical.values() for u in grp}
    by_host_path: dict[str, set[str]] = defaultdict(set)
    for u in survivors:
        by_host_path[_host_path(u)].add(u)
    report.near_dup_clusters = [
        Cluster(key=k, urls=sorted(v)) for k, v in by_host_path.items() if len(v) > 1
    ]
    return report


def rebuild_content_index(
    store: BlobStore, config: ArchiveConfig, *, apply: bool
) -> int:
    """(Re)build ``index/by-content/`` from existing captures. Returns the number
    of content groups (that would be) written. Additive and idempotent."""
    cstore = ContentStore(store, config)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _key, index in iter_url_indexes(store, config.s3_prefix):
        if index.get("alias_of") or not index.get("captures"):
            continue
        uh = index.get("url_hash")
        url = index.get("url")
        ch = index.get("latest_content_hash")
        if uh and url and ch:
            groups[ch].append((uh, url))

    written = 0
    for ch, members in groups.items():
        written += 1
        if not apply:
            continue
        owner_uh, owner_url = members[0]
        # Re-register every member; the first becomes canonical owner.
        for uh, url in members:
            blob_keys = None
            if uh == owner_uh:
                cap = index_blob_keys(store, config, owner_uh, ch)
                blob_keys = cap
            cstore._register_content(ch, uh, url, blob_keys)
    return written


def index_blob_keys(
    store: BlobStore, config: ArchiveConfig, uh: str, ch: str
) -> dict | None:
    cstore = ContentStore(store, config)
    index = cstore._read_index(uh)
    if not index:
        return None
    cap = (index.get("captures") or {}).get(ch)
    if not cap:
        return None
    return {
        "html": cap.get("html_key"),
        "markdown": cap.get("markdown_key"),
        "screenshot": cap.get("screenshot_key"),
    }


def _build_store(local_dir: str | None, bucket: str | None, config: ArchiveConfig):
    if local_dir:
        from forecasting_tools.agents_and_tools.source_archive.storage import (
            LocalBlobStore,
        )

        return LocalBlobStore(local_dir)
    bucket = bucket or config.s3_bucket
    if not bucket:
        sys.exit(
            "No S3 bucket configured. Set WEB_ARCHIVE_S3_BUCKET (or pass --bucket), "
            "or use --local DIR."
        )
    from forecasting_tools.agents_and_tools.source_archive.storage import S3BlobStore

    return S3BlobStore(bucket, config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="source-archive-reindex",
        description="Audit (and optionally rebuild) dedup structures for an "
        "existing archive.",
    )
    parser.add_argument("--local", metavar="DIR", help="audit a local capture dir")
    parser.add_argument("--bucket", help="override WEB_ARCHIVE_S3_BUCKET")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rebuild index/by-content/ for existing captures (additive)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    config = ArchiveConfig.from_env()
    store = _build_store(args.local, args.bucket, config)

    report = analyze(store, config)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(report)

    if args.apply:
        n = rebuild_content_index(store, config, apply=True)
        print(f"\nRebuilt index/by-content/ for {n} content group(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
