"""Per-run citation manifest: one JSONL record per (URL, citation).

This is the provenance layer a bot emits and the input to the capture pipeline.
One manifest per run, stored as ``manifests/<run_id>.jsonl`` in the blob store.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord
from forecasting_tools.agents_and_tools.source_archive.storage.blob_store import (
    BlobStore,
)


def dumps(records: Iterable[CitationRecord]) -> str:
    return "\n".join(json.dumps(r.model_dump(), sort_keys=True) for r in records)


def loads(text: str) -> list[CitationRecord]:
    out: list[CitationRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(CitationRecord.model_validate(json.loads(line)))
    return out


def unique_urls(records: Iterable[CitationRecord]) -> Iterator[str]:
    """Yield each distinct URL once, preserving first-seen order."""
    seen: set[str] = set()
    for r in records:
        if r.url and r.url not in seen:
            seen.add(r.url)
            yield r.url


# --- file io ---------------------------------------------------------------
def read_file(path: str | Path) -> list[CitationRecord]:
    return loads(Path(path).read_text(encoding="utf-8"))


def write_file(path: str | Path, records: Iterable[CitationRecord]) -> None:
    Path(path).write_text(dumps(records), encoding="utf-8")


# --- blob store io ---------------------------------------------------------
def manifest_key(run_id: str, config: ArchiveConfig | None = None) -> str:
    prefix = (config or ArchiveConfig()).s3_prefix.rstrip("/")
    return f"{prefix}/manifests/{run_id}.jsonl"


def read_blob(
    store: BlobStore, run_id: str, config: ArchiveConfig | None = None
) -> list[CitationRecord]:
    return loads(store.get(manifest_key(run_id, config)).decode("utf-8"))


def write_blob(
    store: BlobStore,
    run_id: str,
    records: Iterable[CitationRecord],
    config: ArchiveConfig | None = None,
) -> None:
    store.put(
        manifest_key(run_id, config),
        dumps(records).encode("utf-8"),
        content_type="application/x-ndjson",
    )
