"""Key layout for manifests and run reports in the blob store.

Manifests and reports used to be written flat (``manifests/<run_id>.jsonl``,
``reports/<run_id>.json``), which turns into one giant folder as runs
accumulate. New writes nest by run family instead:

- an explicit ``group`` pins the folder (e.g. ``sprints/myrun``);
- ``daily-YYYY-MM-DD`` run ids nest under ``daily/<YYYY-MM>/``;
- everything else lands under ``adhoc/``.

Readers that list by prefix (catalog, coverage) pick up both layouts for free;
exact-key readers should try :func:`manifest_key_candidates` /
:func:`report_key_candidates` — nested first, then the legacy flat key — so
archives written before the nesting keep working.

Keys here are store-relative (no ``s3_prefix``); callers prepend the prefix.
"""

from __future__ import annotations

import re

_DAILY_RUN_ID = re.compile(r"^daily-(\d{4}-\d{2})-\d{2}$")


def _folder(run_id: str, group: str | None) -> str:
    if group:
        return group.strip("/")
    match = _DAILY_RUN_ID.match(run_id)
    if match:
        return f"daily/{match.group(1)}"
    return "adhoc"


def manifest_key(run_id: str, group: str | None = None) -> str:
    """Nested key for a run's citation manifest."""
    return f"manifests/{_folder(run_id, group)}/{run_id}.jsonl"


def report_key(run_id: str, suffix: str, group: str | None = None) -> str:
    """Nested key for a run report artifact (``suffix`` e.g. ``.json``)."""
    return f"reports/{_folder(run_id, group)}/{run_id}{suffix}"


def legacy_manifest_key(run_id: str) -> str:
    return f"manifests/{run_id}.jsonl"


def legacy_report_key(run_id: str, suffix: str) -> str:
    return f"reports/{run_id}{suffix}"


def manifest_key_candidates(run_id: str, group: str | None = None) -> list[str]:
    """Where a run's manifest may live, preferred (nested) first."""
    return [manifest_key(run_id, group), legacy_manifest_key(run_id)]


def report_key_candidates(
    run_id: str, suffix: str, group: str | None = None
) -> list[str]:
    """Where a run's report may live, preferred (nested) first."""
    return [report_key(run_id, suffix, group), legacy_report_key(run_id, suffix)]
