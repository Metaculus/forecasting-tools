"""Estimate what a capture run cost — per backend and per archived site.

Self-hosted browsers (CloakBrowser / Playwright) and local PDF parsing are ~free;
the managed backends (Hyperbrowser, Firecrawl) bill per page. This turns a run's
outcomes into a cost breakdown so an operator can see what the paid backends are
costing per site archived.

Costs are **estimates** from each vendor's public pricing applied to the
configured proxy mode — we record the backend that produced each capture, not the
live credit count — so treat them as close approximations, not billed amounts.
Only *successful* captures are priced; a paid backend call that then failed the
quality gate isn't attributed to a backend here (so this slightly under-counts).
"""

from __future__ import annotations

import json
from collections import Counter

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive import layout
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig

# $ per vendor credit (public list pricing, 2026-06).
_FIRECRAWL_CREDIT_USD = 0.00083
_HYPERBROWSER_CREDIT_USD = 0.001

# Backends that run on our own machine — no per-page charge.
_FREE_BACKENDS = {"cloakbrowser", "playwright", "pdf", ""}


def price_per_capture(fetcher: str, config: ArchiveConfig) -> float:
    """Estimated $ for one successful capture by ``fetcher`` under ``config``."""
    f = (fetcher or "").lower()
    if f in _FREE_BACKENDS:
        return 0.0
    if f == "hyperbrowser":
        credits = 10 if config.hyperbrowser_use_proxy else 1
        return credits * _HYPERBROWSER_CREDIT_USD
    if f.startswith("firecrawl"):
        basic = (config.firecrawl_proxy or "basic").lower() in ("", "basic")
        return (1 if basic else 5) * _FIRECRAWL_CREDIT_USD
    return 0.0  # unknown backend — assume free rather than invent a number


class BackendCost(BaseModel):
    backend: str
    captures: int
    unit_usd: float
    total_usd: float


class RunCost(BaseModel):
    run_id: str | None = None
    archived: int = 0  # sites we now hold (stored + deduped + cache_hit)
    paid_captures: int = 0  # captures via a paid backend this run
    total_usd: float = 0.0
    usd_per_archived: float = 0.0
    by_backend: list[BackendCost] = []

    def __str__(self) -> str:
        lines = [
            f"RunCost(run_id={self.run_id}, archived={self.archived}, "
            f"paid_captures={self.paid_captures}, total=${self.total_usd:.4f}, "
            f"$/archived=${self.usd_per_archived:.5f})",
            f"  {'backend':<14}{'captures':>9}{'$/capture':>12}{'$ total':>10}",
        ]
        for b in self.by_backend:
            lines.append(
                f"  {b.backend:<14}{b.captures:>9}{b.unit_usd:>12.5f}{b.total_usd:>10.4f}"
            )
        return "\n".join(lines)


def estimate_run_cost(
    summary, config: ArchiveConfig, run_id: str | None = None
) -> RunCost:
    """Estimate a :class:`PipelineSummary`'s cost, broken down by backend.

    Newly fetched captures (``stored`` / ``deduped``) are priced by the backend
    that produced them; ``cache_hit`` re-uses cost nothing (no fetch happened).
    """
    counts: Counter[str] = Counter()
    for o in summary.outcomes:
        if o.status in ("stored", "deduped") and o.stored is not None:
            counts[(o.stored.fetcher or "unknown")] += 1

    by_backend: list[BackendCost] = []
    total = 0.0
    paid = 0
    for backend, n in sorted(counts.items()):
        unit = price_per_capture(backend, config)
        sub = unit * n
        total += sub
        if unit > 0:
            paid += n
        by_backend.append(
            BackendCost(
                backend=backend,
                captures=n,
                unit_usd=round(unit, 6),
                total_usd=round(sub, 4),
            )
        )

    archived = sum(
        1 for o in summary.outcomes if o.status in ("stored", "deduped", "cache_hit")
    )
    return RunCost(
        run_id=run_id,
        archived=archived,
        paid_captures=paid,
        total_usd=round(total, 4),
        usd_per_archived=round(total / archived, 6) if archived else 0.0,
        by_backend=by_backend,
    )


def cost_report_key(
    run_id: str, config: ArchiveConfig, group: str | None = None
) -> str:
    prefix = config.s3_prefix.rstrip("/")
    return f"{prefix}/{layout.report_key(run_id, '_cost.json', group)}"


def write_cost_report(
    store, run_id: str, cost: RunCost, config: ArchiveConfig, group: str | None = None
) -> str:
    """Persist the cost breakdown next to the run report (``<id>_cost.json``)."""
    key = cost_report_key(run_id, config, group)
    store.put(
        key,
        json.dumps(cost.model_dump(), indent=2).encode("utf-8"),
        content_type="application/json",
    )
    return key
