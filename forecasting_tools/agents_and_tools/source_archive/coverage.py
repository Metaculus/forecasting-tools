"""Coverage reports: what fraction of cited sources did we actually archive?

Two **separate** reports, split by ingestion path — they have different
denominators and different notions of ground truth, so they must not be blurred:

- ``trace`` — the complex/template bot's own instrumented runs (metac-ai-sdk).
  Traces record *every* URL the bot touched, so this is a true archival
  success-rate against ground truth.
- ``comments`` — every bot (Metaculus's own + outside bots) harvested from public
  Metaculus comments. Comments are length-truncated, so the denominator is itself
  incomplete: coverage here means "of the links we could *see* in comments, how
  many we archived" — a weaker claim than the trace report.

For each mode: denominator = distinct canonical **page** sources cited (tool/API
calls excluded, same rule as the catalog); numerator = those with a successful
capture in the index. Misses are bucketed by site — the per-URL failure *reason*
isn't persisted yet (that needs each run's pipeline outcomes saved), so we can
say *which* sites we miss, not yet *why*.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive.catalog import (
    Source,
    build_sources,
    exclusion_reason,
)
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.storage.blob_store import (
    BlobStore,
)

MODES = ("trace", "comments")
_COMMENT_ORIGINS = {"metaculus_comment"}


def citation_mode(citation) -> str:
    return "comments" if (citation.origin or "") in _COMMENT_ORIGINS else "trace"


class CoverageRow(BaseModel):
    label: str
    cited: int = 0
    captured: int = 0

    @property
    def pct(self) -> float:
        return round(100 * self.captured / self.cited, 1) if self.cited else 0.0


class CoverageReport(BaseModel):
    mode: str
    cited: int = 0
    captured: int = 0
    excluded: dict[str, int] = {}  # non-source reason -> count
    by_question: list[CoverageRow] = []
    by_bot: list[CoverageRow] = []
    by_tool: list[CoverageRow] = []
    missed_by_domain: list[CoverageRow] = []
    missing_urls: list[str] = []
    # Populated only when per-run outcomes (reports/) exist:
    has_outcomes: bool = False
    missing_never_fetched: int = 0  # the real collection gap
    missing_fetch_failed: int = 0  # tried, failed (Cloudflare/PDF/…)

    @property
    def pct(self) -> float:
        return round(100 * self.captured / self.cited, 1) if self.cited else 0.0

    @property
    def missing(self) -> int:
        return self.cited - self.captured

    def __str__(self) -> str:
        title = {
            "trace": "Trace coverage — complex/template bot (ground truth)",
            "comments": "Comment coverage — all bots (truncated denominator)",
        }.get(self.mode, self.mode)
        excl = (
            "  (excluded as non-sources: "
            + ", ".join(f"{k} {v}" for k, v in sorted(self.excluded.items()))
            + ")"
            if self.excluded
            else ""
        )
        lines = [
            title,
            "=" * len(title),
            # Lead with the collection gap: this report exists to tell us whether
            # there are sources bots are using that we are NOT yet archiving.
            f"{self.missing} of {self.cited} cited page sources are NOT yet in the "
            f"archive — candidates to collect  ({self.captured} archived, "
            f"{self.pct}%).",
            excl,
        ]
        if self.has_outcomes:
            lines.append(
                f"  of those {self.missing}: {self.missing_never_fetched} were "
                f"never fetched (collection gap), {self.missing_fetch_failed} "
                f"were fetched but failed."
            )
        if self.mode == "comments":
            lines.append(
                "  note: comments are length-truncated, so even this denominator "
                "under-counts what bots actually read — the true gap is larger."
            )

        def table(header: str, rows: list[CoverageRow], n: int = 8) -> None:
            if not rows:
                return
            lines.append("")
            lines.append(f"--- {header} ---")
            for r in rows[:n]:
                lines.append(f"  {r.captured:>4}/{r.cited:<4} {r.pct:>5}%  {r.label}")
            if len(rows) > n:
                lines.append(f"  … +{len(rows) - n} more")

        table("by question (most-cited first)", self.by_question)
        table("by bot", self.by_bot)
        if self.mode == "trace":
            table("by tool", self.by_tool)
        table("biggest collection gaps by site (captured/cited)", self.missed_by_domain)
        if self.missing_urls:
            lines.append("")
            lines.append(
                f"--- {len(self.missing_urls)} source(s) to collect (first 10) ---"
            )
            for u in self.missing_urls[:10]:
                lines.append(f"  {u}")
        return "\n".join(lines)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["group", "label", "cited", "captured", "pct"])
        w.writerow(["overall", self.mode, self.cited, self.captured, self.pct])
        for group, rows in (
            ("question", self.by_question),
            ("bot", self.by_bot),
            ("tool", self.by_tool),
            ("missed_domain", self.missed_by_domain),
        ):
            for r in rows:
                w.writerow([group, r.label, r.cited, r.captured, r.pct])
        return buf.getvalue()


def _grouped(scoped: list[tuple[Source, list]], key_of) -> list[CoverageRow]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for source, cits in scoped:
        keys = {k for k in (key_of(c) for c in cits) if k} or {"(none)"}
        for k in keys:
            agg[k][0] += 1
            if source.captured:
                agg[k][1] += 1
    rows = [CoverageRow(label=k, cited=v[0], captured=v[1]) for k, v in agg.items()]
    return sorted(rows, key=lambda r: (-r.cited, r.label))


def coverage_from_sources(
    sources: list[Source], mode: str, outcomes: dict[str, str] | None = None
) -> CoverageReport:
    scoped: list[tuple[Source, list]] = []
    excluded: dict[str, int] = defaultdict(int)
    for s in sources:
        cits = [c for c in s.citations if citation_mode(c) == mode]
        if not cits:
            continue
        reason = exclusion_reason(s.canonical_url, cits)
        if reason:
            excluded[reason] += 1
            continue
        scoped.append((s, cits))

    captured = sum(1 for s, _ in scoped if s.captured)

    never_fetched = failed = 0
    if outcomes is not None:
        from forecasting_tools.agents_and_tools.source_archive.reports import (
            FAILED_STATUSES,
        )

        for s, _ in scoped:
            if s.captured:
                continue
            status = outcomes.get(s.canonical_url)
            if status is None:
                never_fetched += 1
            elif status in FAILED_STATUSES:
                failed += 1
            else:
                failed += 1

    domain_agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s, _ in scoped:
        domain_agg[s.domain][0] += 1
        if s.captured:
            domain_agg[s.domain][1] += 1
    missed_by_domain = sorted(
        (
            CoverageRow(label=d, cited=c, captured=cap)
            for d, (c, cap) in domain_agg.items()
            if cap < c
        ),
        key=lambda r: (-(r.cited - r.captured), r.label),
    )

    return CoverageReport(
        mode=mode,
        cited=len(scoped),
        captured=captured,
        excluded=dict(excluded),
        by_question=_grouped(scoped, lambda c: c.question_id),
        by_bot=_grouped(scoped, lambda c: c.bot),
        by_tool=_grouped(scoped, lambda c: c.tool_name),
        missed_by_domain=missed_by_domain,
        missing_urls=sorted(s.canonical_url for s, _ in scoped if not s.captured),
        has_outcomes=outcomes is not None,
        missing_never_fetched=never_fetched,
        missing_fetch_failed=failed,
    )


def build_coverage(
    store: BlobStore, config: ArchiveConfig, mode: str
) -> CoverageReport:
    from forecasting_tools.agents_and_tools.source_archive.reports import read_outcomes

    sources = build_sources(store, config)
    outcomes = read_outcomes(store, config) or None
    return coverage_from_sources(sources, mode, outcomes)
