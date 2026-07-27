"""Backend bake-off: run each capture backend independently over the same URLs.

This is the harness for deciding *which* backup to put behind Playwright. Unlike
the production :class:`TieredFetcher` (which stops at the first backend that
passes the quality gate), the benchmark runs **every** selected backend over
**every** URL, so you get an apples-to-apples table of reliability, latency, and
estimated cost per backend — broken down by URL category (normal / cloudflare /
pdf).

Run it::

    python -m forecasting_tools.agents_and_tools.source_archive.benchmark \\
        --manifest sample_urls.jsonl \\
        --backends playwright,cloakbrowser,firecrawl,firecrawl-stealth,hyperbrowser,pdf \\
        --out bench.csv

A backend whose dependency or API key is missing is skipped with a note rather
than failing the whole run, so you can benchmark whatever you have configured.

Cost figures are ESTIMATES from a documented pricing model (see ``PRICING``,
sourced 2026-06); they are not billed amounts. Override the credit rates via
CLI flags to match your plan.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import (
    Fetcher,
    FetchError,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.cloakbrowser_fetcher import (
    CloakBrowserFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.firecrawl_fetcher import (
    FirecrawlFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.hyperbrowser_fetcher import (
    HyperbrowserFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.pdf_fetcher import (
    PdfFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers.playwright_fetcher import (
    PlaywrightFetcher,
)
from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult
from forecasting_tools.agents_and_tools.source_archive.quality import evaluate

logger = logging.getLogger(__name__)

GB = 1_000_000_000

# --- Pricing model -----------------------------------------------------------
# $/unit as of 2026-06, from each vendor's public pricing + this repo's prior
# cost experiment. These are the knobs to adjust for your plan.
#
#   - Self-hosted compute (Playwright / CloakBrowser): ~$0.00001/page rendered
#     (measured in bot-sources probe). Marginal service fee is effectively $0.
#   - Firecrawl: 1 credit basic, 5 credits stealth/"enhanced" proxy. Standard
#     plan ≈ $0.00083/credit.
#   - Hyperbrowser: 1 credit ($0.001) basic, 10 credits ($0.01) with proxy,
#     plus $10/GB proxy bandwidth. 1 credit = $0.001.
#   - PDF: PyMuPDF4LLM local = $0; Firecrawl OCR fallback = ~1 credit/PDF page.


@dataclass
class Pricing:
    self_host_per_page: float = 0.00001
    firecrawl_credit_usd: float = 0.00083
    firecrawl_basic_credits: int = 1
    firecrawl_stealth_credits: int = 5
    hyperbrowser_credit_usd: float = 0.001
    hyperbrowser_basic_credits: int = 1
    hyperbrowser_proxy_credits: int = 10
    hyperbrowser_bandwidth_usd_per_gb: float = 10.0


def estimate_cost(
    backend: str, result: CaptureResult, response_bytes: int, pricing: Pricing
) -> float:
    """Estimated $ for one successful capture by ``backend``."""
    meta = result.metadata or {}
    if backend in ("playwright", "cloakbrowser"):
        return pricing.self_host_per_page
    if backend.startswith("firecrawl"):
        proxy = str(meta.get("firecrawl_proxy", "basic")).lower()
        credits = (
            pricing.firecrawl_basic_credits
            if proxy in ("", "basic")
            else pricing.firecrawl_stealth_credits
        )
        return credits * pricing.firecrawl_credit_usd
    if backend == "hyperbrowser":
        credits = (
            pricing.hyperbrowser_proxy_credits
            if meta.get("used_proxy")
            else pricing.hyperbrowser_basic_credits
        )
        bandwidth = (response_bytes / GB) * pricing.hyperbrowser_bandwidth_usd_per_gb
        return credits * pricing.hyperbrowser_credit_usd + bandwidth
    if backend == "pdf":
        if meta.get("pdf_engine") == "firecrawl":
            pages = int(meta.get("pdf_pages") or 1)
            return pages * pricing.firecrawl_credit_usd
        return 0.0  # local PyMuPDF4LLM
    return 0.0


# --- Backend registry --------------------------------------------------------
# Factories so a missing dependency / API key only skips that backend. The
# ``context`` flag marks browser backends that must be entered as a context
# manager (the browser launches once and is reused across URLs).


@dataclass
class BackendSpec:
    name: str
    factory: Callable[[ArchiveConfig], Fetcher]
    context: bool = False
    # Optional pre-flight: return a reason string if the backend can't run
    # (missing key/dep) so the bake-off reports a clean SKIP instead of N/N
    # fetch_errors. ``None`` means "looks runnable".
    precheck: Callable[[ArchiveConfig], str | None] | None = None


def _need_firecrawl_key(config: ArchiveConfig) -> str | None:
    if not config.firecrawl_api_key:
        return "FIRECRAWL_API_KEY not set"
    return None


def _need_hyperbrowser_key(config: ArchiveConfig) -> str | None:
    if not config.hyperbrowser_api_key:
        return "HYPERBROWSER_API_KEY not set"
    return None


def _firecrawl_stealth(config: ArchiveConfig) -> FirecrawlFetcher:
    # Force the proxy/stealth path so this row measures the Cloudflare-grade
    # (5-credit) cost, even if the operator left the default at "basic".
    proxy = config.firecrawl_proxy
    if proxy in ("", "basic"):
        proxy = "auto"
    f = FirecrawlFetcher(config.model_copy(update={"firecrawl_proxy": proxy}))
    f.name = "firecrawl-stealth"
    return f


BACKENDS: dict[str, BackendSpec] = {
    "playwright": BackendSpec("playwright", PlaywrightFetcher, context=True),
    "cloakbrowser": BackendSpec("cloakbrowser", CloakBrowserFetcher, context=True),
    "firecrawl": BackendSpec(
        "firecrawl", FirecrawlFetcher, precheck=_need_firecrawl_key
    ),
    "firecrawl-stealth": BackendSpec(
        "firecrawl-stealth", _firecrawl_stealth, precheck=_need_firecrawl_key
    ),
    "hyperbrowser": BackendSpec(
        "hyperbrowser", HyperbrowserFetcher, precheck=_need_hyperbrowser_key
    ),
    "pdf": BackendSpec("pdf", PdfFetcher),
}


# --- Sample manifest ---------------------------------------------------------
# A curated starter set spanning the three categories the backup must handle.
# Replace/extend with your own real cited URLs for a representative run.
SAMPLE_MANIFEST: list[dict] = [
    {"url": "https://example.com", "category": "normal"},
    {"url": "https://en.wikipedia.org/wiki/Forecasting", "category": "normal"},
    {"url": "https://www.federalregister.gov/", "category": "normal"},
    # Sites commonly fronted by Cloudflare / anti-bot:
    {"url": "https://www.g2.com/", "category": "cloudflare"},
    {"url": "https://www.indeed.com/", "category": "cloudflare"},
    {"url": "https://www.zillow.com/", "category": "cloudflare"},
    # PDFs (the gap Playwright can't render):
    {"url": "https://arxiv.org/pdf/1706.03762", "category": "pdf"},
    {"url": "https://bitcoin.org/bitcoin.pdf", "category": "pdf"},
]


@dataclass
class Row:
    backend: str
    url: str
    category: str
    passed: bool
    reason: str
    seconds: float
    html_bytes: int
    md_bytes: int
    screenshot_bytes: int
    cost_usd: float
    error: str = ""


@dataclass
class BackendRun:
    name: str
    rows: list[Row] = field(default_factory=list)
    skipped: str = ""


def _sizes(result: CaptureResult) -> tuple[int, int, int]:
    html = len(result.html.encode()) if result.html else 0
    md = len(result.markdown.encode()) if result.markdown else 0
    shot = len(result.screenshot) if result.screenshot else 0
    return html, md, shot


def run_backend(
    spec: BackendSpec,
    manifest: list[dict],
    config: ArchiveConfig,
    pricing: Pricing,
) -> BackendRun:
    run = BackendRun(name=spec.name)
    if spec.precheck is not None:
        reason = spec.precheck(config)
        if reason:
            run.skipped = reason
            logger.warning("%s skipped: %s", spec.name, reason)
            return run
    try:
        fetcher = spec.factory(config)
    except Exception as e:  # construction (e.g. missing key) — skip cleanly
        run.skipped = f"could not construct {spec.name}: {e}"
        logger.warning(run.skipped)
        return run

    cm = fetcher if spec.context else nullcontext(fetcher)
    try:
        with cm as live:
            for record in manifest:
                run.rows.append(_capture_one(spec.name, live, record, pricing))
    except FetchError as e:
        # A browser backend can fail to even start (e.g. cloakbrowser not
        # installed). Record it as a skip rather than crashing the bake-off.
        if not run.rows:
            run.skipped = f"{spec.name} unavailable: {e}"
            logger.warning(run.skipped)
        else:
            raise
    return run


def _capture_one(backend: str, fetcher: Fetcher, record: dict, pricing: Pricing) -> Row:
    url = record["url"]
    category = record.get("category", "normal")
    start = time.monotonic()
    try:
        result = fetcher.fetch(url)
    except FetchError as e:
        return Row(
            backend,
            url,
            category,
            False,
            "fetch_error",
            round(time.monotonic() - start, 2),
            0,
            0,
            0,
            0.0,
            error=str(e)[:300],
        )
    except Exception as e:  # backend bug / unexpected SDK error
        return Row(
            backend,
            url,
            category,
            False,
            "exception",
            round(time.monotonic() - start, 2),
            0,
            0,
            0,
            0.0,
            error=str(e)[:300],
        )

    seconds = round(time.monotonic() - start, 2)
    verdict = evaluate(result)
    html_b, md_b, shot_b = _sizes(result)
    response_bytes = html_b + shot_b
    cost = (
        estimate_cost(backend, result, response_bytes, pricing)
        if verdict.passed
        else 0.0
    )
    return Row(
        backend,
        url,
        category,
        verdict.passed,
        verdict.reason or "ok",
        seconds,
        html_b,
        md_b,
        shot_b,
        round(cost, 6),
    )


# --- Reporting ---------------------------------------------------------------
def write_csv(path: str, runs: list[BackendRun]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "backend",
            "url",
            "category",
            "passed",
            "reason",
            "seconds",
            "html_bytes",
            "md_bytes",
            "screenshot_bytes",
            "cost_usd",
            "error",
        ]
    )
    for run in runs:
        for r in run.rows:
            w.writerow(
                [
                    r.backend,
                    r.url,
                    r.category,
                    r.passed,
                    r.reason,
                    r.seconds,
                    r.html_bytes,
                    r.md_bytes,
                    r.screenshot_bytes,
                    r.cost_usd,
                    r.error,
                ]
            )
    Path(path).write_text(buf.getvalue(), encoding="utf-8")


def summarize(runs: list[BackendRun], urls_per_question: int, tail_share: float) -> str:
    cats = ["normal", "cloudflare", "pdf"]
    lines = []
    header = (
        f"{'backend':<18}{'overall':>9}"
        + "".join(f"{c:>11}" for c in cats)
        + f"{'med s':>8}{'$/page':>10}{'proj $/q':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for run in runs:
        if run.skipped:
            lines.append(f"{run.name:<18}  SKIPPED: {run.skipped[:80]}")
            continue
        total = len(run.rows)
        passed = [r for r in run.rows if r.passed]
        overall = f"{len(passed)}/{total}"

        def cat_rate(cat: str) -> str:
            rows = [r for r in run.rows if r.category == cat]
            if not rows:
                return "-"
            ok = sum(1 for r in rows if r.passed)
            return f"{ok}/{len(rows)}"

        med = statistics.median([r.seconds for r in run.rows]) if run.rows else 0
        cost_per = statistics.mean([r.cost_usd for r in passed]) if passed else 0.0
        # Illustrative: if THIS backend alone handled the whole post-Playwright
        # tail of a question. (tail_share × urls × $/successful page.)
        proj = tail_share * urls_per_question * cost_per
        lines.append(
            f"{run.name:<18}{overall:>9}"
            + "".join(f"{cat_rate(c):>11}" for c in cats)
            + f"{med:>8.1f}{cost_per:>10.5f}{proj:>10.3f}"
        )
    note = (
        f"\nproj $/q assumes one backend covers a {tail_share:.0%} tail of "
        f"{urls_per_question} URLs/question, BEFORE the TTL cache (which makes "
        f"re-runs nearly free). Costs are model estimates, not billed amounts."
    )
    return "\n".join(lines) + "\n" + note


def load_manifest(path: str | None) -> list[dict]:
    if not path:
        return SAMPLE_MANIFEST
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Capture-backend bake-off.")
    p.add_argument(
        "--manifest", help="JSONL of {url, category}. Omit for the built-in sample."
    )
    p.add_argument(
        "--backends",
        default="playwright,cloakbrowser,firecrawl,firecrawl-stealth,hyperbrowser,pdf",
        help="Comma-separated subset of: " + ", ".join(BACKENDS),
    )
    p.add_argument("--out", default="benchmark.csv", help="CSV output path.")
    p.add_argument("--urls-per-question", type=int, default=450)
    p.add_argument(
        "--tail-share",
        type=float,
        default=0.30,
        help="Fraction of URLs that fall through Playwright.",
    )
    p.add_argument("--firecrawl-credit-usd", type=float, default=0.00083)
    p.add_argument("--hyperbrowser-credit-usd", type=float, default=0.001)
    args = p.parse_args(argv)

    config = ArchiveConfig.from_env()
    pricing = Pricing(
        firecrawl_credit_usd=args.firecrawl_credit_usd,
        hyperbrowser_credit_usd=args.hyperbrowser_credit_usd,
    )
    manifest = load_manifest(args.manifest)

    selected = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in selected if b not in BACKENDS]
    if unknown:
        p.error(f"unknown backends: {unknown}. Choose from {list(BACKENDS)}")

    runs: list[BackendRun] = []
    for name in selected:
        print(f"running {name} over {len(manifest)} URLs...", file=sys.stderr)
        runs.append(run_backend(BACKENDS[name], manifest, config, pricing))

    write_csv(args.out, runs)
    print("\n" + summarize(runs, args.urls_per_question, args.tail_share))
    print(f"\nper-URL detail written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
