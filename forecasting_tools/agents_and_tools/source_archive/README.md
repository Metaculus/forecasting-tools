# Source Archive

Capture and preserve the web sources a forecasting bot relied on. For every
unique URL a bot cited, this captures **HTML + a full-page screenshot +
markdown** in a single page load and stores it with provenance, so a forecast
can be audited later even if the original page changes or disappears.

## Why this exists

A bot's forecast is only as trustworthy as the sources behind it, and those
sources rot: pages get edited, paywalled, or deleted. This package snapshots
each cited URL at the time it was used.

It is built to be cheap at scale. Two ideas do the heavy lifting:

- **Self-hosted rendering.** A single headless-Chromium page load produces all
  three artifacts (HTML, screenshot, markdown), at a tiny fraction of the cost
  of managed scraping APIs. A hosted fallback (Firecrawl) is used only for sites
  that block headless browsers.
- **A content store with a TTL cache.** Bots re-forecast the same open question
  every 20–30 minutes for weeks, citing the same pages each time. The store is
  keyed by `url + content-hash`: a URL captured within the TTL is *not* refetched,
  and identical content is *not* re-stored. So the first capture costs real money
  and every re-run is nearly free.

## Install

The backends are optional, so they aren't pulled in by a default install:

```bash
pip install "forecasting-tools[source-archive]"
playwright install chromium   # one-time browser download
```

## Configure

Configuration is read from the environment (see the project `.env.template`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `WEB_ARCHIVE_S3_BUCKET` | Destination S3 bucket. Blank → store locally. | — |
| `WEB_ARCHIVE_S3_PREFIX` | Key prefix within the bucket. | `source-archive` |
| `WEB_ARCHIVE_AWS_PROFILE` | Named AWS profile (e.g. an SSO profile). | default chain |
| `WEB_ARCHIVE_TTL_DAYS` | Days before a cached capture is refetched. | `14` |
| `FIRECRAWL_API_KEY` | Enables the Firecrawl fallback. | — (fallback off) |
| `WEB_ARCHIVE_FIRECRAWL_PROXY` | Firecrawl proxy mode for hardened sites: `basic` (1 credit) / `auto` / `stealth` (5 credits). | `basic` |
| `HYPERBROWSER_API_KEY` | Enables the Hyperbrowser managed fallback. | — (off) |
| `WEB_ARCHIVE_CLOAKBROWSER_IMPORT` | Module exposing CloakBrowser's `launch()`. | `cloakbrowser` |
| `WEB_ARCHIVE_PDF_MAX_PAGES` | Cap on PDF pages parsed per document. | `50` |

AWS credentials use the standard AWS resolution chain — environment variables, a
shared config file, or an SSO profile. Nothing secret is committed or baked into
the code.

## Use it from Python

```python
from forecasting_tools.agents_and_tools.source_archive import (
    ArchiveConfig, CapturePipeline, ContentStore, build_default_fetcher,
)
from forecasting_tools.agents_and_tools.source_archive.storage import (
    LocalBlobStore, S3BlobStore,
)

config = ArchiveConfig.from_env()

# Store locally while experimenting...
store = ContentStore(LocalBlobStore("./archive"), config)
# ...or to S3 in production:
# store = ContentStore(S3BlobStore(config.s3_bucket, config=config), config)

with build_default_fetcher(config) as fetcher:
    summary = CapturePipeline(fetcher, store).run([
        "https://example.com",
        "https://www.federalregister.gov/",
    ])

print(summary)
# PipelineSummary(total=2, cache_hit=0, stored=2, deduped=0, quality_failed=0, error=0)
```

## Use it from the command line

```bash
# Inspect the resolved configuration (secrets are masked)
source-archive check

# Capture every URL in a manifest, storing locally (no AWS needed)
source-archive capture run.jsonl --local ./archive

# Capture and upload to S3 (uses WEB_ARCHIVE_S3_BUCKET), plus the manifest itself
source-archive capture run.jsonl --upload-manifest --run-id 2026-06-01_demo

# Skip the Hyperbrowser fallback this run; failures are written to a retry
# manifest so you can come back to just those sites later (e.g. with it on).
source-archive capture run.jsonl --no-hyperbrowser --run-id demo
source-archive capture demo_needs_retry.jsonl --run-id demo   # later, hyperbrowser on

# Build a manifest by harvesting the URLs bots cited on a Metaculus tournament
source-archive harvest 32506 --out run.jsonl
```

Because a failed fetch leaves no cache entry while a success does, re-running the
same manifest only re-attempts the failures — the retry manifest just makes that
explicit and fast (it skips the already-captured majority).

`source-archive` is installed by the extra; the equivalent module form is
`python -m forecasting_tools.agents_and_tools.source_archive.cli`.

## Backup backends & the bake-off

A self-hosted browser is the primary backend and gets ~70% of URLs for ~free,
but two kinds of URL fall through it: **anti-bot/Cloudflare** pages (it detects
the block but can't get past it) and **PDFs** (Chromium downloads them instead of
rendering, so nothing is captured). The package ships these backups, ordered by
marginal cost so the cheap tiers absorb most of the tail:

| Backend | Cost (2026) | Closes | Notes |
| --- | --- | --- | --- |
| `CloakBrowserFetcher` | ~$0/page (self-host) | Cloudflare | **The primary browser tier when installed** (`pip install cloakbrowser`): patched Chromium that beat vanilla Playwright on Cloudflare in 2026 benchmarks. Only one browser runs — cloak *replaces* vanilla Playwright (two `sync_playwright` instances conflict in one process), falling back to vanilla when cloak isn't installed. |
| `PdfFetcher` | $0 local; ~$0.0008/pg OCR | PDFs | PyMuPDF4LLM locally, falls back to Firecrawl OCR on scanned PDFs. |
| `FirecrawlFetcher` | $0.0008 basic / $0.0042 stealth | Cloudflare + PDFs | Native PDF parser; `WEB_ARCHIVE_FIRECRAWL_PROXY=stealth` for hardened sites. |
| `HyperbrowserFetcher` | $0.001 basic / $0.01 proxy | Cloudflare | Consolidates spend onto a vendor already used elsewhere. No PDF support. |

Selenium was evaluated and **rejected**: it drives the same Chromium as
Playwright, so it bypasses nothing Playwright can't, and its stealth ecosystem
(`undetected-chromedriver`) is now legacy. CloakBrowser/Patchright/nodriver are
the credible self-hosted upgrades.

To decide which backup(s) to wire in, run the bake-off — it runs each selected
backend independently over the same URLs (not tiered) and reports reliability,
latency, and estimated cost per backend, broken down by category:

```bash
python -m forecasting_tools.agents_and_tools.source_archive.benchmark \
    --manifest forecasting_tools/agents_and_tools/source_archive/benchmarks/sample_urls.jsonl \
    --backends playwright,cloakbrowser,firecrawl,firecrawl-stealth,hyperbrowser,pdf \
    --out bench.csv
```

Backends whose API key or dependency is missing are skipped cleanly. Cost
figures are model estimates (see `PRICING` in `benchmark.py`); tune the credit
rates with `--firecrawl-credit-usd` / `--hyperbrowser-credit-usd` to match your
plan. Swap the sample manifest for a JSONL of your own cited URLs (one
`{"url", "category"}` per line; categories `normal`/`cloudflare`/`pdf`) for a
representative run.

## Browse what you captured

A Streamlit viewer reads the manifests + index back out of the store and shows
each captured URL's **screenshot, markdown, and HTML** side by side, filterable
by bot and question:

```bash
AWS_PROFILE=default WEB_ARCHIVE_S3_BUCKET=my-web-archive \
  streamlit run forecasting_tools/agents_and_tools/source_archive/viewer.py
```

It uses the same `ArchiveConfig.from_env()` settings as capture, so it points at
whatever bucket/prefix you captured to (no extra configuration).

To browse a **local** capture (no S3/AWS), set `WEB_ARCHIVE_LOCAL_DIR` to the
directory you captured into with `--local`:

```bash
WEB_ARCHIVE_LOCAL_DIR=./archive \
  streamlit run forecasting_tools/agents_and_tools/source_archive/viewer.py
```

## The catalog: a browsable, coworker-legible view

The viewer is interactive (good for us); the **catalog** is a set of static
HTML/CSV pages written into the bucket so a non-technical coworker can browse the
sources without any tooling. It is **question-primary** — the encyclopedia of
every web source used for a question — plus `by-bot/` and `by-domain/`
cross-views, built by joining the manifests with the index:

```bash
# write catalog/ into the bucket (uses WEB_ARCHIVE_S3_BUCKET)
source-archive catalog
# or against a local capture dir
source-archive catalog --local ./archive
```

Start at `catalog/index.html` (or `catalog/READ_ME_FIRST.html` for the plain
explainer). Each source shows its screenshot, who used it (bot + tool), and
whether it was captured; each question also has a CSV. Data/API calls (a bot's
`run_code` pulling a CSV, etc.) are **excluded** from the catalog — it lists web
pages a bot read, not data endpoints — though they remain in the raw manifests.

## Coverage: what fraction did we archive?

The catalog shows what we *have*; the **coverage report** shows what we're
*missing*. It's two separate reports, by ingestion path — different denominators,
different ground truth:

```bash
source-archive coverage                 # both reports
source-archive coverage --mode trace    # just the complex/template bot
source-archive coverage --csv ./cov     # also write cov_<mode>.csv (+ _missing.txt)
```

- **trace** — the complex/template bot's instrumented runs (metac-ai-sdk). Traces
  hold *every* URL the bot touched, so this is a true archival success-rate.
- **comments** — every bot (Metaculus's own + outsiders) harvested from public
  comments. Comments are truncated, so this denominator under-counts — coverage
  here means "of the links visible in comments, how many we archived."

The report is oriented to one question: **are there sources bots are using that
we are not yet archiving?** It leads with that gap, then breaks it down by
question, bot, tool, and the biggest-gap sites, plus the list of sources to
collect. Non-source URLs — search-engine results, `run_code`-style tool/API
calls, and malformed extractor junk — are excluded (same as the catalog).

If capture runs have persisted their outcomes (`reports/<run_id>.json`, written
automatically by `capture`), the gap is split into **never fetched** (the real
collection gap) vs **fetched but failed** (a capture problem).

## The manifest: what to feed it

A run produces a **citation manifest** — a JSONL file with one record per cited
URL. Only `url` is required; the rest is provenance you fill in where you have it:

```json
{"url": "https://example.com/report", "run_id": "2026-06-01_demo", "bot": "my-bot", "question_id": "1234", "question_url": "https://www.metaculus.com/questions/1234/", "tool_name": "web_search", "origin": "research"}
```

The pipeline dedupes URLs within the manifest before fetching.

## Where the manifest comes from

You write a manifest yourself, or generate one from a bot's recorded reasoning.

**From text.** `extract_urls(text)` / `extract_citation_records(...)` in
`ingest.url_extraction` pull URLs out of any markdown/text (markdown links,
autolinks, and bare URLs) — point them at whatever record of a bot's reasoning
you have.

**From instrumented traces.** For bots you control, a trace is the fullest
source. `source-archive ingest-traces <run-dir>` walks a traced run and emits a
manifest of every URL the bot touched, with provenance (trace, tool, search
query).

## How it's organized

| Module | Responsibility |
| --- | --- |
| `config.py` | Environment-driven `ArchiveConfig` |
| `models.py` | `CaptureResult`, `StoredCapture`, `CitationRecord` |
| `ingest/` | Build a manifest: URL extraction from text + traced bot runs |
| `fetchers/` | Playwright (primary) + CloakBrowser / PDF / Firecrawl / Hyperbrowser backups, tiered orchestrator |
| `benchmark.py` | Backend bake-off: reliability + cost per backend over a manifest |
| `quality.py` | Reject 404s, block pages, and thin content before archiving |
| `storage/` | `BlobStore` interface with S3 and local backends |
| `content_store.py` | `url + content-hash` store with the TTL cache and dedup |
| `manifest.py` | Read/write citation manifests |
| `pipeline.py` | `lookup → fetch → quality gate → store` |
| `cli.py` | `source-archive` command |

## Roadmap

Planned and shipped improvements — smarter dedup (URL canonicalization +
redirect/content aliasing), the coworker-legible catalog, and coverage reports —
are written up in [ROADMAP.md](ROADMAP.md).

## What lands in storage

```
<prefix>/index/<url_hash>.json                     per-URL capture history (+ aliases)
<prefix>/index/by-content/<content_hash>.json      reverse index for content dedup
<prefix>/content/<url_hash>/<content_hash>.html
<prefix>/content/<url_hash>/<content_hash>.webp     (screenshot)
<prefix>/content/<url_hash>/<content_hash>.md
<prefix>/manifests/<run_id>.jsonl                  the run's citation manifest
<prefix>/reports/<run_id>.json                     per-URL capture outcomes (for coverage)
<prefix>/catalog/index.html                        browsable catalog (by question/bot/site)
<prefix>/catalog/by-question/<id>.{html,csv}
```
