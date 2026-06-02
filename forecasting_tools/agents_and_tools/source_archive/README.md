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

# Build a manifest by harvesting the URLs bots cited on a Metaculus tournament
source-archive harvest 32506 --out run.jsonl
```

`source-archive` is installed by the extra; the equivalent module form is
`python -m forecasting_tools.agents_and_tools.source_archive.cli`.

## The manifest: what to feed it

A run produces a **citation manifest** — a JSONL file with one record per cited
URL. Only `url` is required; the rest is provenance you fill in where you have it:

```json
{"url": "https://example.com/report", "run_id": "2026-06-01_demo", "bot": "my-bot", "question_id": "1234", "question_url": "https://www.metaculus.com/questions/1234/", "tool_name": "web_search", "origin": "research"}
```

The pipeline dedupes URLs within the manifest before fetching.

## Where the manifest comes from

You can write a manifest yourself, or generate one from a bot's published
reasoning. Both first-party and third-party bots post their reasoning — with the
source links they used — as comments on Metaculus, so the public, no-auth
Metaculus API is the one ingestion path that works across *every* bot:

```python
from forecasting_tools.agents_and_tools.source_archive.ingest import (
    MetaculusCommentHarvester,
)
from forecasting_tools.agents_and_tools.source_archive import manifest

harvester = MetaculusCommentHarvester()        # uses METACULUS_API_BASE_URL
records = harvester.harvest_project(32506)     # a tournament / project id
manifest.write_file("run.jsonl", records)      # -> feed to `capture`
```

Or in one line from the CLI: `source-archive harvest 32506 --out run.jsonl`.

The lower-level `extract_urls(text)` / `extract_citation_records(...)` helpers in
`ingest.url_extraction` pull URLs out of any markdown/text (markdown links,
autolinks, and bare URLs), if you are ingesting from somewhere other than
comments.

Caveat: comments are length-truncated when posted, so a comment-harvested URL
list can be incomplete versus a bot's full research. For bots you control, an
instrumented trace gives a fuller list; comment harvesting is the universal
baseline.

## How it's organized

| Module | Responsibility |
| --- | --- |
| `config.py` | Environment-driven `ArchiveConfig` |
| `models.py` | `CaptureResult`, `StoredCapture`, `CitationRecord` |
| `ingest/` | Build a manifest: URL extraction + Metaculus comment harvester |
| `fetchers/` | Playwright (primary), Firecrawl (fallback), tiered orchestrator |
| `quality.py` | Reject 404s, block pages, and thin content before archiving |
| `storage/` | `BlobStore` interface with S3 and local backends |
| `content_store.py` | `url + content-hash` store with the TTL cache and dedup |
| `manifest.py` | Read/write citation manifests |
| `pipeline.py` | `lookup → fetch → quality gate → store` |
| `cli.py` | `source-archive` command |

## What lands in storage

```
<prefix>/index/<url_hash>.json                     per-URL capture history
<prefix>/content/<url_hash>/<content_hash>.html
<prefix>/content/<url_hash>/<content_hash>.webp     (screenshot)
<prefix>/content/<url_hash>/<content_hash>.md
<prefix>/manifests/<run_id>.jsonl                  the run's citation manifest
```
