"""Source Archive — capture and store the web sources a forecasting bot cited.

For every unique URL a bot used, this captures **HTML + screenshot + markdown**
in a single page load and stores it with provenance, deduplicated by
``url + content-hash`` so re-runs of the same question are nearly free.

Quick start (see ``README.md`` in this package for the full guide)::

    from forecasting_tools.agents_and_tools.source_archive import (
        ArchiveConfig, CapturePipeline, ContentStore, build_default_fetcher,
    )
    from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore

    config = ArchiveConfig.from_env()
    store = ContentStore(LocalBlobStore("./archive"), config)
    with build_default_fetcher(config) as fetcher:
        summary = CapturePipeline(fetcher, store).run(["https://example.com"])
    print(summary)

The heavy backends (Playwright, boto3, Firecrawl, trafilatura) are optional;
install them with ``pip install forecasting-tools[source-archive]``.
"""

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import (
    ContentStore,
    StoreResult,
)
from forecasting_tools.agents_and_tools.source_archive.fetchers import (
    build_default_fetcher,
)
from forecasting_tools.agents_and_tools.source_archive.ingest import (
    MetaculusCommentHarvester,
    extract_urls,
)
from forecasting_tools.agents_and_tools.source_archive.models import (
    CaptureResult,
    CitationRecord,
    StoredCapture,
)
from forecasting_tools.agents_and_tools.source_archive.pipeline import (
    CaptureOutcome,
    CapturePipeline,
    PipelineSummary,
)

__all__ = [
    "ArchiveConfig",
    "CaptureOutcome",
    "CaptureResult",
    "CapturePipeline",
    "CitationRecord",
    "ContentStore",
    "MetaculusCommentHarvester",
    "PipelineSummary",
    "StoreResult",
    "StoredCapture",
    "build_default_fetcher",
    "extract_urls",
]
