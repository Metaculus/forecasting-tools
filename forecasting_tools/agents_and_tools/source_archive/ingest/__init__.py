"""Ingestion: discover the URLs a bot cited and turn them into a manifest.

The capture pipeline needs a citation manifest as input. These helpers build one
from a bot's published reasoning:

  - :mod:`url_extraction` — pull URLs out of free text / markdown.
  - :mod:`metaculus_comments` — harvest bot comments via the public Metaculus API.
"""

from forecasting_tools.agents_and_tools.source_archive.ingest.metaculus_comments import (
    MetaculusCommentHarvester,
)
from forecasting_tools.agents_and_tools.source_archive.ingest.url_extraction import (
    dedupe_records,
    extract_citation_records,
    extract_urls,
)

__all__ = [
    "MetaculusCommentHarvester",
    "dedupe_records",
    "extract_citation_records",
    "extract_urls",
]
