"""Ingestion: discover the URLs a bot cited and turn them into a manifest.

The capture pipeline needs a citation manifest as input. These helpers build one
from a bot's published reasoning:

  - :mod:`url_extraction` — pull URLs out of free text / markdown.
  - :mod:`metaculus_db` — read a bot's cited URLs from the platform database.
  - :mod:`trace_extraction` — build a manifest from a traced bot run (fullest path).
"""

from forecasting_tools.agents_and_tools.source_archive.ingest.metaculus_db import (
    MetaculusDbHarvester,
    resolve_dsn,
)
from forecasting_tools.agents_and_tools.source_archive.ingest.trace_extraction import (
    extract_records_from_events,
    extract_records_from_question_dir,
    extract_records_from_trace_file,
    harvest_run,
)
from forecasting_tools.agents_and_tools.source_archive.ingest.url_extraction import (
    dedupe_records,
    extract_citation_records,
    extract_urls,
)

__all__ = [
    "MetaculusDbHarvester",
    "dedupe_records",
    "extract_citation_records",
    "extract_records_from_events",
    "extract_records_from_question_dir",
    "extract_records_from_trace_file",
    "extract_urls",
    "harvest_run",
    "resolve_dsn",
]
