"""Extract URLs from free text and markdown.

Bots surface their sources as prose with embedded links (e.g. the reasoning
comment they post on a question). This module pulls those URLs out and turns
them into :class:`CitationRecord` provenance rows — the manifest that feeds the
capture pipeline.

It handles markdown links ``[label](url)``, autolinks ``<url>``, and bare URLs,
and trims the trailing punctuation that so often clings to a URL in prose.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord

# Markdown link target: [label](url) or [label](<url>), optionally with a title.
_MD_LINK = re.compile(r"\[[^\]]*\]\(\s*<?(https?://[^)\s>]+)>?[^)]*\)", re.IGNORECASE)
# Autolink: <url>
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>", re.IGNORECASE)
# Bare URL. Parens are allowed in the match and removed by _trim only when
# unbalanced, so trailing prose parens drop but ``..._(disambiguation)`` survives.
_BARE = re.compile(r"(https?://[^\s<>\"'\]]+)", re.IGNORECASE)

# Characters commonly stuck to the end of a URL in prose.
_TRAILING = ".,;:!?'\""


def _trim(url: str) -> str:
    """Strip trailing punctuation, and a closing bracket/paren only when it is
    unbalanced (so Wikipedia-style ``..._(disambiguation)`` URLs survive)."""
    while url:
        last = url[-1]
        if last in _TRAILING:
            url = url[:-1]
        elif last == ")" and url.count("(") < url.count(")"):
            url = url[:-1]
        elif last == "]" and url.count("[") < url.count("]"):
            url = url[:-1]
        else:
            break
    return url


def extract_urls(text: str | None) -> list[str]:
    """Return the distinct http(s) URLs in ``text``, in first-seen order."""
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for pattern in (_MD_LINK, _AUTOLINK, _BARE):
        for match in pattern.finditer(text):
            url = _trim(match.group(1))
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
    return ordered


def extract_citation_records(
    text: str | None,
    *,
    run_id: str | None = None,
    bot: str | None = None,
    question_id: str | None = None,
    metaculus_id: str | None = None,
    question_url: str | None = None,
    trace: str | None = None,
    tool_name: str | None = None,
    origin: str | None = None,
) -> list[CitationRecord]:
    """Extract URLs from ``text`` and wrap each in a CitationRecord with the
    given provenance."""
    return [
        CitationRecord(
            url=url,
            run_id=run_id,
            bot=bot,
            question_id=question_id,
            metaculus_id=metaculus_id,
            question_url=question_url,
            trace=trace,
            tool_name=tool_name,
            origin=origin,
        )
        for url in extract_urls(text)
    ]


def dedupe_records(records: Iterable[CitationRecord]) -> list[CitationRecord]:
    """Keep the first record per URL, preserving order."""
    seen: set[str] = set()
    out: list[CitationRecord] = []
    for r in records:
        if r.url and r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out
