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

from forecasting_tools.agents_and_tools.source_archive.canonicalize import (
    canonicalize_url,
)
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord

# Markdown link target: [label](url) or [label](<url>), optionally with a title.
_MD_LINK = re.compile(r"\[[^\]]*\]\(\s*<?(https?://[^)\s>]+)>?[^)]*\)", re.IGNORECASE)
# Autolink: <url>
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>", re.IGNORECASE)
# Bare URL. Parens are allowed in the match and removed by _trim only when
# unbalanced, so trailing prose parens drop but ``..._(disambiguation)`` survives.
_BARE = re.compile(r"(https?://[^\s<>\"'\]]+)", re.IGNORECASE)

# Characters commonly stuck to the end of a URL in prose (incl. markdown-escape
# residue: a trailing backslash or backtick).
_TRAILING = ".,;:!?'\"\\`"


def _cut_markdown_tail(url: str) -> str:
    """Cut a URL at a markdown reference/link tail the bare-URL scan can swallow.

    Bots sometimes emit ``…/story?id=123)[10](https://other…)`` where ``)[10](…``
    is a markdown reference glued onto a real URL. The leading ``)`` was never
    part of the URL, so cut at the first ``)[`` or ``](`` boundary.
    """
    cut = len(url)
    for marker in (")[", "]("):
        i = url.find(marker)
        if i > 0:
            cut = min(cut, i)
    return url[:cut]


def _trim(url: str) -> str:
    """Strip trailing punctuation, and a closing bracket/paren only when it is
    unbalanced (so Wikipedia-style ``..._(disambiguation)`` URLs survive)."""
    url = _cut_markdown_tail(url)
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
    """Return the distinct http(s) URLs in ``text``, in first-seen order.

    Distinctness is by *canonical* URL (see :func:`canonicalize_url`), so
    ``…/x`` and ``…/x?utm_source=…`` count once; the original first-seen string
    is returned.
    """
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for pattern in (_MD_LINK, _AUTOLINK, _BARE):
        for match in pattern.finditer(text):
            url = _trim(match.group(1))
            if not url:
                continue
            key = canonicalize_url(url)
            if key not in seen:
                seen.add(key)
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
    comment_id: str | None = None,
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
            comment_id=comment_id,
            trace=trace,
            tool_name=tool_name,
            origin=origin,
        )
        for url in extract_urls(text)
    ]


def dedupe_records(records: Iterable[CitationRecord]) -> list[CitationRecord]:
    """Keep the first record per *canonical* URL, preserving order."""
    seen: set[str] = set()
    out: list[CitationRecord] = []
    for r in records:
        if not r.url:
            continue
        key = canonicalize_url(r.url)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
