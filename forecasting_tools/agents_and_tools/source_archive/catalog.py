"""Generate a coworker-legible catalog over the hash-addressed store.

The content store is keyed by URL/content hash — great for dedup, opaque to a
human browsing the bucket. This builds a browsable ``catalog/`` layer on top by
joining the citation manifests (who cited what, on which question, with which
tool) with the per-URL index (what actually got captured). Blobs are never moved
or duplicated; the catalog only writes small HTML/CSV pointer pages.

Views (question-primary, with two cross-views):

    catalog/READ_ME_FIRST.html        plain-language explainer for coworkers
    catalog/index.html                landing page + headline counts
    catalog/by-question/<id>.html     ★ the encyclopedia for one question:
    catalog/by-question/<id>.csv        every source, deduped, tagged with the
                                        bots/tools/queries that used it
    catalog/by-bot/<bot>.html         one bot's sources across questions
    catalog/by-domain/<domain>.html   sources grouped by site

The question view is the default because that's how post-mortems think ("what
sources informed question X?"); ``by-bot`` groups a bot's sources across
questions, and ``by-domain`` groups them by site.
"""

from __future__ import annotations

import csv
import html
import io
from collections import defaultdict
from urllib.parse import urlsplit

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive import manifest as manifest_io
from forecasting_tools.agents_and_tools.source_archive.canonicalize import (
    canonicalize_url,
)
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.models import (
    CitationRecord,
    url_hash,
)
from forecasting_tools.agents_and_tools.source_archive.storage.blob_store import (
    BlobStore,
)

_UNKNOWN_Q = "unknown-question"

# Tools that fetch data/API endpoints, not human-readable web pages. A URL only
# ever touched by one of these is a data call (e.g. a bot's run_code pulling a
# CSV), so it is kept out of the page-oriented catalog (it stays in the raw
# manifests). A URL also seen via search/page-fetch is treated as a real page.
_NON_PAGE_TOOLS = {
    "run_code",
    "code",
    "python",
    "run_python",
    "code_interpreter",
    "execute_code",
    "bash",
    "shell",
}


def tool_call_only(citations: list) -> bool:
    """True if a URL was touched *only* by code-execution tools (a data/API call,
    not a page a bot read)."""
    tools = {(c.tool_name or "").lower() for c in citations}
    code_tools = tools & _NON_PAGE_TOOLS
    other_tools = tools - _NON_PAGE_TOOLS - {""}
    return bool(code_tools) and not other_tools


def _is_tool_call_only(source: "Source") -> bool:
    return tool_call_only(source.citations)


# Search-engine result pages are navigation, not sources — a bot citing a
# google/duckduckgo search URL hasn't handed us a page worth archiving.
_SEARCH_HOSTS = {
    "duckduckgo.com",
    "bing.com",
    "search.brave.com",
    "search.yahoo.com",
    "ecosia.org",
    "startpage.com",
    "baidu.com",
    "ask.com",
    "qwant.com",
    "search.marginalia.nu",
    "kagi.com",
}
# Percent-encoded junk that means the extractor swallowed markdown / a second URL
# / control chars into the URL (legacy captures from before extraction hardening).
_MALFORMED_MARKERS = ("%5b", "%5d", "%5c", "%0a", "%0d", "%28http", "%29%5b")


def is_search_url(url: str) -> bool:
    host = urlsplit(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return host in _SEARCH_HOSTS or host == "google.com" or host.startswith("google.")


def is_malformed_url(url: str) -> bool:
    low = url.lower()
    return url.count("://") > 1 or any(m in low for m in _MALFORMED_MARKERS)


def exclusion_reason(url: str, citations: list) -> str | None:
    """Why a cited URL is kept out of the page catalog / coverage, or ``None`` to
    keep it. ``malformed`` (extractor junk), ``search`` (search-engine results),
    ``tool_call`` (data/API endpoint touched only by code tools)."""
    if is_malformed_url(url):
        return "malformed"
    if is_search_url(url):
        return "search"
    if tool_call_only(citations):
        return "tool_call"
    return None


class Citation(BaseModel):
    bot: str | None = None
    question_id: str | None = None
    question_url: str | None = None
    run_id: str | None = None
    tool_name: str | None = None
    origin: str | None = None
    query: str | None = None
    cited_url: str = ""  # the original URL as cited (pre-canonicalization)


class Source(BaseModel):
    canonical_url: str
    domain: str
    captured: bool = False
    content_hash: str | None = None
    html_key: str | None = None  # store-relative (no prefix)
    screenshot_key: str | None = None
    markdown_key: str | None = None
    citations: list[Citation] = []

    @property
    def bots(self) -> list[str]:
        return sorted({c.bot for c in self.citations if c.bot})

    @property
    def question_ids(self) -> list[str]:
        return sorted({c.question_id for c in self.citations if c.question_id})


class CatalogData(BaseModel):
    sources: list[Source] = []
    excluded: dict[str, int] = {}  # exclusion reason -> count of URLs dropped

    @property
    def hidden_total(self) -> int:
        return sum(self.excluded.values())

    def by_question(self) -> dict[str, list[Source]]:
        out: dict[str, list[Source]] = defaultdict(list)
        for s in self.sources:
            qids = s.question_ids or [_UNKNOWN_Q]
            for qid in qids:
                out[qid].append(s)
        return out

    def by_bot(self) -> dict[str, list[Source]]:
        out: dict[str, list[Source]] = defaultdict(list)
        for s in self.sources:
            for bot in s.bots or ["(no bot)"]:
                out[bot].append(s)
        return out

    def by_domain(self) -> dict[str, list[Source]]:
        out: dict[str, list[Source]] = defaultdict(list)
        for s in self.sources:
            out[s.domain].append(s)
        return out

    def question_url(self, qid: str) -> str | None:
        for s in self.sources:
            for c in s.citations:
                if c.question_id == qid and c.question_url:
                    return c.question_url
        return None


# --------------------------------------------------------------------------- #
# Build (join manifests + index)
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _strip_prefix(key: str | None, prefix: str) -> str | None:
    if not key:
        return None
    p = prefix.rstrip("/") + "/"
    return key[len(p) :] if key.startswith(p) else key


def _latest_capture(store: ContentStore, canonical_url: str) -> dict | None:
    """Return the latest stored capture dict for a URL (ignoring TTL), following
    a redirect alias if present. ``None`` if nothing was ever captured."""
    index = store._read_index(url_hash(canonical_url))
    if not index:
        return None
    if index.get("alias_of"):
        index = store._read_index(index["alias_of"])
        if not index:
            return None
    ch = index.get("latest_content_hash")
    return (index.get("captures") or {}).get(ch)


def _load_all_records(store: BlobStore, prefix: str) -> list[CitationRecord]:
    records: list[CitationRecord] = []
    for key in store.list_keys(f"{prefix.rstrip('/')}/manifests/"):
        if not key.endswith(".jsonl"):
            continue
        try:
            records.extend(manifest_io.loads(store.get(key).decode("utf-8")))
        except (UnicodeDecodeError, ValueError):
            continue
    return records


def build_sources(store: BlobStore, config: ArchiveConfig) -> list[Source]:
    """Join every manifest with the index into one ``Source`` per canonical URL.

    Unfiltered (includes tool/API-call URLs) so other tools — e.g. the coverage
    report — can classify them. The catalog itself filters these out.
    """
    prefix = config.s3_prefix.rstrip("/")
    cstore = ContentStore(store, config)
    records = _load_all_records(store, prefix)

    grouped: dict[str, list[CitationRecord]] = defaultdict(list)
    for r in records:
        if r.url:
            grouped[canonicalize_url(r.url)].append(r)

    sources: list[Source] = []
    for canonical, recs in sorted(grouped.items()):
        cap = _latest_capture(cstore, canonical)
        source = Source(
            canonical_url=canonical,
            domain=_domain(canonical) or "(unknown)",
            captured=cap is not None,
            content_hash=(cap or {}).get("content_hash"),
            html_key=_strip_prefix((cap or {}).get("html_key"), prefix),
            screenshot_key=_strip_prefix((cap or {}).get("screenshot_key"), prefix),
            markdown_key=_strip_prefix((cap or {}).get("markdown_key"), prefix),
            citations=[
                Citation(
                    bot=r.bot,
                    question_id=r.question_id or r.metaculus_id,
                    question_url=r.question_url,
                    run_id=r.run_id,
                    tool_name=r.tool_name,
                    origin=r.origin,
                    query=r.query,
                    cited_url=r.url,
                )
                for r in recs
            ],
        )
        sources.append(source)
    return sources


def build_catalog(store: BlobStore, config: ArchiveConfig) -> CatalogData:
    sources = build_sources(store, config)
    pages: list[Source] = []
    excluded: dict[str, int] = defaultdict(int)
    for s in sources:
        reason = exclusion_reason(s.canonical_url, s.citations)
        if reason:
            excluded[reason] += 1
        else:
            pages.append(s)
    return CatalogData(sources=pages, excluded=dict(excluded))


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#fafafa}
header{background:#1f2937;color:#fff;padding:16px 24px}
header a{color:#cbd5e1}
h1{font-size:20px;margin:0 0 4px}
.wrap{padding:24px;max-width:1100px;margin:0 auto}
.muted{color:#6b7280}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px}
.ok{background:#dcfce7;color:#166534}.no{background:#fee2e2;color:#991b1b}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin:12px 0;display:flex;gap:12px}
.card img{width:160px;height:110px;object-fit:cover;object-position:top;border:1px solid #e5e7eb;border-radius:4px;background:#f3f4f6}
.card .meta{flex:1;min-width:0}
.card .u{font-weight:600;word-break:break-all}
.tags{margin-top:6px}
.tag{display:inline-block;background:#eef2ff;color:#3730a3;font-size:11px;padding:1px 7px;border-radius:10px;margin:2px 4px 2px 0}
.links a{margin-right:10px;font-size:12px}
table{border-collapse:collapse;width:100%;background:#fff}
td,th{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;font-size:13px}
th{background:#f3f4f6}
a.grid{display:inline-block;margin:4px 12px 4px 0}
"""


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def _page(title: str, body: str, rel_root: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<header><h1>Source Archive</h1>"
        f"<a href='{rel_root}catalog/index.html'>← catalog home</a></header>"
        f"<div class='wrap'>{body}</div></body></html>"
    )


class Linker:
    """Turns a store-relative blob key into a link a coworker can open."""

    def __init__(self, store: BlobStore, config: ArchiveConfig):
        from forecasting_tools.agents_and_tools.source_archive.storage import (
            S3BlobStore,
        )

        self.is_s3 = isinstance(store, S3BlobStore)
        self.bucket = config.s3_bucket
        self.region = config.aws_region
        self.prefix = config.s3_prefix.rstrip("/")

    def url(self, rel_key: str | None, rel_root: str) -> str | None:
        if not rel_key:
            return None
        if self.is_s3:
            host = (
                f"{self.bucket}.s3.{self.region}.amazonaws.com"
                if self.region
                else f"{self.bucket}.s3.amazonaws.com"
            )
            return f"https://{host}/{self.prefix}/{rel_key}"
        return f"{rel_root}{rel_key}"  # local: relative within the prefix dir


def _source_card(s: Source, linker: Linker, rel_root: str) -> str:
    shot = linker.url(s.screenshot_key, rel_root)
    html_link = linker.url(s.html_key, rel_root)
    md_link = linker.url(s.markdown_key, rel_root)
    badge = (
        "<span class='badge ok'>captured</span>"
        if s.captured
        else "<span class='badge no'>not captured</span>"
    )
    img = (
        f"<a href='{_esc(shot)}'><img src='{_esc(shot)}' alt='screenshot'></a>"
        if shot
        else "<div class='card-img'></div>"
    )
    tools = sorted({c.tool_name for c in s.citations if c.tool_name})
    tags = "".join(f"<span class='tag'>{_esc(b)}</span>" for b in s.bots)
    tool_tags = "".join(f"<span class='tag'>{_esc(t)}</span>" for t in tools)
    links = []
    if html_link:
        links.append(f"<a href='{_esc(html_link)}'>HTML</a>")
    if md_link:
        links.append(f"<a href='{_esc(md_link)}'>markdown</a>")
    if shot:
        links.append(f"<a href='{_esc(shot)}'>screenshot</a>")
    links.append(f"<a href='{_esc(s.canonical_url)}'>live ↗</a>")
    return (
        f"<div class='card'>{img}<div class='meta'>"
        f"<div class='u'>{_esc(s.canonical_url)}</div>"
        f"<div class='muted'>{_esc(s.domain)} · {badge}</div>"
        f"<div class='tags'>{tags}{tool_tags}</div>"
        f"<div class='links'>{' '.join(links)}</div>"
        f"</div></div>"
    )


def _question_csv(sources: list[Source]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["url", "domain", "captured", "bots", "tools", "screenshot_key"])
    for s in sources:
        tools = sorted({c.tool_name for c in s.citations if c.tool_name})
        w.writerow(
            [
                s.canonical_url,
                s.domain,
                "yes" if s.captured else "no",
                "; ".join(s.bots),
                "; ".join(tools),
                s.screenshot_key or "",
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
class CatalogSummary(BaseModel):
    sources: int = 0
    captured: int = 0
    questions: int = 0
    bots: int = 0
    domains: int = 0
    excluded: dict[str, int] = {}

    def __str__(self) -> str:
        excl = sum(self.excluded.values())
        breakdown = (
            " (" + ", ".join(f"{k}={v}" for k, v in sorted(self.excluded.items())) + ")"
            if self.excluded
            else ""
        )
        return (
            f"Catalog: {self.sources} page sources ({self.captured} captured) across "
            f"{self.questions} questions, {self.bots} bots, {self.domains} domains "
            f"— {excl} non-page URLs excluded{breakdown}"
        )


def _slug(value: str) -> str:
    # Keep dots so domains stay readable (a.test.html); collapse anything else.
    keep = [c if c.isalnum() or c in "-_." else "-" for c in value]
    out = "".join(keep).strip("-.").replace("..", ".")[:80]
    return out or "x"


def write_catalog(
    store: BlobStore,
    config: ArchiveConfig,
    out_store: BlobStore | None = None,
) -> CatalogSummary:
    """Build the catalog from ``store`` and write it to ``out_store`` (default:
    ``store``). Pass a separate ``out_store`` to preview a live bucket's catalog
    into a local directory without mutating the bucket."""
    prefix = config.s3_prefix.rstrip("/")
    data = build_catalog(store, config)
    out = out_store or store
    linker = Linker(out, config)

    def put(rel: str, body: str, ctype: str) -> None:
        out.put(f"{prefix}/catalog/{rel}", body.encode("utf-8"), content_type=ctype)

    by_q = data.by_question()
    by_b = data.by_bot()
    by_d = data.by_domain()

    # Per-question pages (the encyclopedia) + CSVs. rel_root: catalog/<view>/ -> ../../
    rr2 = "../../"
    for qid, sources in sorted(by_q.items()):
        sources = sorted(sources, key=lambda s: s.canonical_url)
        qurl = data.question_url(qid)
        head = f"<h1>Question {_esc(qid)}</h1>"
        if qurl:
            head += f"<p><a href='{_esc(qurl)}'>{_esc(qurl)} ↗</a></p>"
        head += (
            f"<p class='muted'>{len(sources)} source(s); "
            f"{sum(s.captured for s in sources)} captured · "
            f"<a href='{_slug(qid)}.csv'>download CSV</a></p>"
        )
        cards = "".join(_source_card(s, linker, rr2) for s in sources)
        put(
            f"by-question/{_slug(qid)}.html",
            _page(f"Question {qid}", head + cards, rr2),
            "text/html",
        )
        put(f"by-question/{_slug(qid)}.csv", _question_csv(sources), "text/csv")

    # Per-bot and per-domain cross-views.
    for bot, sources in sorted(by_b.items()):
        sources = sorted(sources, key=lambda s: s.canonical_url)
        body = f"<h1>Bot: {_esc(bot)}</h1><p class='muted'>{len(sources)} source(s)</p>"
        body += "".join(_source_card(s, linker, rr2) for s in sources)
        put(f"by-bot/{_slug(bot)}.html", _page(f"Bot {bot}", body, rr2), "text/html")

    for domain, sources in sorted(by_d.items()):
        sources = sorted(sources, key=lambda s: s.canonical_url)
        body = f"<h1>Site: {_esc(domain)}</h1><p class='muted'>{len(sources)} source(s)</p>"
        body += "".join(_source_card(s, linker, rr2) for s in sources)
        put(
            f"by-domain/{_slug(domain)}.html",
            _page(f"Site {domain}", body, rr2),
            "text/html",
        )

    # Landing + readme. rel_root: catalog/ -> ../
    rr1 = "../"
    index_body = _index_body(data, by_q, by_b, by_d)
    put("index.html", _page("Catalog", index_body, rr1), "text/html")
    put("READ_ME_FIRST.html", _page("Read me first", _readme_body(), rr1), "text/html")

    return CatalogSummary(
        sources=len(data.sources),
        captured=sum(s.captured for s in data.sources),
        questions=len(by_q),
        bots=len(by_b),
        domains=len(by_d),
        excluded=data.excluded,
    )


def _index_body(data, by_q, by_b, by_d) -> str:
    captured = sum(s.captured for s in data.sources)

    def links(items: dict, view: str) -> str:
        rows = []
        for key, sources in sorted(items.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            rows.append(
                f"<a class='grid' href='{view}/{_slug(key)}.html'>"
                f"{_esc(key)} <span class='muted'>({len(sources)})</span></a>"
            )
        return "".join(rows)

    hidden_note = (
        f" · {data.hidden_total} non-page URLs hidden "
        f"({', '.join(f'{k} {v}' for k, v in sorted(data.excluded.items()))})"
        if data.hidden_total
        else ""
    )
    return (
        f"<p><a href='READ_ME_FIRST.html'>What is this? →</a></p>"
        f"<p class='muted'>{len(data.sources)} page sources ({captured} captured) · "
        f"{len(by_q)} questions · {len(by_b)} bots · {len(by_d)} sites{hidden_note}</p>"
        f"<h1>By question</h1><p class='muted'>The encyclopedia of sources per "
        f"question — start here.</p>{links(by_q, 'by-question')}"
        f"<h1>By bot</h1>{links(by_b, 'by-bot')}"
        f"<h1>By site</h1>{links(by_d, 'by-domain')}"
    )


def _readme_body() -> str:
    return (
        "<h1>What is this bucket?</h1>"
        "<p>This is a <b>source archive</b>: for every web page a forecasting bot "
        "cited, we save a snapshot — the page's <b>HTML</b>, a full-page "
        "<b>screenshot</b>, and a clean <b>markdown</b> copy — so a forecast can be "
        "audited later even if the original page changes or disappears.</p>"
        "<h1>How to browse it</h1>"
        "<ul>"
        "<li>Open <b>index.html</b> (the catalog home).</li>"
        "<li><b>By question</b> is the main view: pick a question to see every "
        "source used for it, who used it, and a screenshot of each.</li>"
        "<li><b>By bot</b> shows one bot's sources across questions; <b>By site</b> "
        "groups sources by website.</li>"
        "<li>Each question also has a <b>CSV</b> you can open in a spreadsheet.</li>"
        "</ul>"
        "<p class='muted'>The folders with long hash names (content/, index/) are "
        "the machine-readable store — you don't need to open those.</p>"
    )
