from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.catalog import Citation, Source
from forecasting_tools.agents_and_tools.source_archive.coverage import (
    coverage_from_sources,
)


def _src(url, domain, captured, cits):
    return Source(canonical_url=url, domain=domain, captured=captured, citations=cits)


def _trace(bot, q, tool):
    return Citation(bot=bot, question_id=q, tool_name=tool, origin="tool_result")


def _comment(bot, q):
    return Citation(bot=bot, question_id=q, origin="metaculus_comment")


SOURCES = [
    _src(
        "https://a.test/1",
        "a.test",
        True,
        [_trace("template", "100", "scrape_webpage")],
    ),
    _src(
        "https://b.test/2",
        "b.test",
        False,
        [_trace("template", "100", "scrape_webpage")],
    ),
    _src("https://c.test/3", "c.test", True, [_comment("otherbot", "200")]),
    # run_code-only -> excluded as a tool/API call
    _src(
        "https://data.test/x",
        "data.test",
        False,
        [_trace("template", "100", "run_code")],
    ),
    # search-engine result page -> excluded as a non-source
    _src(
        "https://www.google.com/search?q=x",
        "google.com",
        False,
        [_trace("template", "100", "scrape_webpage")],
    ),
    # malformed (extractor junk) -> excluded
    _src(
        "https://a.test/y%5B1%5D",
        "a.test",
        False,
        [_trace("template", "100", "scrape_webpage")],
    ),
]


def test_trace_report_excludes_non_sources_and_counts_pages():
    r = coverage_from_sources(SOURCES, "trace")
    assert r.cited == 2  # a.test/1 + b.test/2 (data/search/malformed excluded)
    assert r.captured == 1
    assert r.pct == 50.0
    assert r.excluded == {"tool_call": 1, "search": 1, "malformed": 1}
    assert r.missing == 1
    assert r.missing_urls == ["https://b.test/2"]

    by_q = {row.label: (row.cited, row.captured) for row in r.by_question}
    assert by_q == {"100": (2, 1)}
    by_tool = {row.label: (row.cited, row.captured) for row in r.by_tool}
    assert by_tool == {"scrape_webpage": (2, 1)}
    missed = {row.label for row in r.missed_by_domain}
    assert missed == {"b.test"}


def test_comment_report_is_separate():
    r = coverage_from_sources(SOURCES, "comments")
    assert r.cited == 1  # only the metaculus_comment source
    assert r.captured == 1
    assert r.pct == 100.0
    assert {row.label for row in r.by_bot} == {"otherbot"}


def test_modes_do_not_bleed():
    trace = coverage_from_sources(SOURCES, "trace")
    comments = coverage_from_sources(SOURCES, "comments")
    assert "https://c.test/3" not in trace.missing_urls  # comment source not in trace
    # the trace bot never appears in the comment report
    assert "template" not in {row.label for row in comments.by_bot}


def test_csv_export_has_overall_row():
    csv_text = coverage_from_sources(SOURCES, "trace").to_csv()
    assert "group,label,cited,captured,pct" in csv_text
    assert "overall,trace,2,1,50.0" in csv_text


def test_outcomes_split_never_fetched_vs_failed():
    # b.test/2 is the only missing page source. With no outcome for it, it's a
    # pure collection gap (never fetched).
    r = coverage_from_sources(SOURCES, "trace", {"https://a.test/1": "stored"})
    assert r.has_outcomes is True
    assert r.missing_never_fetched == 1
    assert r.missing_fetch_failed == 0

    # If a run report shows b.test/2 was fetched and failed, it's a capture
    # problem, not a collection gap.
    r2 = coverage_from_sources(SOURCES, "trace", {"https://b.test/2": "error"})
    assert r2.missing_never_fetched == 0
    assert r2.missing_fetch_failed == 1
