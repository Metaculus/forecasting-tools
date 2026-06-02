from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.ingest.url_extraction import (
    dedupe_records,
    extract_citation_records,
    extract_urls,
)


def test_extracts_markdown_autolink_and_bare():
    text = (
        "See [the report](https://a.test/report) and <https://b.test/page> "
        "plus bare https://c.test/x for details."
    )
    assert extract_urls(text) == [
        "https://a.test/report",
        "https://b.test/page",
        "https://c.test/x",
    ]


def test_trims_trailing_punctuation():
    assert extract_urls("ends a sentence at https://a.test/path.") == [
        "https://a.test/path"
    ]
    assert extract_urls("(see https://a.test/path)") == ["https://a.test/path"]


def test_keeps_balanced_parens_in_url():
    text = "https://en.wikipedia.org/wiki/Forecasting_(disambiguation)"
    assert extract_urls(text) == [
        "https://en.wikipedia.org/wiki/Forecasting_(disambiguation)"
    ]


def test_dedupes_preserving_order():
    text = "https://a.test x https://b.test y https://a.test"
    assert extract_urls(text) == ["https://a.test", "https://b.test"]


def test_ignores_non_http_and_empty():
    assert extract_urls("ftp://a.test mailto:x@y.test nope") == []
    assert extract_urls(None) == []
    assert extract_urls("") == []


def test_extract_citation_records_attaches_provenance():
    records = extract_citation_records(
        "source: https://a.test/r",
        run_id="r1",
        bot="demo",
        question_id="42",
        origin="metaculus_comment",
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.url == "https://a.test/r"
    assert rec.run_id == "r1"
    assert rec.bot == "demo"
    assert rec.question_id == "42"
    assert rec.origin == "metaculus_comment"


def test_dedupe_records_keeps_first():
    records = extract_citation_records("https://a.test https://a.test https://b.test")
    deduped = dedupe_records(records)
    assert [r.url for r in deduped] == ["https://a.test", "https://b.test"]
