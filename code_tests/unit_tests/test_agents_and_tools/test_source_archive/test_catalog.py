from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive import manifest as manifest_io
from forecasting_tools.agents_and_tools.source_archive.catalog import (
    build_catalog,
    write_catalog,
)
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.models import (
    CaptureResult,
    CitationRecord,
)
from forecasting_tools.agents_and_tools.source_archive.storage import LocalBlobStore


def _capture(url: str, html: str) -> CaptureResult:
    return CaptureResult(
        url=url,
        final_url=url,
        status_code=200,
        html=html,
        markdown="md " * 30,
        screenshot=b"img",
        screenshot_content_type="image/png",
        fetcher="fake",
    )


def _seed(tmp_path):
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    cstore = ContentStore(store, config)
    cstore.store(_capture("https://a.test/p", "<p>a</p>"))
    cstore.store(_capture("https://b.test/q", "<p>b</p>"))
    # uncaptured.test/x is cited but never captured.
    records = [
        CitationRecord(
            url="https://a.test/p?utm_source=news",  # canonicalizes to /p
            run_id="r1",
            bot="alpha",
            question_id="100",
            question_url="https://www.metaculus.com/questions/100/",
            tool_name="web_search",
        ),
        CitationRecord(
            url="https://b.test/q",
            run_id="r1",
            bot="beta",
            question_id="100",
            question_url="https://www.metaculus.com/questions/100/",
            tool_name="page_fetch",
        ),
        CitationRecord(
            url="https://uncaptured.test/x",
            run_id="r1",
            bot="alpha",
            question_id="100",
        ),
        # A data/API call made only via run_code -> excluded from the catalog.
        CitationRecord(
            url="https://data.test/api?fmt=csv",
            run_id="r1",
            bot="beta",
            question_id="100",
            tool_name="run_code",
        ),
    ]
    manifest_io.write_blob(store, "r1", records, config)
    return store, config


def test_build_catalog_joins_and_canonicalizes(tmp_path):
    store, config = _seed(tmp_path)
    data = build_catalog(store, config)

    # The two a.test variants collapse to one source; the run_code API call is
    # excluded (tool/API call, not a page).
    urls = {s.canonical_url for s in data.sources}
    assert urls == {
        "https://a.test/p",
        "https://b.test/q",
        "https://uncaptured.test/x",
    }
    assert data.excluded.get("tool_call") == 1
    assert "https://data.test/api?fmt=csv" not in urls
    captured = {s.canonical_url for s in data.sources if s.captured}
    assert captured == {"https://a.test/p", "https://b.test/q"}

    by_q = data.by_question()
    assert set(by_q) == {"100"}
    assert len(by_q["100"]) == 3
    by_bot = data.by_bot()
    assert set(by_bot) == {"alpha", "beta"}


def test_write_catalog_emits_views(tmp_path):
    store, config = _seed(tmp_path)
    summary = write_catalog(store, config)

    assert summary.sources == 3
    assert summary.captured == 2
    assert summary.questions == 1
    assert summary.excluded.get("tool_call") == 1

    keys = set(store.list_keys("t/catalog/"))
    assert "t/catalog/index.html" in keys
    assert "t/catalog/READ_ME_FIRST.html" in keys
    assert "t/catalog/by-question/100.html" in keys
    assert "t/catalog/by-question/100.csv" in keys
    assert "t/catalog/by-bot/alpha.html" in keys
    assert "t/catalog/by-domain/a.test.html" in keys

    q_html = store.get("t/catalog/by-question/100.html").decode("utf-8")
    assert "https://a.test/p" in q_html
    assert "alpha" in q_html  # bot tag present
    # Local links are relative into the content store.
    assert "../../content/" in q_html

    q_csv = store.get("t/catalog/by-question/100.csv").decode("utf-8")
    assert "https://uncaptured.test/x" in q_csv
    assert "no" in q_csv  # uncaptured row marked


def test_nested_views_group_by_day_question_bot(tmp_path):
    store = LocalBlobStore(tmp_path)
    config = ArchiveConfig(s3_prefix="t")
    ContentStore(store, config).store(_capture("https://a.test/p", "<p>a</p>"))
    records = [
        CitationRecord(
            url="https://a.test/p",
            run_id="daily-2026-06-30",
            bot="alpha",
            question_id="100",
            question_url="https://www.metaculus.com/questions/100/",
            first_seen="2026-06-30T12:00:00+00:00",
        ),
        CitationRecord(
            url="https://a.test/p",
            run_id="daily-2026-07-01",
            bot="beta",
            question_id="100",
            first_seen="2026-07-01T12:00:00+00:00",
        ),
    ]
    manifest_io.write_blob(store, "m", records, config)
    summary = write_catalog(store, config)

    assert summary.dates == 2
    keys = set(store.list_keys("t/catalog/"))
    assert "t/catalog/by-date/2026-06-30.html" in keys
    assert "t/catalog/by-date/2026-07-01.html" in keys

    day = store.get("t/catalog/by-date/2026-06-30.html").decode("utf-8")
    assert "Question 100" in day and "Bot alpha" in day  # day -> question -> bot
    q = store.get("t/catalog/by-question/100.html").decode("utf-8")
    assert "2026-06-30" in q and "2026-07-01" in q  # question -> date -> bot
    b = store.get("t/catalog/by-bot/beta.html").decode("utf-8")
    assert "Question 100" in b and "2026-07-01" in b  # bot -> question -> date
