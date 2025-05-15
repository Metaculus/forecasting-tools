from datetime import datetime, timedelta, timezone

import pytest

from forecasting_tools.ai_models.exa_searcher import SearchInput


def test_search_input_valid_times() -> None:
    now = datetime.now(timezone.utc)
    valid_input = SearchInput(
        web_search_query="test",
        highlight_query=None,
        include_domains=[],
        exclude_domains=[],
        include_text=None,
        start_published_date=now - timedelta(days=10),
        end_published_date=now - timedelta(days=1),
    )
    assert valid_input.start_published_date is not None
    assert valid_input.end_published_date is not None
    assert valid_input.start_published_date < valid_input.end_published_date


def test_search_input_invalid_times_start_after_end() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        SearchInput(
            web_search_query="test",
            highlight_query=None,
            include_domains=[],
            exclude_domains=[],
            include_text=None,
            start_published_date=now,
            end_published_date=now - timedelta(days=1),
        )


def test_search_input_invalid_times_start_in_future() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        SearchInput(
            web_search_query="test",
            highlight_query=None,
            include_domains=[],
            exclude_domains=[],
            include_text=None,
            start_published_date=now + timedelta(days=2),
            end_published_date=now + timedelta(days=3),
        )


def test_search_input_invalid_times_end_in_future() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        SearchInput(
            web_search_query="test",
            highlight_query=None,
            include_domains=[],
            exclude_domains=[],
            include_text=None,
            start_published_date=now - timedelta(days=2),
            end_published_date=now + timedelta(days=2),
        )


def test_search_input_valid_include_text() -> None:
    valid_input = SearchInput(
        web_search_query="test",
        highlight_query=None,
        include_domains=[],
        exclude_domains=[],
        include_text="one two three",
        start_published_date=None,
        end_published_date=None,
    )
    assert valid_input.include_text == "one two three"


def test_search_input_invalid_include_text_too_many_words() -> None:
    with pytest.raises(ValueError):
        SearchInput(
            web_search_query="test",
            highlight_query=None,
            include_domains=[],
            exclude_domains=[],
            include_text="one two three four five six",
            start_published_date=None,
            end_published_date=None,
        )


def test_search_input_invalid_include_text_too_few_words() -> None:
    with pytest.raises(ValueError):
        SearchInput(
            web_search_query="test",
            highlight_query=None,
            include_domains=[],
            exclude_domains=[],
            include_text="",
            start_published_date=None,
            end_published_date=None,
        )
