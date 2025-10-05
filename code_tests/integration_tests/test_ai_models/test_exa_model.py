from datetime import datetime
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from forecasting_tools.ai_models.exa_searcher import (
    ExaHighlightQuote,
    ExaSearcher,
    ExaSource,
    SearchInput,
)


async def test_invoke_for_highlights_in_relevance_order(mocker: Mock) -> None:
    mock_return_value = [
        ExaSource(
            original_query="test query",
            auto_prompt_string=None,
            title="Test Title 1",
            url="https://example1.com",
            text="Test text 1",
            author=None,
            published_date=None,
            score=0.9,
            highlights=["Highlight 1A", "Highlight 1B"],
            highlight_scores=[0.8, 0.6],
        ),
        ExaSource(
            original_query="test query",
            auto_prompt_string=None,
            title="Test Title 2",
            url="https://example2.com",
            text="Test text 2",
            author=None,
            published_date=None,
            score=0.85,
            highlights=["Highlight 2A", "Highlight 2B", "Highlight 2C"],
            highlight_scores=[0.75, 0.7, 0.65],
        ),
    ]
    AiModelMockManager.mock_ai_model_direct_call_with_value(
        mocker, ExaSearcher, mock_return_value
    )

    searcher = ExaSearcher()
    cheap_input = searcher._get_cheap_input_for_invoke()
    result = await searcher.invoke_for_highlights_in_relevance_order(cheap_input)

    assert len(result) == 5
    expected_highlights = [
        ("Highlight 1A", 0.8),
        ("Highlight 2A", 0.75),
        ("Highlight 2B", 0.7),
        ("Highlight 2C", 0.65),
        ("Highlight 1B", 0.6),
    ]

    for i, (expected_highlight, expected_score) in enumerate(expected_highlights):
        assert isinstance(result[i], ExaHighlightQuote)
        assert result[i].highlight_text == expected_highlight
        assert (
            abs(result[i].score - expected_score) < 1e-6
        ), f"Score was {result[i].score}, expected {expected_score}"

    # Check that the highlights are in descending order of score
    for i in range(len(result) - 1):
        assert (
            result[i].score >= result[i + 1].score
        ), f"Highlights not in descending order at index {i}"


async def test_general_invoke() -> None:
    num_results = 2
    model = ExaSearcher(
        num_results=num_results, include_highlights=True, include_text=False
    )
    cheap_input = model._get_cheap_input_for_invoke()
    sources = await model.invoke(cheap_input)
    assert len(sources) == num_results
    for source in sources:
        assert isinstance(source, ExaSource)
        assert source.url
        assert source.text is None
        # assert len(source.highlights) == model.num_highlights_per_url # <- Sometimes sources just don't have enough text or failed to scrape


@pytest.mark.skip(
    reason="Filter is not working, but we haven't needed this for a while"
)  # TODO: Fix filter
async def test_filtered_invoke() -> None:
    # TODO: See if Exa has a fix for this failing, but its probably something on their end.
    num_results = 3
    model = ExaSearcher(
        num_results=num_results, include_highlights=False, include_text=True
    )
    search = SearchInput(
        web_search_query="coronavirus",
        highlight_query=None,
        include_domains=[],
        exclude_domains=["alliance.health"],
        include_text="pregnancy",
        start_published_date=datetime(2022, 11, 1),
        end_published_date=datetime(2022, 11, 30),
    )
    sources = await model.invoke(search)

    assert any([source.published_date is not None for source in sources])
    for source in sources:
        assert source.text is not None and source.text != ""

        # As of Dec 19 2024 it seems that Exa sometimes doesn't return a publish date with a source, even if the source is properly published in the date range filter
        if source.published_date is not None:
            assert search.start_published_date is not None
            assert search.end_published_date is not None
            assert source.published_date <= search.end_published_date
            assert source.published_date >= search.start_published_date

        assert search.include_text is not None
        assert search.include_text in source.text
        assert source.url is not None
        assert all(
            exclude_domain not in source.url
            for exclude_domain in search.exclude_domains
        )
        assert len(source.highlights) == 0
        assert len(source.highlight_scores) == 0

    assert len(sources) == num_results


@pytest.mark.skip(reason="Not implemented yet. Currently cost more than is worth it")
async def test_with_only_urls() -> None:
    raise NotImplementedError
