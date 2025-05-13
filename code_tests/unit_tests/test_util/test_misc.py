import pytest
import requests

from forecasting_tools.util.misc import (
    fill_in_citations,
    raise_for_status_with_additional_info,
)


def test_raise_for_status_raises_error_properly() -> None:
    url_that_does_not_exist = "https://www.google.com/this_page_does_not_exist"
    response = requests.get(url_that_does_not_exist)
    with pytest.raises(requests.exceptions.HTTPError):
        raise_for_status_with_additional_info(response)


def test_raise_for_status_does_not_raise_error_improperly() -> None:
    url_that_exists = "https://www.google.com"
    response = requests.get(url_that_exists)
    raise_for_status_with_additional_info(response)


@pytest.mark.parametrize(
    "text, citations, expected_output",
    [
        # Basic case - simple citation
        (
            "This is a test [1]",
            ["https://example.com"],
            "This is a test [1](https://example.com)",
        ),
        # Citation with existing text in parentheses
        (
            "Citation [1](note)",
            ["https://example.com"],
            "Citation [1](https://example.com)",
        ),
        # Escaped brackets
        (
            "Citation \\[[1]\\]",
            ["https://example.com"],
            "Citation [1](https://example.com)",
        ),
        # Empty text
        ("", ["https://example.com"], ""),
        # Empty citation list
        ("This has a citation [1]", [], "This has a citation [1]"),
        # Citation number higher than available URLs
        ("Citation [2]", ["https://example.com"], "Citation [2]"),
        # Multiple citations of different numbers
        (
            "Citation [1] and [2]",
            ["https://example.com", "https://example2.com"],
            "Citation [1](https://example.com) and [2](https://example2.com)",
        ),
        # Multiple instances of same citation
        (
            "Citation [1] and another [1]",
            ["https://example.com"],
            "Citation [1](https://example.com) and another [1](https://example.com)",
        ),
        # Complex text with multiple citation formats
        (
            "Normal [1], escaped \\[[2]\\], with text [3](note)",
            ["https://ex1.com", "https://ex2.com", "https://ex3.com"],
            "Normal [1](https://ex1.com), escaped [2](https://ex2.com), with text [3](https://ex3.com)",
        ),
        # Text with no citations
        (
            "This has no citations",
            ["https://example.com"],
            "This has no citations",
        ),
        # Text with brackets but not citations
        (
            "This has [brackets] but no citations",
            ["https://example.com"],
            "This has [brackets] but no citations",
        ),
    ],
)
def test_populate_citations_works(
    text: str, citations: list[str], expected_output: str
) -> None:

    result = fill_in_citations(
        citations, text, False
    )  # Testing with use_citation_brackets=False
    assert result == expected_output
