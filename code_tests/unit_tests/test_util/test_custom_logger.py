import io
import logging
from collections.abc import Iterator

import httpx
import pytest

from forecasting_tools.util.custom_logger import HttpApiKeyFilter

FAKE_SERPAPI_KEY = "3f9c2a7be41d8e0f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f"
FAKE_GOOGLE_KEY = "AIzaSyD-fake0key1for2tests3only4abcdefg"
SERPAPI_URL = (
    "https://serpapi.com/search.json?engine=google_trends_trending_now&geo=US"
    f"&category_id=3&api_key={FAKE_SERPAPI_KEY}"
)


@pytest.fixture
def captured_log() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    handler.addFilter(HttpApiKeyFilter())
    logger = logging.getLogger("metaculus_ai_sdk.seed.sources.google_trends")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, stream
    logger.removeHandler(handler)


def test_key_in_a_non_httpx_warning_is_redacted(
    captured_log: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = captured_log
    logger.warning(
        f"Google Trends bucket 'climate' failed: Client error '400 Bad Request' for url '{SERPAPI_URL}'"
    )
    output = stream.getvalue()
    assert FAKE_SERPAPI_KEY not in output
    assert "api_key=REDACTED'" in output
    assert "engine=google_trends_trending_now&geo=US&category_id=3" in output


def test_key_passed_as_a_format_argument_is_redacted(
    captured_log: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = captured_log
    logger.info(
        "HTTP Request: %s %s",
        "GET",
        f"https://maps.googleapis.com/maps/api/geocode/json?address=Paris&key={FAKE_GOOGLE_KEY}",
    )
    output = stream.getvalue()
    assert FAKE_GOOGLE_KEY not in output
    assert "address=Paris&key=REDACTED" in output


def test_key_in_logged_exception_text_is_redacted(
    captured_log: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = captured_log
    request = httpx.Request("GET", SERPAPI_URL)
    response = httpx.Response(400, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception("Google Trends request failed")
    output = stream.getvalue()
    assert "HTTPStatusError" in output
    assert FAKE_SERPAPI_KEY not in output
    assert "api_key=REDACTED" in output


def test_httpx_request_lines_are_kept_with_the_key_redacted() -> None:
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        __file__,
        1,
        'HTTP Request: GET %s "HTTP/1.1 200 OK"',
        (SERPAPI_URL,),
        None,
    )
    assert HttpApiKeyFilter().filter(record) is True
    assert FAKE_SERPAPI_KEY not in record.getMessage()


def test_urls_without_secrets_are_unchanged() -> None:
    text = (
        "Fetched https://api.example.com/v1/search?keyword=robotaxi&page=2&monkey_count=3 "
        "and https://www.metaculus.com/api/posts/?tournaments=fall-futureeval-2026&limit=100"
    )
    assert HttpApiKeyFilter.redact(text) == text


def test_token_and_secret_parameters_are_redacted() -> None:
    text = (
        "https://example.com/oauth?client_id=abc&client_secret=s3cr3tValue"
        "&access_token=tok_123.456&X-Amz-Signature=deadbeef"
    )
    redacted = HttpApiKeyFilter.redact(text)
    assert "s3cr3tValue" not in redacted
    assert "tok_123.456" not in redacted
    assert "deadbeef" not in redacted
    assert "client_id=abc" in redacted
