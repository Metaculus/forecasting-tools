import time

import pytest
import requests

from forecasting_tools.util.misc import (
    raise_for_status_with_additional_info,
    retry_with_exponential_backoff,
)


def _make_response(status_code: int, body: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://example.com/api/"
    response.reason = "Reason"
    response.encoding = "utf-8"
    response._content = body.encode()
    return response


def test_client_error_raises_without_retry() -> None:
    call_count = 0
    error_body = (
        '{"non_field_errors": ["Probabilities must be between 0.001 and 0.999"]}'
    )

    @retry_with_exponential_backoff(max_retries=3, initial_delay=0)
    def mock_request() -> None:
        nonlocal call_count
        call_count += 1
        raise_for_status_with_additional_info(_make_response(400, error_body))

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        mock_request()

    assert call_count == 1
    assert exc_info.value.response is not None
    assert exc_info.value.response.status_code == 400
    assert "Probabilities must be between 0.001 and 0.999" in str(exc_info.value)


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.HTTPError(
            "Too many requests", response=_make_response(429, "")
        ),
        requests.exceptions.HTTPError(
            "Service unavailable", response=_make_response(503, "")
        ),
        requests.exceptions.HTTPError("No response attached"),
        requests.exceptions.Timeout("Request timed out"),
        requests.exceptions.ConnectionError("Connection failed"),
    ],
)
def test_transient_errors_are_retried(error: Exception) -> None:
    call_count = 0
    max_retries = 2

    @retry_with_exponential_backoff(max_retries=max_retries, initial_delay=0)
    def mock_request() -> None:
        nonlocal call_count
        call_count += 1
        raise error

    with pytest.raises(type(error)):
        mock_request()

    assert call_count == max_retries + 1


def test_retry_success_on_first_attempt():
    @retry_with_exponential_backoff(max_retries=3)
    def mock_request():
        return "success"

    result = mock_request()
    assert result == "success"
    print("✓ Test 1 passed: Success on first attempt")


def test_retry_success_after_failures():
    call_count = 0

    @retry_with_exponential_backoff(max_retries=3, initial_delay=0.1)
    def mock_request():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.exceptions.ConnectionError("Connection failed")
        return "success"

    result = mock_request()
    assert result == "success"
    assert call_count == 3
    print(f"✓ Test 2 passed: Success after {call_count} attempts")


def test_retry_exhausted():
    call_count = 0

    @retry_with_exponential_backoff(max_retries=2, initial_delay=0.1)
    def mock_request():
        nonlocal call_count
        call_count += 1
        raise requests.exceptions.Timeout("Request timed out")

    try:
        mock_request()
        assert False, "Should have raised an exception"
    except requests.exceptions.Timeout:
        print(f"✓ Test 3 passed: Exception raised after {call_count} attempts")


def test_exponential_backoff_timing():
    call_count = 0
    start_time = time.time()

    @retry_with_exponential_backoff(
        max_retries=3, initial_delay=0.1, exponential_base=2.0
    )
    def mock_request():
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            raise requests.exceptions.ConnectionError("Connection failed")
        return "success"

    result = mock_request()
    elapsed_time = time.time() - start_time

    assert result == "success"
    assert call_count == 4
    print(
        f"✓ Test 4 passed: Exponential backoff timing correct (elapsed: {elapsed_time:.2f}s)"
    )


if __name__ == "__main__":
    print("Testing retry_with_exponential_backoff decorator...\n")

    test_retry_success_on_first_attempt()
    test_retry_success_after_failures()
    test_retry_exhausted()
    test_exponential_backoff_timing()

    print("\n✅ All tests passed!")
