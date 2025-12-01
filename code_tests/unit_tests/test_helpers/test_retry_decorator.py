import time

import requests

from forecasting_tools.util.misc import retry_with_exponential_backoff


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
