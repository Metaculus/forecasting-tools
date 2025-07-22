###################################### IMPORTS ######################################
import asyncio
import logging
import time
from typing import Coroutine

import pytest

from code_tests.utilities_for_tests.coroutine_testing import find_stats_of_coroutine_run
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


################################ Helper Functions ################################
class Counter:
    add_one_function_count = 0


def add_one_and_increment_counter(input: int) -> int:
    Counter.add_one_function_count += 1
    return input + 1


async def add_one_and_wait_set_time(input: int, seconds_to_wait: int) -> int:
    output = add_one_and_increment_counter(input)
    await asyncio.sleep(seconds_to_wait)
    return output


def create_irregular_time_coroutines(
    number_of_coroutines_to_make: int, max_time_wait_function_can_take: int
) -> list[Coroutine]:
    coroutines = []
    input_list = [i for i in range(number_of_coroutines_to_make)]
    possible_times_to_wait = [i for i in range(1, max_time_wait_function_can_take + 1)]

    # Assign times from bottom then top and in toward the middle
    for i, input in enumerate(input_list):
        index_of_times = i % len(possible_times_to_wait)
        if index_of_times % 2 == 0:
            time_to_wait = possible_times_to_wait[index_of_times]
        else:
            time_to_wait = possible_times_to_wait[-index_of_times]
        coroutines.append(add_one_and_wait_set_time(input, time_to_wait))

    return coroutines


def create_set_time_coroutines(
    number_of_coroutines_to_make: int, time_to_wait: int
) -> list[Coroutine]:
    coroutines = []
    input_list = [i for i in range(number_of_coroutines_to_make)]

    for input in input_list:
        coroutines.append(add_one_and_wait_set_time(input, time_to_wait))

    return coroutines


###################################### Tests ######################################
def test_run_coroutine_list_returns_correct_output_in_right_order() -> None:
    number_of_couroutines_to_make = 25
    max_time_wait_function_can_take = 5
    coroutines = create_irregular_time_coroutines(
        number_of_couroutines_to_make, max_time_wait_function_can_take
    )

    all_outputs = async_batching.run_coroutines(coroutines)

    for i in range(number_of_couroutines_to_make):
        expected_output_at_index = i + 1
        actual_output_at_index = all_outputs[i]
        assert (
            expected_output_at_index == actual_output_at_index
        ), f"Output was not correct at index {i}. Expected {expected_output_at_index}, got {actual_output_at_index}"


@pytest.mark.skip(reason="This test works by itself, but interferes with other tests")
def test_run_coroutine_list_runs_expected_duration() -> None:
    number_of_couroutines_to_make = 5000
    time_to_wait = 1
    coroutines = create_set_time_coroutines(number_of_couroutines_to_make, time_to_wait)

    start_time = time.time()
    results = async_batching.run_coroutines(coroutines)
    end_time = time.time()
    duration = end_time - start_time

    assert duration < 2, "The duration took longer than allowed seconds"
    assert (
        len(results) == number_of_couroutines_to_make
    ), f"The number of results was not correct. Expected {number_of_couroutines_to_make}, got {len(results)}"


def test_run_coroutine_list_not_blocked_by_logging() -> None:
    async def log_and_wait() -> int:
        logger.info("Test log")
        await asyncio.sleep(1)
        return 1

    number_of_couroutines_to_make = 2000
    coroutines = [log_and_wait() for _ in range(number_of_couroutines_to_make)]
    start_time = time.time()
    async_batching.run_coroutines(coroutines)
    end_time = time.time()
    duration = end_time - start_time

    assert (
        duration < 10
    ), f"The duration took longer than 10 seconds and thus seems to be blocking. Duration was {duration}"


def create_rate_limited_coroutines(
    number_of_coroutines: int, calls_per_period: int, time_period: int
) -> list[Coroutine]:
    time_to_wait = 1
    coroutines = create_set_time_coroutines(number_of_coroutines, time_to_wait)
    rate_limited_coroutines = async_batching.wrap_coroutines_with_rate_limit(
        coroutines, calls_per_period, time_period
    )
    return rate_limited_coroutines


@pytest.mark.skip(reason="This test takes a while to run")
def test_rate_limit_wrapper_achieves_average_rate_limit() -> None:
    num_coroutines_to_run = 1000
    calls_per_period = 100
    time_period = 1
    target_calls_per_second = calls_per_period / time_period
    allowed_lower_error = 5
    allowed_upper_error = 1.5
    rate_limited_coroutines = create_rate_limited_coroutines(
        num_coroutines_to_run, calls_per_period, time_period
    )

    stats = find_stats_of_coroutine_run(rate_limited_coroutines)
    calls_per_second = stats.calls_per_second
    duration = stats.duration_in_seconds
    logger.info(
        f"Duration was {duration} seconds, calls per second was {calls_per_second}"
    )
    assert (
        calls_per_second < target_calls_per_second + allowed_upper_error
    ), f"The calls per second was too high. Expected {target_calls_per_second}, got {calls_per_second}"
    assert (
        calls_per_second > target_calls_per_second - allowed_lower_error
    ), f"The calls per second was too low. Expected {target_calls_per_second}, got {calls_per_second}"


def test_rate_limit_wrapper_has_initial_call_burst() -> None:
    # 100 calls can be called in 100 seconds at this rate without breaking the limit
    calls_per_period = 1000
    time_period = 100
    num_coroutines_to_run = 100
    rate_limited_coroutines = create_rate_limited_coroutines(
        num_coroutines_to_run, calls_per_period, time_period
    )
    stats = find_stats_of_coroutine_run(rate_limited_coroutines)
    duration = stats.duration_in_seconds
    assert (
        duration < 5
    ), f"The duration took longer than 5 seconds and thus did not burst. Duration was {duration}"


def collect_result_of_3_second_sleep_coroutine_with_timeout(
    timeout_time: int,
) -> list[int]:
    async def long_running_coroutine() -> int:
        await asyncio.sleep(3)
        return 1

    coroutines = [long_running_coroutine() for _ in range(5)]
    timed_coroutines = async_batching.wrap_coroutines_with_timeout(
        coroutines, timeout_time
    )
    results = async_batching.run_coroutines(timed_coroutines)

    return results


def test_coroutine_timeout_error_thrown() -> None:
    with pytest.raises(asyncio.TimeoutError):
        collect_result_of_3_second_sleep_coroutine_with_timeout(1)


def test_no_timeout_error_thrown_when_long_timeout() -> None:
    try:
        collect_result_of_3_second_sleep_coroutine_with_timeout(4)
    except TimeoutError as e:
        assert (
            False
        ), f"A timeout error was thrown when it should not have been. Exception: {e}"
    except Exception as e:
        assert (
            False
        ), f"An exception was thrown when it should not have been. Exception: {e}"

    assert True, "No exception was thrown when it should have been"  # NOSONAR


def test_failed_coroutine_returns_exception_as_result() -> None:
    async def failing_coroutine() -> None:
        raise RuntimeError("Test exception")

    coroutines = [failing_coroutine() for _ in range(5)]
    exception_handled_coroutines = (
        async_batching.wrap_coroutines_to_return_not_raise_exceptions(coroutines)
    )
    results = async_batching.run_coroutines(exception_handled_coroutines)

    for result in results:
        assert isinstance(result, Exception), f"Expected an exception but got {result}"


def test_run_couroutines_with_action_called_on_exception() -> None:
    counter = 0

    def action_on_exception(e: Exception, input: int) -> None:
        nonlocal counter
        counter += 1

    async def failing_coroutine() -> int:
        raise RuntimeError("Test exception")

    async def passing_coroutine() -> int:
        return 1

    num_failures = 5
    num_succeses = 5
    total_calls = num_failures + num_succeses
    inputs = [i for i in range(total_calls)]
    coroutines = [failing_coroutine() for _ in range(num_failures)] + [
        passing_coroutine() for _ in range(num_succeses)
    ]
    results, inputs = (
        async_batching.run_coroutines_while_removing_and_logging_exceptions(
            coroutines, inputs, action_on_exception
        )
    )

    assert (
        counter == num_failures
    ), f"The action was not called the correct number of times. Expected {num_failures}, got {counter}"
    assert (
        len(results) == num_succeses
    ), f"The number of results was not correct. Expected {num_succeses}, got {len(results)}"
    assert len(inputs) == len(
        results
    ), f"The number of inputs and results was not the same. Inputs: {len(inputs)}, Results: {len(results)}"
    assert all(
        [not isinstance(result, Exception) for result in results]
    ), "Not all results were not exceptions"


def test__run_coroutines_with_action_called_on_exception__errors_if_bad_inputs() -> (
    None
):
    async def failing_coroutine() -> int:
        raise RuntimeError("Test exception")

    async def passing_coroutine() -> int:
        return 1

    num_failures = 5
    num_succeses = 5
    total_calls = num_failures + num_succeses
    inputs = [i for i in range(total_calls)]
    coroutines = [failing_coroutine() for _ in range(num_failures)] + [
        passing_coroutine() for _ in range(num_succeses)
    ]

    with pytest.raises(AssertionError):
        async_batching.run_coroutines_while_removing_and_logging_exceptions(
            coroutines, inputs[:1], lambda e, i: None
        )

    with pytest.raises(AssertionError):
        async_batching.run_coroutines_while_removing_and_logging_exceptions(
            coroutines[:1], inputs, lambda e, i: None
        )


def test__run_coroutines_with_action_called_on_exception__handles_no_matching_inputs() -> (
    None
):
    async def failing_coroutine() -> int:
        raise RuntimeError("Test exception")

    async def passing_coroutine() -> int:
        return 1

    num_failures = 5
    num_succeses = 5
    coroutines = [failing_coroutine() for _ in range(num_failures)] + [
        passing_coroutine() for _ in range(num_succeses)
    ]

    results, inputs = (
        async_batching.run_coroutines_while_removing_and_logging_exceptions(coroutines)
    )

    assert (
        len(results) == num_succeses
    ), f"The number of results was not correct. Expected {num_succeses}, got {len(results)}"
    assert len(inputs) == len(
        results
    ), f"The number of inputs and results was not the same. Inputs: {len(inputs)}, Results: {len(results)}"
    assert all(
        [isinstance(result, int) for result in results]
    ), "Not all results were integers"
    assert all(inputs == None for inputs in inputs), "Not all inputs were None"
