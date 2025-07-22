import asyncio
import cProfile
import logging
import pstats
import time
from datetime import datetime
from typing import Any, Coroutine, Tuple

from forecasting_tools.util import async_batching, file_manipulation

logger = logging.getLogger(__name__)


async def time_coroutine(coroutine: Coroutine) -> Tuple[float, float, Any]:
    start_time = time.time()
    result = await coroutine
    end_time = time.time()
    duration = end_time - start_time
    return (duration, end_time, result)


def assert_coroutines_run_under_x_times_duration_of_benchmark(
    benchmark_coroutines: list[Coroutine],
    test_coroutines: list[Coroutine],
    x: int = 2,
    allowed_errors_or_timeouts: int = 0,
) -> None:
    # Time a regular ask_chat
    benchmark_durations = []
    for benchmark_coroutine in benchmark_coroutines:
        duration, _, _ = asyncio.run(time_coroutine(benchmark_coroutine))
        benchmark_durations.append(duration)
    average_benchmark_duration = sum(benchmark_durations) / len(benchmark_durations)
    max_allowed_duration = average_benchmark_duration * x

    # Wrap the coroutines
    timing_coroutines = [time_coroutine(coroutine) for coroutine in test_coroutines]
    timed_timedout_coroutines = async_batching.wrap_coroutines_with_timeout(
        timing_coroutines, max_allowed_duration
    )
    timed_timedout_and_error_handled_coroutines = (
        async_batching.wrap_coroutines_to_return_not_raise_exceptions(
            timed_timedout_coroutines
        )
    )

    # Time the async_batching ask_chat
    list_start_time = time.time()
    duration_result_tuples: list[Tuple[float, float, Any] | Exception] = (
        async_batching.run_coroutines(timed_timedout_and_error_handled_coroutines)
    )
    list_end_time = time.time()

    # Create the stats
    errored_results: list[Exception] = []
    non_errored_durations: list[float] = []
    non_errored_end_times: list[float] = []
    non_errored_results: list[Any] = []
    for duration_result in duration_result_tuples:
        if isinstance(duration_result, Exception):
            errored_results.append(duration_result)
        else:
            non_errored_durations.append(duration_result[0])
            non_errored_end_times.append(duration_result[1])
            non_errored_results.append(duration_result[2])
    max_non_errored_end_time = max(non_errored_end_times)
    non_errored_full_duration = max_non_errored_end_time - list_start_time
    full_duration_of_coroutine_list = list_end_time - list_start_time
    if len(non_errored_durations) == 0:
        average_non_errored_duration = 0
    else:
        average_non_errored_duration = sum(non_errored_durations) / len(
            non_errored_durations
        )

    # Log the stats
    non_errored_end_times_with_start_time_as_0 = [
        end_time - list_start_time for end_time in non_errored_end_times
    ]
    logger.info(f"Benchmark Durations: {benchmark_durations}")
    logger.info(f"Average Benchmark duration: {average_benchmark_duration}")
    logger.info(f"Number of coroutines: {len(test_coroutines)}")
    logger.info(f"Number of non-errored results: {len(non_errored_results)}")
    logger.info(f"Number of errored results: {len(errored_results)}")
    logger.info(f"Non-errored durations: {non_errored_durations}")
    logger.info(f"Non-errored end times {non_errored_end_times_with_start_time_as_0}")
    logger.info(f"Average non-errored duration: {average_non_errored_duration}")
    logger.info(f"Full duration of test: {full_duration_of_coroutine_list}")
    logger.info(f"Full duration of coroutines w/o errors: {non_errored_full_duration}")
    logger.info(f"Max allowed duration: {max_allowed_duration}")
    logger.info(f"Not errored results: {non_errored_results}")
    logger.info(f"Errored results: {errored_results}")

    # Raise errors in results
    if len(errored_results) > allowed_errors_or_timeouts:
        raise RuntimeError(
            f"{len(errored_results)} of the coroutines brought up an error while running. Here are the exceptions: {errored_results}"
        )

    # Assert the time is as expected
    assert (
        non_errored_full_duration < max_allowed_duration
    ), f"Duration of coroutines that did not timeout/error was greater than max allowed duration. Actual Duration: {non_errored_full_duration} Max Allowed Duration: {max_allowed_duration} Time of benchmark: {average_benchmark_duration}"


class CoroutineTestInfo:
    def __init__(self, start_time: float, end_time: float, number_of_runs: int):
        self.start_time = start_time
        self.end_time = end_time
        self.number_of_runs = number_of_runs
        self.duration_in_seconds = end_time - start_time
        self.start_time_as_datetime = datetime.fromtimestamp(start_time)
        self.end_time_as_datetime = datetime.fromtimestamp(end_time)
        self.calls_per_second = number_of_runs / self.duration_in_seconds


def find_stats_of_coroutine_run(
    coroutines: list[Coroutine],
) -> CoroutineTestInfo:
    start_time = time.time()
    async_batching.run_coroutines(coroutines)
    end_time = time.time()
    number_of_function_calls = len(coroutines)
    return CoroutineTestInfo(start_time, end_time, number_of_function_calls)


def profile_coroutine_list(coroutine_list: list[Coroutine]) -> pstats.Stats:
    # Profile the code
    profiler = cProfile.Profile()
    logger.info(f"Profiling {len(coroutine_list)} coroutines")
    profiler.enable()

    # Name of first function in coroutine list
    coroutine_name = coroutine_list[0].cr_frame.f_code.co_name  # type: ignore

    # Run the code to be profiled
    async_batching.run_coroutines(coroutine_list)

    # Stop profiling
    profiler.disable()
    logger.info(f"Finished profiling {len(coroutine_list)} coroutines")

    # Create a statistics object
    log_path = file_manipulation.normalize_package_path(
        f"logs/misc/profile_stats_{coroutine_name}_{time.time()}.txt"
    )
    with open(log_path, "w") as file:
        stats = pstats.Stats(profiler, stream=file)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

    return stats


def assert_resource_rate_not_too_high_or_too_low(
    total_resources_used: int,
    duration_of_coroutines_in_seconds: float,
    allowed_resources_per_period: int,
    period_length_in_seconds: int,
    over_rate_allowed: float = 1.005,
    under_rate_allowed: float = 0.9,
):
    average_resources_per_second = (
        total_resources_used / duration_of_coroutines_in_seconds
    )
    allowed_upper_bound = (
        allowed_resources_per_period / period_length_in_seconds * over_rate_allowed
    )
    allowed_lower_bound = (
        allowed_resources_per_period / period_length_in_seconds * under_rate_allowed
    )

    additional_message = f"Average resources per second: {average_resources_per_second}, duration of coroutines: {duration_of_coroutines_in_seconds}, Total resources used: {total_resources_used}, allowed resources per period: {allowed_resources_per_period}, period length: {period_length_in_seconds}"
    assert (
        average_resources_per_second < allowed_upper_bound
    ), f"Resource rate exceeded upper bound of {allowed_upper_bound}. {additional_message}"
    assert (
        average_resources_per_second > allowed_lower_bound
    ), f"Resource rate was lower than lower bound of {allowed_lower_bound}. {additional_message}"


def assert_resource_burst_is_short(burst_size: int, duration: float) -> None:
    average_time_to_run_a_coroutine_by_computer = (
        0.0567936897277832  # From manutal test of 1000 'add two number' coroutines
    )
    expected_time_to_run_all_coroutines = (
        average_time_to_run_a_coroutine_by_computer * burst_size
    )
    target_duration = expected_time_to_run_all_coroutines * 1.1
    assert (
        duration < target_duration
    ), f"The burst took too long. Expected duration: {expected_time_to_run_all_coroutines}, Actual duration: {duration}"
