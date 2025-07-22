import asyncio
import logging
import random
import time

import pytest

from code_tests.utilities_for_tests import coroutine_testing
from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import (
    LimitReachedResponse,
    RefreshingBucketRateLimiter,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)
from typing import Coroutine


class ResourceLimiterTester:

    def __init__(
        self,
        refresh_rate: float,
        capacity: float,
        limit_reached_response: LimitReachedResponse = LimitReachedResponse.WAIT,
    ) -> None:
        self.resource_limiter = RefreshingBucketRateLimiter(
            capacity, refresh_rate, limit_reached_response
        )
        self.lock = asyncio.Lock()
        self.total_resource_counter = 0
        self.total_calls = 0

    async def increment_counter(self, resources_used: int) -> None:
        async with self.lock:
            self.total_resource_counter += resources_used
            self.total_calls += 1

    async def function_that_use_random_resources(
        self, resources_to_acquire_range: tuple[int, int]
    ) -> None:
        resources_acquired = random.randint(*resources_to_acquire_range)
        await self.resource_limiter.wait_till_able_to_acquire_resources(
            resources_acquired
        )
        await self.increment_counter(resources_acquired)

    async def function_using_fixed_resources(self, resources_to_acquire: int) -> None:
        await self.resource_limiter.wait_till_able_to_acquire_resources(
            resources_to_acquire
        )
        await self.increment_counter(resources_to_acquire)


def test_resource_manager_refreshes_at_right_rate() -> None:
    refresh_rate = 1
    capacity = 2
    tester = ResourceLimiterTester(refresh_rate, capacity)

    tester.resource_limiter.zero_out_resources()
    time.sleep(1)
    available_resources_1 = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )
    time.sleep(1)
    available_resources_2 = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )
    time.sleep(1)
    available_resources_3 = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )

    allowed_error = 0.03
    assert (
        abs(available_resources_1 - refresh_rate) < allowed_error
    ), "Resources did not refresh"
    assert (
        abs(available_resources_2 - refresh_rate * 2) < allowed_error
    ), "Resources did not refresh"
    assert (
        abs(available_resources_3 - capacity) < allowed_error
    ), "Resources refreshed too much"


def test_resource_manager_blocks_when_no_resources_for_single_coroutine() -> None:
    refresh_rate = 5
    capacity = 20
    resources_to_consume = capacity
    tester = ResourceLimiterTester(refresh_rate, capacity)
    tester.resource_limiter.zero_out_resources()

    stats = coroutine_testing.find_stats_of_coroutine_run(
        [tester.function_using_fixed_resources(resources_to_consume)]
    )
    duration = stats.duration_in_seconds
    seconds_it_should_take = resources_to_consume / refresh_rate

    allowed_error = 0.5
    assert (
        abs(seconds_it_should_take - duration) < allowed_error
    ), "Resource limiter did not block"


@pytest.mark.skip(reason="This test takes a while to run")
def test_resource_manager_blocks_when_no_resources_for_many_coroutines() -> None:
    refresh_rate = 1
    capacity = 10
    resources_to_consume = 1
    tester = ResourceLimiterTester(refresh_rate, capacity)
    tester.resource_limiter.zero_out_resources()
    number_of_coroutines_to_run = 20

    coroutines = [
        tester.function_using_fixed_resources(resources_to_consume)
        for _ in range(number_of_coroutines_to_run)
    ]
    stats = coroutine_testing.find_stats_of_coroutine_run(coroutines)
    duration = stats.duration_in_seconds
    seconds_it_should_take = (
        number_of_coroutines_to_run * resources_to_consume / refresh_rate
    )

    allowed_error = 1.3
    assert (
        abs(seconds_it_should_take - duration) < allowed_error
    ), "Resource limiter did not block"


def test_resources_taken_out_correctly_for_single_coroutine() -> None:
    refresh_rate = 0
    capacity = 10
    resources_to_consume = 5
    tester = ResourceLimiterTester(refresh_rate, capacity)

    asyncio.run(tester.function_using_fixed_resources(resources_to_consume))
    resources_afterwards_1 = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )
    correct_resources_1 = capacity - resources_to_consume

    asyncio.run(tester.function_using_fixed_resources(resources_to_consume))
    resources_afterwards_2 = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )
    correct_resources_2 = correct_resources_1 - resources_to_consume

    assert (
        resources_afterwards_1 == correct_resources_1
    ), "Resources not taken out correctly"
    assert (
        resources_afterwards_2 == correct_resources_2
    ), "Resources not taken out correctly"


def test_resources_taken_out_correctly_for_a_lot_of_calls_that_dont_zero_out_bucket() -> (
    None
):
    refresh_rate = 0
    capacity = 1000
    resources_to_consume = 1
    tester = ResourceLimiterTester(refresh_rate, capacity)
    number_of_coroutines_to_run_each_half = int(capacity / resources_to_consume / 2)

    first_coroutines = [
        tester.function_using_fixed_resources(resources_to_consume)
        for _ in range(number_of_coroutines_to_run_each_half)
    ]
    async_batching.run_coroutines(first_coroutines)
    resources_after_first_half = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )

    second_half_of_coroutines = [
        tester.function_using_fixed_resources(resources_to_consume)
        for _ in range(number_of_coroutines_to_run_each_half)
    ]
    async_batching.run_coroutines(second_half_of_coroutines)
    resources_after_second_half = (
        tester.resource_limiter.refresh_and_then_get_available_resources()
    )

    assert (
        resources_after_first_half == capacity / 2
    ), "Resources not taken out correctly"
    assert resources_after_second_half == 0, "Resources not taken out correctly"


def test_error_thrown_if_capacity_reached_with_zero_refresh() -> None:
    refresh_rate = 0
    capacity = 1
    resources_to_consume = 2
    tester = ResourceLimiterTester(refresh_rate, capacity)

    # This would wait forever if there was not an error
    with pytest.raises(Exception):
        asyncio.run(
            tester.resource_limiter.wait_till_able_to_acquire_resources(
                resources_to_consume
            )
        )


def test_error_mode_errors_when_resource_limit_reached_with_no_refresh() -> None:
    limit_reached_response = LimitReachedResponse.RAISE_EXCEPTION
    refresh_rate = 0
    capacity = 10
    resources_to_consume = 6
    tester = ResourceLimiterTester(refresh_rate, capacity, limit_reached_response)

    try:
        asyncio.run(
            tester.resource_limiter.wait_till_able_to_acquire_resources(
                resources_to_consume
            )
        )
    except Exception:
        raise AssertionError("Resource limiter errored when there was enough capacity")

    with pytest.raises(Exception):
        asyncio.run(
            tester.resource_limiter.wait_till_able_to_acquire_resources(
                resources_to_consume
            )
        )


def test_error_mode_errors_when_resource_limit_reached_with_refresh() -> None:
    limit_reached_response = LimitReachedResponse.RAISE_EXCEPTION
    refresh_rate = 1
    capacity = 10
    resources_to_consume = 6
    tester = ResourceLimiterTester(refresh_rate, capacity, limit_reached_response)

    try:
        asyncio.run(
            tester.resource_limiter.wait_till_able_to_acquire_resources(
                resources_to_consume
            )
        )
    except Exception:
        raise AssertionError("Resource limiter errored when there was enough capacity")

    time.sleep(2)

    try:
        asyncio.run(
            tester.resource_limiter.wait_till_able_to_acquire_resources(
                resources_to_consume
            )
        )
    except Exception:
        raise AssertionError("Resource limiter did not refresh correctly")

    with pytest.raises(Exception):
        asyncio.run(
            tester.resource_limiter.wait_till_able_to_acquire_resources(
                resources_to_consume
            )
        )


def test_waits_till_bucket_is_full_if_it_runs_out_of_resources() -> None:
    refresh_rate = 1
    capacity = 10
    resources_to_consume = 1
    tester = ResourceLimiterTester(refresh_rate, capacity)

    asyncio.run(tester.resource_limiter.wait_till_able_to_acquire_resources(capacity))
    assert tester.resource_limiter.refresh_and_then_get_available_resources() < 0.001

    start_time = time.time()
    asyncio.run(
        tester.resource_limiter.wait_till_able_to_acquire_resources(
            resources_to_consume
        )
    )
    end_time = time.time()
    duration = end_time - start_time

    expected_time = capacity / refresh_rate

    allowed_error = 0.1
    assert (
        abs(duration - expected_time) < allowed_error
    ), "Did not wait till bucket was full"


def test_rate_during_burst_is_respected() -> None:
    refresh_rate = 1
    capacity = 10
    resources_to_consume = 1
    tester = ResourceLimiterTester(refresh_rate, capacity)
    number_of_coroutines_to_run = 10

    coroutines = [
        tester.function_using_fixed_resources(resources_to_consume)
        for _ in range(number_of_coroutines_to_run)
    ]
    stats = coroutine_testing.find_stats_of_coroutine_run(coroutines)
    duration = stats.duration_in_seconds

    coroutine_testing.assert_resource_burst_is_short(
        burst_size=capacity, duration=duration
    )


@pytest.mark.skip(reason="This test takes a while to run")
def test_rate_is_respected_between_burst_and_after() -> None:
    allowed_resources_per_period = 100
    period_length_in_seconds = 10
    resources_to_consume = 1
    refresh_rate = allowed_resources_per_period / period_length_in_seconds

    tester = ResourceLimiterTester(refresh_rate, allowed_resources_per_period)

    number_of_burst_coroutines_to_run = (
        allowed_resources_per_period // resources_to_consume
    )
    number_of_regular_coroutines_to_run = int(number_of_burst_coroutines_to_run * 0.2)
    total_number_of_coroutines_to_run = (
        number_of_burst_coroutines_to_run + number_of_regular_coroutines_to_run
    )

    coroutines_to_run = [
        tester.function_using_fixed_resources(resources_to_consume)
        for _ in range(total_number_of_coroutines_to_run)
    ]
    assert_whether_rate_limit_is_respected_and_resources_tracked_right(
        tester,
        coroutines_to_run,
        period_length_in_seconds,
        allowed_resources_per_period,
    )


@pytest.mark.skip(reason="This test takes a while to run")
def test_resource_manager_works_with_a_lot_of_random_resource_requests() -> None:
    allowed_resources_per_period = 5000
    period_length_in_seconds = 2
    resources_to_use_range = (20, 50)
    number_of_coroutines_to_run = 1000

    refresh_rate = allowed_resources_per_period / period_length_in_seconds
    capacity = allowed_resources_per_period

    tester = ResourceLimiterTester(refresh_rate, capacity)
    coroutines = [
        tester.function_that_use_random_resources(resources_to_use_range)
        for _ in range(number_of_coroutines_to_run)
    ]
    assert_whether_rate_limit_is_respected_and_resources_tracked_right(
        tester,
        coroutines,
        period_length_in_seconds,
        allowed_resources_per_period,
    )


def assert_whether_rate_limit_is_respected_and_resources_tracked_right(
    tester: ResourceLimiterTester,
    coroutines: list[Coroutine],
    period_length_in_seconds: int,
    allowed_resources_per_period: int,
) -> None:
    stats = coroutine_testing.find_stats_of_coroutine_run(coroutines)
    duration = stats.duration_in_seconds

    if duration < period_length_in_seconds:
        raise ValueError("Test did not take long enough")

    total_resources_used = (
        tester.resource_limiter.calculate_resources_passed_into_acquire_in_time_range(
            stats.start_time_as_datetime, stats.end_time_as_datetime
        )
    )

    assert total_resources_used != 0, "No resources used in last period"
    assert (
        total_resources_used <= tester.total_resource_counter
    ), "Resource counters conflict"
    assert len(coroutines) == tester.total_calls, "Not all coroutines ran"

    log_message = "\n"
    log_message += f"Duration of coroutines: {duration}\n"
    log_message += f"Allowed resources per period: {allowed_resources_per_period}\n"
    log_message += f"Period length in seconds: {period_length_in_seconds}\n"
    log_message += f"Allowed resources per second: {allowed_resources_per_period / period_length_in_seconds}\n"
    log_message += (
        f"Resources used by coroutines over last period: {total_resources_used}\n"
    )
    log_message += f"Number of coroutines run: {len(coroutines)}\n"
    log_message += (
        f"Total resources used by coroutines: {tester.total_resource_counter}\n"
    )
    log_message += f"Actual resources used per second over last period: {total_resources_used / period_length_in_seconds}\n"
    logger.info(log_message)
    coroutine_testing.assert_resource_rate_not_too_high_or_too_low(
        total_resources_used,
        duration,
        allowed_resources_per_period,
        period_length_in_seconds,
        over_rate_allowed=1.2,
        under_rate_allowed=0.9,
    )
