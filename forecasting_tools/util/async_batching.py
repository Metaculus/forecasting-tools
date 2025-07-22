import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

import nest_asyncio
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T")
T2 = TypeVar("T2")

nest_asyncio.apply()  # Make sure asyncio can be nested inside of other asyncio calls


def wrap_coroutines_with_rate_limit(
    coroutine_list: list[Coroutine[Any, Any, T]],
    calls_per_period: int,
    time_period_in_seconds: int = 60,
) -> list[Coroutine[Any, Any, T]]:
    """
    Rate Limiting is only applied to the coroutines in the list, and not between calls of this function.
    """
    limiter = AsyncLimiter(
        max_rate=calls_per_period, time_period=time_period_in_seconds
    )
    return [
        apply_limiter_to_coroutine(coroutine, limiter) for coroutine in coroutine_list
    ]


async def apply_limiter_to_coroutine(
    coroutine: Coroutine, limiter: AsyncLimiter
) -> Any:
    await limiter.acquire()
    return await coroutine


def wrap_coroutines_with_timeout(
    coroutine_list: list[Coroutine[Any, Any, T]], timeout_time: float
) -> list[Coroutine[Any, Any, T]]:
    async def coroutine_with_timeout(coroutine: Coroutine, timeout_time: float) -> Any:
        try:
            result = await asyncio.wait_for(coroutine, timeout=timeout_time)
            return result
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError(
                f"Timeout of {timeout_time} seconds exceeded while running coroutine. Here is the exception: {e.__class__.__name__}: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Exception while running coroutine with timeout wrapper. Here is the exception: {e.__class__.__name__}: {e}"
            )

    return [
        coroutine_with_timeout(coroutine, timeout_time) for coroutine in coroutine_list
    ]


def wrap_coroutines_to_return_not_raise_exceptions(
    coroutine_list: list[Coroutine[Any, Any, T]]
) -> list[Coroutine[Any, Any, T | Exception]]:
    async def coroutine_where_exception_is_returned_not_raised(
        coroutine: Coroutine,
    ) -> Any | Exception:
        try:
            return await coroutine
        except Exception as e:
            return e

    return [
        coroutine_where_exception_is_returned_not_raised(coroutine)
        for coroutine in coroutine_list
    ]


def wrap_coroutines_with_limit_timeout_and_returning_exceptions(
    coroutine_list: list[Coroutine[Any, Any, T]],
    calls_per_period: int,
    time_period: int = 60,
    timeout_time: float = 120,
) -> list[Coroutine[Any, Any, T | Exception]]:
    rate_limited_coroutines = wrap_coroutines_with_rate_limit(
        coroutine_list, calls_per_period, time_period
    )
    limited_and_timed_coroutines = wrap_coroutines_with_timeout(
        rate_limited_coroutines, timeout_time
    )
    limited_timed_error_handled_coroutines = (
        wrap_coroutines_to_return_not_raise_exceptions(limited_and_timed_coroutines)
    )
    return limited_timed_error_handled_coroutines


def run_coroutines(coroutines: list[Coroutine[Any, Any, T]]) -> list[T]:
    async def run_coroutines(coroutines: list[Coroutine[Any, Any, T]]) -> list[T]:
        tasks = []
        for coroutine in coroutines:
            tasks.append(loop.create_task(coroutine))
        results = await asyncio.gather(*tasks)
        return results

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_coroutines(coroutines))


def run_coroutines_while_removing_and_logging_exceptions(
    coroutines: list[Coroutine[Any, Any, T]],
    matching_inputs: list[T2] | T2 = None,
    action_on_exception: Callable[[Exception, T2], None] | None = None,
) -> tuple[list[T], list[T2]]:
    """
    Runs a list of coroutines and returns only the results (and their corresponding inputs) that did not raise an exception.
    A list of "None" is returned as the corresponding input if no matching_inputs are provided.
    A default log message is given on the case of an exception. You can switch out this with a custom function if desired.
    """
    if matching_inputs is None:
        modified_inputs = [None] * len(coroutines)
    elif not isinstance(matching_inputs, list):
        modified_inputs = [matching_inputs] * len(coroutines)
    else:
        modified_inputs = matching_inputs

    assert len(modified_inputs) == len(
        coroutines
    ), "The number of inputs must match the number of coroutines"

    exception_wrapped_coroutines = wrap_coroutines_to_return_not_raise_exceptions(
        coroutines
    )
    results = run_coroutines(exception_wrapped_coroutines)

    results_that_did_not_error: list[T] = []
    inputs_that_did_not_error: list[T2] = []
    for input, result, coroutine in zip(modified_inputs, results, coroutines):
        if isinstance(result, Exception):
            error = result
            if action_on_exception is None:
                action_on_exception = lambda error, _, coroutine=coroutine: logger.error(
                    f"Error while running coroutine '{coroutine.cr_code.co_name}': {error.__class__.__name__} Exception - {error}"
                )
            action_on_exception(error, input)  # type: ignore - Linter improperly thinks that input can't be of type 'None' even if None is assigned to Generic type. It works if the default value for inputs is set to an int
        else:
            results_that_did_not_error.append(result)
            inputs_that_did_not_error.append(input)  # type: ignore - Linter improperly thinks that input can't be of type 'None' even if None is assigned to Generic type. It works if the default value for inputs is set to an int

    return results_that_did_not_error, inputs_that_did_not_error
