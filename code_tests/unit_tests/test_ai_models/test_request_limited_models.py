import asyncio
import logging
from typing import Coroutine
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest

logger = logging.getLogger(__name__)
import time
from enum import Enum

from code_tests.utilities_for_tests import coroutine_testing
from forecasting_tools.ai_models.model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.token_limited_model import (
    TokenLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.tokens_are_calculatable import (
    TokensAreCalculatable,
)
from forecasting_tools.util import async_batching

REQUEST_LIMITED_ERROR_MESSAGE = "Model must be RequestLimited"

######################################## CREATE TESTING INSTANCES ########################################


class NumModelMode(Enum):
    CALLS_FROM_ONE_MODEL = "calls_from_one_model"
    CALLS_FROM_MULTIPLE_MODELS = "calls_from_multiple_models"


class BurstMode(Enum):
    TESTING_ONLY_BURST = "testing_only_burst"
    TESTING_BURST_AND_AFTER_IS_CORRECT_RATE = "testing_burst_and_after_rate"
    TESTING_BURST_AND_AFTER_TAKES_TIME = "testing_burst_and_after_rate_takes_time"


class RunSetUp:
    def __init__(self, subclass: type[RequestLimitedModel], mode: NumModelMode) -> None:
        self.subclass = subclass
        self.mode = mode

    @property
    def name_of_test(self) -> str:
        return f"{self.subclass.__name__} {self.mode.value}"


class RequestLimitTestResults:
    def __init__(
        self,
        duration: float,
        number_of_invokes: int,
        number_of_direct_calls: int,
    ) -> None:
        self.duration = duration
        self.number_of_invokes = number_of_invokes
        self.number_of_direct_calls = number_of_direct_calls

    @property
    def invokes_per_second(self) -> float:
        return self.number_of_invokes / self.duration

    @property
    def direct_calls_per_second(self) -> float:
        return self.number_of_direct_calls / self.duration


def get_testing_instances() -> list[tuple[str, RunSetUp]]:
    models_to_test = ModelsToTest.REQUEST_LIMITED_LIST
    tests = []
    for model in models_to_test:
        for mode in [
            NumModelMode.CALLS_FROM_ONE_MODEL,
            NumModelMode.CALLS_FROM_MULTIPLE_MODELS,
        ]:
            run_setup = RunSetUp(model, mode)
            name_of_run = run_setup.name_of_test
            tests.append(
                (name_of_run, run_setup)
            )  # Name of run apppended so it shows up in VSCode tests runner in an understandable way
    return tests


################################ TESTS ################################


@pytest.mark.skip(reason="Skipping test because it's slow")
@pytest.mark.parametrize(("_", "run_setup"), get_testing_instances())
def test_burst_happens_quickly(mocker: Mock, _: str, run_setup: RunSetUp) -> None:
    number_of_calls = get_number_of_coroutines_for_initial_burst(run_setup.subclass)
    run_request_limit_test(
        mocker, run_setup, number_of_calls, BurstMode.TESTING_ONLY_BURST
    )


@pytest.mark.skip(reason="Skipping test because it's slow")
@pytest.mark.parametrize(("_", "run_setup"), get_testing_instances())
def test_calls_after_burst_take_time(mocker: Mock, _: str, run_setup: RunSetUp) -> None:
    burst_calls = get_number_of_coroutines_for_initial_burst(run_setup.subclass)
    regular_calls = get_number_of_calls_that_takes_10s_to_run_given_model_rate(
        run_setup.subclass
    )
    number_of_calls = burst_calls + regular_calls
    run_request_limit_test(
        mocker,
        run_setup,
        number_of_calls,
        BurstMode.TESTING_BURST_AND_AFTER_TAKES_TIME,
    )


############################################## Helper Functions ##############################################
def run_request_limit_test(
    mocker: Mock,
    run_setup: RunSetUp,
    number_of_calls_to_make: int,
    burst_mode: BurstMode,
) -> None:
    coroutines_to_test = setup_coroutines(run_setup, number_of_calls_to_make)
    mocked_direct_call = set_up_mocking_for_request_limit_tests(
        mocker, run_setup.subclass
    )

    if burst_mode == BurstMode.TESTING_BURST_AND_AFTER_TAKES_TIME:
        run_test_for__burst_and_after_takes_time(coroutines_to_test)
    else:
        run_other_burst_modes(
            run_setup, coroutines_to_test, burst_mode, mocked_direct_call
        )


def setup_coroutines(
    run_setup: RunSetUp, number_of_calls_to_make: int
) -> list[Coroutine]:
    assert issubclass(
        run_setup.subclass, RequestLimitedModel
    ), REQUEST_LIMITED_ERROR_MESSAGE
    AiModelMockManager.reinitialize_limiters(run_setup.subclass)
    coroutines_to_test = create_invoke_coroutines_for_test(
        run_setup, number_of_calls_to_make
    )
    return coroutines_to_test


def run_test_for__burst_and_after_takes_time(
    coroutines_to_test: list[Coroutine],
) -> None:
    make_sure_to_take_longer_than_seconds = 5
    timed_coroutines = async_batching.wrap_coroutines_with_timeout(
        coroutines_to_test, make_sure_to_take_longer_than_seconds
    )
    if len(timed_coroutines) > 4000:
        pytest.skip(
            "Skipping test because with our current usage we won't realistically hit this usage amount, and over this many coroutines messes up timeout because logging files become a resource bottleneck"
        )

    logger.info(f"Running {len(timed_coroutines)} test coroutines")
    start_time = time.time()
    with pytest.raises(asyncio.TimeoutError):
        async_batching.run_coroutines(timed_coroutines)
    end_time = time.time()
    duration = end_time - start_time
    assert (
        duration >= make_sure_to_take_longer_than_seconds
    ), f"Didn't wait long enough. Duration: {duration}, Timeout: {make_sure_to_take_longer_than_seconds}"
    assert (
        duration < make_sure_to_take_longer_than_seconds + 2
    ), f"Waited too long. Duration: {duration}, Timeout: {make_sure_to_take_longer_than_seconds}"


def run_other_burst_modes(
    run_setup: RunSetUp,
    coroutines_to_test: list[Coroutine],
    burst_mode: BurstMode,
    mocked_direct_call: Mock,
) -> None:
    test_results = get_request_limit_test_results(
        run_setup.subclass, coroutines_to_test, mocked_direct_call
    )
    if burst_mode == BurstMode.TESTING_ONLY_BURST:
        logger.info(
            f"Got duration {test_results.duration}, with {test_results.number_of_direct_calls} direct calls over {test_results.number_of_invokes} coroutines"
        )
        coroutine_testing.assert_resource_burst_is_short(
            run_setup.subclass.REQUESTS_PER_PERIOD_LIMIT, test_results.duration
        )
    elif burst_mode == BurstMode.TESTING_BURST_AND_AFTER_IS_CORRECT_RATE:
        assert_rate_limit_not_exceeded_or_missed_for_model(
            run_setup.subclass, test_results
        )


def run_burst_for_model_subclass(run_setup: RunSetUp) -> None:
    number_of_calls_to_make = get_number_of_coroutines_for_initial_burst(
        run_setup.subclass
    )
    coroutines_to_test = create_invoke_coroutines_for_test(
        run_setup, number_of_calls_to_make
    )
    async_batching.run_coroutines(coroutines_to_test)


def set_up_mocking_for_request_limit_tests(
    mocker: Mock, subclass: type[RequestLimitedModel]
) -> Mock:
    mocked_direct_call = (
        AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
            mocker, subclass
        )
    )

    if issubclass(subclass, TokensAreCalculatable):
        AiModelMockManager.mock_input_to_tokens_with_value(mocker, subclass)

    return mocked_direct_call


def get_number_of_calls_that_takes_10s_to_run_given_model_rate(
    subclass: type[RequestLimitedModel],
) -> int:
    time_tests_should_take_in_seconds = 10
    allowed_calls_per_second = (
        subclass.REQUESTS_PER_PERIOD_LIMIT / subclass.REQUEST_PERIOD_IN_SECONDS
    )
    number_of_calls_to_make = int(
        time_tests_should_take_in_seconds * allowed_calls_per_second
    )
    return number_of_calls_to_make


def get_number_of_coroutines_for_initial_burst(
    subclass: type[RequestLimitedModel],
) -> int:
    return subclass.REQUESTS_PER_PERIOD_LIMIT


def create_invoke_coroutines_for_test(
    run_setup: RunSetUp, number_of_coroutines: int
) -> list[Coroutine]:
    if run_setup.mode == NumModelMode.CALLS_FROM_MULTIPLE_MODELS:
        models_to_create = 2
    else:
        models_to_create = 1

    models = [run_setup.subclass() for _ in range(models_to_create)]

    coroutines = []
    for model in models:
        number_of_calls_for_model = number_of_coroutines // len(models)
        cheap_input = model._get_cheap_input_for_invoke()
        coroutines.extend(
            model.invoke(cheap_input) for _ in range(number_of_calls_for_model)
        )

    return coroutines


def get_request_limit_test_results(
    subclass: type[RequestLimitedModel],
    coroutines_to_test: list[Coroutine],
    mocked_direct_call: Mock,
) -> RequestLimitTestResults:
    if issubclass(subclass, TokenLimitedModel):
        subclass._make_token_limiter_have_large_rate()
    number_of_calls_before_this_test = mocked_direct_call.call_count
    stats = coroutine_testing.find_stats_of_coroutine_run(coroutines_to_test)
    duration = stats.duration_in_seconds
    call_count_of_direct_call_mock = (
        mocked_direct_call.call_count - number_of_calls_before_this_test
    )
    number_of_invokes = len(coroutines_to_test)
    return RequestLimitTestResults(
        duration, number_of_invokes, call_count_of_direct_call_mock
    )


def assert_rate_limit_not_exceeded_or_missed_for_model(
    subclass: type[RequestLimitedModel], test_results: RequestLimitTestResults
) -> None:
    allowed_calls_per_second = (
        subclass.REQUESTS_PER_PERIOD_LIMIT / subclass.REQUEST_PERIOD_IN_SECONDS
    )
    allowed_upper_bound = allowed_calls_per_second * 1.50
    allowed_lower_bound = allowed_calls_per_second * 0.9

    results_message = f"Expected {round(allowed_calls_per_second,2)} direct calls per second, got {round(test_results.direct_calls_per_second,2)} direct calls per second, and {round(test_results.invokes_per_second,4)} invokes per second over {round(test_results.duration,2)} seconds, and {test_results.number_of_direct_calls} direct calls, over {test_results.number_of_invokes} coroutines"
    logger.info(results_message)
    assert (
        test_results.direct_calls_per_second < allowed_upper_bound
    ), f"The calls per second was too high. {results_message}"
    assert (
        test_results.direct_calls_per_second > allowed_lower_bound
    ), f"The calls per second was too low. {results_message}"
