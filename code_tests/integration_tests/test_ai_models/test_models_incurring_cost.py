import asyncio
import logging
import random
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.ai_models.deprecated_model_classes.deepseek_r1 import DeepSeekR1
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)
from forecasting_tools.ai_models.model_interfaces.incurs_cost import IncursCost
from forecasting_tools.ai_models.model_interfaces.retryable_model import RetryableModel
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


############################### HELPERS ########################################
NOT_INCURS_COST_ERROR_MESSAGE = "Model must have interface IncursCost"


async def async_run_cheap_invoke(subclass: type[AiModel]) -> None:
    if issubclass(subclass, RetryableModel):
        model = subclass(allowed_tries=1)
    else:
        model = subclass()
    model_input = model._get_cheap_input_for_invoke()
    random_time_to_wait = random.uniform(0.05, 0.1)
    await asyncio.sleep(random_time_to_wait)
    await model.invoke(model_input)


def run_cheap_invoke(subclass: type[AiModel]) -> None:
    asyncio.run(async_run_cheap_invoke(subclass))


def run_cheap_invoke_and_track_cost(subclass: type[AiModel], max_cost: float) -> float:
    running_cost: float = 0
    with MonetaryCostManager(max_cost) as cost_manager:
        run_cheap_invoke(subclass)
        running_cost = cost_manager.current_usage

    return running_cost


def find_number_of_calls_to_make(subclass: type[AiModel]) -> int:
    return 10


async def find_number_of_hard_limit_exceptions_in_run(
    mocker: Mock,
    subclass: type[AiModel],
    max_cost: float,
    number_of_calls_to_make: int,
    number_of_expected_exceptions: int,
) -> int:
    AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
        mocker, subclass
    )

    coroutines = [
        async_run_cheap_invoke(subclass) for _ in range(number_of_calls_to_make)
    ]
    exception_handled_coroutines = (
        async_batching.wrap_coroutines_to_return_not_raise_exceptions(coroutines)
    )

    with MonetaryCostManager(max_cost):
        results = []
        for coroutine in exception_handled_coroutines:
            result = await coroutine
            results.append(result)
        logger.info(results)

    number_of_exceptions = len(
        [result for result in results if isinstance(result, Exception)]
    )
    number_of_hard_limit_exceptions = len(
        [result for result in results if isinstance(result, HardLimitExceededError)]
    )
    assert (
        number_of_hard_limit_exceptions == number_of_exceptions
    ), "Some exceptions were not HardLimitExceededError"

    return number_of_hard_limit_exceptions


############################### TESTS ########################################


@pytest.mark.parametrize("subclass", ModelsToTest.INCURS_COST_LIST)
def test_cost_manager_notices_cost_without_mocks(
    subclass: type[AiModel],
) -> None:
    if not issubclass(subclass, IncursCost):
        raise ValueError(NOT_INCURS_COST_ERROR_MESSAGE)

    max_cost = 10
    cost = run_cheap_invoke_and_track_cost(subclass, max_cost)
    assert cost > 0, "No cost was incurred"


@pytest.mark.parametrize("subclass", ModelsToTest.INCURS_COST_LIST)
async def test_cost_calculated_matches_actual_cost(
    subclass: type[AiModel],
) -> None:
    if not issubclass(subclass, CombinedLlmArchetype):
        pytest.skip("Model does not have calculate_cost_from_tokens method")
    if issubclass(subclass, DeepSeekR1):
        pytest.skip(
            "DeepSeekR1 does not have correct token-cost estimation due to reasoning tokens"
        )
    model = subclass()
    direct_response = await model._mockable_direct_call_to_model(
        model._get_cheap_input_for_invoke()
    )
    assert isinstance(direct_response, TextTokenCostResponse)
    actual_cost = direct_response.cost
    calculated_cost = model.calculate_cost_from_tokens(
        direct_response.prompt_tokens_used,
        direct_response.completion_tokens_used,
    )
    assert calculated_cost == pytest.approx(
        actual_cost,
    ), "Cost calculated does not match actual cost"


@pytest.mark.parametrize("subclass", ModelsToTest.INCURS_COST_LIST)
def test_cost_manager_notices_cost_with_mocks(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, IncursCost):
        raise ValueError(NOT_INCURS_COST_ERROR_MESSAGE)

    AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
        mocker, subclass
    )
    max_cost = 100
    cost = run_cheap_invoke_and_track_cost(subclass, max_cost)
    assert cost > 0, "No cost was incurred"


@pytest.mark.parametrize("subclass", ModelsToTest.INCURS_COST_LIST)
def test_error_thrown_when_limit_reached(mocker: Mock, subclass: type[AiModel]) -> None:
    if not issubclass(subclass, IncursCost):
        raise ValueError(NOT_INCURS_COST_ERROR_MESSAGE)

    AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
        mocker, subclass
    )
    max_cost = 0.0000000000001
    with MonetaryCostManager(max_cost):
        with pytest.raises(HardLimitExceededError):
            run_cheap_invoke(
                subclass
            )  # This first line might not need to raise an error, since the cost is not exceeded before the calls is made. Depends on current cost implementation
            run_cheap_invoke(subclass)


@pytest.mark.parametrize("subclass", ModelsToTest.INCURS_COST_LIST)
async def test_error_thrown_with_many_calls_and_tiny_limit(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, IncursCost):
        raise ValueError(NOT_INCURS_COST_ERROR_MESSAGE)

    max_cost = 0.0000000000001
    number_of_calls_to_make = find_number_of_calls_to_make(subclass)
    number_of_expected_exceptions = number_of_calls_to_make - 1
    num_hard_limit_errors = await find_number_of_hard_limit_exceptions_in_run(
        mocker,
        subclass,
        max_cost,
        number_of_calls_to_make,
        number_of_expected_exceptions,
    )
    assert (
        num_hard_limit_errors >= number_of_expected_exceptions
    ), "Improper number of exceptions thrown. Only first call should be allowed to possibly exceed since cost would be exceeded after this"


@pytest.mark.parametrize("subclass", ModelsToTest.INCURS_COST_LIST)
async def test_error_thrown_when_many_calls_with_a_larger_than_0_cost_limit(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, IncursCost):
        raise ValueError(NOT_INCURS_COST_ERROR_MESSAGE)

    number_of_calls_to_make = find_number_of_calls_to_make(subclass)
    mock_return_value = (
        subclass._get_mock_return_for_direct_call_to_model_using_cheap_input()
    )
    if isinstance(mock_return_value, TextTokenCostResponse):
        cost_of_mock_call = mock_return_value.cost
    else:
        cost_of_mock_call = 0.005  # Estimate

    percentage_we_want_to_error = 0.5
    max_cost = cost_of_mock_call * number_of_calls_to_make * percentage_we_want_to_error
    number_of_expected_exceptions_given_perfect_cost_prediction = (
        int(number_of_calls_to_make * percentage_we_want_to_error) - 1
    )

    num_hard_limit_errors = await find_number_of_hard_limit_exceptions_in_run(
        mocker,
        subclass,
        max_cost,
        number_of_calls_to_make,
        number_of_expected_exceptions_given_perfect_cost_prediction,
    )
    assert (
        num_hard_limit_errors
        >= number_of_expected_exceptions_given_perfect_cost_prediction
    )

    number_of_calls_that_did_not_error = number_of_calls_to_make - num_hard_limit_errors
    assert number_of_calls_that_did_not_error > 1
