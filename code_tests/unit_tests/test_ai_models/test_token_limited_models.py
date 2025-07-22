import logging
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.token_limited_model import (
    TokenLimitedModel,
)

logger = logging.getLogger(__name__)
import asyncio

from forecasting_tools.util import async_batching

TOKEN_LIMITED_ERROR_MESSAGE = "Model must be TokenLimited"
MOCK_INPUT_TO_TOKEN_RETURN_VALUE = 5000

# TODO: Also check that the burst works by itself


@pytest.mark.skip(reason="Skipping test because it's slow")
@pytest.mark.parametrize("subclass", ModelsToTest.TOKEN_LIMITED_LIST)
def test_token_amount_past_burst_doesnt_happen_instantly(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, TokenLimitedModel):
        raise ValueError(TOKEN_LIMITED_ERROR_MESSAGE)

    AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
        mocker, subclass
    )
    AiModelMockManager.reinitialize_limiters(subclass)

    model = subclass()
    invoke_input = subclass._get_cheap_input_for_invoke()
    tokens_to_reach_burst = get_number_of_tokens_to_deplete_burst(subclass)
    tokens_after_burst = get_number_of_tokens_that_takes_10s_to_run_given_model_rate(
        subclass
    )
    total_tokens = tokens_to_reach_burst + tokens_after_burst
    number_of_calls_to_make = mock_input_to_tokens_and_get_call_count(
        mocker, subclass, total_tokens
    )
    coroutines_to_test = [
        model.invoke(invoke_input) for _ in range(number_of_calls_to_make)
    ]

    with pytest.raises(asyncio.TimeoutError):
        seconds_coroutines_should_take_longer_than = 5
        timed_coroutines = async_batching.wrap_coroutines_with_timeout(
            coroutines_to_test, seconds_coroutines_should_take_longer_than
        )
        async_batching.run_coroutines(timed_coroutines)


def get_number_of_tokens_to_deplete_burst(
    subclass: type[TokenLimitedModel],
) -> int:
    tokens_to_reach_burst = subclass.TOKENS_PER_PERIOD_LIMIT
    return tokens_to_reach_burst


def get_number_of_tokens_that_takes_10s_to_run_given_model_rate(
    subclass: type[TokenLimitedModel],
) -> int:
    time_tests_should_take_in_seconds = 10
    allowed_tokens_per_second = (
        subclass.TOKENS_PER_PERIOD_LIMIT / subclass.TOKEN_PERIOD_IN_SECONDS
    )
    number_of_tokens_to_ask_for = int(
        time_tests_should_take_in_seconds * allowed_tokens_per_second
    )
    return number_of_tokens_to_ask_for


def mock_input_to_tokens_and_get_call_count(
    mocker: Mock, subclass: type[TokenLimitedModel], tokens: int
) -> int:
    mock_input_to_token_return_value = int(subclass.TOKENS_PER_PERIOD_LIMIT / 1000)

    AiModelMockManager.mock_input_to_tokens_with_value(
        mocker, subclass, mock_input_to_token_return_value
    )

    number_of_calls_to_make = tokens // mock_input_to_token_return_value
    return number_of_calls_to_make
