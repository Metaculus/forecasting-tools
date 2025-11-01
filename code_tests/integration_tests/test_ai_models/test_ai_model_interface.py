import asyncio
import logging
import os
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest
from code_tests.utilities_for_tests import coroutine_testing
from forecasting_tools.ai_models.deprecated_model_classes.gpto1preview import (
    GptO1Preview,
)
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("subclass", ModelsToTest.BASIC_MODEL_LIST)
def test_ai_model_returns_response_with_invoke(
    subclass: type[AiModel],
) -> None:
    model = subclass()
    model_input = model._get_cheap_input_for_invoke()
    response = asyncio.run(model.invoke(model_input))
    assert response is not None, "Response is None"


@pytest.mark.parametrize("subclass", ModelsToTest.BASIC_MODEL_LIST)
def test_ai_model_async_is_not_blocking(subclass: type[AiModel]) -> None:
    # NOTE: Don't mock this unless costs get bad since blocking could be caused at any code level below the mock
    number_of_coroutines_to_run = 5
    list_should_run_under_x_times_first_coroutine = 3

    if issubclass(subclass, GptO1Preview):
        pytest.skip("GptO1 is around 2c per call, so this test would be too expensive")

    model_input = subclass._get_cheap_input_for_invoke()
    first_coroutine = subclass().invoke(model_input)
    list_of_coroutines = [
        subclass().invoke(model_input) for _ in range(number_of_coroutines_to_run)
    ]
    coroutine_testing.assert_coroutines_run_under_x_times_duration_of_benchmark(
        [first_coroutine],
        list_of_coroutines,
        x=list_should_run_under_x_times_first_coroutine,
        allowed_errors_or_timeouts=5,
    )


@pytest.mark.parametrize("subclass", ModelsToTest.BASIC_MODEL_LIST)
def test_call_limit_error_not_raise_if_not_in_test_env(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    pytest_current_test = os.environ.get("PYTEST_CURRENT_TEST")
    if pytest_current_test is None:
        raise RuntimeError("PYTEST_CURRENT_TEST is not set")
    try:
        del os.environ["PYTEST_CURRENT_TEST"]
        model = subclass()
        max_calls = 1

        model._increment_calls_then_error_if_testing_call_limit_reached(max_calls)
        model._increment_calls_then_error_if_testing_call_limit_reached(max_calls)
        model._increment_calls_then_error_if_testing_call_limit_reached(max_calls)
        os.environ["PYTEST_CURRENT_TEST"] = pytest_current_test
    except Exception as e:
        os.environ["PYTEST_CURRENT_TEST"] = pytest_current_test
        assert False, f"Error: {e}"


@pytest.mark.parametrize("subclass", ModelsToTest.BASIC_MODEL_LIST)
def test_call_limit_error_raised_so_tests_cant_accidentally_create_insane_costs(
    subclass: type[AiModel],
) -> None:
    model = subclass()
    calls_before = model._num_calls_to_dependent_model.get()
    try:
        model._num_calls_to_dependent_model.set(0)
        max_calls = 1
        model._increment_calls_then_error_if_testing_call_limit_reached(max_calls)

        with pytest.raises(RuntimeError):
            model._increment_calls_then_error_if_testing_call_limit_reached(max_calls)
    finally:
        model._num_calls_to_dependent_model.set(calls_before)


@pytest.mark.parametrize("subclass", ModelsToTest.BASIC_MODEL_LIST)
async def test_special_functions_called_with_direct_call(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    model = subclass()
    mock_limiting_function = (
        AiModelMockManager.mock_function_that_throws_error_if_test_limit_reached(mocker)
    )
    await model._mockable_direct_call_to_model(model._get_cheap_input_for_invoke())
    limit_function_num_calls = mock_limiting_function.call_count
    assert (
        limit_function_num_calls == 1
    ), f"Model function was not called n=1 times. It was called {limit_function_num_calls} times"
