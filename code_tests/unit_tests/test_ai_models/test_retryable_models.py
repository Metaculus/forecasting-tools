import asyncio
import logging
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.retryable_model import RetryableModel

logger = logging.getLogger(__name__)

RETRYABLE_ERROR_MESSAGE = "Model must be Retryable"


@pytest.mark.skip(reason="Skipping test because it's slow")
@pytest.mark.parametrize("subclass", ModelsToTest.RETRYABLE_LIST)
def test_ai_model_successfully_retries(mocker: Mock, subclass: type[AiModel]) -> None:
    if not issubclass(subclass, RetryableModel):
        raise ValueError(RETRYABLE_ERROR_MESSAGE)

    AiModelMockManager.mock_ai_model_direct_call_with_2_errors_then_success(
        mocker, subclass
    )
    number_of_retries = 3

    model = subclass()
    model.allowed_tries = number_of_retries
    model_input = model._get_cheap_input_for_invoke()
    response = asyncio.run(model.invoke(model_input))
    assert response is not None


@pytest.mark.skip(reason="Skipping test because it's slow")
@pytest.mark.parametrize("subclass", ModelsToTest.RETRYABLE_LIST)
def test_errors_when_runs_out_of_tries(mocker: Mock, subclass: type[AiModel]) -> None:
    if not issubclass(subclass, RetryableModel):
        raise ValueError(RETRYABLE_ERROR_MESSAGE)

    AiModelMockManager.mock_ai_model_direct_call_with_2_errors_then_success(
        mocker, subclass
    )
    model = subclass()
    model.allowed_tries = 2  # It should run out of retries with the first 2 errors
    model_input = model._get_cheap_input_for_invoke()
    with pytest.raises(Exception):
        asyncio.run(model.invoke(model_input))


@pytest.mark.parametrize("subclass", ModelsToTest.RETRYABLE_LIST)
def test_raises_error_on_tries_setter_if_invalid(
    subclass: type[AiModel],
) -> None:
    if not issubclass(subclass, RetryableModel):
        raise ValueError(RETRYABLE_ERROR_MESSAGE)

    model = subclass()

    with pytest.raises(ValueError):
        model.allowed_tries = -1

    with pytest.raises(ValueError):
        model.allowed_tries = 0

    try:
        model.allowed_tries = 1
    except Exception:
        pytest.fail("Should not raise error on positive allowed_tries")


@pytest.mark.parametrize("subclass", ModelsToTest.RETRYABLE_LIST)
def test_raises_error_on_tries_init_if_invalid(
    subclass: type[AiModel],
) -> None:
    if not issubclass(subclass, RetryableModel):
        raise ValueError(RETRYABLE_ERROR_MESSAGE)

    with pytest.raises(ValueError):
        subclass(allowed_tries=0)

    with pytest.raises(ValueError):
        subclass(allowed_tries=-1)

    try:
        subclass(allowed_tries=1)
    except Exception:
        pytest.fail("Should not raise error on positive allowed_tries")
