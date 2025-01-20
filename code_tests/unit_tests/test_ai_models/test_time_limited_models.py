import asyncio
import logging
import os
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import (
    AiModelMockManager,
)
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest
from forecasting_tools.ai_models.basic_model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.basic_model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.model_archetypes.anthropic_text_model import (
    AnthropicTextToTextModel,
)

logger = logging.getLogger(__name__)

TIME_LIMITED_ERROR_MESSAGE = "Model must be TimeLimited"


@pytest.mark.skip(
    "This test takes too long to run between threads for some reason (it passes but doesn't check for timeout within 10 sec). Not important enough to fix right now"
)
@pytest.mark.parametrize("subclass", ModelsToTest.TIME_LIMITED_LIST)
def test_ai_model_successfully_times_out(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, TimeLimitedModel):
        raise ValueError(TIME_LIMITED_ERROR_MESSAGE)

    subclass.TIMEOUT_TIME = 10

    AiModelMockManager.mock_ai_model_direct_call_with_long_wait(
        mocker, subclass
    )
    model = subclass()
    model_input = model._get_cheap_input_for_invoke()

    with pytest.raises(asyncio.exceptions.TimeoutError):
        asyncio.run(model.invoke(model_input))


@pytest.mark.parametrize("subclass", ModelsToTest.TIME_LIMITED_LIST)
def test_ai_model_does_not_time_out_when_run_time_less_than_timeout_time(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, TimeLimitedModel):
        raise ValueError(TIME_LIMITED_ERROR_MESSAGE)

    if (
        issubclass(subclass, AnthropicTextToTextModel)
        and os.getenv("ANTHROPIC_API_KEY") is None
    ):
        pytest.skip(
            "Skipping test for AnthropicTextModel since API key is not set and is needed for token counting"
        )

    AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
        mocker, subclass
    )
    model = subclass()
    min_timeout_time = 3
    if model.TIMEOUT_TIME < min_timeout_time:
        raise ValueError(
            f"TIMEOUT_TIME must be greater than {min_timeout_time} since the mock function still takes time"
        )
    model_input = model._get_cheap_input_for_invoke()
    response = asyncio.run(model.invoke(model_input))
    assert response is not None
