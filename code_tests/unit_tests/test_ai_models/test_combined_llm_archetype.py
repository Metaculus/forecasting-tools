from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)


async def test_invoke_counts_call_cost_once(mocker: Mock) -> None:
    call_cost = 0.0005
    AiModelMockManager.mock_general_llm_litellm_call_with_value(
        mocker,
        TextTokenCostResponse(
            data="Hello",
            prompt_tokens_used=1,
            completion_tokens_used=1,
            total_tokens_used=2,
            model=Gpt4o.MODEL_NAME,
            cost=call_cost,
        ),
    )

    with MonetaryCostManager(10) as cost_manager:
        await Gpt4o().invoke("Hi")

    assert cost_manager.current_usage == pytest.approx(call_cost)
