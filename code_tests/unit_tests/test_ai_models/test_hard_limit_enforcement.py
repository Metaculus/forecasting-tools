from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, Mock

import pytest
from agents.extensions.models.litellm_model import LitellmModel
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from forecasting_tools.ai_models.agent_wrappers import AgentSdkLlm
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

COST_LIMIT = 0.01


def make_litellm_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                message=Message(role="assistant", content="Hello"),
                finish_reason="stop",
                index=0,
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.parametrize("prior_usage", [0.005, COST_LIMIT])
async def test_general_llm_calls_model_while_usage_is_within_limit(
    mocker: Mock, prior_usage: float
) -> None:
    litellm_call = mocker.patch(
        "forecasting_tools.ai_models.general_llm.acompletion",
        AsyncMock(return_value=make_litellm_response()),
    )

    with MonetaryCostManager(COST_LIMIT):
        MonetaryCostManager.increase_current_usage_in_parent_managers(prior_usage)
        await GeneralLlm(model="openrouter/openai/gpt-4.1-nano").invoke("Hi")

    litellm_call.assert_called_once()


async def test_general_llm_makes_no_call_and_no_retry_once_limit_is_exceeded(
    mocker: Mock,
) -> None:
    litellm_call = mocker.patch(
        "forecasting_tools.ai_models.general_llm.acompletion",
        AsyncMock(return_value=make_litellm_response()),
    )
    direct_call = mocker.spy(GeneralLlm, "_mockable_direct_call_to_model")
    llm = GeneralLlm(model="openrouter/openai/gpt-4.1-nano", allowed_tries=3)

    with MonetaryCostManager(COST_LIMIT):
        MonetaryCostManager.increase_current_usage_in_parent_managers(2 * COST_LIMIT)
        with pytest.raises(HardLimitExceededError):
            await llm.invoke("Hi")

    litellm_call.assert_not_called()
    assert direct_call.call_count == 1


async def get_response(llm: AgentSdkLlm) -> None:
    await llm.get_response()


async def consume_stream_response(llm: AgentSdkLlm) -> None:
    async for _ in llm.stream_response():
        pass


@pytest.mark.parametrize(
    "litellm_model_method, call_llm",
    [("get_response", get_response), ("stream_response", consume_stream_response)],
)
async def test_agent_sdk_llm_makes_no_call_once_limit_is_exceeded(
    mocker: Mock,
    litellm_model_method: str,
    call_llm: Callable[[AgentSdkLlm], Awaitable[None]],
) -> None:
    parent_method = mocker.patch.object(LitellmModel, litellm_model_method)
    llm = AgentSdkLlm(model="openrouter/openai/gpt-4.1-nano")

    with MonetaryCostManager(COST_LIMIT):
        MonetaryCostManager.increase_current_usage_in_parent_managers(2 * COST_LIMIT)
        with pytest.raises(HardLimitExceededError):
            await call_llm(llm)

    parent_method.assert_not_called()
