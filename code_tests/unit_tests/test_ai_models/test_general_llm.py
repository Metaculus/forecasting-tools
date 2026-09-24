import logging
from unittest.mock import AsyncMock, Mock

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from forecasting_tools.ai_models.general_llm import GeneralLlm


def make_litellm_response(serving_provider: str | None) -> ModelResponse:
    provider_field = {} if serving_provider is None else {"provider": serving_provider}
    return ModelResponse(
        choices=[
            Choices(
                message=Message(role="assistant", content="Hello"),
                finish_reason="stop",
                index=0,
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        **provider_field,
    )


async def test_serving_provider_is_returned_and_logged_once_per_model(
    mocker: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch(
        "forecasting_tools.ai_models.general_llm.acompletion",
        AsyncMock(return_value=make_litellm_response("DeepInfra")),
    )
    llm = GeneralLlm(model="openrouter/test-lab/provider-logging-test-model")

    with caplog.at_level(logging.INFO):
        first_response = await llm._mockable_direct_call_to_model("Hi")
        second_response = await llm._mockable_direct_call_to_model("Hi")

    assert first_response.serving_provider == "DeepInfra"
    assert second_response.serving_provider == "DeepInfra"
    provider_logs = [
        record
        for record in caplog.records
        if "served by provider 'DeepInfra'" in record.getMessage()
    ]
    assert len(provider_logs) == 1


async def test_serving_provider_is_none_when_response_has_no_provider(
    mocker: Mock,
) -> None:
    mocker.patch(
        "forecasting_tools.ai_models.general_llm.acompletion",
        AsyncMock(return_value=make_litellm_response(None)),
    )
    llm = GeneralLlm(model="openai/gpt-4o-mini")

    response = await llm._mockable_direct_call_to_model("Hi")

    assert response.serving_provider is None
