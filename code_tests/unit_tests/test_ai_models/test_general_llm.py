import logging
from unittest.mock import AsyncMock, Mock

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from typeguard import TypeCheckError

from forecasting_tools.ai_models.general_llm import GeneralLlm


def make_litellm_response(serving_provider: str | dict | None) -> ModelResponse:
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


def mock_litellm_response(mocker: Mock, serving_provider: str | dict | None) -> None:
    mocker.patch(
        "forecasting_tools.ai_models.general_llm.acompletion",
        AsyncMock(return_value=make_litellm_response(serving_provider)),
    )


def make_pinned_llm(allow_fallbacks: bool) -> GeneralLlm:
    return GeneralLlm(
        model="openrouter/openai/gpt-oss-120b",
        extra_body={
            "provider": {
                "order": ["deepinfra/bf16"],
                "allow_fallbacks": allow_fallbacks,
                "require_parameters": True,
            }
        },
    )


async def test_serving_provider_is_returned_and_logged_once_per_model(
    mocker: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_litellm_response(mocker, "DeepInfra")
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
    mock_litellm_response(mocker, None)
    llm = GeneralLlm(model="openai/gpt-4o-mini")

    response = await llm._mockable_direct_call_to_model("Hi")

    assert response.serving_provider is None


async def test_non_string_serving_provider_errors(mocker: Mock) -> None:
    mock_litellm_response(mocker, {"name": "DeepInfra"})
    llm = GeneralLlm(model="openrouter/openai/gpt-oss-120b")

    with pytest.raises(TypeCheckError):
        await llm._mockable_direct_call_to_model("Hi")


async def test_pinned_llm_accepts_response_from_pinned_provider(mocker: Mock) -> None:
    mock_litellm_response(mocker, "DeepInfra")

    response = await make_pinned_llm(
        allow_fallbacks=False
    )._mockable_direct_call_to_model("Hi")

    assert response.serving_provider == "DeepInfra"


@pytest.mark.parametrize("served_by", ["Novita", None])
async def test_pinned_llm_errors_when_not_served_by_pinned_provider(
    mocker: Mock, served_by: str | None
) -> None:
    mock_litellm_response(mocker, served_by)

    with pytest.raises(RuntimeError, match="pinned to"):
        await make_pinned_llm(allow_fallbacks=False)._mockable_direct_call_to_model(
            "Hi"
        )


async def test_llm_with_fallbacks_allowed_accepts_any_provider(mocker: Mock) -> None:
    mock_litellm_response(mocker, "Novita")

    response = await make_pinned_llm(
        allow_fallbacks=True
    )._mockable_direct_call_to_model("Hi")

    assert response.serving_provider == "Novita"
