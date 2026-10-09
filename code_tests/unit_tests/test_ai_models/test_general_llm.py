import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from litellm import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from typeguard import TypeCheckError

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    LitellmCostTracker,
    MonetaryCostManager,
)


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


def make_pinned_llm(
    allow_fallbacks: bool | None, endpoint_slug: str = "deepinfra/bf16"
) -> GeneralLlm:
    return GeneralLlm(
        model="openrouter/openai/gpt-oss-120b",
        extra_body={
            "provider": {
                "order": [endpoint_slug],
                "allow_fallbacks": allow_fallbacks,
                "require_parameters": True,
            }
        },
    )


async def test_serving_provider_is_returned_and_logged_once_per_model(
    mocker: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch.object(GeneralLlm, "_logged_model_provider_pairs", set())
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


@pytest.mark.parametrize(
    "served_by, endpoint_slug",
    [
        ("DeepInfra", "deepinfra/bf16"),
        ("Z.AI", "z-ai/fp8"),
        ("Moonshot AI", "moonshotai/int4"),
        ("Google", "google-vertex/us-east5"),
        ("Mancer 2", "mancer"),
    ],
)
async def test_pinned_llm_accepts_response_from_pinned_provider(
    mocker: Mock, served_by: str, endpoint_slug: str
) -> None:
    mock_litellm_response(mocker, served_by)

    response = await make_pinned_llm(
        allow_fallbacks=False, endpoint_slug=endpoint_slug
    )._mockable_direct_call_to_model("Hi")

    assert response.serving_provider == served_by


@pytest.mark.parametrize("served_by", ["Novita", "", None])
async def test_pinned_llm_errors_when_not_served_by_pinned_provider(
    mocker: Mock, served_by: str | None
) -> None:
    mock_litellm_response(mocker, served_by)

    with pytest.raises(RuntimeError, match="pinned to"):
        await make_pinned_llm(allow_fallbacks=False)._mockable_direct_call_to_model(
            "Hi"
        )


@pytest.mark.parametrize(
    "served_by, is_allowed",
    [("AkashML", True), ("DeepInfra", True), ("Crusoe", False)],
)
async def test_llm_pinned_to_multiple_hosts_accepts_only_those_hosts(
    mocker: Mock, served_by: str, is_allowed: bool
) -> None:
    mock_litellm_response(mocker, served_by)
    llm = GeneralLlm(
        model="openrouter/openai/gpt-oss-120b",
        extra_body={
            "provider": {
                "order": ["akashml/bf16", "deepinfra/bf16"],
                "allow_fallbacks": False,
            }
        },
    )

    if is_allowed:
        response = await llm._mockable_direct_call_to_model("Hi")
        assert response.serving_provider == served_by
    else:
        with pytest.raises(RuntimeError, match="pinned to"):
            await llm._mockable_direct_call_to_model("Hi")


@pytest.mark.parametrize("allow_fallbacks", [True, None])
async def test_llm_with_fallbacks_allowed_accepts_any_provider(
    mocker: Mock, allow_fallbacks: bool | None
) -> None:
    mock_litellm_response(mocker, "Novita")

    response = await make_pinned_llm(
        allow_fallbacks=allow_fallbacks
    )._mockable_direct_call_to_model("Hi")

    assert response.serving_provider == "Novita"


async def test_llm_with_empty_provider_routing_accepts_any_provider(
    mocker: Mock,
) -> None:
    mock_litellm_response(mocker, "Novita")
    llm = GeneralLlm(
        model="openrouter/openai/gpt-oss-120b", extra_body={"provider": None}
    )

    response = await llm._mockable_direct_call_to_model("Hi")

    assert response.serving_provider == "Novita"


FAKE_CREDENTIAL = "fake-credential-value"


@pytest.mark.parametrize(
    "model, credential_env_var",
    [("exa/exa", "EXA_API_KEY"), ("metaculus/gpt-4o", "METACULUS_TOKEN")],
)
def test_to_dict_excludes_credential_loaded_from_env(
    monkeypatch: pytest.MonkeyPatch, model: str, credential_env_var: str
) -> None:
    monkeypatch.setenv(credential_env_var, FAKE_CREDENTIAL)
    llm = GeneralLlm(model=model)

    assert FAKE_CREDENTIAL in str(llm.litellm_kwargs)
    assert FAKE_CREDENTIAL not in str(llm.to_dict())


@pytest.mark.parametrize(
    "credential_kwarg, credential_value",
    [
        ("api_key", FAKE_CREDENTIAL),
        ("extra_headers", {"Authorization": f"Bearer {FAKE_CREDENTIAL}"}),
        ("aws_secret_access_key", FAKE_CREDENTIAL),
        ("vertex_credentials", FAKE_CREDENTIAL),
    ],
)
def test_to_dict_excludes_explicitly_passed_credential(
    credential_kwarg: str, credential_value: str | dict[str, str]
) -> None:
    llm = GeneralLlm(model="openai/gpt-4o-mini", **{credential_kwarg: credential_value})

    assert FAKE_CREDENTIAL in str(llm.litellm_kwargs)
    assert FAKE_CREDENTIAL not in str(llm.to_dict())


CALL_COST = 0.0005


def mock_litellm_call_with_late_success_callback(
    mocker: Mock,
    litellm_function_name: str,
    response: ModelResponse | ResponsesAPIResponse,
) -> Callable[[], Awaitable[None]]:
    """
    Like litellm's logging worker, the returned function runs the success callback in the
    context captured when each call finished, after the call has already returned.
    """
    response._hidden_params = {"response_cost": CALL_COST}
    contexts_at_call_end: list[contextvars.Context] = []

    async def litellm_call(**kwargs: Any) -> ModelResponse | ResponsesAPIResponse:
        contexts_at_call_end.append(contextvars.copy_context())
        return response

    async def fire_success_callbacks() -> None:
        for context in contexts_at_call_end:
            await context.run(
                asyncio.create_task,
                LitellmCostTracker().async_log_success_event(
                    {"response_cost": CALL_COST}, response, None, None
                ),
            )

    mocker.patch(
        f"forecasting_tools.ai_models.general_llm.{litellm_function_name}",
        litellm_call,
    )
    return fire_success_callbacks


async def test_cost_is_counted_once_when_litellm_callback_fires_after_call_returns(
    mocker: Mock,
) -> None:
    fire_success_callbacks = mock_litellm_call_with_late_success_callback(
        mocker, "acompletion", make_litellm_response("OpenAI")
    )
    llm = GeneralLlm(model="openrouter/openai/gpt-4.1-nano")

    with MonetaryCostManager(10) as cost_manager:
        responses = await asyncio.gather(
            llm._mockable_direct_call_to_model("Hi"),
            llm._mockable_direct_call_to_model("Hi"),
        )
        await fire_success_callbacks()

    assert [response.cost for response in responses] == pytest.approx(
        [CALL_COST, CALL_COST]
    )
    assert cost_manager.current_usage == pytest.approx(2 * CALL_COST)


async def test_cost_of_response_rejected_by_provider_pin_is_counted_once(
    mocker: Mock,
) -> None:
    fire_success_callbacks = mock_litellm_call_with_late_success_callback(
        mocker, "acompletion", make_litellm_response("Novita")
    )
    llm = make_pinned_llm(allow_fallbacks=False)

    with MonetaryCostManager(10) as cost_manager:
        with pytest.raises(RuntimeError, match="pinned to"):
            await llm._mockable_direct_call_to_model("Hi")
        await fire_success_callbacks()

    assert cost_manager.current_usage == pytest.approx(CALL_COST)


async def test_cost_of_incomplete_responses_api_response_is_counted_once(
    mocker: Mock,
) -> None:
    incomplete_response = ResponsesAPIResponse(
        id="resp_test",
        created_at=0,
        output=[],
        incomplete_details={"reason": "max_output_tokens"},
    )
    fire_success_callbacks = mock_litellm_call_with_late_success_callback(
        mocker, "aresponses", incomplete_response
    )
    llm = GeneralLlm(model="openai/o4-mini-deep-research", responses_api=True)

    with MonetaryCostManager(10) as cost_manager:
        with pytest.raises(ValueError, match="unable to complete request"):
            await llm._mockable_direct_call_to_model("Hi")
        await fire_success_callbacks()

    assert cost_manager.current_usage == pytest.approx(CALL_COST)


async def test_litellm_callback_still_counts_calls_made_outside_general_llm(
    mocker: Mock,
) -> None:
    fire_success_callbacks = mock_litellm_call_with_late_success_callback(
        mocker, "acompletion", make_litellm_response("OpenAI")
    )
    llm = GeneralLlm(model="openrouter/openai/gpt-4.1-nano")

    with MonetaryCostManager(10) as cost_manager:
        await llm._mockable_direct_call_to_model("Hi")
        await LitellmCostTracker().async_log_success_event(
            {"response_cost": CALL_COST}, make_litellm_response(None), None, None
        )
        await fire_success_callbacks()

    assert cost_manager.current_usage == pytest.approx(2 * CALL_COST)
