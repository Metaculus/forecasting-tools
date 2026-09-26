import asyncio
import logging

import pytest

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from run_bots import get_default_bot_dict

logger = logging.getLogger(__name__)


def _pinned_llms_of_active_bots() -> list[GeneralLlm]:
    unique_llms: dict[str, GeneralLlm] = {}
    for config in get_default_bot_dict().values():
        if not config.tournaments or config.bot is None:
            continue
        for llm in config.bot._llms.values():
            if isinstance(llm, GeneralLlm) and llm._get_pinned_provider_slugs():
                unique_llms[str(llm.to_dict())] = llm
    return list(unique_llms.values())


@pytest.mark.parametrize(
    "llm",
    _pinned_llms_of_active_bots(),
    ids=lambda llm: f"{llm.model}@{llm._get_pinned_provider_slugs()[0]}",
)
async def test_active_bot_llm_is_served_by_its_pinned_openrouter_host(
    llm: GeneralLlm,
) -> None:
    with MonetaryCostManager(1) as cost_manager:
        response = await llm._invoke_with_request_cost_time_and_token_limits_and_retry(
            "Reply with only the word 'OK'."
        )
    logger.info(
        f"{llm.model} pinned to {llm._get_pinned_provider_slugs()} was served by "
        f"{response.serving_provider} for ${cost_manager.current_usage:.5f}"
    )

    assert response.serving_provider is not None
    assert cost_manager.current_usage > 0


async def test_opus_bot_reasoning_effort_is_applied_through_openrouter() -> None:
    bot_llm = get_default_bot_dict()["METAC_CLAUDE_OPUS_5_5_HIGH"].bot._llms["default"]
    assert isinstance(bot_llm, GeneralLlm)
    assert bot_llm.litellm_kwargs["reasoning_effort"] == "high"
    low_effort_kwargs = {
        key: value for key, value in bot_llm.litellm_kwargs.items() if key != "model"
    }
    low_effort_kwargs["reasoning_effort"] = "low"
    low_effort_llm = GeneralLlm(model=bot_llm.model, **low_effort_kwargs)
    # Effort changes the length of open-ended answers far more than of short puzzle answers
    prompt = "What should a forecaster consider when predicting whether a central bank will cut interest rates at its next meeting?"

    with MonetaryCostManager(1) as cost_manager:
        high_effort_response, low_effort_response = await asyncio.gather(
            bot_llm._invoke_with_request_cost_time_and_token_limits_and_retry(prompt),
            low_effort_llm._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            ),
        )
    logger.info(
        f"{bot_llm.model} used {high_effort_response.completion_tokens_used} completion tokens at high effort "
        f"and {low_effort_response.completion_tokens_used} at low effort for ${cost_manager.current_usage:.5f}"
    )

    assert (
        high_effort_response.completion_tokens_used
        > 1.25 * low_effort_response.completion_tokens_used
    )
