import logging
import re

import pytest

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from run_bots import get_default_bot_dict

logger = logging.getLogger(__name__)


def _get_pinned_endpoint_slug(llm: GeneralLlm) -> str | None:
    provider_settings = llm.litellm_kwargs.get("extra_body", {}).get("provider")
    if provider_settings is None:
        return None
    return provider_settings["order"][0]


def _pinned_llms_of_active_bots() -> list[GeneralLlm]:
    unique_llms: dict[str, GeneralLlm] = {}
    for config in get_default_bot_dict().values():
        if not config.tournaments or config.bot is None:
            continue
        for llm in config.bot._llms.values():
            if isinstance(llm, GeneralLlm) and _get_pinned_endpoint_slug(llm):
                unique_llms[str(llm.to_dict())] = llm
    return list(unique_llms.values())


def _normalize_provider_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@pytest.mark.parametrize(
    "llm",
    _pinned_llms_of_active_bots(),
    ids=lambda llm: f"{llm.model}@{_get_pinned_endpoint_slug(llm)}",
)
async def test_active_bot_llm_is_served_by_its_pinned_openrouter_host(
    llm: GeneralLlm,
) -> None:
    pinned_slug = _get_pinned_endpoint_slug(llm)
    assert pinned_slug is not None

    with MonetaryCostManager(1) as cost_manager:
        response = await llm._invoke_with_request_cost_time_and_token_limits_and_retry(
            "Reply with only the word 'OK'."
        )
    logger.info(
        f"{llm.model} pinned to {pinned_slug} was served by "
        f"{response.serving_provider} for ${cost_manager.current_usage:.5f}"
    )

    assert response.serving_provider is not None
    assert _normalize_provider_name(
        response.serving_provider
    ) == _normalize_provider_name(pinned_slug.split("/")[0])
