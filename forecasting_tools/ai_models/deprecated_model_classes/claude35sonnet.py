from typing import Final

import typing_extensions

from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)


@typing_extensions.deprecated(
    "LLM calls will slowly be moved to the GeneralLlm class", category=None
)
class Claude35Sonnet(CombinedLlmArchetype):
    # See Anthropic Limit on the account dashboard for most up-to-date limit
    # Latest as of Nov 6 2024 is claude-2-5-sonnet-20241022
    # Latest in general is claude-3-5-sonnet-latest
    # See models here https://docs.anthropic.com/en/docs/about-claude/models
    MODEL_NAME: Final[str] = "claude-3-5-sonnet-20241022"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 1_750
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 140_000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
