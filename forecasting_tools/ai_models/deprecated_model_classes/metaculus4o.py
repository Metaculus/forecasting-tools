from typing import Final

import typing_extensions

from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)


@typing_extensions.deprecated(
    "LLM calls will slowly be moved to the GeneralLlm class", category=None
)
class Gpt4oMetaculusProxy(CombinedLlmArchetype):
    """
    This model sends gpt4o requests to the Metaculus proxy server.
    """

    # See OpenAI Limit on the account dashboard for most up-to-date limit
    _USE_METACULUS_PROXY: bool = True

    MODEL_NAME: Final[str] = "gpt-4o"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 10000
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 800000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
