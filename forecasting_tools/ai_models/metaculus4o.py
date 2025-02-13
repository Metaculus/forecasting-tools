from typing import Final

from forecasting_tools.ai_models.model_archetypes.general_text_to_text_llm import (
    BaseLlmArchetype,
)


class Gpt4oMetaculusProxy(BaseLlmArchetype):
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
