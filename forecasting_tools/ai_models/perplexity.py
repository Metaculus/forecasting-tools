from typing import Final

from forecasting_tools.ai_models.model_archetypes.general_llm import (
    GeneralTextToTextLlm,
)


class Perplexity(GeneralTextToTextLlm):
    MODEL_NAME: Final[str] = "perplexity/sonar-pro"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = (
        40  # Technically 50, but giving wiggle room
    )
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 120
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 2000000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
