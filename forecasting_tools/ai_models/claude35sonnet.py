from typing import Final

from forecasting_tools.ai_models.model_archetypes.anthropic_text_model import (
    AnthropicTextToTextModel,
)


class Claude35Sonnet(AnthropicTextToTextModel):
    # See Anthropic Limit on the account dashboard for most up-to-date limit
    # Latest as of Nov 6 2024 is claude-2-5-sonnet-20241022
    # Latest in general is claude-3-5-sonnet-latest
    # See models here https://docs.anthropic.com/en/docs/about-claude/models
    MODEL_NAME: Final[str] = "claude-3-5-sonnet-20240620"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 1_750
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 140_000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
