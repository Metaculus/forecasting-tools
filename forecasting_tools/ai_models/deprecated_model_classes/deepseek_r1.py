import typing_extensions

from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)


@typing_extensions.deprecated(
    "LLM calls will slowly be moved to the GeneralLlm class", category=None
)
class DeepSeekR1(CombinedLlmArchetype):
    MODEL_NAME = (
        "openrouter/deepseek/deepseek-r1"  # "deepseek/deepseek-reasoner"
    )
    REQUESTS_PER_PERIOD_LIMIT: int = 8_000
    REQUEST_PERIOD_IN_SECONDS: int = 60
    TIMEOUT_TIME: int = 125
    TOKENS_PER_PERIOD_LIMIT: int = 8_000_000
    TOKEN_PERIOD_IN_SECONDS: int = 60
