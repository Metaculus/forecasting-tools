from forecasting_tools.ai_models.model_archetypes.general_llm import (
    GeneralTextToTextLlm,
)


class DeepSeekR1(GeneralTextToTextLlm):
    MODEL_NAME = "deepseek/deepseek-reasoner"
    REQUESTS_PER_PERIOD_LIMIT: int = 8_000
    REQUEST_PERIOD_IN_SECONDS: int = 60
    TIMEOUT_TIME: int = 125
    TOKENS_PER_PERIOD_LIMIT: int = 8_000_000
    TOKEN_PERIOD_IN_SECONDS: int = 60