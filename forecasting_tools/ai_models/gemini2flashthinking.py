from typing import Final

from forecasting_tools.ai_models.model_archetypes.google_model import (
    GoogleTextToTextModel,
)


class Gemini2FlashThinking(GoogleTextToTextModel):
    """
    Represents the Gemini 2.0 Flash Thinking model for text generation.
    """

    MODEL_NAME: Final[str] = "gemini-2.0-flash-thinking-exp-1219"
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 2048,
        "candidate_count": 1,
        "stop_sequences": [],
        "safety_settings": {
            "category": "HARM_CATEGORY_UNSPECIFIED",
            "threshold": "BLOCK_NONE",
        },
    }

    # Rate limits
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 10
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 60
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 30000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
