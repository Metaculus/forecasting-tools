from typing import Final

from forecasting_tools.ai_models.model_archetypes.google_model import (
    GoogleTextToTextModel,
)


class Gemini2Flash(GoogleTextToTextModel):
    """
    Represents the Gemini 2.0 Flash model for text generation.
    """

    MODEL_NAME: Final[str] = "gemini-2.0-flash-exp"
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Rate limits based on Google's documentation
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 10
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 30
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 30000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
