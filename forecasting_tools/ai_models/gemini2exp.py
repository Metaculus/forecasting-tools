from typing import Final

from forecasting_tools.ai_models.model_archetypes.google_model import (
    GoogleTextToTextModel,
)


class Gemini2Exp(GoogleTextToTextModel):
    """
    Represents the Gemini 2.0 (experimental) model for text generation.
    """

    MODEL_NAME: Final[str] = "gemini-exp-1206"
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Rate limits based on Google's documentation
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 4
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 60
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 30000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
