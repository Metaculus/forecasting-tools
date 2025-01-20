from typing import Final

from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.ai_models.model_archetypes.openai_vision_model import (
    OpenAiVisionToTextModel,
)


class Gpt4VisionInput(VisionMessageData):
    # This class is just to allow additional fields later, and easier imports so you can import all you need from one file
    pass


class Gpt4oVision(OpenAiVisionToTextModel):
    MODEL_NAME: Final[str] = "gpt-4o"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = (
        4000  # Errors said the limit is 4k, but it says 100k online See OpenAI Limit on the account dashboard for most up-to-date limit
    )
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = (
        35000  # Actual rate is 40k, but lowered for wiggle room. See OpenAI Limit on the account dashboard for most up-to-date limit
    )
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
