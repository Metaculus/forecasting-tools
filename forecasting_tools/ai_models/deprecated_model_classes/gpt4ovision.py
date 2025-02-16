from typing import Final

import typing_extensions

from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)


class Gpt4VisionInput(VisionMessageData):
    # This class is just to allow additional fields later, and easier imports so you can import all you need from one file
    pass


@typing_extensions.deprecated(
    "LLM calls will slowly be moved to the GeneralLlm class", category=None
)
class Gpt4oVision(CombinedLlmArchetype):
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

    SMALL_BASE_64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    CHEAP_VISION_MESSAGE_DATA = VisionMessageData(
        prompt="Hi", b64_image=SMALL_BASE_64_IMAGE, image_resolution="low"
    )

    @classmethod
    def _get_cheap_input_for_invoke(cls) -> VisionMessageData:
        return cls.CHEAP_VISION_MESSAGE_DATA
