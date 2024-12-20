import os
from typing import Final

from openai import AsyncOpenAI

from forecasting_tools.ai_models.model_archetypes.openai_text_model import (
    OpenAiTextToTextModel,
)


class Gpt4oMetaculusProxy(OpenAiTextToTextModel):
    """
    This model sends gpt4o requests to the Metaculus proxy server.
    """

    METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
    _OPENAI_ASYNC_CLIENT = AsyncOpenAI(
        base_url="https://llm-proxy.metaculus.com/proxy/openai/v1",
        default_headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {METACULUS_TOKEN}",
        },
        api_key="Fake API Key since openai requires this not to be NONE. This isn't used",
        max_retries=0,  # Retry is implemented locally
    )

    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: Final[str] = "gpt-4o"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 10000
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 800000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
