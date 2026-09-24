from typing import Any

from pydantic import BaseModel


class ModelResponse(BaseModel):
    data: Any


class TextTokenResponse(ModelResponse):
    data: str
    prompt_tokens_used: int
    completion_tokens_used: int
    total_tokens_used: int
    model: str


class TextTokenCostResponse(TextTokenResponse):
    cost: float
    serving_provider: str | None = None
