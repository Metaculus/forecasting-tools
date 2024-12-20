from __future__ import annotations

import logging
import os
from abc import ABC

from langchain_community.chat_models.perplexity import ChatPerplexity
from langchain_core.messages.utils import convert_to_messages
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.basic_model_interfaces.priced_per_request import (
    PricedPerRequest,
)
from forecasting_tools.ai_models.model_archetypes.openai_text_model import (
    OpenAiTextToTextModel,
)

logger = logging.getLogger(__name__)


class PerplexityTextModel(OpenAiTextToTextModel, PricedPerRequest, ABC):
    PRICE_PER_TOKEN: float
    PERPLEXITY_API_KEY = (
        os.getenv("PERPLEXITY_API_KEY")
        if os.getenv("PERPLEXITY_API_KEY") is not None
        else "fake_key_so_it_doesn't_error_on_initialization"
    )
    _OPENAI_ASYNC_CLIENT = AsyncOpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai",
        max_retries=0,  # Retry is implemented locally
    )

    def __init_subclass__(cls: type[PerplexityTextModel], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if ABC not in cls.__bases__:
            if cls.PRICE_PER_TOKEN is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define PRICE_PER_TOKEN"
                )
            if cls.PRICE_PER_REQUEST is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define PRICE_PER_REQUEST"
                )

    def input_to_tokens(self, prompt: str) -> int:
        messages: list[ChatCompletionMessageParam] = (
            self._turn_model_input_into_messages(prompt)
        )
        chat = ChatPerplexity(
            client=self._OPENAI_ASYNC_CLIENT,
            api_key=self.PERPLEXITY_API_KEY,
            timeout=self.TIMEOUT_TIME,
        )
        langchain_messages = convert_to_messages(messages)  # type: ignore
        tokens = chat.get_num_tokens_from_messages(langchain_messages)
        adjustment = -2 * len(
            messages
        )  # Experimentally the actual tokens always seem to be 2 less more than the tokens calculated (it used to be 47 more)
        adjusted_tokens = tokens + adjustment
        if adjusted_tokens < 0:
            raise ValueError(
                f"adjusted_tokens is less than 0. tokens: {tokens}, adjustment: {adjustment}"
            )
        return adjusted_tokens

    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        """
        NOTE: Perplexity cost is not dependent on completion versus prompt differences
        NOTE: There is a Per-Request cost added to this function
        """
        total_tokens = prompt_tkns + completion_tkns
        cost = total_tokens * self.PRICE_PER_TOKEN + self.PRICE_PER_REQUEST
        return cost

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        data = 'The term "hi" has multiple meanings and uses:\n\n1. **Interjection**: It is commonly used as a greeting, similar to "hello" or "hey." For example, "Hi, how are you?".\n\n2. **Abbreviation**: It can stand for "Hawaii" or "Hawaiian Islands," especially in contexts like zip codes. In meteorology, it can also mean "heat index" or "humidity index".\n\n3. **Informal Adjective**: It is an informal, simplified spelling of "high," often used in terms like "hi-fi" (high fidelity).\n\n4. **Pop Culture**: "Hi" is also the title of a song by Hannah Diamond, a popstar and image-maker known for her work with PC Music.\n\n5. **Brand Name**: "Hi" is the name of a Web3 neobank that allows users to trade, save, and spend cryptocurrencies and fiat currencies. However, the UK operations of this platform have been ceased.\n\n6. **YouTube Content**: There are various YouTube videos titled "Hi" that address different topics, including personal messages from content creators and music videos.'
        prompt_tokens = 1
        completion_tokens = 254
        total_cost = cls().calculate_cost_from_tokens(
            prompt_tokens, completion_tokens
        )
        total_tokens = prompt_tokens + completion_tokens
        return TextTokenCostResponse(
            data=data,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=cls.MODEL_NAME,
            cost=total_cost,
        )
