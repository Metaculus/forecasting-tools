import logging
import os
from abc import ABC

from langchain_community.callbacks.openai_info import (
    TokenType,
    get_openai_token_cost_for_model,
)
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam

from forecasting_tools.ai_models.ai_utils.openai_utils import OpenAiUtils
from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_archetypes.traditional_online_llm import (
    TraditionalOnlineLlm,
)

logger = logging.getLogger(__name__)


class OpenAiTextToTextModel(TraditionalOnlineLlm, ABC):
    _OPENAI_ASYNC_CLIENT = AsyncOpenAI(
        api_key=(
            os.getenv("OPENAI_API_KEY")
            if os.getenv("OPENAI_API_KEY") is not None
            else "fake_key_so_it_doesn't_error_on_initialization"
        ),
        max_retries=0,  # Retry is implemented locally
    )

    async def invoke(self, prompt: str) -> str:
        response: TextTokenCostResponse = (
            await self._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            )
        )
        return response.data

    async def _mockable_direct_call_to_model(
        self, prompt: str
    ) -> TextTokenCostResponse:
        self._everything_special_to_call_before_direct_call()
        messages = self._turn_model_input_into_messages(prompt)
        response: TextTokenCostResponse = (
            await self._call_online_model_using_api(messages, self.temperature)
        )
        return response

    def _turn_model_input_into_messages(
        self, prompt: str
    ) -> list[ChatCompletionMessageParam]:
        if self.system_prompt is None:
            return OpenAiUtils.put_single_user_message_in_list_using_prompt(
                prompt
            )
        else:
            return OpenAiUtils.create_system_and_user_message_from_prompt(
                prompt, self.system_prompt
            )

    async def _call_online_model_using_api(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int | NotGiven = NOT_GIVEN,
    ) -> TextTokenCostResponse:
        client = self._OPENAI_ASYNC_CLIENT

        response = await client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.choices[0].message.content is None:
            raise RuntimeError(
                "The model failed to give an answer. response.choices[0].message.content is None"
            )

        answer = response.choices[0].message.content
        usage_stats = response.usage
        if usage_stats is None:
            raise RuntimeError("usage_stats is None")
        prompt_tokens = usage_stats.prompt_tokens
        completion_tokens = usage_stats.completion_tokens
        total_tokens = usage_stats.total_tokens

        cost = self.calculate_cost_from_tokens(
            prompt_tkns=prompt_tokens, completion_tkns=completion_tokens
        )

        return TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=self.MODEL_NAME,
            cost=cost,
        )

    ################################## Methods For Mocking/Testing ##################################

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        cheap_input = cls._get_cheap_input_for_invoke()
        probable_output = "Hello! How can I assist you today?"

        model = cls()
        prompt_tokens = model.input_to_tokens(cheap_input)
        completion_tokens = OpenAiUtils.text_to_tokens_direct(
            probable_output, cls.MODEL_NAME
        )

        total_cost = model.calculate_cost_from_tokens(
            prompt_tkns=prompt_tokens, completion_tkns=completion_tokens
        )
        total_tokens = prompt_tokens + completion_tokens
        return TextTokenCostResponse(
            data=probable_output,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=cls.MODEL_NAME,
            cost=total_cost,
        )

    @staticmethod
    def _get_cheap_input_for_invoke() -> str:
        return "Hi"

    ############################# Cost and Token Tracking Methods #############################

    def input_to_tokens(self, prompt: str) -> int:
        messages = self._turn_model_input_into_messages(prompt)
        tokens = OpenAiUtils.messages_to_tokens(messages, self.MODEL_NAME)
        return tokens

    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        prompt_cost = get_openai_token_cost_for_model(
            self.MODEL_NAME, prompt_tkns, token_type=TokenType.PROMPT
        )
        completion_cost = get_openai_token_cost_for_model(
            self.MODEL_NAME, completion_tkns, token_type=TokenType.COMPLETION
        )
        cost = prompt_cost + completion_cost
        return cost
