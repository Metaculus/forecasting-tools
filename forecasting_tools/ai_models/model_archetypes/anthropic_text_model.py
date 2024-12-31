import logging
import os
from abc import ABC

from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks.bedrock_anthropic_callback import (
    MODEL_COST_PER_1K_INPUT_TOKENS,
    _get_anthropic_claude_token_cost,
)
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_archetypes.traditional_online_llm import (
    TraditionalOnlineLlm,
)

logger = logging.getLogger(__name__)


class AnthropicTextToTextModel(TraditionalOnlineLlm, ABC):
    API_KEY_MISSING = True if os.getenv("ANTHROPIC_API_KEY") is None else False
    ANTHROPIC_API_KEY = SecretStr(
        os.getenv("ANTHROPIC_API_KEY")  # type: ignore
        if not API_KEY_MISSING
        else "fake-api-key-so-tests-dont-fail-to-initialize"
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
        response: TextTokenCostResponse = (
            await self._call_online_model_using_api(prompt)
        )
        return response

    async def _call_online_model_using_api(
        self, prompt: str
    ) -> TextTokenCostResponse:
        anthropic_llm = ChatAnthropic(
            model_name=self.MODEL_NAME,
            temperature=self.temperature,
            timeout=None,
            stop=None,
            base_url=None,
            api_key=self.ANTHROPIC_API_KEY,
        )
        messages = self._turn_model_input_into_messages(prompt)
        answer_message = await anthropic_llm.ainvoke(messages)
        answer = answer_message.content

        response_metadata = answer_message.response_metadata
        prompt_tokens = response_metadata["usage"]["input_tokens"]  # type: ignore
        completion_tokens = response_metadata["usage"]["output_tokens"]  # type: ignore
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost_from_tokens(
            prompt_tkns=prompt_tokens, completion_tkns=completion_tokens
        )

        assert isinstance(answer, str), "Answer is not a string"
        assert cost >= 0, "Cost is less than 0"
        assert prompt_tokens >= 0, "Prompt Tokens is less than 0"
        assert completion_tokens >= 0, "Completion Tokens is less than 0"

        return TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=self.MODEL_NAME,
            cost=cost,
        )

    def _turn_model_input_into_messages(
        self, prompt: str
    ) -> list[BaseMessage]:
        if self.system_prompt is None:
            return [HumanMessage(prompt)]
        else:
            return [SystemMessage(self.system_prompt), HumanMessage(prompt)]

    ################################## Methods For Mocking/Testing ##################################

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        cheap_input = cls._get_cheap_input_for_invoke()
        probable_output = "Hello! How can I assist you today? Feel free to ask any questions or let me know if you need help with anything."

        model = cls()
        prompt_tokens = (
            model.input_to_tokens(cheap_input)
            if not cls.API_KEY_MISSING
            else 13
        )
        anthropic_llm = ChatAnthropic(
            model_name=model.MODEL_NAME,
            timeout=None,
            stop=None,
            base_url=None,
            api_key=cls.ANTHROPIC_API_KEY,
        )
        completion_tokens = (
            anthropic_llm.get_num_tokens(probable_output)
            if not cls.API_KEY_MISSING
            else 26
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
        llm = ChatAnthropic(
            model_name=self.MODEL_NAME,
            timeout=None,
            stop=None,
            base_url=None,
            api_key=self.ANTHROPIC_API_KEY,
        )
        messages = self._turn_model_input_into_messages(prompt)
        tokens = llm.get_num_tokens_from_messages(messages)
        return tokens

    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        possible_detailed_model_names = MODEL_COST_PER_1K_INPUT_TOKENS.keys()
        detailed_model_name = [
            name
            for name in possible_detailed_model_names
            if self.MODEL_NAME in name
        ][0]
        cost = _get_anthropic_claude_token_cost(
            prompt_tkns, completion_tkns, detailed_model_name
        )
        return cost
