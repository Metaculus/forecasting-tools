import logging
import os
from abc import ABC

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAI
from pydantic import SecretStr

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_archetypes.traditional_online_llm import (
    TraditionalOnlineLlm,
)

logger = logging.getLogger(__name__)


class GoogleTextToTextModel(TraditionalOnlineLlm, ABC):
    API_KEY_MISSING = True if os.getenv("GOOGLE_API_KEY") is None else False
    GOOGLE_API_KEY = SecretStr(
        os.getenv("GOOGLE_API_KEY")  # type: ignore
        if not API_KEY_MISSING
        else "fake-api-key-so-tests-dont-fail-to-initialize"
    )

    async def invoke(self, prompt: str) -> str:
        response: TextTokenCostResponse = (
            await self._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            )
        )
        print(response.data)
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
        try:
            google_llm = GoogleGenerativeAI(
                model=self.MODEL_NAME,
                temperature=self.temperature,
                generation_config=self.GENERATION_CONFIG,
                google_api_key=str(self.GOOGLE_API_KEY.get_secret_value()),
            )

            logger.debug(
                f"Sending prompt to model {self.MODEL_NAME}: {prompt}"
            )
            answer = await google_llm.ainvoke(prompt)

            prompt_tokens = (
                self.input_to_tokens(prompt) if not self.API_KEY_MISSING else 0
            )
            completion_tokens = (
                self.output_to_tokens(answer)
                if not self.API_KEY_MISSING
                else 0
            )
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
        except Exception as e:
            logger.error(f"Error in Google model call: {str(e)}")
            logger.error(f"Model: {self.MODEL_NAME}")
            logger.error(f"Raw prompt: {prompt}")
            raise

    def _turn_model_input_into_messages(
        self, prompt: str
    ) -> list[BaseMessage]:
        if self.system_prompt is None:
            return [HumanMessage(content=prompt)]
        else:
            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]

    ################################## Methods For Mocking/Testing ##################################

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        cheap_input = cls._get_cheap_input_for_invoke()
        probable_output = "Hello! How can I assist you today?"

        model = cls()
        prompt_tokens = (
            model.input_to_tokens(cheap_input)
            if not cls.API_KEY_MISSING
            else 0
        )
        completion_tokens = (
            model.output_to_tokens(probable_output)
            if not cls.API_KEY_MISSING
            else 0
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
        llm = GoogleGenerativeAI(
            model=self.MODEL_NAME,
            google_api_key=self.GOOGLE_API_KEY,
        )
        tokens = llm.get_num_tokens(prompt)
        return tokens

    def output_to_tokens(self, output: str) -> int:
        llm = GoogleGenerativeAI(
            model=self.MODEL_NAME,
            google_api_key=self.GOOGLE_API_KEY,
        )
        tokens = llm.get_num_tokens(output)
        return tokens

    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        # Google AI Studio doesn't have a public pricing model yet.
        # This is a placeholder for future cost calculations.
        # For now, we assume the cost is 0.
        return 0.0
