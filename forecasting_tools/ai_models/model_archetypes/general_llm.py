from __future__ import annotations

import logging
from abc import ABC
from typing import Any

import litellm
import typeguard
from litellm import acompletion, model_cost
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices, Usage
from litellm.utils import token_counter

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.basic_model_interfaces.named_model import (
    NamedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.ai_models.basic_model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.retryable_model import (
    RetryableModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.token_limited_model import (
    TokenLimitedModel,
)
from forecasting_tools.ai_models.basic_model_interfaces.tokens_incur_cost import (
    TokensIncurCost,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


class GeneralTextToTextLlm(
    TokenLimitedModel,
    RequestLimitedModel,
    TimeLimitedModel,
    TokensIncurCost,
    RetryableModel,
    OutputsText,
    NamedModel,
    ABC,
):

    def __init__(
        self,
        temperature: float = 0,
        allowed_tries: int = RetryableModel._DEFAULT_ALLOWED_TRIES,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(allowed_tries=allowed_tries)
        self.temperature: float = temperature
        self.system_prompt: str | None = system_prompt

    def __init_subclass__(cls: type[GeneralTextToTextLlm], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        assert isinstance(model_cost, dict)
        supported_model_names = model_cost.keys()
        model_not_supported = cls.MODEL_NAME not in supported_model_names
        if model_not_supported:
            logger.warning(
                f"Model {cls.MODEL_NAME} is not supported by model_cost"
            )

    async def invoke(self, prompt: str) -> str:
        response: TextTokenCostResponse = (
            await self._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            )
        )
        return response.data

    @RequestLimitedModel._wait_till_request_capacity_available
    @TokenLimitedModel._wait_till_token_capacity_available
    @RetryableModel._retry_according_to_model_allowed_tries
    async def _invoke_with_request_cost_time_and_token_limits_and_retry(
        self, *args, **kwargs
    ) -> Any:
        logger.debug(f"Invoking model with args: {args} and kwargs: {kwargs}")
        direct_call_response = await self._mockable_direct_call_to_model(
            *args, **kwargs
        )
        response_to_log = (
            direct_call_response[:1000]
            if isinstance(direct_call_response, str)
            else direct_call_response
        )
        logger.debug(f"Model responded with: {response_to_log}...")
        return direct_call_response

    @classmethod
    def _initialize_rate_limiters(cls) -> None:
        cls._reinitialize_request_rate_limiter()
        cls._reinitialize_token_limiter()

    async def _mockable_direct_call_to_model(
        self, prompt: str
    ) -> TextTokenCostResponse:
        self._everything_special_to_call_before_direct_call()
        assert self.MODEL_NAME is not None
        litellm.drop_params = True
        response = await acompletion(
            model=self.MODEL_NAME,
            messages=self._get_messages(prompt),
            temperature=self.temperature,
            stream=False,
            timeout=self.TIMEOUT_TIME,
        )
        assert isinstance(response, ModelResponse)
        choices = response.choices
        choices = typeguard.check_type(choices, list[Choices])
        answer = choices[0].message.content
        assert isinstance(answer, str)
        usage = response.usage
        assert isinstance(usage, Usage)
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        cost = response._hidden_params[
            "response_cost"
        ]  # If this has problems, consider using the budgetmanager class
        if cost is None:
            cost = 0
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)

        return TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=self.MODEL_NAME,
            cost=cost,
        )

    def _get_messages(self, prompt: str) -> list[dict[str, str]]:
        user_message = {"role": "user", "content": prompt}
        if self.system_prompt is not None:
            return [
                {"role": "system", "content": self.system_prompt},
                user_message,
            ]
        return [user_message]

    ################################## Methods For Mocking/Testing ##################################

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        cheap_input = cls._get_cheap_input_for_invoke()
        probable_output = "Hello! How can I assist you today?"

        model = cls()
        prompt_tokens = model.input_to_tokens(cheap_input)
        completion_tokens = model.text_to_tokens_direct(probable_output)

        try:
            total_cost = model.calculate_cost_from_tokens(
                prompt_tkns=prompt_tokens, completion_tkns=completion_tokens
            )
        except ValueError:
            total_cost = 0.0

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
        return token_counter(
            model=self.MODEL_NAME, messages=self._get_messages(prompt)
        )

    def text_to_tokens_direct(self, text: str) -> int:
        return token_counter(model=self.MODEL_NAME, text=text)

    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        assert self.MODEL_NAME is not None
        # litellm.model_cost contains cost per 1k tokens for input/output
        model_cost_data = model_cost.get(self.MODEL_NAME)
        if model_cost_data is None:
            raise ValueError(
                f"Model {self.MODEL_NAME} is not supported by model_cost"
            )

        input_cost_per_1k = (
            model_cost_data.get("input_cost_per_token", 0) * 1000
        )
        output_cost_per_1k = (
            model_cost_data.get("output_cost_per_token", 0) * 1000
        )

        prompt_cost = (prompt_tkns / 1000) * input_cost_per_1k
        completion_cost = (completion_tkns / 1000) * output_cost_per_1k

        return prompt_cost + completion_cost
