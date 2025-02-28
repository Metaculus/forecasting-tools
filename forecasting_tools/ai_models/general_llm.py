from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import litellm
import typeguard
from litellm import acompletion, model_cost
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices, Usage
from litellm.utils import token_counter

from forecasting_tools.ai_models.ai_utils.openai_utils import (
    OpenAiUtils,
    VisionMessageData,
)
from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.ai_models.model_interfaces.retryable_model import (
    RetryableModel,
)
from forecasting_tools.ai_models.model_interfaces.tokens_incur_cost import (
    TokensIncurCost,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)
ModelInputType = str | VisionMessageData | list[dict[str, str]]


class ModelTracker:
    def __init__(self, model: str) -> None:
        self.model = model
        self.gave_cost_tracking_warning = False


class GeneralLlm(
    TokensIncurCost,
    RetryableModel,
    OutputsText,
):
    """
    A wrapper around litellm's acompletion function that adds some functionality
    like rate limiting, retry logic, metaculus proxy, and cost callback handling.

    Litellm support every model, most every parameter, and acts as one interface for every provider.
    """

    _model_trackers: dict[str, ModelTracker] = {}
    _defaults: dict[str, Any] = {
        "gpt-4o": {
            "timeout": 40,
        },
        "gpt-4o-mini": {
            "timeout": 40,
        },
        "o1-preview": {
            "timeout": 80,
        },
        "o1": {
            "timeout": 80,
        },
        "o3-mini": {
            "timeout": 80,
        },
        "perplexity/": {
            "timeout": 120,
        },
        "claude": {
            "timeout": 40,
        },
        "deepseek/": {
            "timeout": 80,
        },
    }

    def __init__(
        self,
        model: str,
        allowed_tries: int = RetryableModel._DEFAULT_ALLOWED_TRIES,
        temperature: float | int | None = 0,
        timeout: float | int | None = None,
        pass_through_unknown_kwargs: bool = True,
        **kwargs,
    ) -> None:
        """
        Pass in litellm kwargs as needed. Below are the available kwargs as of Feb 13 2025.

        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        functions: list | None = None,
        function_call: str | None = None,
        timeout: float | int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool | None = None,
        stream_options: dict | None = None,
        stop: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        modalities: list[ChatCompletionModality] | None = None,
        prediction: ChatCompletionPredictionContentParam | None = None,
        audio: ChatCompletionAudioParam | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict | None = None,
        user: str | None = None,
        # openai v1.0+ new params
        response_format: dict | Type[BaseModel] | None = None,
        seed: int | None = None,
        tools: list | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        # set api_base, api_version, api_key
        base_url: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        model_list: list | None = None,  # pass in a list of api_base,keys, etc.
        extra_headers: dict | None = None,
        # Optional liteLLM function params
        **kwargs,
        """
        super().__init__(allowed_tries=allowed_tries)
        self.model = model

        metaculus_prefix = "metaculus/"
        self._use_metaculus_proxy = model.startswith(metaculus_prefix)
        self._litellm_model = (
            model[len(metaculus_prefix) :]
            if self._use_metaculus_proxy
            else model
        )
        default_timeout = self._get_default_timeout(self._litellm_model)

        self.litellm_kwargs = kwargs
        self.litellm_kwargs["model"] = self._litellm_model
        self.litellm_kwargs["temperature"] = temperature
        self.litellm_kwargs["timeout"] = timeout or default_timeout

        if self._use_metaculus_proxy:
            assert (
                self.litellm_kwargs.get("base_url") is None
            ), "base_url should not be set if use_metaculus_proxy is True"
            assert (
                self.litellm_kwargs.get("extra_headers") is None
            ), "extra_headers should not be set if use_metaculus_proxy is True"
            if "claude" in self._litellm_model:
                self.litellm_kwargs["base_url"] = (
                    "https://llm-proxy.metaculus.com/proxy/anthropic"
                )
            else:
                self.litellm_kwargs["base_url"] = (
                    "https://llm-proxy.metaculus.com/proxy/openai/v1"
                )
            METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
            self.litellm_kwargs["extra_headers"] = {
                "Content-Type": "application/json",
                "Authorization": f"Token {METACULUS_TOKEN}",
            }
            self.litellm_kwargs["api_key"] = METACULUS_TOKEN

        valid_acompletion_params = set(
            inspect.signature(acompletion).parameters.keys()
        )
        invalid_params = (
            set(self.litellm_kwargs.keys()) - valid_acompletion_params
        )
        if invalid_params and not pass_through_unknown_kwargs:
            raise ValueError(
                f"The following parameters are not valid for litellm's acompletion: {invalid_params}"
            )

        self._give_cost_tracking_warning_if_needed()

    @classmethod
    def _get_default_timeout(cls, model: str) -> int:
        all_keys = cls._defaults.keys()
        matching_keys = [key for key in all_keys if key in model]
        if not matching_keys:
            return 60
        return cls._defaults[matching_keys[0]]["timeout"]

    def _give_cost_tracking_warning_if_needed(self) -> None:
        model = self._litellm_model
        model_tracker = self._model_trackers.get(model)
        if model_tracker is None:
            self._model_trackers[model] = ModelTracker(model)
        model_tracker = self._model_trackers[model]

        if model_tracker.gave_cost_tracking_warning:
            return

        assert isinstance(model_cost, dict)
        supported_model_names = model_cost.keys()
        model_not_supported = model not in supported_model_names
        if model_not_supported:
            message = f"Warning: Model {model} does not support cost tracking."
            print(message)
            logger.warning(message)

        model_tracker.gave_cost_tracking_warning = True

    async def invoke(self, prompt: ModelInputType) -> str:
        response: TextTokenCostResponse = (
            await self._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            )
        )
        return response.data

    @RetryableModel._retry_according_to_model_allowed_tries
    async def _invoke_with_request_cost_time_and_token_limits_and_retry(
        self, *args, **kwargs
    ) -> Any:
        logger.debug(f"Invoking model with args: {args} and kwargs: {kwargs}")
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        direct_call_response = await self._mockable_direct_call_to_model(
            *args, **kwargs
        )
        response_to_log = (
            direct_call_response[:1000]
            if isinstance(direct_call_response, str)
            else direct_call_response
        )
        logger.debug(f"Model responded with: {response_to_log}...")
        cost = direct_call_response.cost
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)
        return direct_call_response

    async def _mockable_direct_call_to_model(
        self, prompt: ModelInputType
    ) -> TextTokenCostResponse:
        self._everything_special_to_call_before_direct_call()
        assert self._litellm_model is not None
        litellm.drop_params = True

        response = await acompletion(
            messages=self.model_input_to_message(prompt),
            **self.litellm_kwargs,
        )
        assert isinstance(response, ModelResponse)
        choices = response.choices
        choices = typeguard.check_type(choices, list[Choices])
        answer = choices[0].message.content
        assert isinstance(
            answer, str
        ), f"Answer is not a string and is of type: {type(answer)}. Answer: {answer}"
        usage = response.usage  # type: ignore
        assert isinstance(usage, Usage)
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        cost = response._hidden_params.get(
            "response_cost"
        )  # If this has problems, consider using the budgetmanager class
        if cost is None:
            cost = 0

        cost += self.calculate_per_request_cost(self.model)

        return TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=self.model,
            cost=cost,
        )

    def model_input_to_message(
        self, user_input: ModelInputType, system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if isinstance(user_input, list):
            assert (
                system_prompt is None
            ), "System prompt cannot be used with list of messages since the list may include a system message"
            user_input = typeguard.check_type(user_input, list[dict[str, str]])
            messages = user_input
        elif isinstance(user_input, str):
            user_message: dict[str, str] = {
                "role": "user",
                "content": user_input,
            }
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    user_message,
                ]
            else:
                messages = [user_message]
        elif isinstance(user_input, VisionMessageData):
            if system_prompt is not None:
                messages = (
                    OpenAiUtils.create_system_and_image_message_from_prompt(
                        user_input, system_prompt
                    )
                )  # type: ignore
            else:
                messages = OpenAiUtils.put_single_image_message_in_list_using_gpt_vision_input(
                    user_input
                )  # type: ignore
        else:
            raise TypeError("Unexpected model input type")

        messages = typeguard.check_type(messages, list[dict[str, str]])
        return messages

    ################################## Methods For Mocking/Testing ##################################

    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        self,
    ) -> TextTokenCostResponse:
        cheap_input = self._get_cheap_input_for_invoke()
        probable_output = "Hello! How can I assist you today?"

        prompt_tokens = self.input_to_tokens(cheap_input)
        completion_tokens = self.text_to_tokens_direct(probable_output)

        try:
            total_cost = self.calculate_cost_from_tokens(
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
            model=self.model,
            cost=total_cost,
        )

    @staticmethod
    def _get_cheap_input_for_invoke() -> str:
        return "Hi"

    ############################# Cost and Token Tracking Methods #############################

    def input_to_tokens(self, prompt: ModelInputType) -> int:
        return token_counter(
            model=self._litellm_model,
            messages=self.model_input_to_message(prompt),
        )

    def text_to_tokens_direct(self, text: str) -> int:
        return token_counter(model=self._litellm_model, text=text)

    def calculate_cost_from_tokens(
        self,
        prompt_tkns: int,
        completion_tkns: int,
        calculate_full_cost: bool = True,
    ) -> float:
        assert self._litellm_model is not None
        # litellm.model_cost contains cost per 1k tokens for input/output
        model_cost_data = model_cost.get(self._litellm_model)
        if model_cost_data is None:
            raise ValueError(
                f"Model {self._litellm_model} is not supported by model_cost"
            )

        input_cost_per_1k = (
            model_cost_data.get("input_cost_per_token", 0) * 1000
        )
        output_cost_per_1k = (
            model_cost_data.get("output_cost_per_token", 0) * 1000
        )

        prompt_cost = (prompt_tkns / 1000) * input_cost_per_1k
        completion_cost = (completion_tkns / 1000) * output_cost_per_1k

        total_cost = prompt_cost + completion_cost
        if calculate_full_cost:
            total_cost += self.calculate_per_request_cost(self.model)
        return total_cost

    @classmethod
    def calculate_per_request_cost(cls, model: str) -> float:
        cost = 0
        if "perplexity" in model:
            cost += 0.005  # There is at least one search costing $0.005 per perplexity request
            if "pro" in model:
                cost += 0.005  # There is probably more than one search in pro models
        return cost
