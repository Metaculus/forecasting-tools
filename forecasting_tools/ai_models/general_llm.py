from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Literal

import litellm
import nest_asyncio
import typeguard
from litellm import acompletion, model_cost
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices, Usage
from litellm.utils import token_counter
from openai import AsyncOpenAI

from forecasting_tools.ai_models.agent_wrappers import track_generation
from forecasting_tools.ai_models.ai_utils.openai_utils import (
    OpenAiUtils,
    VisionMessageData,
)
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.ai_models.model_interfaces.outputs_text import OutputsText
from forecasting_tools.ai_models.model_interfaces.retryable_model import RetryableModel
from forecasting_tools.ai_models.model_interfaces.tokens_incur_cost import (
    TokensIncurCost,
)
from forecasting_tools.ai_models.model_tracker import ModelTracker
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    LitellmCostTracker,
)
from forecasting_tools.util.misc import fill_in_citations

nest_asyncio.apply()

logger = logging.getLogger(__name__)
ModelInputType = str | VisionMessageData | list[dict[str, str]]


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

    _defaults: dict[str, Any] = {
        "gpt-4o": {
            "timeout": 40,
        },
        "gpt-4o-mini": {
            "timeout": 40,
        },
        "o1": {
            "timeout": 80,
        },
        "o3": {
            "timeout": 80,
        },
        "o4": {
            "timeout": 80,
        },
        "claude-3-5-sonnet": {
            "timeout": 40,
        },
        "gemini-2.5": {
            "timeout": 80,
        },
        "deepseek/": {
            "timeout": 80,
        },
        "perplexity/": {
            "timeout": 120,
        },
        "perplexity/sonar-deep-research": {
            "timeout": 60
            * 60,  # Can take 30min+ at times according to Perplexity https://docs.perplexity.ai/models/model-cards#search-models:~:text=models%20may%20take-,30%2B%20minutes,-to%20process%20and
        },
    }

    def __init__(
        self,
        model: str,
        allowed_tries: int = RetryableModel._DEFAULT_ALLOWED_TRIES,
        temperature: float | int | None = 0,
        timeout: float | int | None = None,
        pass_through_unknown_kwargs: bool = True,
        populate_citations: bool = True,
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
        litellm.suppress_debug_info = (
            True  # Disables print statements that escape the litellm logger
        )
        super().__init__(allowed_tries=allowed_tries)
        self.model = model
        self.populate_citations = populate_citations

        metaculus_prefix = "metaculus/"
        exa_prefix = "exa/"
        openai_prefix = "openai/"
        anthropic_prefix = "anthropic/"
        self._use_metaculus_proxy = model.startswith(metaculus_prefix)
        self._use_exa = model.startswith(exa_prefix)
        prefixes_in_operational_order = [
            metaculus_prefix,
            exa_prefix,
            openai_prefix,
            anthropic_prefix,
        ]

        # prefix removal is to help with matching with model cost lists
        self._litellm_model = model
        for prefix in prefixes_in_operational_order:
            if self._litellm_model.startswith(prefix):
                self._litellm_model = self._litellm_model.removeprefix(prefix)

        self.litellm_kwargs = kwargs
        self.litellm_kwargs["model"] = self._litellm_model
        self.litellm_kwargs["temperature"] = temperature
        self.litellm_kwargs["timeout"] = timeout or self._get_default_timeout(
            self._litellm_model
        )

        if self._use_metaculus_proxy:
            if self.litellm_kwargs.get("base_url") is not None:
                raise ValueError(
                    "base_url should not be set if use_metaculus_proxy is True"
                )
            if self.litellm_kwargs.get("extra_headers") is not None:
                raise ValueError(
                    "extra_headers should not be set if use_metaculus_proxy is True"
                )
            if "claude" in self.model or "anthropic" in self.model:
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
            if self.litellm_kwargs.get("api_key") is None:
                self.litellm_kwargs["api_key"] = METACULUS_TOKEN
        elif self._use_exa and self.litellm_kwargs.get("api_key") is None:
            self.litellm_kwargs["api_key"] = os.getenv("EXA_API_KEY")

        valid_acompletion_params = set(inspect.signature(acompletion).parameters.keys())
        invalid_params = set(self.litellm_kwargs.keys()) - valid_acompletion_params
        if invalid_params and not pass_through_unknown_kwargs:
            raise ValueError(
                f"The following parameters are not valid for litellm's acompletion: {invalid_params}"
            )

        ModelTracker.give_cost_tracking_warning_if_needed(self._litellm_model)

    async def invoke(self, prompt: ModelInputType) -> str:
        response: TextTokenCostResponse = (
            await self._invoke_with_request_cost_time_and_token_limits_and_retry(prompt)
        )
        data = response.data
        return data

    @RetryableModel._retry_according_to_model_allowed_tries
    async def _invoke_with_request_cost_time_and_token_limits_and_retry(
        self, prompt: ModelInputType
    ) -> Any:
        logger.debug(f"Invoking model with prompt: {prompt}")

        with track_generation(
            input=self.model_input_to_message(prompt),
            model=self.model,
        ) as span:
            direct_call_response = await self._mockable_direct_call_to_model(prompt)
            answer = direct_call_response.data
            span.span_data.output = [{"role": "assistant", "content": answer}]
            # span.span_data.usage = usage.model_dump()
            span.span_data.model = self.model
            span.span_data.model_config = self.litellm_kwargs

        logger.debug(f"Model responded with: {direct_call_response}")
        return direct_call_response

    async def _mockable_direct_call_to_model(
        self, prompt: ModelInputType
    ) -> TextTokenCostResponse:
        self._everything_special_to_call_before_direct_call()
        assert self._litellm_model is not None

        if self._use_exa:
            return await self._call_exa_model(prompt)

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

        if answer == "":
            logger.warning(
                f"Model {self.model} returned an empty string as an answer. Raising exception (though this will probably result in a retry)"
            )
            raise RuntimeError(
                f"LLM answer is an empty string. The model was {self.model} and the prompt was: {prompt}"
            )

        cost = LitellmCostTracker.calculate_cost(response._hidden_params)

        if (
            response.model_extra
            and "citations" in response.model_extra
            and self.populate_citations
        ):
            citations = response.model_extra.get("citations")
            citations = typeguard.check_type(citations, list[str])
            answer = fill_in_citations(citations, answer, use_citation_brackets=False)

        await asyncio.sleep(
            0.00001
        )  # For whatever reason, you need to await a coroutine to get the litellm cost call back to work

        response = TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=total_tokens,
            model=self.model,
            cost=cost,
        )

        return response

    async def _call_exa_model(self, prompt: ModelInputType) -> TextTokenCostResponse:
        # TODO: Move this back to ussing the exa or OpenAI sdk.
        # I thought that a direct call might reveal the costDollars field but it didn't
        assert self._litellm_model is not None, "litellm model is not set"
        assert self.model.startswith(
            "exa/"
        ), f"model {self.model} is not an exa model but is being called like one"

        if len(str(prompt)) > 10_000:
            raise ValueError(
                f"Prompt is too long. Exa models have a 10k token limit. "
                f"Prompt length: {len(str(prompt))}. "
                f"Prompt:\n{str(prompt)[:1000]}...{str(prompt)[-1000:]}"
            )

        api_key = self.litellm_kwargs.get("api_key")
        timeout = self.litellm_kwargs.get("timeout")
        temperature = self.litellm_kwargs.get("temperature")
        extra_headers = self.litellm_kwargs.get("extra_headers")

        async with AsyncOpenAI(
            base_url="https://api.exa.ai",
            api_key=api_key,
        ) as client:

            completion = await client.chat.completions.create(
                model=self._litellm_model,
                messages=self.model_input_to_message(prompt),  # type: ignore
                temperature=temperature,
                timeout=timeout,
                extra_headers=extra_headers,
            )

        response_text = completion.choices[0].message.content
        if response_text is None:
            raise ValueError("Response text is None for exa model")

        usage = completion.usage
        if usage is None:
            raise ValueError("Usage is None for exa model")
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # TODO: API claims that there is completion["costDollars"], but I can't find it
        # Additionally we will need to log this separately to monetary cost manager (since litellm uses callbacks)
        cost = 0

        return TextTokenCostResponse(
            data=response_text,
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
                messages = OpenAiUtils.create_system_and_image_message_from_prompt(
                    user_input, system_prompt
                )  # type: ignore
            else:
                messages = (
                    OpenAiUtils.put_single_image_message_in_list_using_gpt_vision_input(
                        user_input
                    )
                )  # type: ignore
        else:
            raise TypeError("Unexpected model input type")

        messages = typeguard.check_type(messages, list[dict[str, str]])
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_model": self.model,
            "allowed_tries": self.allowed_tries,
            **{k: v for k, v in self.litellm_kwargs.items()},
        }

    @classmethod
    def _get_default_timeout(cls, model: str) -> int:
        all_keys = cls._defaults.keys()
        matching_keys = [key for key in all_keys if key in model]
        if not matching_keys:
            return 60
        return cls._defaults[matching_keys[-1]]["timeout"]

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
                f"Model {self._litellm_model} is not supported by litellm's model_cost dictionary"
            )

        input_cost_per_1k = model_cost_data.get("input_cost_per_token", 0) * 1000
        output_cost_per_1k = model_cost_data.get("output_cost_per_token", 0) * 1000

        prompt_cost = (prompt_tkns / 1000) * input_cost_per_1k
        completion_cost = (completion_tkns / 1000) * output_cost_per_1k

        total_cost = prompt_cost + completion_cost
        return total_cost

    ################################### Convenience Methods ###################################

    @classmethod
    def to_llm(cls, model_name_or_instance: str | GeneralLlm) -> GeneralLlm:
        if isinstance(model_name_or_instance, str):
            return GeneralLlm(model=model_name_or_instance)
        return model_name_or_instance

    @classmethod
    def to_model_name(cls, model_name_or_instance: str | GeneralLlm) -> str:
        if isinstance(model_name_or_instance, str):
            return model_name_or_instance
        return model_name_or_instance.model

    @classmethod
    def grounded_model(cls, model: str, temperature: float = 0) -> GeneralLlm:
        grounding_llm = GeneralLlm(
            model=model,
            temperature=temperature,
            generationConfig={
                "thinkingConfig": {
                    "thinkingBudget": 0,
                },
                "responseMimeType": "text/plain",
            },
            tools=[
                {"googleSearch": {}},
            ],
        )
        return grounding_llm

    @classmethod
    def thinking_budget_model(
        cls,
        model: str,
        temperature: float = 1,
        budget_tokens: int = 32000,
        max_tokens: int = 40000,
        timeout: float = 160,
    ) -> GeneralLlm:
        thinking_budget_llm = GeneralLlm(
            model=model,
            temperature=temperature,
            thinking={
                "type": "enabled",
                "budget_tokens": budget_tokens,
            },
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return thinking_budget_llm

    @classmethod
    def search_context_model(
        cls,
        model: str,
        temperature: float = 0,
        search_context_size: Literal["high", "medium", "low"] = "high",
    ) -> GeneralLlm:
        search_model = GeneralLlm(
            model=model,
            temperature=temperature,
            web_search_options={"search_context_size": search_context_size},
            reasoning_effort="high",
        )
        return search_model


if __name__ == "__main__":
    llm = GeneralLlm(model="openai/gpt-4o-mini", temperature=None)
    print(asyncio.run(llm.invoke("What is the capital of France?")))
