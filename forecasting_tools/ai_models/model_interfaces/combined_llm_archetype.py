from __future__ import annotations

import logging
from abc import ABC

from litellm import model_cost

from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.model_interfaces.named_model import NamedModel
from forecasting_tools.ai_models.model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.ai_models.model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.retryable_model import (
    RetryableModel,
)
from forecasting_tools.ai_models.model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.token_limited_model import (
    TokenLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.tokens_incur_cost import (
    TokensIncurCost,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


class CombinedLlmArchetype(
    TokenLimitedModel,
    RequestLimitedModel,
    TimeLimitedModel,
    TokensIncurCost,
    RetryableModel,
    OutputsText,
    NamedModel,
    ABC,
):
    """
    This class will eventually be deprecated in favor of GeneralLlm.
    """

    _USE_METACULUS_PROXY: bool = False

    def __init__(
        self,
        temperature: float = 0,
        allowed_tries: int = RetryableModel._DEFAULT_ALLOWED_TRIES,
        system_prompt: str | None = None,
    ) -> None:
        if self._USE_METACULUS_PROXY:
            model_name = f"metaculus/{self.MODEL_NAME}"
        else:
            model_name = self.MODEL_NAME
        self.llm = GeneralLlm(
            model=model_name,
            temperature=temperature,
            allowed_tries=allowed_tries,
        )
        self.system_prompt = system_prompt

    @classmethod
    def _give_cost_tracking_warning(cls) -> None:
        assert isinstance(model_cost, dict)
        supported_model_names = model_cost.keys()
        model_not_supported = cls.MODEL_NAME not in supported_model_names
        if model_not_supported:
            logger.warning(
                f"Model {cls.MODEL_NAME} does not support cost tracking. "
            )

    async def invoke(
        self, prompt: str | VisionMessageData | list[dict[str, str]]
    ) -> str:
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        if self.system_prompt is not None:
            prompt = self.llm.model_input_to_message(
                prompt, self.system_prompt
            )
        response: TextTokenCostResponse = (
            await self._mockable_direct_call_to_model(prompt)
        )
        MonetaryCostManager.increase_current_usage_in_parent_managers(
            response.cost
        )
        return response.data

    @classmethod
    def _initialize_rate_limiters(cls) -> None:
        cls._reinitialize_request_rate_limiter()
        cls._reinitialize_token_limiter()

    async def _mockable_direct_call_to_model(
        self, prompt: str | VisionMessageData | list[dict[str, str]]
    ) -> TextTokenCostResponse:
        return await self.llm._mockable_direct_call_to_model(prompt)

    ################################## Methods For Mocking/Testing ##################################

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        llm = GeneralLlm(model=cls.MODEL_NAME)
        mock_response = (
            llm._get_mock_return_for_direct_call_to_model_using_cheap_input()
        )
        return mock_response

    @staticmethod
    def _get_cheap_input_for_invoke() -> str:
        return "Hi"

    ############################# Cost and Token Tracking Methods #############################

    def input_to_tokens(self, prompt: str | VisionMessageData) -> int:
        return self.llm.input_to_tokens(prompt)

    def text_to_tokens_direct(self, text: str) -> int:
        return self.llm.text_to_tokens_direct(text)

    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        return self.llm.calculate_cost_from_tokens(
            prompt_tkns, completion_tkns
        )
