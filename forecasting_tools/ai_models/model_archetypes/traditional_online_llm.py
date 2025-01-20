import logging
from abc import ABC
from typing import Any

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

logger = logging.getLogger(__name__)


class TraditionalOnlineLlm(
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
        assert (
            temperature >= 0
        ), "Temperature must be greater than or equal to 0"
        assert temperature <= 1, "Temperature must be less than or equal to 1"
        self.temperature: float = temperature
        self.system_prompt: str | None = system_prompt

    async def invoke(self, *args, **kwargs) -> Any:
        result = await self._invoke_with_request_cost_time_and_token_limits_and_retry(
            *args, **kwargs
        )
        return result

    @RequestLimitedModel._wait_till_request_capacity_available
    @TokenLimitedModel._wait_till_token_capacity_available
    @RetryableModel._retry_according_to_model_allowed_tries
    @TokensIncurCost._wrap_in_cost_limiting_and_tracking
    @TimeLimitedModel._wrap_in_model_defined_timeout
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
