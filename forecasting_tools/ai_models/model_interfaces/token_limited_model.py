from __future__ import annotations

import logging
from abc import ABC

from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel

logger = logging.getLogger(__name__)
import functools
from typing import Any, Callable, Coroutine, TypeVar

from forecasting_tools.ai_models.model_interfaces.tokens_are_calculatable import (
    TokensAreCalculatable,
)
from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import (
    RefreshingBucketRateLimiter,
)

T = TypeVar("T")


class TokenLimitedModel(AiModel, TokensAreCalculatable, ABC):
    TOKENS_PER_PERIOD_LIMIT: int = NotImplemented
    TOKEN_PERIOD_IN_SECONDS: int = NotImplemented
    _token_limiter: RefreshingBucketRateLimiter = NotImplemented

    def __init_subclass__(cls: type[TokenLimitedModel], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        is_not_abstract = ABC not in cls.__bases__
        if is_not_abstract:
            if cls.TOKENS_PER_PERIOD_LIMIT is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define TOKENS_PER_PERIOD_LIMIT"
                )
            if cls.TOKEN_PERIOD_IN_SECONDS is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define TOKEN_PERIOD_IN_SECONDS"
                )
            cls._reinitialize_token_limiter()

    @classmethod
    def _reinitialize_token_limiter(cls) -> None:
        cls._token_limiter = NotImplemented
        cls._token_limiter = RefreshingBucketRateLimiter(
            cls.TOKENS_PER_PERIOD_LIMIT,
            cls.TOKENS_PER_PERIOD_LIMIT / cls.TOKEN_PERIOD_IN_SECONDS,
        )

    @staticmethod
    def _wait_till_token_capacity_available(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(self: TokenLimitedModel, *args, **kwargs) -> T:
            tokens_of_prompt = self.input_to_tokens(*args, **kwargs)
            await self._token_limiter.wait_till_able_to_acquire_resources(
                tokens_of_prompt
            )
            result = await func(self, *args, **kwargs)
            return result

        return wrapper

    @classmethod
    def _make_token_limiter_have_large_rate(cls) -> None:
        """
        NOTE: This method is only for testing purposes.
        """
        absurdly_large_capacity = 100000000
        cls._token_limiter = RefreshingBucketRateLimiter(
            absurdly_large_capacity, absurdly_large_capacity / 1
        )
