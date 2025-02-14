from __future__ import annotations

import functools
import logging
from abc import ABC
from typing import Any, Callable, Coroutine, TypeVar

from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import (
    RefreshingBucketRateLimiter,
)

logger = logging.getLogger(__name__)


T = TypeVar("T")


class RequestLimitedModel(AiModel, ABC):
    # For more thoughts on abstract class properties see https://stackoverflow.com/questions/45248243/most-pythonic-way-to-declare-an-abstract-class-property
    REQUESTS_PER_PERIOD_LIMIT: int = NotImplemented
    REQUEST_PERIOD_IN_SECONDS: int = NotImplemented
    _request_limiter: RefreshingBucketRateLimiter = NotImplemented

    def __init_subclass__(cls: type[RequestLimitedModel], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        class_is_not_abstract = ABC not in cls.__bases__
        if class_is_not_abstract:
            if cls.REQUESTS_PER_PERIOD_LIMIT is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define REQUESTS_PER_PERIOD_LIMIT"
                )
            if cls.REQUEST_PERIOD_IN_SECONDS is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define REQUEST_PERIOD_IN_SECONDS"
                )
            cls._reinitialize_request_rate_limiter()

    @classmethod
    def _reinitialize_request_rate_limiter(cls) -> None:
        cls._request_limiter = NotImplemented
        cls._request_limiter = RefreshingBucketRateLimiter(
            cls.REQUESTS_PER_PERIOD_LIMIT,
            cls.REQUESTS_PER_PERIOD_LIMIT / cls.REQUEST_PERIOD_IN_SECONDS,
        )

    @staticmethod
    def _wait_till_request_capacity_available(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(self: RequestLimitedModel, *args, **kwargs) -> T:
            number_of_requests_being_made = 1
            await self._request_limiter.wait_till_able_to_acquire_resources(
                number_of_requests_being_made
            )
            result = await func(self, *args, **kwargs)
            return result

        return wrapper
