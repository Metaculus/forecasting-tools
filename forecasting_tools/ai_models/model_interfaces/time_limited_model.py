from __future__ import annotations

import logging
from abc import ABC

from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)
import functools
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")


class TimeLimitedModel(AiModel, ABC):
    TIMEOUT_TIME: int = NotImplemented

    def __init_subclass__(cls: type[TimeLimitedModel], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if ABC not in cls.__bases__ and cls.TIMEOUT_TIME is NotImplemented:
            raise NotImplementedError("You forgot to define TIMEOUT_TIME")

    @staticmethod
    def _wrap_in_model_defined_timeout(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(self: TimeLimitedModel, *args, **kwargs) -> T:
            async def wrapper2(self: TimeLimitedModel, *args, **kwargs) -> T:
                result = await func(self, *args, **kwargs)
                return result

            coroutine = wrapper2(self, *args, **kwargs)
            timed_coroutine = async_batching.wrap_coroutines_with_timeout(
                [coroutine], self.TIMEOUT_TIME
            )[0]
            return await timed_coroutine

        return wrapper
