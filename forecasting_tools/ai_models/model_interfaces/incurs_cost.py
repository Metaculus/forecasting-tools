import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, TypeVar

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

T = TypeVar("T")


class IncursCost(ABC):
    """
    Interface indicating a model should track its cost in the monetary cost manager.
    """

    @abstractmethod
    async def _track_cost_in_manager_using_model_response(
        self, response_from_direct_call: Any
    ) -> None:
        pass

    @staticmethod
    def _wrap_in_cost_limiting_and_tracking(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(self: IncursCost, *args, **kwargs) -> T:
            MonetaryCostManager.raise_error_if_limit_would_be_reached()

            direct_call_response = await func(self, *args, **kwargs)

            await self._track_cost_in_manager_using_model_response(
                direct_call_response
            )
            return direct_call_response

        return wrapper
