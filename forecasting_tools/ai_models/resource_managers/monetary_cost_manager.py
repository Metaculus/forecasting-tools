from __future__ import annotations

import litellm
from litellm.integrations.custom_logger import (
    CustomLogger as LitellmCustomLogger,
)

from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (  # For other files to easily import from this file #NOSONAR
    HardLimitManager,
)


class MonetaryCostManager(HardLimitManager):
    """
    This class is a subclass of HardLimitManager that is specifically for monetary costs.
    Assume every cost is in USD

    As of Aug 27 2024, the manager does not track predicted costs.
    For instance if you run 50 coroutines in parallel that cost 10c, and your limit is $1,
    all 50 will be let through (not 10).
    The cost will not register until the coroutines finish.
    """

    def __enter__(self) -> MonetaryCostManager:
        super().__enter__()
        LitellmCostTracker.initialize_cost_tracking()
        return self


class LitellmCostTracker(LitellmCustomLogger):
    """
    A callback handler for litellm cost tracking.
    See LitellmCustomLogger for more callback functions (on failure, post/pre API call, etc)
    """

    _initialized = False

    @staticmethod
    def initialize_cost_tracking() -> None:
        if LitellmCostTracker._initialized:
            return
        custom_handler = LitellmCostTracker()
        litellm.callbacks.append(custom_handler)
        LitellmCostTracker._initialized = True

    def log_pre_api_call(self, model, messages, kwargs):  # NOSONAR
        MonetaryCostManager.raise_error_if_limit_would_be_reached()

    async def async_log_pre_api_call(self, model, messages, kwargs):  # NOSONAR
        MonetaryCostManager.raise_error_if_limit_would_be_reached()

    def log_success_event(
        self, kwargs: dict, response_obj, start_time, end_time  # NOSONAR
    ) -> None:
        self._track_cost(kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time  # NOSONAR
    ) -> None:
        """
        For acompletion/aembeddings
        """
        self._track_cost(kwargs, response_obj)

    def _track_cost(self, kwargs: dict, response_obj) -> None:  # NOSONAR
        cost = self.calculate_cost(kwargs)
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)

    @classmethod
    def calculate_cost(cls, kwargs: dict) -> float:
        """
        Calculate the cost of the API call.
        """
        cost = kwargs.get("response_cost", 0)
        if cost is None:
            cost = 0
        return cost
