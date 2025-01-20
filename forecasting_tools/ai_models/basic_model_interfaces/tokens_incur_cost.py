from abc import ABC, abstractmethod
from typing import Any

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.basic_model_interfaces.incurs_cost import (
    IncursCost,
)
from forecasting_tools.ai_models.basic_model_interfaces.tokens_are_calculatable import (
    TokensAreCalculatable,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)


class TokensIncurCost(TokensAreCalculatable, IncursCost, ABC):

    @abstractmethod
    def calculate_cost_from_tokens(
        self, prompt_tkns: int, completion_tkns: int
    ) -> float:
        pass

    @abstractmethod
    def input_to_tokens(self, *args, **kwargs) -> int:
        pass

    async def _track_cost_in_manager_using_model_response(
        self, response_from_direct_call: Any
    ) -> None:
        if isinstance(response_from_direct_call, TextTokenCostResponse):
            cost = response_from_direct_call.cost
        else:
            raise NotImplementedError(
                f"This method has not been implemented for response type {type(response_from_direct_call)}"
            )
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)

    @property
    def cost_per_token_completion(self) -> float:
        return self.calculate_cost_from_tokens(
            prompt_tkns=0, completion_tkns=1
        )

    @property
    def cost_per_token_prompt(self) -> float:
        return self.calculate_cost_from_tokens(
            prompt_tkns=1, completion_tkns=0
        )
