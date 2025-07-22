from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

import pytest

from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
    HardLimitManager,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)
import sys

# List of hard limit manager classes to parameterize on
HARD_LIMIT_MANAGER_LIST = [HardLimitManager, MonetaryCostManager]


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_error_raised_properly_with_after_cost_check(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_per_call = 75

    with hard_limit_subclass(max_cost):
        try:
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
            hard_limit_subclass.raise_error_if_limit_would_be_reached(0)
        except Exception:
            assert False, "Error raised before max cost is exceeded"

        with pytest.raises(HardLimitExceededError):
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
            hard_limit_subclass.raise_error_if_limit_would_be_reached(0)


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_error_raised_properly_with_before_cost_check(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_per_call = 75

    with hard_limit_subclass(max_cost):
        try:
            hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_per_call)
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
        except Exception:
            assert False, "Error raised before max cost is exceeded"

        with pytest.raises(HardLimitExceededError):
            hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_per_call)


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_no_error_raised_when_only_adding_costs_without_a_cost_check(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_per_call = 75

    with hard_limit_subclass(max_cost):
        try:
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
        except Exception:
            assert False, "Error raised when only adding costs without a check"


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_cost_incurred_in_nested_cost_managers(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 200
    cost_per_call = 75
    expected_cost = 150

    with hard_limit_subclass(max_cost) as cost_manager_1:
        with hard_limit_subclass(max_cost) as cost_manager_2:
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)

            assert cost_manager_1.current_usage == expected_cost
            assert cost_manager_2.current_usage == expected_cost


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_costs_dont_conflict_between_unested_cost_managers(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_per_call = 75

    with hard_limit_subclass(max_cost):
        hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)

    with hard_limit_subclass(max_cost):
        hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_error_thrown_in_nested_cost_managers_when_max_cost_would_be_exceeded(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    unexceed_max_cost = 100
    exceeded_max_cost = 1
    cost_incurred = 75

    with hard_limit_subclass(unexceed_max_cost):
        with hard_limit_subclass(exceeded_max_cost):
            with pytest.raises(HardLimitExceededError):
                hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_incurred)

    with hard_limit_subclass(exceeded_max_cost):
        with hard_limit_subclass(unexceed_max_cost):
            with pytest.raises(HardLimitExceededError):
                hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_incurred)


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_error_raised_if_cost_is_negative_for_tracking(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_per_call = -75

    with hard_limit_subclass(max_cost):
        with pytest.raises(ValueError):
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_error_raised_if_cost_is_negative_for_checking(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_to_check = -75

    with hard_limit_subclass(max_cost):
        with pytest.raises(ValueError):
            hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_to_check)


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_no_error_raise_if_cost_is_0_for_tracking(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_per_call = 0

    with hard_limit_subclass(max_cost) as cost_manager:
        try:
            cost_manager.increase_current_usage_in_parent_managers(cost_per_call)
        except Exception:
            assert False, "Error raised on a cost of zero"


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_no_error_raise_if_cost_is_0_for_checking(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 100
    cost_to_check = 0

    with hard_limit_subclass(max_cost):
        try:
            hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_to_check)
        except Exception:
            assert False, "Error raised on a cost of zero"


# Specific test for HardLimitManager
@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_hard_limit_manager_works_with_async_tasks(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    async def charge_cost_function(cost_to_incur: float) -> None:
        hard_limit_subclass.increase_current_usage_in_parent_managers(cost_to_incur)

    run_async_tests(hard_limit_subclass, charge_cost_function)


def run_async_tests(
    hard_limit_subclass: type[HardLimitManager],
    charge_cost_function: Callable[[float], Coroutine[Any, Any, None]],
) -> None:
    sys.setrecursionlimit(2000)
    number_of_coroutines_at_each_level = 10
    cost_to_incur = 1
    max_cost = 1000000000

    class CostWithLevel:
        def __init__(
            self, level: int, cost: float, previous_levels: list[CostWithLevel]
        ) -> None:
            self.level = level
            self.cost = cost
            self.previous_levels = previous_levels

        def to_json(self) -> dict:
            return {
                "level": self.level,
                "cost": self.cost,
                "previous_levels": [level.to_json() for level in self.previous_levels],
            }

    async def charge_cost() -> None:
        await charge_cost_function(cost_to_incur)

    async def charge_cost_in_cost_manager() -> CostWithLevel:
        CALL_LEVEL = 0
        with hard_limit_subclass(max_cost) as cost_manager:
            coroutine = charge_cost()
            async_batching.run_coroutines([coroutine])
            current_cost = cost_manager.current_usage
        return CostWithLevel(CALL_LEVEL, current_cost, [])

    async def run_managers_in_a_manager() -> CostWithLevel:
        CALL_LEVEL = 1
        coroutines = [
            charge_cost_in_cost_manager()
            for _ in range(number_of_coroutines_at_each_level)
        ]
        with hard_limit_subclass(max_cost) as cost_manager:
            previous_levels = async_batching.run_coroutines(coroutines)
            current_cost = cost_manager.current_usage
        return CostWithLevel(CALL_LEVEL, current_cost, previous_levels)

    async def run_manger_managers_in_a_manager() -> CostWithLevel:
        CALL_LEVEL = 2
        coroutines = [
            run_managers_in_a_manager()
            for _ in range(number_of_coroutines_at_each_level)
        ]
        with hard_limit_subclass(max_cost) as cost_manager:
            previous_levels = async_batching.run_coroutines(coroutines)
            current_cost = cost_manager.current_usage
        return CostWithLevel(CALL_LEVEL, current_cost, previous_levels)

    level_data = asyncio.run(run_manger_managers_in_a_manager())
    logger.info(level_data.to_json())

    level_2_costs: list[CostWithLevel] = [level_data]
    level_1_costs: list[CostWithLevel] = level_2_costs[0].previous_levels
    nested_level_0: list[list[CostWithLevel]] = [
        level_1_item.previous_levels for level_1_item in level_1_costs
    ]
    level_0_costs: list[CostWithLevel] = [
        item for sublist in nested_level_0 for item in sublist
    ]

    for cost_with_level in level_0_costs:
        assert cost_with_level.cost == cost_to_incur

    for cost_with_level in level_1_costs:
        assert (
            cost_with_level.cost == number_of_coroutines_at_each_level * cost_to_incur
        )

    for cost_with_level in level_2_costs:
        assert (
            cost_with_level.cost
            == number_of_coroutines_at_each_level
            * number_of_coroutines_at_each_level
            * cost_to_incur
        )


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_no_error_raised_when_hard_limit_is_zero(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    max_cost = 0
    cost_per_call = 0

    with hard_limit_subclass(max_cost) as cost_manager:
        try:
            hard_limit_subclass.raise_error_if_limit_would_be_reached(cost_per_call)
            hard_limit_subclass.increase_current_usage_in_parent_managers(cost_per_call)
        except Exception:
            assert False, "Error raised when hard limit is zero"

    assert cost_manager.current_usage == 0


@pytest.mark.parametrize("hard_limit_subclass", HARD_LIMIT_MANAGER_LIST)
def test_error_raised_when_hard_limit_is_negative(
    hard_limit_subclass: type[HardLimitManager],
) -> None:
    negative_limit = -1

    with pytest.raises(AssertionError):
        hard_limit_subclass(negative_limit)
