from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)
from contextvars import ContextVar


class HardLimitExceededError(Exception):
    """Raised when the hardlimit is exceeded"""


class HardLimitManager:
    _active_limit_managers: ContextVar[list[HardLimitManager]] = ContextVar(
        "_active_limit_managers", default=[]
    )
    _id_counter: int = 0

    def __init__(
        self, hard_limit: float = 0, log_usage_when_called: bool = False
    ) -> None:
        assert hard_limit >= 0
        self.hard_limit: Final[float] = hard_limit
        self._current_usage: float = 0
        self.__log_usage_when_called: bool = log_usage_when_called
        HardLimitManager._id_counter += 1
        self.id = HardLimitManager._id_counter

    @property
    def current_usage(self) -> float:
        return self._current_usage

    @property
    def amount_left(self) -> float:
        return self.hard_limit - self._current_usage

    @classmethod
    def get_active_cost_managers(cls) -> list[HardLimitManager]:
        return cls._active_limit_managers.get()

    def __enter__(self) -> HardLimitManager:
        cost_managers = self._active_limit_managers.get().copy()
        cost_managers.append(self)
        self._active_limit_managers.set(cost_managers)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # NOSONAR
        # This stuff might be redundant
        cost_managers = self._active_limit_managers.get()
        cost_managers.remove(self)
        self._active_limit_managers.set(cost_managers)

    @classmethod
    def raise_error_if_limit_would_be_reached(
        cls, amount_to_check_room_for: float = 0
    ) -> None:
        """
        You can put in 0 for the amount to check if the current usage is at the limit
        """
        if amount_to_check_room_for < 0:
            raise ValueError("Amount should be a positive number or zero")
        for cost_manager in cls._active_limit_managers.get():
            if (
                cost_manager.amount_left < amount_to_check_room_for
                and cost_manager.hard_limit != 0
            ):
                raise HardLimitExceededError(
                    f"Usage amount {amount_to_check_room_for} would push current usage to {cost_manager.current_usage + amount_to_check_room_for} exceeding the hard limit of {cost_manager.hard_limit}"
                )

    @classmethod
    def increase_current_usage_in_parent_managers(cls, amount: float) -> None:
        if amount < 0:
            raise ValueError("Cost should be a positive number or zero")
        if amount == 0:
            logger.debug(
                "The cost inputted is zero which may or may not be a problem"
            )
        for cost_manager in cls._active_limit_managers.get():
            cost_manager._current_usage += amount
            if (
                cost_manager._current_usage > cost_manager.hard_limit
                and cost_manager.hard_limit != 0
            ):
                logger.warning(
                    f"Usage increase exceeded the hard limit of {cost_manager.hard_limit}"
                )
            if cost_manager.__log_usage_when_called:
                logger.info(
                    f"{cost_manager.__class__}.ID{cost_manager.id}. Current usage now {cost_manager._current_usage}. Cost of {amount} added"
                )
