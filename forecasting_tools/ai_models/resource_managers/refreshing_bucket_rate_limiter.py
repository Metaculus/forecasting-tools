import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
import asyncio
from enum import Enum
from typing import Final


class LimitReachedResponse(Enum):
    RAISE_EXCEPTION = 1
    WAIT = 2


class ResourceUnavailableError(RuntimeError):
    """Raised when resources are unavailable and cannot continue execution."""


class ResourceUseEntry:
    def __init__(self, resources_used: int, time: datetime) -> None:
        self.resources_used: int = resources_used
        self.time: datetime = time


class RefreshingBucketRateLimiter:
    """
    The refreshing bucket rate limiter is a way of limiting resource use over time.
    - It sets up a capacity and a refresh rate.
    - Refresh rate is per second.
    - It will wait to fill up when the bucket is emptied.


    For example:
    - requests per minute
    - tokens per second
    - etc.

    The capacity acts as a burst limit. You can spend X resources before
    being forced to slow down as the bottom of the bucket is hit and you
    can't use more resources.
    If you use only X capacity, it will take the (refresh rate * X) to get back to full capacity.
    If you reach the bottom of the bucket, the bucket will fill all the way up before you can use resources again.
    This is to make sure something like a "requests per minute" limit is not exceeded even after a burst
    (since averaging out the burst over the full recharge period would successfully hold to the limit).
    """

    def __init__(
        self,
        capacity: float,
        refresh_rate: float,
        limit_reached_response: LimitReachedResponse = LimitReachedResponse.WAIT,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self.capacity: Final[float] = capacity

        if refresh_rate < 0:
            raise ValueError("refresh_rate must not be negative")
        elif refresh_rate == 0:
            logger.info("refresh_rate is 0, resources will not refresh")
        self.refresh_rate: Final[float] = refresh_rate

        self.__limit_reached_response: LimitReachedResponse = (
            limit_reached_response
        )
        self.__available_resources: float = capacity
        self.__resource_history: list[ResourceUseEntry] = []
        self.__last_replenish_time: datetime = datetime.now()
        self.__available_resource_lock = asyncio.Lock()
        self.__resource_history_lock = asyncio.Lock()
        self.__fill_the_bucket_mode = False

    def refresh_and_then_get_available_resources(self) -> float:
        asyncio.run(self._refresh_resource_count())
        return self._available_resources

    def zero_out_resources(self) -> None:
        self._available_resources = 0
        asyncio.run(self.__update_fill_the_bucket_mode(resources_ran_out=True))

    @property
    def _available_resources(self) -> float:
        return self.__available_resources

    @_available_resources.setter
    def _available_resources(self, value: float) -> None:
        if value > self.capacity:
            raise ValueError("value must be less than or equal to capacity")
        elif value < 0:
            raise ValueError("value must be greater than or equal to 0")
        self.__available_resources = value

    def calculate_resources_passed_into_acquire_in_time_range(
        self, start_time: datetime, end_time: datetime
    ) -> int:
        resources_used = 0
        for entry in self.__resource_history:
            if start_time < entry.time < end_time:
                resources_used += entry.resources_used

        return resources_used

    async def wait_till_able_to_acquire_resources(
        self, resources_being_consumed: int
    ) -> None:
        if resources_being_consumed > self.capacity:
            raise ValueError(
                f"resources_being_consumed must be less than or equal to capacity. Capacity: {self.capacity}, resources_being_consumed: {resources_being_consumed}"
            )

        resources_are_available: bool = (
            await self._determine_if_resources_available(
                resources_being_consumed
            )
        )
        await self.__update_fill_the_bucket_mode(
            resources_ran_out=(not resources_are_available)
        )

        if (
            not resources_are_available
            and self.__limit_reached_response
            == LimitReachedResponse.RAISE_EXCEPTION
        ):
            raise ResourceUnavailableError(
                "Resources not available. Limit Reached Response is RAISE_EXCEPTION"
            )

        if not resources_are_available and self.refresh_rate == 0:
            raise RuntimeError(
                "Resources not available. Would have waited indefinitely. refresh_rate is 0"
            )

        while not resources_are_available or self.__fill_the_bucket_mode:
            seconds_to_sleep = await self.__calculate_seconds_to_sleep(
                resources_being_consumed
            )
            await asyncio.sleep(seconds_to_sleep)
            resources_are_available = (
                await self._determine_if_resources_available(
                    resources_being_consumed
                )
            )
            await self.__update_fill_the_bucket_mode(
                resources_ran_out=(not resources_are_available)
            )

        async with self.__available_resource_lock:
            self._available_resources -= resources_being_consumed

        await self.__add_resource_use_entry(resources_being_consumed)

    async def _determine_if_resources_available(
        self, resources_being_consumed: int
    ) -> bool:
        await self._refresh_resource_count()
        async with self.__available_resource_lock:
            resources_are_available = (
                resources_being_consumed <= self._available_resources
            )
            return resources_are_available

    async def __update_fill_the_bucket_mode(
        self, resources_ran_out: bool
    ) -> None:
        await self._refresh_resource_count()
        if self._available_resources >= self.capacity:
            if resources_ran_out == True:
                raise ValueError(
                    "resources_ran_out should not be True when resources are available"
                )
            self.__fill_the_bucket_mode = False
        if resources_ran_out:
            self.__fill_the_bucket_mode = True

    async def _refresh_resource_count(self) -> None:
        async with self.__available_resource_lock:
            time_since_last_replenish: timedelta = (
                datetime.now() - self.__last_replenish_time
            )
            seconds_since_last_replenish: float = (
                time_since_last_replenish.total_seconds()
            )
            replenish_amount = seconds_since_last_replenish * self.refresh_rate
            new_total = self._available_resources + replenish_amount
            self._available_resources = min(new_total, self.capacity)
            self.__last_replenish_time = datetime.now()

    async def __calculate_seconds_to_sleep(
        self, resources_being_consumed: int
    ) -> float:
        if self.__fill_the_bucket_mode:
            seconds_till_bucket_is_full = (
                self.capacity - self._available_resources
            ) / self.refresh_rate
            return seconds_till_bucket_is_full
        elif resources_being_consumed > self._available_resources:
            seconds_till_resources_available = (
                resources_being_consumed - self._available_resources
            ) / self.refresh_rate
            return seconds_till_resources_available
        else:
            return 0

    async def __add_resource_use_entry(self, resource_amount: int) -> None:
        async with self.__resource_history_lock:
            new_entry = ResourceUseEntry(resource_amount, datetime.now())
            self.__resource_history.append(new_entry)
