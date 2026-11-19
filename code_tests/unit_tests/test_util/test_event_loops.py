import asyncio
import logging
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from forecasting_tools.util import async_batching
from forecasting_tools.util.event_loops import run_coroutine_synchronously


async def double(number: int) -> int:
    await asyncio.sleep(0)
    return number * 2


async def raise_value_error() -> None:
    await asyncio.sleep(0)
    raise ValueError("expected failure")


def test_runs_coroutine_when_no_event_loop_is_running() -> None:
    assert run_coroutine_synchronously(double(2)) == 4


async def test_runs_coroutine_and_warns_when_event_loop_is_already_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert run_coroutine_synchronously(double(2)) == 4
    assert "already running" in caplog.text


def test_exception_propagates_when_no_event_loop_is_running() -> None:
    with pytest.raises(ValueError, match="expected failure"):
        run_coroutine_synchronously(raise_value_error())


async def test_exception_propagates_when_event_loop_is_already_running() -> None:
    with pytest.raises(ValueError, match="expected failure"):
        run_coroutine_synchronously(raise_value_error())


async def test_cost_incurred_in_worker_thread_counts_toward_callers_cost_manager() -> (
    None
):
    async def incur_cost() -> None:
        MonetaryCostManager.increase_current_usage_in_parent_managers(0.25)

    with MonetaryCostManager(1.0) as cost_manager:
        run_coroutine_synchronously(incur_cost())
    assert cost_manager.current_usage == pytest.approx(0.25)


def test_run_coroutines_works_after_an_earlier_asyncio_run() -> None:
    asyncio.run(double(1))
    assert async_batching.run_coroutines([double(1), double(2)]) == [2, 4]


async def test_sync_tournament_fetch_works_inside_running_event_loop(
    mocker: Mock,
) -> None:
    fake_questions = [ForecastingTestManager.get_fake_binary_question()]
    mocker.patch.object(
        MetaculusClient, "get_questions_matching_filter", return_value=fake_questions
    )
    questions = MetaculusClient().get_all_open_questions_from_tournament(1)
    assert questions == fake_questions
