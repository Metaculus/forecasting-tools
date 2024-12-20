import os
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.helpers.benchmarker import Benchmarker
from forecasting_tools.util import file_manipulation


async def test_file_is_made_for_benchmark(mocker: Mock) -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    bot_type = TemplateBot
    bot = bot_type()

    ForecastingTestManager.mock_forecast_bot_run_forecast(bot_type, mocker)

    file_path_to_save_reports = "logs/forecasts/benchmarks/"
    absolute_path = file_manipulation.get_absolute_path(
        file_path_to_save_reports
    )

    files_before = set(
        f
        for f in os.listdir(absolute_path)
        if os.path.isfile(os.path.join(absolute_path, f))
    )

    await Benchmarker.run_benchmark([bot], 10)

    files_after = set(
        f
        for f in os.listdir(absolute_path)
        if os.path.isfile(os.path.join(absolute_path, f))
    )

    new_files = files_after - files_before
    assert len(new_files) > 0, "No new benchmark report file was created"

    for new_file in new_files:
        os.remove(os.path.join(absolute_path, new_file))


async def test_each_benchmark_mode_calls_forecaster_more_time(
    mocker: Mock,
) -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    bot_type = TemplateBot
    bot = bot_type()

    mock_run_forecast = ForecastingTestManager.mock_forecast_bot_run_forecast(
        bot_type, mocker
    )

    modes: list[int] = [1, 5, 10]
    num_calls_for_modes = []
    for mode in modes:
        score = await Benchmarker.run_benchmark([bot], mode)
        assert isinstance(score, float), "The score should be a float"

        previous_calls = num_calls_for_modes[-1] if num_calls_for_modes else 0
        current_calls = mock_run_forecast.call_count - previous_calls
        num_calls_for_modes.append(current_calls)

        assert (
            current_calls > previous_calls
        ), "No new forecast calls were made"
