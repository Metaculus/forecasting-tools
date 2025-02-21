import os
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.forecast_bots.template_bot import TemplateBot
from forecasting_tools.forecast_helpers.benchmarker import Benchmarker
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

    await Benchmarker(
        forecast_bots=[bot],
        number_of_questions_to_use=10,
        file_path_to_save_reports=file_path_to_save_reports,
    ).run_benchmark()

    files_after = set(
        f
        for f in os.listdir(absolute_path)
        if os.path.isfile(os.path.join(absolute_path, f))
    )

    new_files = files_after - files_before
    assert len(new_files) > 0, "No new benchmark report file was created"

    for new_file in new_files:
        os.remove(os.path.join(absolute_path, new_file))


@pytest.mark.parametrize("num_questions", [10])
async def test_benchmarks_run_properly_with_mocked_bot(
    mocker: Mock,
    num_questions: int,
) -> None:
    bot_type = TemplateBot
    bot = TemplateBot()
    mock_run_forecast = ForecastingTestManager.mock_forecast_bot_run_forecast(
        bot_type, mocker
    )

    benchmarks = await Benchmarker(
        forecast_bots=[bot],
        number_of_questions_to_use=num_questions,
    ).run_benchmark()
    assert isinstance(benchmarks, list)
    assert all(
        isinstance(benchmark, BenchmarkForBot) for benchmark in benchmarks
    )
    assert mock_run_forecast.call_count == num_questions

    for benchmark in benchmarks:
        assert_all_benchmark_object_fields_are_not_none(
            benchmark, num_questions
        )


def assert_all_benchmark_object_fields_are_not_none(
    benchmark: BenchmarkForBot, num_questions: int
) -> None:
    expected_time_taken = 0.5
    assert benchmark.name is not None, "Name is not set"
    assert benchmark.description is not None, "Description is not set"
    assert (
        benchmark.timestamp < datetime.now()
        and benchmark.timestamp
        > datetime.now() - timedelta(minutes=expected_time_taken)
    ), ("Timestamp is not set properly")
    assert (
        benchmark.time_taken_in_minutes is not None
        and benchmark.time_taken_in_minutes > 0
        and benchmark.time_taken_in_minutes
        < expected_time_taken  # The mocked benchmark should be quick
    ), "Time taken in minutes is not set"
    assert (
        benchmark.total_cost is not None and benchmark.total_cost >= 0
    ), "Total cost is not set"
    assert benchmark.git_commit_hash is not None, "Git commit hash is not set"
    assert "False" in str(
        benchmark.forecast_bot_config
    ), "Forecast bot config is not set"
    assert (
        benchmark.code is not None and "class" in benchmark.code
    ), "Code is not set"
    assert (
        len(benchmark.forecast_reports) == num_questions
    ), "Forecast reports is not set"
    assert (
        benchmark.average_expected_baseline_score > 0
    ), "Average inverse expected log score is not set"
