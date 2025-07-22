import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.forecasting_test_manager import (
    ForecastingTestManager,
    MockBot,
)
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.cp_benchmarking.benchmarker import Benchmarker
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_bots.template_bot import TemplateBot

logger = logging.getLogger(__name__)


async def test_file_is_made_for_benchmark(mocker: Mock, tmp_path: Path) -> None:
    if ForecastingTestManager.metaculus_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    class MockBot1(MockBot):
        pass

    class MockBot2(MockBot):
        pass

    bots_to_use = [MockBot1(), MockBot2()]

    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()

    chosen_questions = [
        ForecastingTestManager.get_fake_binary_question() for _ in range(10)
    ]
    await Benchmarker(
        forecast_bots=bots_to_use,
        questions_to_use=chosen_questions,
        file_path_to_save_reports=str(benchmark_dir),
        concurrent_question_batch_size=2,
    ).run_benchmark()

    created_files = list(benchmark_dir.iterdir())
    assert len(created_files) == 1, "Only one benchmark report file should be created"
    benchmark_file = created_files[0]
    assert benchmark_file.name.startswith("benchmarks_")
    assert benchmark_file.name.endswith(".jsonl")

    benchmarks_created = BenchmarkForBot.load_json_from_file_path(str(benchmark_file))
    assert len(benchmarks_created) == len(
        bots_to_use
    ), "Number of benchmarks created should be equal to number of bots"
    bots_benchmarked = set()
    for benchmark in benchmarks_created:
        bot_name = benchmark.forecast_bot_class_name
        assert (
            bot_name not in bots_benchmarked
        ), "There should only be one benchmark object per bot"
        bots_benchmarked.add(bot_name)

    for created_file in created_files:
        created_file.unlink()


@pytest.mark.skip(
    reason="Reducing the number of calls to metaculus api due to rate limiting"
)
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
    assert all(isinstance(benchmark, BenchmarkForBot) for benchmark in benchmarks)
    assert mock_run_forecast.call_count == num_questions

    for benchmark in benchmarks:
        assert_all_benchmark_object_fields_are_not_none(benchmark, num_questions)


async def test_correct_number_of_final_forecasts_for_multiple_bots() -> None:
    class Bot1(MockBot):
        pass

    class Bot2(MockBot):
        pass

    class Bot3(MockBot):
        pass

    bot1 = Bot1()
    bot2 = Bot2()
    bot3 = Bot3()

    chosen_questions = [
        ForecastingTestManager.get_fake_binary_question() for _ in range(30)
    ]
    benchmarks = await Benchmarker(
        forecast_bots=[bot1, bot2, bot3],
        questions_to_use=chosen_questions,
        concurrent_question_batch_size=4,
    ).run_benchmark()

    assert len(benchmarks) == 3
    for i, benchmark in enumerate(benchmarks):
        assert benchmarks[i].forecast_bot_class_name == f"Bot{i + 1}"
        assert len(benchmark.forecast_reports) == 30
        assert benchmark.num_input_questions == 30
        assert benchmark.num_failed_forecasts == 0
        assert_all_benchmark_object_fields_are_not_none(benchmark, 30)

    bot1_questions = [r.question.question_text for r in benchmarks[0].forecast_reports]
    bot2_questions = [r.question.question_text for r in benchmarks[1].forecast_reports]
    bot3_questions = [r.question.question_text for r in benchmarks[2].forecast_reports]
    assert (
        len(set(bot1_questions))
        == len(set(bot2_questions))
        == len(set(bot3_questions))
        == len(set(bot1_questions + bot2_questions + bot3_questions))
    )


def assert_all_benchmark_object_fields_are_not_none(
    benchmark: BenchmarkForBot, num_questions: int
) -> None:
    expected_time_taken = 0.5
    assert benchmark.name is not None, "Name is not set"
    assert len(benchmark.name) <= 75, "Name is too long"
    assert (
        "n/a" not in benchmark.name and "None" not in benchmark.name
    ), "All fields for name should be set (none should fail)"
    assert benchmark.description is not None, "Description is not set"
    assert (
        "n/a" not in benchmark.description and "None" not in benchmark.description
    ), "All fields for description should be set (none should fail)"
    assert (
        benchmark.explicit_name is None
    ), "Explicit name is set without someone explicitly setting it"
    assert (
        benchmark.explicit_description is None
    ), "Explicit description is set without someone explicitly setting it"
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
    assert benchmark.code is not None and "class" in benchmark.code, "Code is not set"
    assert (
        len(benchmark.forecast_reports) == num_questions
    ), "Forecast reports is not set"
    assert (
        benchmark.average_expected_baseline_score <= 100
    ), "Average expected baseline score is not set"
    assert (
        benchmark.num_input_questions == num_questions
    ), "Number of input questions is not set"
    assert (
        benchmark.forecast_bot_class_name is not None
    ), "Forecast bot class name is not set"
    assert (
        len(benchmark.forecast_bot_class_name) > 0
    ), "Forecast bot class name is not set correctly"
    assert (
        len(benchmark.forecast_reports) == num_questions
    ), "Forecast reports is not of the correct length"


async def test_benchmarks_run_properly_with_provided_questions(
    mocker: Mock,
) -> None:
    bot_type = TemplateBot
    bot = TemplateBot()
    mock_run_forecast = ForecastingTestManager.mock_forecast_bot_run_forecast(
        bot_type, mocker
    )

    test_questions = [
        ForecastingTestManager.get_fake_binary_question() for _ in range(4)
    ]

    benchmarks = await Benchmarker(
        forecast_bots=[bot],
        questions_to_use=test_questions,
    ).run_benchmark()

    assert isinstance(benchmarks, list)
    assert all(isinstance(benchmark, BenchmarkForBot) for benchmark in benchmarks)
    assert mock_run_forecast.call_count == len(test_questions)

    for benchmark in benchmarks:
        assert_all_benchmark_object_fields_are_not_none(benchmark, len(test_questions))


async def test_code_is_saved_for_benchmark(mocker: Mock) -> None:
    bot_type = Q2TemplateBot2025
    bot = bot_type()
    ForecastingTestManager.mock_forecast_bot_run_forecast(bot_type, mocker)
    questions = [ForecastingTestManager.get_fake_binary_question() for _ in range(10)]
    benchmarks = await Benchmarker(
        forecast_bots=[bot],
        additional_code_to_snapshot=[ForecastingTestManager],
        questions_to_use=questions,
    ).run_benchmark()
    assert len(benchmarks) == 1
    benchmark = benchmarks[0]

    logger.info(benchmark.code)
    assert benchmark.code is not None
    assert f"{Q2TemplateBot2025.__name__}" in benchmark.code
    assert f"{Q2TemplateBot2025._run_forecast_on_binary.__name__}" in benchmark.code
    assert f"{ForecastingTestManager.__name__}" in benchmark.code
    assert (
        f"{ForecastingTestManager.mock_forecast_bot_run_forecast.__name__}"
        in benchmark.code
    )


def test_benchmarker_initialization_errors() -> None:
    bot = TemplateBot()

    with pytest.raises(
        ValueError,
    ):
        Benchmarker(forecast_bots=[bot])

    with pytest.raises(
        ValueError,
    ):
        Benchmarker(
            forecast_bots=[bot],
            number_of_questions_to_use=10,
            questions_to_use=[ForecastingTestManager.get_fake_binary_question()],
        )

    # Make sure these do not error
    Benchmarker(
        forecast_bots=[bot],
        number_of_questions_to_use=10,
    )
    Benchmarker(
        forecast_bots=[bot],
        questions_to_use=[ForecastingTestManager.get_fake_binary_question()],
    )


def test_benchmark_for_bot_naming_and_description() -> None:
    benchmark = BenchmarkForBot(
        explicit_name="explicit name",
        explicit_description="explicit description",
        forecast_reports=[],
        forecast_bot_config={},
        time_taken_in_minutes=0,
        total_cost=0,
        git_commit_hash="",
    )
    assert benchmark.name == "explicit name"
    assert benchmark.description == "explicit description"


class FailingBot(MockBot):
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ):  # NOSONAR
        if question.question_text == "fail":
            raise RuntimeError("Simulated failure")
        return await super()._run_forecast_on_binary(question, research)


async def test_failed_and_missing_forecasts_are_tracked(mocker: Mock) -> None:
    bot = FailingBot()

    questions = [
        ForecastingTestManager.get_fake_binary_question() for _ in range(2)
    ] + [ForecastingTestManager.get_fake_binary_question() for _ in range(2)]
    questions[1].question_text = "fail"
    questions[3].question_text = "fail"

    benchmarks = await Benchmarker(
        forecast_bots=[bot],
        questions_to_use=questions,
        concurrent_question_batch_size=2,
    ).run_benchmark()
    benchmark = benchmarks[0]

    assert benchmark.num_input_questions == 4
    assert len(benchmark.forecast_reports) == 2
    assert benchmark.num_failed_forecasts == 2
    assert all("Simulated failure" in err for err in benchmark.failed_report_errors)
