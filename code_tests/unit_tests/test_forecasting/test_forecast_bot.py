from pathlib import Path

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
    MockBot,
)
from forecasting_tools.forecasting.forecast_bots.bot_lists import (
    get_all_official_bot_classes,
)
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
)


async def test_forecast_questions_returns_exceptions_when_specified() -> None:
    bot = MockBot()
    test_questions = [
        ForecastingTestManager.get_fake_binary_questions(),
        ForecastingTestManager.get_fake_binary_questions(),
    ]

    original_research = bot.run_research
    research_call_count = 0

    async def mock_research(*args, **kwargs):
        nonlocal research_call_count
        research_call_count += 1
        if research_call_count > 1:
            raise RuntimeError("Test error")
        return await original_research(*args, **kwargs)

    bot.run_research = mock_research

    results = await bot.forecast_questions(
        test_questions, return_exceptions=True
    )
    assert len(results) == 2
    assert isinstance(results[0], ForecastReport)
    assert isinstance(results[1], RuntimeError)
    assert "Test error" in str(results[1])

    with pytest.raises(RuntimeError, match="Test error"):
        await bot.forecast_questions(test_questions, return_exceptions=False)


async def test_forecast_question_returns_exception_when_specified() -> None:
    bot = MockBot()
    test_question = ForecastingTestManager.get_fake_binary_questions()

    async def mock_research(*args, **kwargs):
        raise RuntimeError("Test error")

    bot.run_research = mock_research

    result = await bot.forecast_question(test_question, return_exceptions=True)
    assert isinstance(result, RuntimeError)
    assert "Test error" in str(result)

    with pytest.raises(RuntimeError, match="Test error"):
        await bot.forecast_question(test_question, return_exceptions=False)


@pytest.mark.parametrize("failing_function", ["prediction", "research"])
async def test_forecast_report_contains_errors_from_failed_operations(
    failing_function: str,
) -> None:
    bot = MockBot(
        research_reports_per_question=2,
        predictions_per_research_report=2,
    )
    test_question = ForecastingTestManager.get_fake_binary_questions()

    error_message = "Test error"
    mock_call_count = 0

    async def mock_with_error(*args, **kwargs):
        nonlocal mock_call_count
        mock_call_count += 1
        should_error = mock_call_count % 2 == 0
        if should_error:
            raise RuntimeError(error_message)
        original_result = await original_function(*args, **kwargs)
        return original_result

    if failing_function == "prediction":
        original_function = bot._run_forecast_on_binary
        bot._run_forecast_on_binary = mock_with_error  # type: ignore
    else:
        original_function = bot.run_research
        bot.run_research = mock_with_error  # type: ignore

    result = await bot.forecast_question(test_question)
    assert isinstance(result, ForecastReport)
    expected_num_errors = 2 if failing_function == "prediction" else 1
    assert len(result.errors) == expected_num_errors
    assert error_message in str(result.errors[0])
    assert "RuntimeError" in str(result.errors[0])


async def test_forecast_fails_with_all_predictions_erroring() -> None:
    bot = MockBot(
        research_reports_per_question=2,
        predictions_per_research_report=3,
    )
    test_question = ForecastingTestManager.get_fake_binary_questions()

    async def mock_forecast(*args, **kwargs):
        raise RuntimeError("Test prediction error")

    bot._run_forecast_on_binary = mock_forecast

    with pytest.raises(RuntimeError):
        await bot.forecast_question(test_question)


async def test_research_reports_and_predictions_per_question_counts() -> None:
    research_reports = 3
    predictions_per_report = 2
    bot = MockBot(
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_report,
    )
    test_question = ForecastingTestManager.get_fake_binary_questions()

    research_call_count = 0
    prediction_call_count = 0

    async def count_research(*args, **kwargs):
        nonlocal research_call_count
        research_call_count += 1
        return "test research"

    async def count_predictions(*args, **kwargs):
        nonlocal prediction_call_count
        prediction_call_count += 1
        return ReasonedPrediction(
            prediction_value=0.5, reasoning="test reasoning"
        )

    bot.run_research = count_research
    bot._run_forecast_on_binary = count_predictions

    await bot.forecast_question(test_question)
    assert research_call_count == research_reports
    assert prediction_call_count == research_reports * predictions_per_report


async def test_use_research_summary_for_forecast() -> None:
    bot = MockBot(use_research_summary_to_forecast=True)
    test_question = ForecastingTestManager.get_fake_binary_questions()

    full_research = "Full research content"
    summary = "Summary content"
    received_research = None

    async def mock_research(*args, **kwargs):
        return full_research

    async def mock_summary(*args, **kwargs):
        return summary

    async def mock_forecast(question: BinaryQuestion, research: str):
        nonlocal received_research
        received_research = research
        return ReasonedPrediction(
            prediction_value=0.5, reasoning="test reasoning"
        )

    bot.run_research = mock_research
    bot.summarize_research = mock_summary
    bot._run_forecast_on_binary = mock_forecast

    await bot.forecast_question(test_question)
    assert received_research == summary


async def test_saves_reports_to_specified_folder(tmp_path: Path) -> None:
    folder_path = str(tmp_path)
    bot = MockBot(folder_to_save_reports_to=folder_path)
    test_questions = [
        ForecastingTestManager.get_fake_binary_questions(),
        ForecastingTestManager.get_fake_binary_questions(),
    ]

    await bot.forecast_questions(test_questions)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith("Forecasts-for-")
    assert "-2-questions.json" in files[0].name


async def test_skip_previously_forecasted_questions() -> None:
    bot = MockBot(skip_previously_forecasted_questions=True)
    forecasted_question = ForecastingTestManager.get_fake_binary_questions()
    unforecasted_question = ForecastingTestManager.get_fake_binary_questions()

    forecasted_question.already_forecasted = True
    unforecasted_question.already_forecasted = False

    research_call_count = 0

    async def count_research(*args, **kwargs):
        nonlocal research_call_count
        research_call_count += 1
        return "test research"

    bot.run_research = count_research

    await bot.forecast_questions([forecasted_question, unforecasted_question])
    assert research_call_count == 1

    with pytest.raises(AssertionError):
        await bot.forecast_question(forecasted_question)


@pytest.mark.parametrize("bot", get_all_official_bot_classes())
def test_bot_has_config(bot: type[ForecastBot]):
    probable_minimum_number_of_bot_params = 3
    bot_config = bot().get_config()
    assert bot_config is not None
    assert len(bot_config.keys()) > probable_minimum_number_of_bot_params
