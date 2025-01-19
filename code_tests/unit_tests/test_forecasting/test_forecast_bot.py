import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
    MockBot,
)
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastReport,
)


async def test_forecast_questions_returns_exceptions_when_specified() -> None:
    # Setup
    bot = MockBot()
    test_questions = [
        ForecastingTestManager.get_fake_binary_questions(),
        ForecastingTestManager.get_fake_binary_questions(),
    ]

    # Mock research to raise an error for the second question
    original_research = bot.run_research
    research_call_count = 0

    async def mock_research(*args, **kwargs):
        nonlocal research_call_count
        research_call_count += 1
        if research_call_count > 1:
            raise RuntimeError("Test error")
        return await original_research(*args, **kwargs)

    bot.run_research = mock_research

    # Test with return_exceptions=True
    results = await bot.forecast_questions(
        test_questions, return_exceptions=True
    )
    assert len(results) == 2
    assert isinstance(results[0], ForecastReport)
    assert isinstance(results[1], RuntimeError)
    assert "Test error" in str(results[1])

    # Test with return_exceptions=False
    with pytest.raises(RuntimeError, match="Test error"):
        await bot.forecast_questions(test_questions, return_exceptions=False)


async def test_forecast_question_returns_exception_when_specified() -> None:
    # Setup
    bot = MockBot()
    test_question = ForecastingTestManager.get_fake_binary_questions()

    # Mock research to raise an error
    async def mock_research(*args, **kwargs):
        raise RuntimeError("Test error")

    bot.run_research = mock_research

    # Test with return_exceptions=True
    result = await bot.forecast_question(test_question, return_exceptions=True)
    assert isinstance(result, RuntimeError)
    assert "Test error" in str(result)

    # Test with return_exceptions=False
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
