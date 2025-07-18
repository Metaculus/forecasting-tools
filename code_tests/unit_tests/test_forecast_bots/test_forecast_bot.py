import json
from pathlib import Path
from typing import Any

import pytest

from code_tests.unit_tests.forecasting_test_manager import (
    ForecastingTestManager,
    MockBot,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.bot_lists import get_all_important_bot_classes
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot, ForecastReport


async def test_forecast_questions_returns_exceptions_when_specified() -> None:
    bot = MockBot()
    test_questions = [
        ForecastingTestManager.get_fake_binary_question(),
        ForecastingTestManager.get_fake_binary_question(),
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

    results = await bot.forecast_questions(test_questions, return_exceptions=True)
    assert len(results) == 2
    assert isinstance(results[0], ForecastReport)
    assert isinstance(results[1], Exception)
    assert "Test error" in str(results[1])

    with pytest.raises(Exception):
        await bot.forecast_questions(test_questions, return_exceptions=False)


async def test_forecast_question_returns_exception_when_specified() -> None:
    bot = MockBot()
    test_question = ForecastingTestManager.get_fake_binary_question()

    async def mock_research(*args, **kwargs):
        raise RuntimeError("Test error")

    bot.run_research = mock_research

    result = await bot.forecast_question(test_question, return_exceptions=True)
    assert isinstance(result, Exception)
    assert "Test error" in str(result)

    with pytest.raises(Exception):
        await bot.forecast_question(test_question, return_exceptions=False)


@pytest.mark.parametrize("failing_function", ["prediction", "research"])
async def test_forecast_report_contains_errors_from_failed_operations(
    failing_function: str,
) -> None:
    bot = MockBot(
        research_reports_per_question=2,
        predictions_per_research_report=2,
    )
    test_question = ForecastingTestManager.get_fake_binary_question()

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
    test_question = ForecastingTestManager.get_fake_binary_question()

    async def mock_forecast(*args, **kwargs):
        raise RuntimeError("Test prediction error")

    bot._run_forecast_on_binary = mock_forecast

    with pytest.raises(Exception):
        await bot.forecast_question(test_question)


async def test_research_reports_and_predictions_per_question_counts() -> None:
    research_reports = 3
    predictions_per_report = 2
    bot = MockBot(
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_report,
    )
    test_question = ForecastingTestManager.get_fake_binary_question()

    research_call_count = 0
    prediction_call_count = 0

    async def count_research(*args, **kwargs):
        nonlocal research_call_count
        research_call_count += 1
        return "test research"

    async def count_predictions(*args, **kwargs):
        nonlocal prediction_call_count
        prediction_call_count += 1
        return ReasonedPrediction(prediction_value=0.5, reasoning="test reasoning")

    bot.run_research = count_research
    bot._run_forecast_on_binary = count_predictions

    await bot.forecast_question(test_question)
    assert research_call_count == research_reports
    assert prediction_call_count == research_reports * predictions_per_report


async def test_use_research_summary_for_forecast() -> None:
    bot = MockBot(use_research_summary_to_forecast=True)
    test_question = ForecastingTestManager.get_fake_binary_question()

    full_research = "Full research content"
    summary = "Summary content"
    received_research = None

    async def mock_research(*args, **kwargs) -> str:
        return full_research

    async def mock_summary(*args, **kwargs) -> str:
        return summary

    async def mock_forecast(
        question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        nonlocal received_research
        received_research = research
        return ReasonedPrediction(prediction_value=0.5, reasoning="test reasoning")

    bot.run_research = mock_research
    bot.summarize_research = mock_summary
    bot._run_forecast_on_binary = mock_forecast

    await bot.forecast_question(test_question)
    assert received_research == summary, "Research should be the summary"


async def test_saves_reports_to_specified_folder(tmp_path: Path) -> None:
    folder_path = str(tmp_path)
    bot = MockBot(folder_to_save_reports_to=folder_path)
    test_questions = [
        ForecastingTestManager.get_fake_binary_question(),
        ForecastingTestManager.get_fake_binary_question(),
    ]

    await bot.forecast_questions(test_questions)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith("Forecasts-for-")
    assert "-2-questions.json" in files[0].name


async def test_skip_previously_forecasted_questions() -> None:
    bot = MockBot(skip_previously_forecasted_questions=True)
    forecasted_question = ForecastingTestManager.get_fake_binary_question()
    unforecasted_question = ForecastingTestManager.get_fake_binary_question()

    forecasted_question.already_forecasted = True
    unforecasted_question.already_forecasted = False

    research_call_count = 0

    async def count_research(*args, **kwargs) -> str:
        nonlocal research_call_count
        research_call_count += 1
        return "test research"

    bot.run_research = count_research

    await bot.forecast_questions([forecasted_question, unforecasted_question])
    assert research_call_count == 1

    with pytest.raises(AssertionError):
        await bot.forecast_question(forecasted_question)


@pytest.mark.parametrize("bot", get_all_important_bot_classes())
def test_bot_has_config(bot: type[ForecastBot]) -> None:
    probable_minimum_number_of_bot_params = 3
    bot_config = bot().get_config()
    assert bot_config is not None
    assert len(bot_config.keys()) > probable_minimum_number_of_bot_params

    # Verify config can be JSON serialized

    try:
        json.dumps(bot_config)
    except Exception as e:
        pytest.fail(f"Bot config is not JSON serializable: {e}")

    assert (
        bot_config["research_reports_per_question"] > 0
    ), "Did not have parameter for expected config"


async def test_llm_returns_general_llm_when_llm_is_str() -> None:
    bot = MockBot(llms={"default": "gpt-4o"})
    assert isinstance(bot.get_llm("default"), str)


async def test_get_llm_returns_correct_type_when_set() -> None:
    bot = MockBot(
        llms={
            "default": "gpt-4o",
            "summarizer": GeneralLlm(model="gpt-4o-mini", temperature=0.3),
        }
    )

    str_llm = bot.get_llm("default")
    general_llm = bot.get_llm("summarizer")
    assert isinstance(str_llm, str)
    assert isinstance(general_llm, GeneralLlm)

    general_llm_2 = bot.get_llm("default", guarantee_type="llm")
    str_llm_2 = bot.get_llm("summarizer", guarantee_type="string_name")
    assert isinstance(general_llm_2, GeneralLlm)
    assert isinstance(str_llm_2, str)

    str_llm_3 = bot.get_llm("default", guarantee_type=None)
    general_llm_3 = bot.get_llm("summarizer", guarantee_type=None)
    assert isinstance(str_llm_3, str)
    assert isinstance(general_llm_3, GeneralLlm)


async def test_get_llm_returns_none_when_not_set() -> None:
    bot = MockBot()
    with pytest.raises(ValueError):
        bot.get_llm("non_existent_llm")


async def test_get_config_includes_default_llms_when_not_set() -> None:
    bot = MockBot(
        llms={
            "default": "gpt-4o",
            "summarizer": GeneralLlm(model="gpt-4o-mini", temperature=0.3),
        }
    )
    config = bot.get_config()

    llm_dict: dict[str, Any] = config["llms"]

    assert "llms" in config
    assert "default" in llm_dict
    assert "summarizer" in llm_dict

    assert llm_dict["default"] == "gpt-4o"
    assert "temperature" in llm_dict["summarizer"]
    assert "model" in llm_dict["summarizer"]


async def test_get_llm_edge_case_behavior() -> None:
    bot = MockBot()
    non_existent_purpose = "non_existent_purpose"

    # Should return None when guarantee_type is None
    with pytest.raises(ValueError):
        bot.get_llm(non_existent_purpose, guarantee_type=None)

    # Should raise ValueError when guarantee_type is specified
    with pytest.raises(Exception):
        bot.get_llm(non_existent_purpose, guarantee_type="string_name")

    with pytest.raises(Exception):
        bot.get_llm(non_existent_purpose, guarantee_type="llm")


async def test_default_used_for_missing_llm_key() -> None:
    bot = MockBot(llms={})
    assert bot.get_llm("default") is not None


async def test_notepad_counts_research_and_prediction_attempts() -> None:
    research_reports = 2
    predictions_per_report = 3
    bot = MockBot(
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_report,
    )
    test_question = ForecastingTestManager.get_fake_binary_question()

    async def mock_remove_notepad(*args, **kwargs) -> None:
        # Do nothing
        pass

    bot._remove_notepad = mock_remove_notepad

    await bot.forecast_question(test_question)
    notepad = await bot._get_notepad(test_question)

    assert notepad.total_research_reports_attempted == research_reports
    assert (
        notepad.total_predictions_attempted == predictions_per_report * research_reports
    )


async def test_summarize_research_returns_disabled_message_when_false() -> None:
    bot = MockBot(enable_summarize_research=False)
    question = ForecastingTestManager.get_fake_binary_question()
    result = await bot.summarize_research(question, "some research")
    assert "disabled" in result.lower()


async def test_summarize_research_calls_llm_and_returns_output() -> None:
    bot = MockBot(enable_summarize_research=True)
    question = ForecastingTestManager.get_fake_binary_question()
    result = await bot.summarize_research(question, "some research")
    assert "mock" in result.lower()


def test_conflicting_summarize_research_and_use_summary_raises() -> None:
    with pytest.raises(Exception):
        MockBot(
            enable_summarize_research=False,
            use_research_summary_to_forecast=True,
        )


async def test_research_fits_with_research_markdown_section() -> None:
    bot = MockBot()
    test_question = ForecastingTestManager.get_fake_binary_question()
    expected_research = "### Finding 1\nContent 1\n### Finding 2\nContent 2"

    async def mock_research(*args, **kwargs) -> str:  # NOSONAR
        return "# Finding 1\nContent 1\n# Finding 2\nContent 2"

    bot.run_research = mock_research
    report = await bot.forecast_question(test_question)
    assert (
        expected_research in report.research.strip()
    ), "Assuming research section has heading of 2, there should be a heading of 3 for the inner part"

    async def mock_research_2(*args, **kwargs) -> str:  # NOSONAR
        return "#### Finding 1\nContent 1\n#### Finding 2\nContent 2"

    bot.run_research = mock_research_2
    report = await bot.forecast_question(test_question)
    assert (
        expected_research in report.research.strip()
    ), "Assuming research section has heading of 2, there should be a heading of 3 for the inner part"

    async def mock_research_3(*args, **kwargs) -> str:  # NOSONAR
        return "### Finding 1\nContent 1\n### Finding 2\nContent 2"

    bot.run_research = mock_research_3
    report = await bot.forecast_question(test_question)
    assert (
        expected_research in report.research.strip()
    ), "Assuming research section has heading of 2, there should be a heading of 3 for the inner part"
