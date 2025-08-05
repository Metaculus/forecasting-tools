import asyncio
import importlib.metadata
import json
import os
import urllib.request

import pytest

from forecasting_tools import (
    DataOrganizer,
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    TemplateBot,
)
from forecasting_tools.forecast_bots.bot_lists import get_all_bots_for_doing_cheap_tests


def test_example_questions_forecasted_saved_and_loaded() -> None:
    folder_to_save_reports_to = "logs/reports/"
    template_bot = TemplateBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=folder_to_save_reports_to,
        skip_previously_forecasted_questions=False,
    )

    questions = [
        MetaculusApi.get_question_by_url(question_url)
        for question_url in MetaculusApi.TEST_QUESTION_URLS
    ]
    forecast_reports = asyncio.run(
        template_bot.forecast_questions(questions, return_exceptions=False)
    )
    urls_forecasted = [
        forecast_report.question.page_url for forecast_report in forecast_reports
    ]
    assert len(urls_forecasted) == len(MetaculusApi.TEST_QUESTION_URLS)
    for url in urls_forecasted:
        assert url, "URL is empty"
        assert any(
            url in test_url for test_url in MetaculusApi.TEST_QUESTION_URLS
        ), f"URL {url} is not in the list of test URLs"

    for forecast_report in forecast_reports:
        assert forecast_report.prediction

    assert os.path.exists(folder_to_save_reports_to)
    files = os.listdir(folder_to_save_reports_to)
    assert len(files) == 1, f"Expected 1 file, got {files}"
    file = files[0]
    file_name = os.path.basename(file)

    loaded_forecast_reports = DataOrganizer.load_reports_from_file_path(
        os.path.join(folder_to_save_reports_to, file_name)
    )
    assert (
        len(loaded_forecast_reports) > 0
    ), f"Expected at least 1 forecast report, got {loaded_forecast_reports}"

    for loaded_forecast_report in loaded_forecast_reports:
        assert loaded_forecast_report.question.id_of_question in [
            forecast_report.question.id_of_question
            for forecast_report in forecast_reports
        ], f"Loaded forecast report {loaded_forecast_report.question.id_of_question} is not in the list of forecasted questions ({[forecast_report.question.id_of_question for forecast_report in forecast_reports]})"


def test_forecasting_tools_is_latest_version() -> None:

    installed_version = importlib.metadata.version("forecasting_tools")
    with urllib.request.urlopen(
        "https://pypi.org/pypi/forecasting-tools/json"
    ) as response:
        data = json.load(response)
        latest_version = data["info"]["version"]
    assert (
        installed_version == latest_version
    ), f"Installed: {installed_version}, Latest: {latest_version}"


def test_saving_and_loading_question() -> None:
    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
    )
    question.save_object_list_to_file_path([question], "test_question.json")
    loaded_questions = MetaculusQuestion.load_json_from_file_path("test_question.json")
    assert len(loaded_questions) == 1
    loaded_question = loaded_questions[0]
    assert question.question_text == loaded_question.question_text
    assert question.id_of_question == loaded_question.id_of_question
    assert question.page_url == loaded_question.page_url
    os.remove("test_question.json")


@pytest.mark.parametrize(
    "bot",
    get_all_bots_for_doing_cheap_tests(),
)
async def test_predicts_ai_2027_tournament(bot: ForecastBot) -> None:
    # This tournament has all questions end in 2 years,
    # and has at least one of every question type (binary, numeric, multiple choice, discrete, date)
    original_publish_status = bot.publish_reports_to_metaculus
    try:
        bot.publish_reports_to_metaculus = True
        reports = await bot.forecast_on_tournament("ai-2027", return_exceptions=True)
        bot.log_report_summary(reports, raise_errors=True)
        assert len(reports) == 15, "Expected 15 reports"
    except Exception as e:
        pytest.fail(f"Forecasting on ai-2027 tournament failed: {e}")
    finally:
        bot.publish_reports_to_metaculus = original_publish_status
