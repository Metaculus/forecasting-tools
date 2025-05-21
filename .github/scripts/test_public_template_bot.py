import asyncio
import importlib.metadata
import json
import urllib.request

from forecasting_tools import MetaculusApi, TemplateBot


def test_example_questions_forecasted() -> None:
    template_bot = TemplateBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        # llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
        #     "default": GeneralLlm(
        #         model="metaculus/anthropic/claude-3-5-sonnet-20241022",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        # },
    )

    questions = [
        MetaculusApi.get_question_by_url(question_url)
        for question_url in MetaculusApi.TEST_QUESTION_URLS
    ]
    forecast_reports = asyncio.run(
        template_bot.forecast_questions(questions, return_exceptions=False)
    )
    urls_forecasted = [
        forecast_report.question.page_url
        for forecast_report in forecast_reports
    ]
    assert len(urls_forecasted) == len(MetaculusApi.TEST_QUESTION_URLS)
    for url in urls_forecasted:
        assert url, "URL is empty"
        assert any(
            url in test_url for test_url in MetaculusApi.TEST_QUESTION_URLS
        ), f"URL {url} is not in the list of test URLs"

    for forecast_report in forecast_reports:
        assert forecast_report.prediction


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
