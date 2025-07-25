import logging
from unittest.mock import Mock

import pytest
import typeguard

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.bot_lists import (
    get_all_bot_question_type_pairs_for_cheap_tests,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.template_bot import TemplateBot
from forecasting_tools.helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "question_type, bot", get_all_bot_question_type_pairs_for_cheap_tests()
)
async def test_predicts_test_question(
    question_type: type[MetaculusQuestion],
    bot: ForecastBot,
) -> None:
    question = DataOrganizer.get_live_example_question_of_type(question_type)
    assert isinstance(question, question_type)
    target_cost_in_usd = 0.3
    with MonetaryCostManager() as cost_manager:
        report = await bot.forecast_question(question)
        logger.info(f"Cost of forecast: {cost_manager.current_usage}")
        logger.info(f"Report Explanation: \n{report.explanation}")
        expected_report_type = DataOrganizer.get_report_type_for_question_type(
            question_type
        )
    await report.publish_report_to_metaculus()
    assert isinstance(report, expected_report_type)
    assert cost_manager.current_usage <= target_cost_in_usd
    assert len(report.report_sections) > 1
    assert report.prediction is not None
    assert report.explanation is not None
    assert report.price_estimate is not None
    assert report.minutes_taken is not None
    assert report.question is not None
    assert question.id_of_post is not None

    updated_question = MetaculusApi.get_question_by_post_id(question.id_of_post)
    assert updated_question.already_forecasted


async def test_collects_reports_on_open_questions(mocker: Mock) -> None:
    if ForecastingTestManager.metaculus_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    bot_type = TemplateBot
    bot = bot_type()
    ForecastingTestManager.mock_forecast_bot_run_forecast(bot_type, mocker)
    tournament_id = ForecastingTestManager.TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN
    reports = await bot.forecast_on_tournament(tournament_id)
    questions_that_should_be_being_forecast_on = (
        MetaculusApi.get_all_open_questions_from_tournament(tournament_id)
    )
    assert len(reports) == len(
        questions_that_should_be_being_forecast_on
    ), "Not all questions were forecasted on"


async def test_no_reports_when_questions_already_forecasted(
    mocker: Mock,
) -> None:
    bot_type = TemplateBot
    bot = bot_type(skip_previously_forecasted_questions=True)
    ForecastingTestManager.mock_forecast_bot_run_forecast(bot_type, mocker)
    questions = [ForecastingTestManager.get_fake_binary_question()]
    questions = typeguard.check_type(questions, list[MetaculusQuestion])

    for question in questions:
        question.already_forecasted = True

    reports = await bot.forecast_questions(questions)
    assert (
        len(reports) == 0
    ), "Expected no reports since all questions were already forecasted on"

    for question in questions:
        question.already_forecasted = False

    reports = await bot.forecast_questions(questions)
    assert len(reports) == len(questions), "Expected all questions to be forecasted on"


async def test_works_with_configured_llm() -> None:
    bot_type = TemplateBot
    researcher_model = "openrouter/perplexity/sonar-pro"
    bot = bot_type(
        llms={
            "default": GeneralLlm(model="gpt-4o-mini", timeout=42),
            "summarizer": "gpt-4o-mini",
            "researcher": GeneralLlm(model=researcher_model),
        }
    )

    default_llm = bot.get_llm("default")
    assert isinstance(default_llm, GeneralLlm)
    assert default_llm.litellm_kwargs["timeout"] == 42
    assert bot.get_llm("summarizer") == "gpt-4o-mini"
    assert bot.get_llm("researcher", "string_name") == researcher_model

    question = ForecastingTestManager.get_fake_binary_question()
    report = await bot.forecast_question(question)
    assert report is not None


@pytest.mark.parametrize(
    "research_llm",
    [
        "asknews/news-summaries",
        "openrouter/perplexity/sonar",
        GeneralLlm("perplexity/sonar"),
        "smart-searcher/gpt-4o-mini",
        "",
        "non-existent-llm",
    ],
)
async def test_research(research_llm: GeneralLlm | str) -> None:
    bot = TemplateBot(llms={"researcher": research_llm})
    question = ForecastingTestManager.get_fake_binary_question()

    if research_llm == "non-existent-llm":
        with pytest.raises(Exception):
            await bot.run_research(question)
    else:
        research = await bot.run_research(question)
        if not research_llm:
            research = ""
        else:
            assert len(research) > 0, "Expected research to return a non-empty string"
            assert (
                "https:" in research or "www." in research or "[1]"
            ), "Expected research to contain a URL"
