import logging
from unittest.mock import Mock

import pendulum
import pytest
import typeguard

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools import ApiFilter, MetaculusClient
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.conditional_report import ConditionalReport
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import DateQuestion, MetaculusQuestion
from forecasting_tools.forecast_bots.bot_lists import (
    get_all_bot_question_type_pairs_for_cheap_tests,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.uniform_probability_bot import (
    UniformProbabilityBot,
)
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
    ten_minutes_ago = pendulum.now().subtract(minutes=10)
    assert (
        updated_question.timestamp_of_my_last_forecast is not None
    ), "Timestamp of my last forecast is None"
    assert (
        updated_question.timestamp_of_my_last_forecast > ten_minutes_ago
    ), f"Timestamp of my last forecast is not recent enough: {updated_question.timestamp_of_my_last_forecast}"


@pytest.mark.parametrize(
    "bot",
    # get_all_bots_for_doing_cheap_tests(),
    [UniformProbabilityBot()],
)
async def test_predicts_ai_2027_tournament(bot: ForecastBot) -> None:
    # This tournament has all questions end in 2 years,
    # and has at least one of every question type (binary, numeric, multiple choice, discrete, date)
    original_publish_status = bot.publish_reports_to_metaculus
    try:
        bot.publish_reports_to_metaculus = True
        reports = await bot.forecast_on_tournament("ai-2027")
        bot.log_report_summary(reports)

        assert len(reports) == 15, "Expected 19 reports"

    except Exception as e:
        pytest.fail(f"Forecasting on ai-2027 tournament failed: {e}")
    finally:
        bot.publish_reports_to_metaculus = original_publish_status


async def test_taiwan_tournament_uniform_probability_bot() -> None:
    bot = UniformProbabilityBot(
        publish_reports_to_metaculus=True, skip_previously_forecasted_questions=False
    )
    reports = await bot.forecast_on_tournament("taiwan")
    bot.log_report_summary(reports)
    assert len(reports) > 10, "Expected some reports"
    assert all(
        not isinstance(report, Exception) for report in reports
    ), "Expected no exceptions"
    assert any(
        isinstance(report, ConditionalReport) for report in reports
    ), "Expected some conditional reports"


async def test_conditional_forecasts() -> None:
    bot = TemplateBot(
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=False,
        llms={
            "default": GeneralLlm(model="openai/o4-mini", temperature=1),
            "summarizer": GeneralLlm(model="openai/o4-mini", temperature=1),
            "researcher": GeneralLlm(model="openai/o4-mini", temperature=1),
            "parser": GeneralLlm(model="openai/o4-mini", temperature=1),
        },
        # TODO: Make sure that template bot uses "Dev" MetaculusClient inside the bot itself
    )
    questions = await MetaculusClient.dev().get_questions_matching_filter(
        ApiFilter(allowed_types=["conditional"], allowed_subquestion_types=["binary"]),
        num_questions=1,
    )
    url = "https://dev.metaculus.com/questions/40107/conditional-someone-born-before-2001-lives-to-150/"
    url_questions = MetaculusClient.dev().get_question_by_url(
        url, group_question_mode="unpack_subquestions"
    )
    url_questions = typeguard.check_type(url_questions, MetaculusQuestion)
    questions.append(url_questions)
    reports = await bot.forecast_questions(questions)
    assert len(reports) == len(questions)


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
    date_questions = [
        question
        for question in questions_that_should_be_being_forecast_on
        if isinstance(question, DateQuestion)
    ]
    assert len(reports) == len(questions_that_should_be_being_forecast_on) - len(
        date_questions
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
        "no_research",
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
        if not research_llm or research_llm == "no_research":
            research = ""
        else:
            assert len(research) > 0, "Expected research to return a non-empty string"
            assert (
                "https:" in research or "www." in research or "[1]"
            ), "Expected research to contain a URL"
