import asyncio

from agents import Tool, function_tool

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.forecast_helpers.metaculus_api import (
    MetaculusApi,
    MetaculusQuestion,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.util.misc import get_schema_of_base_model


@function_tool
async def get_general_news_with_asknews(topic: str) -> str:
    """
    Get general news context for a topic using AskNews.
    This will provide a list of news articles and their summaries
    """
    # TODO: Insert an if statement that will use Exa summaries rather than AskNews if AskNews keys are not enabled
    return await AskNewsSearcher().get_formatted_news_async(topic)


@function_tool
async def perplexity_search(query: str) -> str:
    """
    Use Perplexity (sonar-reasoning-pro) to search for information on a topic.
    This will provide a LLM answer with citations.
    """
    llm = GeneralLlm(
        model="perplexity/sonar-reasoning-pro",
        reasoning_effort="high",
        web_search_options={"search_context_size": "high"},
        populate_citations=True,
    )
    return await llm.invoke(query)


@function_tool
async def smart_searcher_search(query: str) -> str:
    """
    Use SmartSearcher to search for information on a topic.
    This will provide a LLM answer with citations.
    Citations will include url text fragments for faster fact checking.
    """
    return await SmartSearcher(model="o4-mini").invoke(query)


@function_tool
def grab_question_details_from_metaculus(url: str) -> MetaculusQuestion:
    """
    This function grabs the details of a question from a Metaculus URL.
    """
    question = MetaculusApi.get_question_by_url(url)
    question.api_json = {}
    return question


@function_tool
def grab_open_questions_from_tournament(
    tournament_id_or_slug: int | str,
) -> list[MetaculusQuestion]:
    """
    This function grabs the details of all questions from a Metaculus tournament.
    """
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        tournament_id_or_slug
    )
    for question in questions:
        question.api_json = {}
    return questions


def create_tool_for_forecasting_bot(
    bot_or_class: type[ForecastBot] | ForecastBot,
) -> Tool:
    if isinstance(bot_or_class, type):
        bot = bot_or_class()
    else:
        bot = bot_or_class

    description = clean_indents(
        f"""
        Forecast a SimpleQuestion (simplified binary, numeric, or multiple choice question) using a forecasting bot.
        Output: Forecast and research report

        A simple question has the following format:
        {get_schema_of_base_model(SimpleQuestion)}
        """
    )

    @function_tool(description_override=description)
    def forecast_question_tool(question: SimpleQuestion) -> str:
        metaculus_question = (
            SimpleQuestion.simple_questions_to_metaculus_questions([question])[
                0
            ]
        )
        task = bot.forecast_question(metaculus_question)
        report = asyncio.run(task)
        return report.explanation

    return forecast_question_tool
