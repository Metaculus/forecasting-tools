import asyncio
import logging
import random

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.ai_models.agent_wrappers import AgentTool, agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.helpers.metaculus_api import MetaculusApi, MetaculusQuestion
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents, get_schema_of_base_model

logger = logging.getLogger(__name__)


@agent_tool
async def query_asknews(topic: str) -> str:
    """
    Get an overview of news context for a topic using AskNews. Can search international news from other languages.
    This will provide a list of ~16 news articles and their summaries with fields:
    - Title
    - Summary
    - URL
    - Date
    """
    logger.info(f"TOOL: Querying AskNews for topic: {topic}")
    return await AskNewsSearcher().get_formatted_news_async(topic)


@agent_tool
async def perplexity_reasoning_pro_search(query: str) -> str:
    """
    Use Perplexity (sonar-reasoning-pro) to search for information on a topic.
    This will provide a LLM answer with citations.
    This is Perplexity's highest quality search model.
    """
    logger.info(f"TOOL: Querying Perplexity (sonar-reasoning-pro) for query: {query}")
    return await GeneralLlm(
        model="openrouter/perplexity/sonar-reasoning-pro",
        reasoning_effort="high",
        web_search_options={"search_context_size": "high"},
        populate_citations=True,
    ).invoke(query)


@agent_tool
async def perplexity_quick_search_high_context(query: str) -> str:
    """
    Use Perplexity (sonar) to search for information on a topic.
    This will provide a LLM answer with citations.
    This is Perplexity's fastest but lowest quality search model.
    Good for getting a simple and quick answer to a question
    """
    logger.info(f"TOOL: Querying Perplexity (sonar) for query: {query}")
    llm = GeneralLlm(
        model="openrouter/perplexity/sonar",
        web_search_options={"search_context_size": "high"},
        populate_citations=True,
    )
    return await llm.invoke(query)


@agent_tool
async def perplexity_quick_search_low_context(query: str) -> str:
    """
    Use Perplexity (sonar) to search for information on a topic.
    This will provide a LLM answer with citations.
    This is Perplexity's fastest but lowest quality search model.
    Good for getting a simple and quick answer to a question
    """
    logger.info(f"TOOL: Querying Perplexity (sonar) for query: {query}")
    llm = GeneralLlm(
        model="openrouter/perplexity/sonar",
        web_search_options={"search_context_size": "low"},
        populate_citations=True,
    )
    return await llm.invoke(query)


@agent_tool
async def smart_searcher_search(query: str) -> str:
    """
    Use SmartSearcher to search for information on a topic.
    This will provide a LLM answer with citations.
    Citations will include url text fragments for faster fact checking.
    """
    logger.info(f"TOOL: Querying SmartSearcher for query: {query}")
    return await SmartSearcher(model="openrouter/openai/o4-mini").invoke(query)


@agent_tool
def grab_question_details_from_metaculus(
    url_or_id: str | int,
) -> MetaculusQuestion:
    """
    This function grabs the details of a question from a Metaculus URL or ID.
    """
    logger.info(
        f"TOOL: Grabbing question details from Metaculus for URL or ID: {url_or_id}"
    )
    if isinstance(url_or_id, str):
        try:
            url_or_id = int(url_or_id)
        except ValueError:
            pass

    if isinstance(url_or_id, int):
        question = MetaculusApi.get_question_by_post_id(url_or_id)
    else:
        question = MetaculusApi.get_question_by_url(url_or_id)
    question.api_json = {}
    return question


@agent_tool
def grab_open_questions_from_tournament(
    tournament_id_or_slug: int | str,
) -> list[MetaculusQuestion]:
    """
    This function grabs the details of all questions from a Metaculus tournament.
    """
    logger.info(
        f"TOOL: Grabbing open questions from Metaculus tournament: {tournament_id_or_slug}"
    )
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        tournament_id_or_slug
    )
    for question in questions:
        question.api_json = {}
    return questions


def create_tool_for_forecasting_bot(
    bot_or_class: type[ForecastBot] | ForecastBot,
) -> AgentTool:
    logger.info(f"TOOL: Creating tool for forecasting bot: {bot_or_class}")
    if isinstance(bot_or_class, type):
        bot = bot_or_class()
    else:
        bot = bot_or_class

    description = clean_indents(
        f"""
        Forecast a question and get back a report that contains probabilities for "yes/no", "options", or a number in a range

        The input is a SimpleQuestion json object (binary, numeric, or multiple choice question).
        If you have a MetaculusQuestion, there should be a MetaculusQuestion field for every field in the SimpleQuestion. Please populate these 1 for 1.

        Here is the structure of the SimpleQuestion:
        ```
        {get_schema_of_base_model(SimpleQuestion)}
        ```
        """
    )

    @agent_tool(description_override=description)
    def forecast_question_tool(question: str) -> str:
        logger.info(f"TOOL: Forecasting question: {question}")
        question_object = asyncio.run(
            structure_output(
                question,
                SimpleQuestion,
                additional_instructions=(
                    "Please note that if the input you are given is "
                    "already a json, it is probably structured wrong. "
                    "Please structure it correctly"
                ),
            )
        )

        metaculus_question = SimpleQuestion.simple_questions_to_metaculus_questions(
            [question_object]
        )[0]
        task = bot.forecast_question(metaculus_question)
        report = asyncio.run(task)
        return report.explanation

    return forecast_question_tool


@agent_tool
def roll_dice(
    probability_as_decimal: float,
) -> str:
    """
    Roll the dice to determine if an event occurred based on its probability.

    This simulates whether an event with a given probability actually happened.
    For example, if a forecast says "35% chance of X", this tool rolls the dice
    to determine if X actually occurred in this simulated future.

    Args:
        probability_as_decimal: The probability as a decimal (e.g., 0.35 for 35%)

    Returns:
        A string indicating whether the event occurred
    """
    if not (0 <= probability_as_decimal <= 1):
        raise ValueError("Probability must be between 0 and 1")

    roll = random.random()
    occurred = roll < probability_as_decimal

    result_emoji = "✅" if occurred else "❌"
    result_text = "OCCURRED" if occurred else "DID NOT OCCUR"

    message = f"{result_emoji} EVENT {result_text}"
    logger.info(
        f"TOOL: Probability: {probability_as_decimal}, Roll: {roll:.2f}, Occurred: {occurred}, Message: {message}"
    )
    return message
