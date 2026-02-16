import json
import logging
import random

from forecasting_tools.ai_models.agent_wrappers import AgentTool, agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)


def create_search_tool(search_model: str) -> AgentTool:
    description = (
        f"Search for information on a topic using {search_model}. "
        "This will provide an LLM answer with citations. "
        "Use this tool extensively to research the policy question, "
        "gather evidence for forecasts, and verify claims."
    )

    @agent_tool(description_override=description)
    async def search(query: str) -> str:
        logger.info(f"TOOL: Searching with {search_model} for query: {query}")
        return await GeneralLlm(
            model=search_model,
            reasoning_effort="high",
            web_search_options={"search_context_size": "high"},
            populate_citations=True,
        ).invoke(query)

    return search


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
    from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

    logger.info(f"TOOL: Querying AskNews for topic: {topic}")
    return await AskNewsSearcher().get_formatted_news_async(topic)


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
        f"TOOL: Probability: {probability_as_decimal}, Roll: {roll:.2f}, "
        f"Occurred: {occurred}, Message: {message}"
    )
    return message


def roll_multiple_dice_raw(forecasts_json: str) -> str:
    try:
        forecasts = json.loads(forecasts_json)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}. Please provide valid JSON."

    if not isinstance(forecasts, list):
        return "Error: Input must be a JSON array of forecast objects."

    results: list[str] = []
    results.append("| ID | Title | Probability | Roll | Outcome |")
    results.append("|---|---|---|---|---|")

    for forecast in forecasts:
        forecast_id = str(forecast.get("id", "?"))
        title = str(forecast.get("title", "Unknown"))
        probability = float(forecast.get("probability", 0.5))

        if not (0 <= probability <= 1):
            results.append(
                f"| {forecast_id} | {title} | {probability} | ERROR | "
                f"Probability must be between 0 and 1 |"
            )
            continue

        roll = random.random()
        occurred = roll < probability
        outcome = "✅ OCCURRED" if occurred else "❌ DID NOT OCCUR"

        results.append(
            f"| {forecast_id} | {title} | {probability:.0%} | {roll:.2f} | {outcome} |"
        )
        logger.info(
            f"TOOL: Batch dice - ID: {forecast_id}, Probability: {probability}, "
            f"Roll: {roll:.2f}, Occurred: {occurred}"
        )

    return "\n".join(results)


ROLL_MULTIPLE_DICE_DESCRIPTION = (
    "Roll dice for multiple forecasts at once, returning all outcomes with clear ID mapping. "
    "This is more efficient than calling roll_dice repeatedly. Pass a JSON array of objects, "
    'each with: "id" (unique identifier like "[^1]"), "title" (forecast question title), '
    'and "probability" (decimal 0.0 to 1.0). '
    "Returns a formatted table of all outcomes."
)


@agent_tool(description_override=ROLL_MULTIPLE_DICE_DESCRIPTION)
def roll_multiple_dice(forecasts_json: str) -> str:
    return roll_multiple_dice_raw(forecasts_json)
