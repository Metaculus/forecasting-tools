from agents import function_tool

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher

# TODO: Make tools for Exa-Pro, AskNews DeepNews, Perplexity DeepResearch, and an agent that combines all of these


@function_tool
async def get_general_news(topic: str) -> str:
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
    )
    return await llm.invoke(query)


@function_tool
async def smart_searcher_search(query: str) -> str:
    """
    Use SmartSearcher to search for information on a topic.
    This will provide a LLM answer with citations.
    """
    return await SmartSearcher(model="o4-mini").invoke(query)
