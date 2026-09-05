import logging
import os
import time

import pytest

from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)


def test_asknews_connection() -> None:
    if not os.getenv("ASKNEWS_CLIENT_ID") or not os.getenv("ASKNEWS_SECRET"):
        pytest.skip("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")
    logger.debug("Testing AskNews connection")
    start_time = time.time()
    news = AskNewsSearcher().get_formatted_news(
        "Will the US stock market crash in 2025?"
    )
    end_time = time.time()
    logger.debug(f"Time taken: {end_time - start_time} seconds")
    logger.debug(f"News: {news}")
    assert news is not None
    assert len(news) > 100


async def test_formatted_deep_research() -> None:
    if not os.getenv("ASKNEWS_CLIENT_ID") or not os.getenv("ASKNEWS_SECRET"):
        pytest.skip("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")
    logger.debug("Testing AskNews connection")
    start_time = time.time()
    news = await AskNewsSearcher().call_preconfigured_version(
        "asknews/deep-research/low-depth/o3-mini",
        "Will the US stock market crash in 2025?",
    )
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")
    logger.info(f"News: {news}")
    assert news is not None, "News is None"
    assert len(news) > 100, "News is too short"
    assert "<final_answer>" not in news, "<final_answer> is in the final answer"
    assert "<think>" not in news, "<think> is in the news"


async def test_deep_research_accepts_podcasts_source() -> None:
    if not os.getenv("ASKNEWS_CLIENT_ID") or not os.getenv("ASKNEWS_SECRET"):
        pytest.skip("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")
    response = await AskNewsSearcher().run_deep_research(
        "In one sentence, what is the latest news about the US federal funds rate?",
        sources=["asknews", "podcasts"],
        model="open-source-best",
        search_depth=1,
        max_depth=1,
    )
    content = response.choices[0].message.content
    logger.info(f"Deep research with podcasts source: {content}")
    assert content is not None, "Content is None"
    assert len(content) > 100, "Content is too short"
