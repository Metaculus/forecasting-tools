import logging
import os
import time

import pytest

from forecasting_tools.forecasting.helpers.asknews_searcher import (
    AskNewsSearcher,
)

logger = logging.getLogger(__name__)


def test_asknews_connection():
    if not os.getenv("ASKNEWS_CLIENT_ID") or not os.getenv("ASKNEWS_SECRET"):
        pytest.skip("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")
    logger.debug("Testing AskNews connection")
    start_time = time.time()
    news = AskNewsSearcher.get_formatted_news(
        "Will the US stock market crash in 2025?"
    )
    end_time = time.time()
    logger.debug(f"Time taken: {end_time - start_time} seconds")
    logger.debug(f"News: {news}")
    assert news is not None
    assert len(news) > 100
