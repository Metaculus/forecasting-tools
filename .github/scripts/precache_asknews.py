"""
Script to pre-cache AskNews results for all open tournament questions.
This should be run before the main bot runs to populate the cache.
"""

import asyncio
import logging
import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from typing import Literal

from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from run_bots import TournConfig, get_questions_for_allowed_tournaments

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


async def precache_all_questions() -> None:
    cache_mode: Literal["use_cache", "use_cache_with_fallback", "no_cache"] = os.getenv("ASKNEWS_CACHE_MODE", "no_cache")  # type: ignore
    assert (
        cache_mode != "no_cache"
    ), "Cache mode must be set to use_cache or use_cache_with_fallback"
    searcher = AskNewsSearcher(cache_mode=cache_mode)

    logger.info("Fetching all questions from all tournaments")
    questions = await get_questions_for_allowed_tournaments(
        allowed_tournaments=TournConfig.everything,
        max_questions=200,
    )

    logger.info(f"Found {len(questions)} questions to cache")

    tasks = [
        searcher.get_formatted_news_async(question.question_text)
        for question in questions
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(precache_all_questions())

    logger.info("Pre-caching complete!")
