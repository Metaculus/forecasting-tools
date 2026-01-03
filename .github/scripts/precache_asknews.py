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


from forecasting_tools.forecast_bots.template_bot import TemplateBot
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from run_bots import TournConfig, get_questions_for_allowed_tournaments

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


async def precache_all_questions() -> None:
    cache_mode = "use_cache_with_fallback"
    searcher = AskNewsSearcher(cache_mode=cache_mode)

    logger.info("Fetching all questions from all tournaments")
    questions = await get_questions_for_allowed_tournaments(
        allowed_tournaments=TournConfig.everything, max_questions=200, mode=None
    )

    logger.info(f"Found {len(questions)} questions to cache")

    tasks = []
    for question in questions:
        query = TemplateBot._get_research_prompt(question, "asknews/news-summaries")
        tasks.append(searcher.get_formatted_news_async(query))

    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True), timeout=120
    )
    # Log any failures
    failed = sum(1 for r in results if isinstance(r, Exception))
    if failed:
        logger.warning(f"Failed to cache {failed}/{len(results)} questions")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(precache_all_questions())

    logger.info("Pre-caching complete!")
