import logging
from datetime import datetime, timedelta

from forecasting_tools.helpers.adjacent_news_api import AdjacentFilter, AdjacentNewsApi

logger = logging.getLogger(__name__)


def test_adjacent_news_api() -> None:
    min_volume = 50000
    one_year_ago = datetime.now() - timedelta(days=365)
    api_filter = AdjacentFilter(
        include_closed=False,
        platform=["polymarket"],
        market_type=["binary"],
        volume_min=min_volume,
        created_after=one_year_ago,
    )
    requested_markets = 10
    markets = AdjacentNewsApi.get_questions_matching_filter(
        api_filter,
        num_questions=requested_markets,
        error_if_market_target_missed=False,
    )
    all_markets = ""
    for market in markets:
        all_markets += f"{market.question_text} - {market.tags} - {market.category} - {market.link} - {market.volume} - {market.probability_at_access_time} \n"

    assert (
        len(markets) == requested_markets
    ), f"Expected {requested_markets} markets, got {len(markets)}"
    for market in markets:
        assert market.volume is not None
        assert min_volume <= market.volume
        assert market.probability_at_access_time
        assert market.status == "active"
        assert market.platform == "polymarket"
        assert market.market_type == "binary"
        assert market.created_at is not None
        assert market.created_at >= one_year_ago

    logger.info(all_markets)
