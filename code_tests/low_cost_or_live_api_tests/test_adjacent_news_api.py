import logging

from forecasting_tools.forecast_helpers.adjacent_news_api import (
    AdjacentFilter,
    AdjacentNewsApi,
)

logger = logging.getLogger(__name__)


def test_adjacent_news_api() -> None:
    min_volume = 50000
    api_filter = AdjacentFilter(
        include_closed=False,
        platform=["polymarket"],
        volume_min=min_volume,
    )
    requested_markets = 100
    markets = AdjacentNewsApi.get_questions_matching_filter(
        api_filter, num_questions=requested_markets
    )
    assert len(markets) == requested_markets
    for market in markets:
        assert market.volume is not None
        assert min_volume <= market.volume
        assert market.probability_at_access_time
        assert market.status == "active"
        assert market.platform == "polymarket"

    all_markets = ""
    for market in markets:
        all_markets += f"{market.question_text} - {market.platform} - {market.volume} - {market.probability_at_access_time} \n"

    logger.info(all_markets)
