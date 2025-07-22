import logging

import pytest

from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "question_url",
    [
        "https://www.metaculus.com/questions/28841/will-eric-adams-be-the-nyc-mayor-on-january-1-2025/",
        "https://www.metaculus.com/questions/28658/russia-to-ban-youtube-by-2025/",
        "https://www.metaculus.com/questions/28657/china-announce-metal-export-restrictions-2025/",
        "https://www.metaculus.com/questions/28423/change-in-givedirectlys-cost-effectiveness/",
    ],
)
async def test_find_key_factors_end_to_end(question_url: str) -> None:
    num_factors_to_return = 5
    num_questions_to_research_with = 16

    question = MetaculusApi.get_question_by_url(question_url)

    with MonetaryCostManager() as cost_manager:
        key_factors = await KeyFactorsResearcher.find_and_sort_key_factors(
            question,
            num_key_factors_to_return=num_factors_to_return,
            num_questions_to_research_with=num_questions_to_research_with,
        )
        key_factors_markdown = ScoredKeyFactor.turn_key_factors_into_markdown_list(
            key_factors
        )
        logger.info(f"\nCost: {cost_manager.current_usage}\n{key_factors_markdown}")

    assert len(key_factors) == num_factors_to_return
    assert all(isinstance(factor, ScoredKeyFactor) for factor in key_factors)
    assert cost_manager.current_usage < 2, "Cost should be less than 2 dollars"

    for factor in key_factors:
        assert factor.text, "Factor should have text"
        assert factor.citation, "Factor should have citation"
        assert factor.source_publish_date, "Factor should have publish date"
        assert factor.score > 0, "Factor should have positive score"
        assert factor.score_card, "Factor should have score card"
