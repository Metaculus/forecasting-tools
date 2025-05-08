import logging
from datetime import datetime

import pytest

from forecasting_tools.agents_and_tools.key_factors_researcher import (
    KeyFactor,
    KeyFactorType,
    ScoreCard,
    ScoreCardGrade,
)

logger = logging.getLogger(__name__)


def test_score_card_calculated_score() -> None:
    score_card = ScoreCard(
        recency=ScoreCardGrade.GOOD,
        relevance=ScoreCardGrade.VERY_GOOD,
        specificness=ScoreCardGrade.OK,
        predictive_power_and_applicability=ScoreCardGrade.GOOD,
        reputable_source=ScoreCardGrade.GOOD,
        overall_quality=ScoreCardGrade.GOOD,
        is_outdated=False,
        includes_number=True,
        includes_date=True,
        is_key_person_quote=True,
    )

    expected_score = (
        4  # recency
        + 5  # relevance
        + 3  # specificness
        + 8  # predictive_power (weighted x2)
        + 4  # reputable_source
        + 8  # overall_quality (weighted x2)
        + 5  # includes_number
        + 3  # includes_date
        + 3  # is_key_person_quote
    )
    assert score_card.calculated_score == expected_score


def test_score_card_calculated_score_with_outdated() -> None:
    score_card = ScoreCard(
        recency=ScoreCardGrade.GOOD,
        relevance=ScoreCardGrade.VERY_GOOD,
        specificness=ScoreCardGrade.OK,
        predictive_power_and_applicability=ScoreCardGrade.GOOD,
        reputable_source=ScoreCardGrade.GOOD,
        overall_quality=ScoreCardGrade.GOOD,
        is_outdated=True,  # This should halve the score
        includes_number=True,
        includes_date=True,
        is_key_person_quote=True,
    )

    expected_score = round(
        (
            4  # recency
            + 5  # relevance
            + 3  # specificness
            + 8  # predictive_power (weighted x2)
            + 4  # reputable_source
            + 8  # overall_quality (weighted x2)
            + 5  # includes_number
            + 3  # includes_date
            + 3  # is_key_person_quote
        )
        * 0.5  # halved due to being outdated
    )
    assert score_card.calculated_score == expected_score


def test_key_factor_validation() -> None:
    valid_citation = "[1](http://example.com)"
    KeyFactor(
        text="Test factor",
        factor_type=KeyFactorType.PRO,
        citation=valid_citation,
        source_publish_date=datetime.now(),
    )

    invalid_citation = "http://example.com"
    with pytest.raises(ValueError):
        KeyFactor(
            text="Test factor",
            factor_type=KeyFactorType.PRO,
            citation=invalid_citation,
            source_publish_date=datetime.now(),
        )
