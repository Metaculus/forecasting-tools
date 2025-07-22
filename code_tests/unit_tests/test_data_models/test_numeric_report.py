import logging

import pytest

from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion, QuestionState

logger = logging.getLogger(__name__)


def test_percentile_validation() -> None:
    # Valid percentiles
    valid_percentile = Percentile(value=10.0, percentile=0.5)
    assert valid_percentile.value == pytest.approx(10.0)
    assert valid_percentile.percentile == pytest.approx(0.5)

    # Invalid percentiles
    with pytest.raises(ValueError, match="Percentile must be between 0 and 1"):
        Percentile(value=10.0, percentile=1.5)

    with pytest.raises(ValueError, match="Percentile must be between 0 and 1"):
        Percentile(value=10.0, percentile=-0.1)


def test_numeric_distribution_validation() -> None:
    valid_percentiles = [
        Percentile(value=10.0, percentile=0.1),
        Percentile(value=20.0, percentile=0.5),
        Percentile(value=30.0, percentile=0.9),
    ]

    # Valid distribution
    distribution = NumericDistribution(
        declared_percentiles=valid_percentiles,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )
    assert len(distribution.declared_percentiles) == 3

    # Test non-increasing percentiles
    invalid_percentiles = [
        Percentile(value=10.0, percentile=0.5),
        Percentile(value=20.0, percentile=0.3),  # Decreasing percentile
        Percentile(value=30.0, percentile=0.9),
    ]
    with pytest.raises(ValueError):
        NumericDistribution(
            declared_percentiles=invalid_percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )

    # Test non-increasing values
    invalid_values = [
        Percentile(value=10.0, percentile=0.1),
        Percentile(value=5.0, percentile=0.5),  # Decreasing value
        Percentile(value=30.0, percentile=0.9),
    ]
    with pytest.raises(Exception):
        NumericDistribution(
            declared_percentiles=invalid_values,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )


def test_get_representative_percentiles() -> None:
    percentiles = [
        Percentile(value=10.0, percentile=0.1),
        Percentile(value=20.0, percentile=0.3),
        Percentile(value=30.0, percentile=0.5),
        Percentile(value=40.0, percentile=0.7),
        Percentile(value=50.0, percentile=0.9),
    ]
    distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )

    # Test with valid number of percentiles
    rep_percentiles = distribution.get_representative_percentiles(3)
    assert len(rep_percentiles) == 3
    assert rep_percentiles[0] == percentiles[0]
    assert rep_percentiles[1] == percentiles[2]
    assert rep_percentiles[2] == percentiles[4]

    # Test with invalid number of percentiles
    with pytest.raises(ValueError, match="Number of percentiles must be at least 2"):
        distribution.get_representative_percentiles(1)

    # Test with too many percentiles
    rep_percentiles = distribution.get_representative_percentiles(10)
    assert len(rep_percentiles) == len(percentiles)


async def test_aggregate_predictions() -> None:
    percentiles1 = [
        Percentile(value=10.0, percentile=0.4),
        Percentile(value=20.0, percentile=0.5),
        Percentile(value=30.0, percentile=0.6),
    ]
    percentiles2 = [
        Percentile(value=20.0, percentile=0.4),
        Percentile(value=30.0, percentile=0.5),
        Percentile(value=40.0, percentile=0.6),
    ]
    percentiles3 = [
        Percentile(value=30.0, percentile=0.4),
        Percentile(value=40.0, percentile=0.5),
        Percentile(value=50.0, percentile=0.6),
    ]

    dist1 = NumericDistribution(
        declared_percentiles=percentiles1,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )
    dist2 = NumericDistribution(
        declared_percentiles=percentiles2,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )
    dist3 = NumericDistribution(
        declared_percentiles=percentiles3,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )

    question = NumericQuestion(
        id_of_post=1,
        id_of_question=1,
        question_text="Test question",
        background_info="Test background",
        resolution_criteria="Test criteria",
        fine_print="Test fine print",
        state=QuestionState.OPEN,
        upper_bound=100.0,
        lower_bound=0.0,
        open_upper_bound=False,
        open_lower_bound=False,
        zero_point=None,
    )

    aggregated = await NumericReport.aggregate_predictions(
        [dist1, dist2, dist3], question
    )
    assert isinstance(aggregated, NumericDistribution)
    assert len(aggregated.cdf) == 201  # Full CDF should have 201 points
    # Find median (50th percentile)
    median_value = next(
        p.value
        for p in aggregated.declared_percentiles
        if p.percentile == pytest.approx(0.5)
    )
    assert median_value == pytest.approx(
        30.0
    )  # Median of 20, 30, 40 from the input distributions

    # Test empty predictions list
    with pytest.raises(Exception):
        await NumericReport.aggregate_predictions([], question)


@pytest.mark.parametrize(
    "percentiles",
    [
        [
            Percentile(value=0.0, percentile=0.1),
            Percentile(value=20.0, percentile=0.5),
            Percentile(value=100.0, percentile=0.9),
        ],
        [
            Percentile(value=5.0, percentile=0.1),
            Percentile(value=20.0, percentile=0.5),
            Percentile(value=95.0, percentile=0.9),
        ],
    ],
)
def test_close_bound_distribution(percentiles: list[Percentile]) -> None:
    distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )

    assert distribution.cdf[0].percentile == pytest.approx(0.0)
    assert distribution.cdf[0].value == pytest.approx(0.0)
    assert distribution.cdf[-1].percentile == pytest.approx(1.0)
    assert distribution.cdf[-1].value == pytest.approx(100.0)

    for i in range(len(distribution.cdf) - 1):
        assert distribution.cdf[i + 1].value - distribution.cdf[i].value > 0.00001


def test_error_on_too_little_probability_assigned_in_range() -> None:
    prediction = NumericDistribution(
        declared_percentiles=[
            Percentile(percentile=0.1, value=1.1),
            Percentile(percentile=0.2, value=1.2),
            Percentile(percentile=0.3, value=1.3),
            Percentile(percentile=0.4, value=1.4),
            Percentile(percentile=0.5, value=1.5),
            Percentile(percentile=0.6, value=1.6),
            Percentile(percentile=0.7, value=1.7),
            Percentile(percentile=0.8, value=1.8),
            Percentile(percentile=0.9, value=1.9),
        ],
        lower_bound=14,
        upper_bound=15,
        zero_point=None,
        open_lower_bound=True,
        open_upper_bound=True,
    )
    with pytest.raises(Exception):
        logger.info(prediction.cdf)
        prediction.cdf
