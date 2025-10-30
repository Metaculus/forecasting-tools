import logging

import pytest

from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion, QuestionState

logger = logging.getLogger(__name__)


# TODO: Lower priority tests
# - Make discrete tests for all the above edge cases
# - Test open bound distributions
# - Test zero point


def test_percentile_validation() -> None:
    # Valid percentiles
    valid_percentile = Percentile(value=10.0, percentile=0.5)
    assert valid_percentile.value == pytest.approx(10.0)
    assert valid_percentile.percentile == pytest.approx(0.5)

    # Invalid percentiles
    with pytest.raises(ValueError):
        Percentile(value=10.0, percentile=1.5)

    with pytest.raises(ValueError):
        Percentile(value=10.0, percentile=-0.1)


class TestNumericDistributionValidation:
    def test_valid_distribution_creation(self) -> None:
        valid_percentiles = [
            Percentile(value=10.0, percentile=0.1),
            Percentile(value=20.0, percentile=0.5),
            Percentile(value=30.0, percentile=0.9),
        ]

        distribution = NumericDistribution(
            declared_percentiles=valid_percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )
        assert len(distribution.declared_percentiles) == 3

    def test_non_increasing_percentiles_raises_error(self) -> None:
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
                standardize_cdf=False,
            )

    def test_non_increasing_values_raises_error(self) -> None:
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
                standardize_cdf=False,
            )

    def test_valid_repeating_values_at_lower_bound(self) -> None:
        valid_repeating_values_at_bounds = [
            Percentile(value=0.0, percentile=0.1),
            Percentile(value=0.0, percentile=0.5),
            Percentile(value=100.0, percentile=0.9),
        ]
        distribution = NumericDistribution(
            declared_percentiles=valid_repeating_values_at_bounds,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )
        assert len(distribution.declared_percentiles) == 3
        assert (
            distribution.declared_percentiles[0].value
            < distribution.declared_percentiles[1].value
        )

    def test_valid_repeating_values_at_upper_bound(self) -> None:
        valid_repeating_values_at_bounds = [
            Percentile(value=0.0, percentile=0.1),
            Percentile(value=100.0, percentile=0.5),
            Percentile(value=100.0, percentile=0.9),
        ]
        distribution = NumericDistribution(
            declared_percentiles=valid_repeating_values_at_bounds,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )
        assert len(distribution.declared_percentiles) == 3
        assert (
            distribution.declared_percentiles[2].value
            > distribution.declared_percentiles[1].value
        )

    def test_valid_repeating_percentiles_in_middle_of_distribution(self) -> None:
        valid_percentiles = [
            Percentile(value=10, percentile=0.1),
            Percentile(value=11, percentile=0.2),
            Percentile(value=11, percentile=0.4),
            Percentile(value=11, percentile=0.6),
            Percentile(value=11, percentile=0.8),
            Percentile(value=11, percentile=0.9),
        ]
        distribution = NumericDistribution(
            declared_percentiles=valid_percentiles,
            open_upper_bound=False,
            open_lower_bound=True,
            upper_bound=220,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )
        assert len(distribution.declared_percentiles) == 6
        assert distribution.declared_percentiles[0].value == 10
        assert 10.999 < distribution.declared_percentiles[1].value <= 11
        assert 10.999 < distribution.declared_percentiles[2].value <= 11
        assert 10.999 < distribution.declared_percentiles[3].value <= 11
        assert 10.999 < distribution.declared_percentiles[4].value <= 11
        assert 10.999 < distribution.declared_percentiles[5].value <= 11
        assert distribution.declared_percentiles[5].percentile == 0.9


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
        standardize_cdf=False,
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
        standardize_cdf=False,
    )
    dist2 = NumericDistribution(
        declared_percentiles=percentiles2,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        standardize_cdf=False,
    )
    dist3 = NumericDistribution(
        declared_percentiles=percentiles3,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        standardize_cdf=False,
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
    for cdf_size in [10, 100, 201]:
        distribution = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            cdf_size=cdf_size,
            standardize_cdf=False,
        )

        assert distribution.get_cdf()[0].percentile == pytest.approx(0.0)
        assert distribution.get_cdf()[0].value == pytest.approx(0.0)
        assert distribution.get_cdf()[-1].percentile == pytest.approx(1.0)
        assert distribution.cdf[-1].value == pytest.approx(100.0)

        for i in range(len(distribution.get_cdf()) - 1):
            assert (
                distribution.get_cdf()[i + 1].value - distribution.get_cdf()[i].value
                > 0.00001
            )

        assert len(distribution.get_cdf()) == cdf_size


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
        standardize_cdf=False,
    )
    with pytest.raises(Exception):
        logger.info(prediction.get_cdf())
        prediction.get_cdf()


def test_numeric_edge_of_bin_edge_case() -> None:
    """
    If all of the probability is assigned to 12, we want to make sure that if it resolves "12"
    that probaiblity was assigned to the bucket that is scored.

    See discussion here on how bounds are handled: https://discord.com/channels/694850840200216657/1248850491773812821/1412537502543118420
    Also see discussion here: https://metaculus.slack.com/archives/C01Q9AQBVHB/p1761766286194479?thread_ts=1761675067.315789&cid=C01Q9AQBVHB
    """
    percentiles = [
        Percentile(value=12, percentile=0.1),
        Percentile(value=12, percentile=0.2),
        Percentile(value=12, percentile=0.4),
        Percentile(value=12, percentile=0.6),
        Percentile(value=12, percentile=0.8),
        Percentile(value=12, percentile=0.9),
    ]
    # Question URL: https://www.metaculus.com/questions/39617/opec-member-countries-in-2025/
    numeric_distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=True,
        open_lower_bound=False,
        upper_bound=20,
        lower_bound=0,
        zero_point=None,
        # standardize_cdf=True,
    )
    pmf_diffs = _get_and_log_pmf_diffs(numeric_distribution)

    correct_bucket_diff = pmf_diffs[120]
    wrong_bucket_diff = pmf_diffs[121]
    assert (
        correct_bucket_diff > 0.5 > wrong_bucket_diff
    ), f"The bucket for (12, 12.1] has more probability than the bucket for (11.9, 12] ({wrong_bucket_diff:.5f} > {correct_bucket_diff:.5f})"


def test_discrete_distribution_repeated_value() -> None:
    percentiles = [
        Percentile(value=12, percentile=0.1),
        Percentile(value=12, percentile=0.2),
        Percentile(value=12, percentile=0.4),
        Percentile(value=12, percentile=0.6),
        Percentile(value=12, percentile=0.8),
        Percentile(value=12, percentile=0.9),
    ]
    numeric_distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=20,
        lower_bound=0,
        zero_point=None,
        # standardize_cdf=True,
        cdf_size=21,
    )
    pmf_diffs = _get_and_log_pmf_diffs(numeric_distribution)

    assert pmf_diffs[12] > 0.5 > pmf_diffs[13]
    assert pmf_diffs[12] > 0.5 > pmf_diffs[11]


def test_log_scale_distribution_repeated_value() -> None:
    percentiles = [
        Percentile(value=11, percentile=0.1),
        Percentile(value=11, percentile=0.2),
        Percentile(value=11, percentile=0.4),
        Percentile(value=11, percentile=0.6),
        Percentile(value=11, percentile=0.8),
        Percentile(value=11, percentile=0.9),
    ]
    # https://dev.metaculus.com/questions/6609/non-tesla-vehicles-w-tesla-software-by-2030/
    numeric_distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100_000_000,
        lower_bound=1,
        zero_point=0,
        # standardize_cdf=True,
    )
    pmf_diffs = _get_and_log_pmf_diffs(numeric_distribution)
    assert (
        pmf_diffs[27] > 0.5 > pmf_diffs[26]
    ), "Not enough probability in bucket (10.96, 12.02]. Should be at least 0.5"
    assert (
        pmf_diffs[27] > 0.5 > pmf_diffs[28]
    ), "Not enough probability in bucket (10.96, 12.02]. Should be at least 0.5"


def _get_and_log_pmf_diffs(distribution: NumericDistribution) -> list[float]:
    cdf = distribution.get_cdf()
    pmf_diffs = []
    for i, percentile in enumerate(cdf):
        previous_percentile = cdf[i - 1] if i > 0 else cdf[0]
        diff = percentile.percentile - previous_percentile.percentile
        logger.info(
            f"Index {i} | Value {percentile.value:.4f} | Percentile {percentile.percentile:.4f} | Diff {diff:.6f} (Index {i-1} to {i})"
        )
        pmf_diffs.append(diff)
    return pmf_diffs
