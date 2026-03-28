import logging
from typing import Literal

import pytest

from forecasting_tools.data_models.numeric_report import (
    NumericDefaults,
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
    expected_max_pmf_value = (
        NumericDefaults.get_max_pmf_value(
            len(numeric_distribution.get_cdf()), include_wiggle_room=True
        )
        * 0.95
    )
    assert (
        correct_bucket_diff > expected_max_pmf_value > wrong_bucket_diff
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
    expected_max_pmf_value = NumericDefaults.get_max_pmf_value(
        len(numeric_distribution.get_cdf()), include_wiggle_room=True
    )
    assert (
        pmf_diffs[27] > expected_max_pmf_value > pmf_diffs[26]
    ), "Not enough probability in bucket (10.96, 12.02]. Should be at least 0.19"
    assert (
        pmf_diffs[27] > expected_max_pmf_value > pmf_diffs[28]
    ), "Not enough probability in bucket (10.96, 12.02]. Should be at least 0.19"


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


@pytest.mark.parametrize(
    "orientation",
    ["far_left", "center", "far_right"],
)
@pytest.mark.parametrize(
    "spread",
    ["very wide", "normal", "very narrow"],
)
@pytest.mark.parametrize(
    "bounds_and_cdf_size",
    [
        (-5, 5, 10),
        (0, 100, 100),
        (1, 200, 201),
        (-30_000, 30_000, 201),
    ],  # TODO: Should discrete questions have bounds at 0.5 intervals?
)
@pytest.mark.parametrize(
    "open_upper_bound",
    [True, False],
)
@pytest.mark.parametrize(
    "open_lower_bound",
    [True],
)
@pytest.mark.parametrize(
    "zero_point",
    [None, -1, 0],
)
def test_distribution_variations(
    orientation: Literal["far_left", "left", "center", "right", "far_right"],
    spread: Literal["very wide", "wide", "normal", "narrow", "very narrow"],
    bounds_and_cdf_size: tuple[float, float, int],
    open_upper_bound: bool,
    open_lower_bound: bool,
    zero_point: float | None,
):
    check_distribution_variations(
        orientation,
        spread,
        bounds_and_cdf_size,
        open_upper_bound,
        open_lower_bound,
        zero_point,
    )


def check_distribution_variations(
    orientation: Literal["far_left", "left", "center", "right", "far_right"],
    spread: Literal["very wide", "wide", "normal", "narrow", "very narrow"],
    bounds_and_cdf_size: tuple[float, float, int],
    open_upper_bound: bool,
    open_lower_bound: bool,
    zero_point: float | None,
) -> None:
    lower_bound, upper_bound, cdf_size = bounds_and_cdf_size

    if zero_point is not None and cdf_size != 201:
        pytest.skip("zero_point is not supported for discrete questions")

    if zero_point is not None and lower_bound < zero_point:
        lower_bound = zero_point + 1

    range_size = upper_bound - lower_bound

    orientation_fractions = {
        "far_left": 0.02,
        "left": 0.3,
        "center": 0.5,
        "right": 0.7,
        "far_right": 0.98,
    }
    center_fraction = orientation_fractions[orientation]
    center = lower_bound + range_size * center_fraction

    spread_fractions = {
        "very wide": 0.99,
        "wide": 0.25,
        "normal": 0.15,
        "narrow": 0.08,
        "very narrow": 0.01,
    }
    spread_fraction = spread_fractions[spread]
    half_spread = spread_fraction * range_size

    percentile_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    percentiles = []
    for p in percentile_points:
        offset = (p - 0.5) * 2 * half_spread
        value = center + offset
        percentiles.append(Percentile(value=value, percentile=p))
        if zero_point is not None and value < zero_point:
            # TODO: Get this edge case to pass
            pytest.skip(
                "value is less than zero_point which is currently not supported"
            )

    distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
        cdf_size=cdf_size,
        standardize_cdf=True,
    )

    cdf = distribution.get_cdf()
    cdf_distribution = NumericDistribution(
        declared_percentiles=cdf,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
        cdf_size=cdf_size,
    )
    # Let the model validator do most the asserts
    assert cdf_distribution.cdf_size == cdf_size


class TestGetPercentilesAtTargetHeights:
    def _make_distribution(
        self,
        percentiles: list[Percentile],
        lower_bound: float = 0.0,
        upper_bound: float = 100.0,
        open_lower_bound: bool = False,
        open_upper_bound: bool = False,
    ) -> NumericDistribution:
        return NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            zero_point=None,
            standardize_cdf=False,
        )

    def test_basic_interpolation(self) -> None:
        percentiles = [
            Percentile(value=10.0, percentile=0.1),
            Percentile(value=50.0, percentile=0.5),
            Percentile(value=90.0, percentile=0.9),
        ]
        dist = self._make_distribution(percentiles)
        result = dist.get_percentiles_at_target_heights([0.1, 0.5, 0.9])
        assert result[0].value == pytest.approx(10.0)
        assert result[1].value == pytest.approx(50.0)
        assert result[2].value == pytest.approx(90.0)
        assert result[0].percentile == pytest.approx(0.1)
        assert result[1].percentile == pytest.approx(0.5)
        assert result[2].percentile == pytest.approx(0.9)

    def test_interpolation_between_declared_points(self) -> None:
        percentiles = [
            Percentile(value=0.0, percentile=0.0),
            Percentile(value=100.0, percentile=1.0),
        ]
        dist = self._make_distribution(percentiles)
        result = dist.get_percentiles_at_target_heights([0.25, 0.5, 0.75])
        assert result[0].value == pytest.approx(25.0)
        assert result[1].value == pytest.approx(50.0)
        assert result[2].value == pytest.approx(75.0)

    def test_default_target_heights(self) -> None:
        percentiles = [
            Percentile(value=0.0, percentile=0.0),
            Percentile(value=100.0, percentile=1.0),
        ]
        dist = self._make_distribution(percentiles)
        result = dist.get_percentiles_at_target_heights()
        assert len(result) == 6
        expected_heights = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        for r, h in zip(result, expected_heights):
            assert r.percentile == pytest.approx(h)

    def test_clamping_beyond_range(self) -> None:
        percentiles = [
            Percentile(value=30.0, percentile=0.3),
            Percentile(value=70.0, percentile=0.7),
        ]
        dist = self._make_distribution(percentiles)
        result = dist.get_percentiles_at_target_heights([0.1, 0.5, 0.9])
        assert result[0].value == pytest.approx(30.0)
        assert result[1].value == pytest.approx(50.0)
        assert result[2].value == pytest.approx(70.0)


class TestAggregateReadablePrediction:
    def _make_question(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 100_000_000_000.0,
        open_lower_bound: bool = True,
        open_upper_bound: bool = True,
    ) -> NumericQuestion:
        return NumericQuestion(
            id_of_post=1,
            id_of_question=1,
            question_text="Test question",
            background_info="Test background",
            resolution_criteria="Test criteria",
            fine_print="Test fine print",
            state=QuestionState.OPEN,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            zero_point=None,
        )

    def _make_forecaster_distribution(
        self,
        percentile_values: list[tuple[float, float]],
        question: NumericQuestion,
    ) -> NumericDistribution:
        percentiles = [Percentile(percentile=p, value=v) for p, v in percentile_values]
        return NumericDistribution.from_question(percentiles, question)

    async def test_aggregate_readable_matches_forecaster_format(self) -> None:
        """
        Reproduces the bug from NVIDIA forward guidance question where
        the aggregate showed weird percentile values like 5.04%, 78.24%
        instead of clean values like 10%, 20%, 40%, 60%, 80%, 90%.
        """
        question = self._make_question()
        forecaster_percentile_values = [
            [
                (0.1, 55e9),
                (0.2, 58e9),
                (0.4, 61e9),
                (0.6, 63e9),
                (0.8, 66e9),
                (0.9, 70e9),
            ],
            [
                (0.1, 54e9),
                (0.2, 57e9),
                (0.4, 60e9),
                (0.6, 62.5e9),
                (0.8, 65.5e9),
                (0.9, 69e9),
            ],
            [
                (0.1, 55e9),
                (0.2, 58e9),
                (0.4, 61e9),
                (0.6, 62.5e9),
                (0.8, 65e9),
                (0.9, 68e9),
            ],
            [
                (0.1, 55e9),
                (0.2, 58e9),
                (0.4, 60e9),
                (0.6, 62e9),
                (0.8, 65e9),
                (0.9, 68e9),
            ],
        ]
        predictions = [
            self._make_forecaster_distribution(pvs, question)
            for pvs in forecaster_percentile_values
        ]

        aggregated = await NumericReport.aggregate_predictions(predictions, question)
        readable = NumericReport.make_readable_prediction(aggregated)

        assert "10.00%" in readable
        assert "20.00%" in readable
        assert "40.00%" in readable
        assert "60.00%" in readable
        assert "80.00%" in readable
        assert "90.00%" in readable
        assert "5.04%" not in readable

    async def test_aggregate_readable_values_are_reasonable(self) -> None:
        """
        The interpolated values at standard percentile heights should be
        close to the median of individual forecasters' values at those heights.
        """
        question = self._make_question()
        predictions = [
            self._make_forecaster_distribution(
                [
                    (0.1, 10e9),
                    (0.2, 20e9),
                    (0.4, 40e9),
                    (0.6, 60e9),
                    (0.8, 80e9),
                    (0.9, 90e9),
                ],
                question,
            ),
            self._make_forecaster_distribution(
                [
                    (0.1, 12e9),
                    (0.2, 22e9),
                    (0.4, 42e9),
                    (0.6, 62e9),
                    (0.8, 82e9),
                    (0.9, 92e9),
                ],
                question,
            ),
        ]

        aggregated = await NumericReport.aggregate_predictions(predictions, question)
        result = aggregated.get_percentiles_at_target_heights()
        for p in result:
            assert p.percentile in [
                pytest.approx(h) for h in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            ]
        tenth_percentile_value = result[0].value
        assert 9e9 < tenth_percentile_value < 15e9

    async def test_aggregate_with_identical_forecasters(self) -> None:
        question = self._make_question()
        same_pvs = [
            (0.1, 30e9),
            (0.2, 40e9),
            (0.4, 50e9),
            (0.6, 60e9),
            (0.8, 70e9),
            (0.9, 80e9),
        ]
        predictions = [
            self._make_forecaster_distribution(same_pvs, question) for _ in range(4)
        ]

        aggregated = await NumericReport.aggregate_predictions(predictions, question)
        result = aggregated.get_percentiles_at_target_heights()
        assert result[0].value == pytest.approx(30e9, rel=0.1)
        assert result[2].value == pytest.approx(50e9, rel=0.1)
        assert result[5].value == pytest.approx(80e9, rel=0.1)

    async def test_aggregate_with_spread_out_forecasters(self) -> None:
        """
        Even with very different forecasters, the aggregate should display
        with clean percentile heights.
        """
        question = self._make_question()
        predictions = [
            self._make_forecaster_distribution(
                [
                    (0.1, 5e9),
                    (0.2, 10e9),
                    (0.4, 20e9),
                    (0.6, 30e9),
                    (0.8, 40e9),
                    (0.9, 50e9),
                ],
                question,
            ),
            self._make_forecaster_distribution(
                [
                    (0.1, 50e9),
                    (0.2, 60e9),
                    (0.4, 70e9),
                    (0.6, 80e9),
                    (0.8, 90e9),
                    (0.9, 95e9),
                ],
                question,
            ),
        ]

        aggregated = await NumericReport.aggregate_predictions(predictions, question)
        readable = NumericReport.make_readable_prediction(aggregated)
        assert "10.00%" in readable
        assert "90.00%" in readable

    def test_individual_forecaster_readable_unchanged(self) -> None:
        """Individual forecasters with <=10 percentiles still show all their values."""
        percentiles = [
            Percentile(value=10.0, percentile=0.1),
            Percentile(value=20.0, percentile=0.2),
            Percentile(value=40.0, percentile=0.4),
            Percentile(value=60.0, percentile=0.6),
            Percentile(value=80.0, percentile=0.8),
            Percentile(value=90.0, percentile=0.9),
        ]
        dist = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )
        readable = NumericReport.make_readable_prediction(dist)
        assert "10.00%" in readable
        assert "20.00%" in readable
        assert "40.00%" in readable
        assert "60.00%" in readable
        assert "80.00%" in readable
        assert "90.00%" in readable
        assert readable.count("chance of value below") == 6

    async def test_aggregate_readable_with_narrow_distribution(self) -> None:
        """
        When all forecasters predict a very narrow range relative to the question bounds,
        the aggregate should still show clean percentiles (not weird values from x-axis spacing).
        """
        question = self._make_question(lower_bound=0.0, upper_bound=1_000_000_000_000.0)
        predictions = [
            self._make_forecaster_distribution(
                [
                    (0.1, 99e9),
                    (0.2, 99.5e9),
                    (0.4, 100e9),
                    (0.6, 100.5e9),
                    (0.8, 101e9),
                    (0.9, 101.5e9),
                ],
                question,
            ),
            self._make_forecaster_distribution(
                [
                    (0.1, 99e9),
                    (0.2, 99.5e9),
                    (0.4, 100e9),
                    (0.6, 100.5e9),
                    (0.8, 101e9),
                    (0.9, 101.5e9),
                ],
                question,
            ),
        ]

        aggregated = await NumericReport.aggregate_predictions(predictions, question)
        readable = NumericReport.make_readable_prediction(aggregated)
        assert "10.00%" in readable
        assert "90.00%" in readable
