"""
Regression tests anchored to open numeric-distribution issues:

- #166: "add more robust solution for _check_and_update_repeating_values"
- #209: "Investigate numeric distribution percentiles being placed to close"
- #210: "Occasional spikes on sides of numeric distribution"
- #212: "Numeric forecasting has problems when bots forecast the same
  value for percentiles"

Every test passes on current code and locks in behaviour that the issues
questioned, so that a future fix (or regression) is checked against it.
"""

import logging

import pytest

from forecasting_tools.data_models.numeric_report import (
    NumericDefaults,
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion, QuestionState

logger = logging.getLogger(__name__)


def _make_distribution(
    percentiles: list[tuple[float, float]],
    open_upper_bound: bool = False,
    open_lower_bound: bool = False,
    upper_bound: float = 100.0,
    lower_bound: float = 0.0,
    zero_point: float | None = None,
    cdf_size: int | None = None,
) -> NumericDistribution:
    return NumericDistribution(
        declared_percentiles=[
            Percentile(value=value, percentile=percentile)
            for value, percentile in percentiles
        ],
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
        cdf_size=cdf_size,
    )


def _pmf(distribution: NumericDistribution) -> list[float]:
    cdf = distribution.get_cdf()
    return [
        percentile.percentile - (cdf[i - 1].percentile if i > 0 else 0.0)
        for i, percentile in enumerate(cdf)
    ]


def _adjacent_height_diffs(distribution: NumericDistribution) -> list[float]:
    cdf = distribution.get_cdf()
    return [cdf[i + 1].percentile - cdf[i].percentile for i in range(len(cdf) - 1)]


class TestRepeatingValuesRobustness:
    """Issue #166: the repeating-values repair must not corrupt the list."""

    def test_repeating_values_with_ample_separation_keep_order(self) -> None:
        # Locks the normal path: well-separated repeats stay sorted and are
        # moved by at most the documented 1e-6 nudge.
        distribution = _make_distribution(
            [
                (10.0, 0.1),
                (50.0, 0.4),
                (50.0, 0.6),
                (90.0, 0.9),
            ]
        )
        values = [p.value for p in distribution.declared_percentiles]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))
        assert all(abs(value - 50.0) <= 1e-6 for value in values[1:3])


class TestSameValuePercentiles:
    """Issue #212: many percentiles sharing one value."""

    def test_bin_count_matching_range_keeps_mass_in_correct_bucket(self) -> None:
        # Problem 1 of #212 was about the number of bins (200) fitting the
        # range (0-20) exactly. test_discrete_distribution_repeated_value
        # covers cdf_size=21; this extends it to the default 201-point CDF,
        # where each bucket is 0.1 wide and the mass at 12 must land in the
        # bucket ending at 12.0 rather than the one after it.
        distribution = _make_distribution(
            [
                (12, 0.1),
                (12, 0.2),
                (12, 0.4),
                (12, 0.6),
                (12, 0.8),
                (12, 0.9),
            ],
            upper_bound=20.0,
        )
        pmf = _pmf(distribution)
        bucket_ending_at_12 = 120  # cdf value at index 120 is 12.0
        assert distribution.get_cdf()[bucket_ending_at_12].value == pytest.approx(12.0)
        assert pmf[bucket_ending_at_12] > 0.15
        assert pmf[bucket_ending_at_12] > pmf[bucket_ending_at_12 - 1]
        assert pmf[bucket_ending_at_12] > pmf[bucket_ending_at_12 + 1]

    @pytest.mark.parametrize(
        "percentiles",
        [
            # Sharply peaked forecast
            [(49.9999, 0.1), (50.0, 0.5), (50.0001, 0.9)],
            # Almost all mass at the top of the range ("0% for multiple
            # times in a row" across most of the CDF, per #212 problem 2)
            [(98.0, 0.001), (99.0, 0.5), (99.9, 0.999)],
            # Almost all mass at the bottom of the range
            [(0.1, 0.001), (1.0, 0.5), (2.0, 0.999)],
            # Very narrow normal-shaped forecast
            [
                (49.99, 0.1),
                (49.999, 0.3),
                (50.0, 0.5),
                (50.001, 0.7),
                (50.01, 0.9),
            ],
        ],
    )
    def test_get_cdf_heights_strictly_increasing_for_concentrated_forecasts(
        self, percentiles: list[tuple[float, float]]
    ) -> None:
        # Problem 2 of #212 showed a workflow dying inside get_cdf with
        # "Percentiles must be in strictly increasing order". The generated
        # CDF must stay strictly increasing (and respect the model's own
        # 5e-05 spacing floor) even for concentrated forecasts.
        distribution = _make_distribution(percentiles)
        diffs = _adjacent_height_diffs(distribution)
        assert all(diff > 0 for diff in diffs)
        assert min(diffs) >= 5e-05

    async def test_aggregate_identical_near_bound_forecasters(self) -> None:
        # The workflow shape behind #212 problem 2: several bots whose
        # forecasts sit near the top of the range, so their CDFs spend most
        # bins near 0%. Aggregating them must not raise a ValidationError.
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
        predictions = [
            NumericDistribution.from_question(
                [
                    Percentile(value=90.0, percentile=0.5),
                    Percentile(value=95.0, percentile=0.9),
                ],
                question,
            )
            for _ in range(3)
        ]
        aggregated = await NumericReport.aggregate_predictions(predictions, question)
        diffs = _adjacent_height_diffs(aggregated)
        assert all(diff > 0 for diff in diffs)


class TestPercentileSpacing:
    """Issue #209: percentiles placed too close together."""

    def test_declared_percentile_heights_too_close_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="too close"):
            _make_distribution([(10.0, 0.1), (20.0, 0.10001), (30.0, 0.9)])

    @pytest.mark.parametrize(
        "percentiles,kwargs",
        [
            (
                [
                    (10.0, 0.1),
                    (20.0, 0.2),
                    (30.0, 0.3),
                    (40.0, 0.4),
                    (50.0, 0.5),
                    (60.0, 0.6),
                    (70.0, 0.7),
                    (80.0, 0.8),
                    (90.0, 0.9),
                ],
                {},
            ),
            # Long plateau: declared heights move only 0.45 -> 0.55
            ([(1.0, 0.45), (99.0, 0.55)], {}),
            # Heights stay near zero across the whole range
            (
                [(10.0, 0.01), (50.0, 0.02), (90.0, 0.03)],
                {"open_lower_bound": True, "open_upper_bound": True},
            ),
        ],
    )
    def test_generated_cdf_respects_minimum_height_spacing(
        self,
        percentiles: list[tuple[float, float]],
        kwargs: dict,
    ) -> None:
        # Whatever a bot declares (within validation), the generated CDF
        # must itself satisfy the 5e-05 minimum spacing the model demands
        # of any NumericDistribution.
        distribution = _make_distribution(percentiles, **kwargs)
        diffs = _adjacent_height_diffs(distribution)
        assert min(diffs) >= 5e-05


class TestEdgeSpikes:
    """Issue #210: spikes on the sides of the distribution."""

    def test_no_edge_spike_for_smooth_closed_bound_forecast(self) -> None:
        # An evenly spread forecast on closed bounds should produce an
        # essentially flat PMF, edges included: nothing in the bound
        # handling may pile extra mass into the first or last buckets.
        distribution = _make_distribution(
            [
                (10.0, 0.1),
                (20.0, 0.2),
                (30.0, 0.3),
                (40.0, 0.4),
                (50.0, 0.5),
                (60.0, 0.6),
                (70.0, 0.7),
                (80.0, 0.8),
                (90.0, 0.9),
            ]
        )
        pmf = _pmf(distribution)
        interior = sorted(pmf[5:-5])
        median_interior = interior[len(interior) // 2]
        edge_bins = pmf[1:4] + pmf[-4:-1]
        assert all(bin_mass <= 1.5 * median_interior for bin_mass in edge_bins)

    def test_no_edge_spike_for_log_scaled_forecast(self) -> None:
        distribution = _make_distribution(
            [
                (10.0, 0.1),
                (100.0, 0.3),
                (1000.0, 0.5),
                (10000.0, 0.7),
                (100000.0, 0.9),
            ],
            lower_bound=1.0,
            upper_bound=100_000_000.0,
            zero_point=0.0,
        )
        pmf = _pmf(distribution)
        interior_max = max(pmf[5:-5])
        assert pmf[1] <= interior_max
        assert pmf[2] <= interior_max
        assert pmf[-2] <= interior_max

    def test_declared_edge_mass_is_capped_by_standardization(self) -> None:
        # When a bot genuinely declares heavy mass next to a bound, the
        # standardized CDF must still respect the per-bin cap everywhere,
        # so the visible "spike" stays bounded by the documented maximum.
        distribution = _make_distribution([(1.0, 0.4), (50.0, 0.5), (99.0, 0.6)])
        pmf = _pmf(distribution)
        cap = NumericDefaults.get_max_pmf_value(len(pmf), include_wiggle_room=True)
        assert max(pmf) <= cap + 1e-9
