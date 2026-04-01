"""
This  file is AI Generated. If it causes problems for a human we can probably just delete it.
"""

import numpy as np
import pytest

from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion, QuestionState


def _make_question(
    lower_bound: float = 0.0,
    upper_bound: float = 100e9,
    open_lower_bound: bool = True,
    open_upper_bound: bool = True,
) -> NumericQuestion:
    return NumericQuestion(
        id_of_post=1,
        id_of_question=1,
        question_text="Test question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        state=QuestionState.OPEN,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        zero_point=None,
    )


def _make_distribution(
    percentile_value_pairs: list[tuple[float, float]],
    question: NumericQuestion,
) -> NumericDistribution:
    percentiles = [Percentile(percentile=p, value=v) for p, v in percentile_value_pairs]
    return NumericDistribution.from_question(percentiles, question)


CENTERED_DISTRIBUTION = [
    (0.1, 30e9),
    (0.2, 40e9),
    (0.4, 50e9),
    (0.6, 60e9),
    (0.8, 70e9),
    (0.9, 80e9),
]
SHIFTED_DISTRIBUTION = [
    (0.1, 35e9),
    (0.2, 45e9),
    (0.4, 55e9),
    (0.6, 65e9),
    (0.8, 75e9),
    (0.9, 85e9),
]


async def test_aggregate_shows_six_lines() -> None:
    question = _make_question()
    predictions = [
        _make_distribution(CENTERED_DISTRIBUTION, question),
        _make_distribution(SHIFTED_DISTRIBUTION, question),
    ]
    aggregated = await NumericReport.aggregate_predictions(predictions, question)
    readable = NumericReport.make_readable_prediction(aggregated)
    assert readable.count("chance of value below") == 6


async def test_single_forecaster_aggregate() -> None:
    question = _make_question()
    predictions = [_make_distribution(CENTERED_DISTRIBUTION, question)]
    aggregated = await NumericReport.aggregate_predictions(predictions, question)
    readable = NumericReport.make_readable_prediction(aggregated)
    assert "10.00%" in readable
    assert "90.00%" in readable
    result = aggregated.get_percentiles_at_target_heights()
    assert result[0].value == pytest.approx(30e9, rel=0.1)
    assert result[5].value == pytest.approx(80e9, rel=0.1)


async def test_values_monotonically_increasing() -> None:
    question = _make_question()
    predictions = [
        _make_distribution(CENTERED_DISTRIBUTION, question),
        _make_distribution(SHIFTED_DISTRIBUTION, question),
    ]
    aggregated = await NumericReport.aggregate_predictions(predictions, question)
    result = aggregated.get_percentiles_at_target_heights()
    for i in range(len(result) - 1):
        assert result[i].value < result[i + 1].value


async def test_closed_bounds() -> None:
    question = _make_question(open_lower_bound=False, open_upper_bound=False)
    predictions = [
        _make_distribution(CENTERED_DISTRIBUTION, question),
        _make_distribution(SHIFTED_DISTRIBUTION, question),
    ]
    aggregated = await NumericReport.aggregate_predictions(predictions, question)
    readable = NumericReport.make_readable_prediction(aggregated)
    assert "10.00%" in readable


async def test_nvidia_values_near_median() -> None:
    question = _make_question()
    forecaster_percentile_values = [
        [(0.1, 55e9), (0.2, 58e9), (0.4, 61e9), (0.6, 63e9), (0.8, 66e9), (0.9, 70e9)],
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
        [(0.1, 55e9), (0.2, 58e9), (0.4, 60e9), (0.6, 62e9), (0.8, 65e9), (0.9, 68e9)],
    ]
    predictions = [
        _make_distribution(pvs, question) for pvs in forecaster_percentile_values
    ]
    aggregated = await NumericReport.aggregate_predictions(predictions, question)
    result = aggregated.get_percentiles_at_target_heights()
    for i in range(6):
        expected = np.median([pvs[i][1] for pvs in forecaster_percentile_values])
        assert result[i].value == pytest.approx(expected, rel=0.15)


async def test_asymmetric_skew() -> None:
    question = _make_question()
    right_skewed_a = [
        (0.1, 5e9),
        (0.2, 8e9),
        (0.4, 10e9),
        (0.6, 12e9),
        (0.8, 20e9),
        (0.9, 50e9),
    ]
    right_skewed_b = [
        (0.1, 6e9),
        (0.2, 9e9),
        (0.4, 11e9),
        (0.6, 13e9),
        (0.8, 22e9),
        (0.9, 55e9),
    ]
    predictions = [
        _make_distribution(right_skewed_a, question),
        _make_distribution(right_skewed_b, question),
    ]
    aggregated = await NumericReport.aggregate_predictions(predictions, question)
    readable = NumericReport.make_readable_prediction(aggregated)
    assert "10.00%" in readable
    result = aggregated.get_percentiles_at_target_heights()
    for i in range(len(result) - 1):
        assert result[i].value < result[i + 1].value
