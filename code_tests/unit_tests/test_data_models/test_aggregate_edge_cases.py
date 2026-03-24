import numpy as np
import pytest

from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion, QuestionState


def _q(lb=0.0, ub=100e9, olb=True, oub=True):
    return NumericQuestion(
        id_of_post=1,
        id_of_question=1,
        question_text="T",
        background_info="",
        resolution_criteria="",
        fine_print="",
        state=QuestionState.OPEN,
        upper_bound=ub,
        lower_bound=lb,
        open_upper_bound=oub,
        open_lower_bound=olb,
        zero_point=None,
    )


def _d(pvs, q):
    return NumericDistribution.from_question(
        [Percentile(percentile=p, value=v) for p, v in pvs], q
    )


S1 = [(0.1, 30e9), (0.2, 40e9), (0.4, 50e9), (0.6, 60e9), (0.8, 70e9), (0.9, 80e9)]
S2 = [(0.1, 35e9), (0.2, 45e9), (0.4, 55e9), (0.6, 65e9), (0.8, 75e9), (0.9, 85e9)]


async def test_aggregate_shows_six_lines():
    q = _q()
    agg = await NumericReport.aggregate_predictions([_d(S1, q), _d(S2, q)], q)
    assert (
        NumericReport.make_readable_prediction(agg).count("chance of value below") == 6
    )


async def test_single_forecaster_aggregate():
    q = _q()
    agg = await NumericReport.aggregate_predictions([_d(S1, q)], q)
    readable = NumericReport.make_readable_prediction(agg)
    assert "10.00%" in readable
    assert "90.00%" in readable
    r = agg.get_percentiles_at_target_heights()
    assert r[0].value == pytest.approx(30e9, rel=0.1)
    assert r[5].value == pytest.approx(80e9, rel=0.1)


async def test_values_monotonically_increasing():
    q = _q()
    agg = await NumericReport.aggregate_predictions([_d(S1, q), _d(S2, q)], q)
    r = agg.get_percentiles_at_target_heights()
    for i in range(len(r) - 1):
        assert r[i].value < r[i + 1].value


async def test_closed_bounds():
    q = _q(olb=False, oub=False)
    agg = await NumericReport.aggregate_predictions([_d(S1, q), _d(S2, q)], q)
    readable = NumericReport.make_readable_prediction(agg)
    assert "10.00%" in readable


async def test_nvidia_values_near_median():
    q = _q()
    fd = [
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
    agg = await NumericReport.aggregate_predictions([_d(p, q) for p in fd], q)
    r = agg.get_percentiles_at_target_heights()
    for i in range(6):
        expected = np.median([pvs[i][1] for pvs in fd])
        assert r[i].value == pytest.approx(expected, rel=0.15)


async def test_asymmetric_skew():
    q = _q()
    a = [(0.1, 5e9), (0.2, 8e9), (0.4, 10e9), (0.6, 12e9), (0.8, 20e9), (0.9, 50e9)]
    b = [(0.1, 6e9), (0.2, 9e9), (0.4, 11e9), (0.6, 13e9), (0.8, 22e9), (0.9, 55e9)]
    agg = await NumericReport.aggregate_predictions([_d(a, q), _d(b, q)], q)
    readable = NumericReport.make_readable_prediction(agg)
    assert "10.00%" in readable
    r = agg.get_percentiles_at_target_heights()
    for i in range(len(r) - 1):
        assert r[i].value < r[i + 1].value
