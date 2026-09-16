from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.data_models.questions import DiscreteQuestion


_BASE = dict(
    question_text="How many X will happen?",
    resolution_criteria="Counts of X reported by source.",
    fine_print="",
    background_information="",
    expected_resolution_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
)


def _make_discrete(**overrides):
    return SimpleQuestion(
        **{
            **_BASE,
            "question_type": "discrete",
            "open_lower_bound": False,
            "open_upper_bound": False,
            **overrides,
        }
    )


def test_coworker_canonical_example_0_to_10():
    sq = _make_discrete(min_value=0, max_value=10, step=1)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    assert isinstance(mq, DiscreteQuestion)
    assert mq.lower_bound == -0.5
    assert mq.upper_bound == 10.5
    assert mq.nominal_lower_bound == 0
    assert mq.nominal_upper_bound == 10
    assert mq.cdf_size == 12  # inbound_outcome_count (11) + 1


def test_round_trip_preserves_step():
    sq = _make_discrete(min_value=0, max_value=10, step=1)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    back = SimpleQuestion.full_questions_to_simple_questions([mq])[0]
    assert back.question_type == "discrete"
    assert back.step == 1.0
    assert back.min_value == 0
    assert back.max_value == 10


def test_round_trip_preserves_all_fields_non_integer_step():
    """Full SimpleQuestion -> DiscreteQuestion -> SimpleQuestion round-trip
    with a non-1 step that actually exercises the reverse float division."""
    sq = _make_discrete(
        min_value=0,
        max_value=100,
        step=5,
        open_lower_bound=False,
        open_upper_bound=True,
    )
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    back = SimpleQuestion.full_questions_to_simple_questions([mq])[0]
    assert back.question_type == "discrete"
    assert back.step == pytest.approx(5)
    assert back.min_value == pytest.approx(0)
    assert back.max_value == pytest.approx(100)
    assert back.open_lower_bound is False
    assert back.open_upper_bound is True
    assert back.question_text == sq.question_text
    assert back.resolution_criteria == sq.resolution_criteria
    assert back.expected_resolution_date == sq.expected_resolution_date
    assert back.options == []


def test_round_trip_with_float_step():
    """Float step (0.1) — exercises float-arithmetic in both directions."""
    sq = _make_discrete(min_value=0, max_value=1, step=0.1)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    back = SimpleQuestion.full_questions_to_simple_questions([mq])[0]
    assert back.step == pytest.approx(0.1)
    assert back.min_value == pytest.approx(0)
    assert back.max_value == pytest.approx(1)


def test_rejects_max_equal_to_min():
    with pytest.raises(ValidationError, match="must be greater than"):
        _make_discrete(min_value=5, max_value=5, step=1)


def test_back_compute_when_nominal_bounds_missing():
    """API may return a DiscreteQuestion without nominal_*_bound populated;
    full_questions_to_simple_questions should recover step + nominal bounds
    from actual bounds and cdf_size."""
    sq = _make_discrete(min_value=0, max_value=10, step=1)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    mq.nominal_lower_bound = None
    mq.nominal_upper_bound = None
    back = SimpleQuestion.full_questions_to_simple_questions([mq])[0]
    assert back.question_type == "discrete"
    assert back.step == 1.0
    assert back.min_value == 0
    assert back.max_value == 10


def test_binned_percentage_step_5():
    sq = _make_discrete(min_value=0, max_value=100, step=5)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    assert mq.cdf_size == 22  # 21 outcomes + 1
    assert mq.lower_bound == -2.5
    assert mq.upper_bound == 102.5


def test_tenths_step_with_float_arithmetic():
    sq = _make_discrete(min_value=0, max_value=1, step=0.1)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    assert mq.cdf_size == 12  # 11 outcomes + 1


def test_min_outcomes_edge_3_outcomes():
    sq = _make_discrete(min_value=0, max_value=10, step=5)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    assert mq.cdf_size == 4


def test_max_outcomes_edge_200_outcomes():
    sq = _make_discrete(min_value=0, max_value=199, step=1)
    mq = SimpleQuestion.simple_questions_to_metaculus_questions([sq])[0]
    assert mq.cdf_size == 201


def test_rejects_step_too_large_only_two_outcomes():
    with pytest.raises(ValidationError, match="step too large"):
        _make_discrete(min_value=0, max_value=10, step=10)


def test_rejects_step_too_small_over_200_outcomes():
    with pytest.raises(ValidationError, match="step too small"):
        _make_discrete(min_value=0, max_value=100, step=0.1)


def test_rejects_non_integer_divisor():
    with pytest.raises(ValidationError, match="must be an integer"):
        _make_discrete(min_value=0, max_value=10, step=3)


def test_rejects_negative_step():
    with pytest.raises(ValidationError, match="step must be positive"):
        _make_discrete(min_value=0, max_value=10, step=-1)


def test_rejects_missing_step():
    with pytest.raises(ValidationError, match="step must be provided"):
        _make_discrete(min_value=0, max_value=10)


def test_rejects_missing_bounds():
    with pytest.raises(ValidationError, match="Upper bound must be provided"):
        _make_discrete(min_value=0, step=1)


def test_rejects_step_on_binary():
    with pytest.raises(ValidationError, match="step must not be provided"):
        SimpleQuestion(
            **{
                **_BASE,
                "question_type": "binary",
                "step": 1,
            }
        )


def test_rejects_step_on_numeric():
    with pytest.raises(ValidationError, match="step must not be provided"):
        SimpleQuestion(
            **{
                **_BASE,
                "question_type": "numeric",
                "min_value": 0,
                "max_value": 100,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "step": 1,
            }
        )


def test_rejects_step_on_multiple_choice():
    with pytest.raises(ValidationError, match="step must not be provided"):
        SimpleQuestion(
            **{
                **_BASE,
                "question_type": "multiple_choice",
                "options": ["a", "b"],
                "step": 1,
            }
        )
