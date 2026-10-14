import logging

import pytest
from pydantic import ValidationError

from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.questions import (
    MultipleChoiceQuestion,
    QuestionState,
)


def create_test_mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Test Question",
        id_of_post=1234,
        id_of_question=5678,
        options=["Option A", "Option B", "Option C"],
        state=QuestionState.OPEN,
    )


def create_test_prediction() -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Option A", probability=0.3),
            PredictedOption(option_name="Option B", probability=0.5),
            PredictedOption(option_name="Option C", probability=0.2),
        ]
    )


def test_predicted_option_validation() -> None:
    # Valid probabilities
    PredictedOption(option_name="Test", probability=0.0)
    PredictedOption(option_name="Test", probability=1.0)
    PredictedOption(option_name="Test", probability=0.5)

    # Invalid probabilities
    with pytest.raises(ValidationError):
        PredictedOption(option_name="Test", probability=-0.1)

    with pytest.raises(ValidationError):
        PredictedOption(option_name="Test", probability=1.1)


def test_add_to_one_validation() -> None:
    with pytest.raises(ValidationError):
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.5),
            ]
        )
    with pytest.raises(ValidationError):
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.5),
                PredictedOption(option_name="Option B", probability=0.6),
                PredictedOption(option_name="Option C", probability=0.7),
            ]
        )
    with pytest.raises(ValidationError):
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.98),
                PredictedOption(option_name="Option B", probability=0.02),
                PredictedOption(option_name="Option C", probability=0.02),
            ]
        )

    PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Option A", probability=0.98),
            PredictedOption(option_name="Option B", probability=0.01),
            PredictedOption(option_name="Option C", probability=0.001),
        ]
    )


async def test_aggregate_predictions_with_same_options() -> None:
    predictions = [
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.3),
                PredictedOption(option_name="Option B", probability=0.5),
                PredictedOption(option_name="Option C", probability=0.2),
            ]
        ),
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.1),
                PredictedOption(option_name="Option B", probability=0.7),
                PredictedOption(option_name="Option C", probability=0.2),
            ]
        ),
    ]

    question = create_test_mc_question()
    result = await MultipleChoiceReport.aggregate_predictions(predictions, question)

    assert len(result.predicted_options) == 3
    option_probs = {
        opt.option_name: opt.probability for opt in result.predicted_options
    }
    assert option_probs["Option A"] == pytest.approx(0.2)  # (0.3 + 0.1) / 2
    assert option_probs["Option B"] == pytest.approx(0.6)  # (0.5 + 0.7) / 2
    assert option_probs["Option C"] == pytest.approx(0.2)  # (0.2 + 0.2) / 2


async def test_aggregate_predictions_with_different_options() -> None:
    predictions = [
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.5),
                PredictedOption(option_name="Option B", probability=0.5),
            ]
        ),
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.3),
                PredictedOption(option_name="Option C", probability=0.7),
            ]
        ),
    ]

    question = create_test_mc_question()
    with pytest.raises(Exception):
        await MultipleChoiceReport.aggregate_predictions(predictions, question)


def test_clamping_of_probabilities() -> None:
    predicted_option_list = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Option A", probability=0),
            PredictedOption(option_name="Option B", probability=1),
        ],
    )
    assert predicted_option_list.predicted_options[0].probability == 0.001
    assert predicted_option_list.predicted_options[1].probability == 0.999

    predicted_option_list = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Option A", probability=0.001),
            PredictedOption(option_name="Option B", probability=1),
            PredictedOption(option_name="Option C", probability=0),
        ],
    )
    assert predicted_option_list.predicted_options[0].probability == 0.001
    assert predicted_option_list.predicted_options[1].probability == pytest.approx(
        0.998
    )
    assert predicted_option_list.predicted_options[2].probability == 0.001

    predicted_option_list = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Option A", probability=0),
            PredictedOption(option_name="Option B", probability=1),
            PredictedOption(option_name="Option C", probability=0),
            PredictedOption(option_name="Option D", probability=0),
            PredictedOption(option_name="Option E", probability=0),
        ],
    )
    assert predicted_option_list.predicted_options[0].probability != 0
    assert predicted_option_list.predicted_options[1].probability != 1
    assert predicted_option_list.predicted_options[2].probability != 0
    assert predicted_option_list.predicted_options[3].probability != 0
    assert predicted_option_list.predicted_options[4].probability != 0


def test_probabilities_between_api_minimum_and_one_percent_are_kept() -> None:
    predicted_option_list = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Option A", probability=0.003),
            PredictedOption(option_name="Option B", probability=0.992),
            PredictedOption(option_name="Option C", probability=0.005),
        ],
    )
    assert predicted_option_list.to_dict() == pytest.approx(
        {"Option A": 0.003, "Option B": 0.992, "Option C": 0.005}
    )


def test_many_options_below_one_percent_do_not_raise() -> None:
    small_options = [
        PredictedOption(option_name=f"Option {index}", probability=0.005)
        for index in range(39)
    ]
    predicted_option_list = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Main Option", probability=0.805),
            *small_options,
        ],
    )
    assert predicted_option_list.predicted_options[0].probability == pytest.approx(
        0.805
    )
    for option in predicted_option_list.predicted_options[1:]:
        assert option.probability == pytest.approx(0.005)


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0005, 0.9995, 0.0005, 0.0],
        [0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0009, 0.0009, 0.0009, 0.497, 0.497],
        [0.0] * 40 + [1.0],
        [0.0005245095586030892, 0.9904266597797196],
        [5.199386370047876e-05, 0.9901402836987023],
    ],
)
def test_normalized_probabilities_stay_within_api_bounds(
    probabilities: list[float],
) -> None:
    predicted_option_list = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=f"Option {index}", probability=probability)
            for index, probability in enumerate(probabilities)
        ],
    )
    normalized = [
        option.probability for option in predicted_option_list.predicted_options
    ]
    assert sum(normalized) == pytest.approx(1)
    assert all(0.001 <= probability <= 0.999 for probability in normalized)


@pytest.mark.parametrize(
    "probabilities, should_warn",
    [
        ([0.3, 0.5, 0.2], False),
        ([0.1] * 10, False),
        ([0.3, 0.5, 0.205], True),
    ],
)
def test_sum_warning_only_logged_when_sum_is_not_one(
    probabilities: list[float], should_warn: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=f"Option {index}", probability=probability)
                for index, probability in enumerate(probabilities)
            ],
        )
    warned = any(
        "Sum of option probabilities" in record.message for record in caplog.records
    )
    assert warned == should_warn
