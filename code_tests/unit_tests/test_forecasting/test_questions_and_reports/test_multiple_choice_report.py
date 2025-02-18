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
    result = await MultipleChoiceReport.aggregate_predictions(
        predictions, question
    )

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
