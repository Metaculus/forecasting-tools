import pytest

from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)


@pytest.mark.parametrize(
    "reasoning, options, expected_probabilities",
    [
        (
            """
            Option A: 30
            Option B: 40
            Option C: 30
            """,
            ["Option A", "Option B", "Option C"],
            [0.3, 0.4, 0.3],
        ),
        (
            """
            Option Financial_Growth: 50
            Option consumer_demand: 25%
            Option_Government_Spending: 25
            """,
            [" Financial Growth ", "Consumer Demand", "Government Spending"],
            [0.5, 0.25, 0.25],
        ),
        (
            """
            Introduction text.
            Option X: 20
            More stuff.
            Option Y: 30
            Some more details.
            Option Z: 50
            Final notes.
            """,
            ["X", "Y", "Option Z"],
            [0.2, 0.3, 0.5],
        ),
        (
            """
            In this forecast, we consider three options: Blue, Green, and Yellow.
            I think that Blue is 30%
            And Option Yellow is 20%

            Option Blue: 20
            Option Green: 30
            Option Yellow: 50
            """,
            ["Blue", "Green", "Yellow"],
            [0.2, 0.3, 0.5],
        ),
        (
            """
            Forecast breakdown:
            Option One: 33.33
            Option Two: 33.33
            Option Three: 33.34
            """,
            ["One", "Two", "Three"],
            [33.33 / 100, 33.33 / 100, 33.34 / 100],
        ),
        (
            """
            Option One: 0.5
            Option Two: 0.2
            Option Three: 0.3
            """,
            ["One", "Two", "Three"],
            [0.5, 0.2, 0.3],
        ),
        (
            """
            Option One: 0.05
            Option Two: 0.02
            Option Three: 0.03
            """,
            ["One", "Two", "Three"],
            [0.5, 0.2, 0.3],  # Probabilities are normalized
        ),
        (
            """
            Option One: 5
            Option Two: 2
            Option Three: 3
            """,
            ["One", "Two", "Three"],
            [0.5, 0.2, 0.3],  # Probabilities are normalized
        ),
        (
            """
            I'm thinking that Option A, option B and option C are 50%, 20% and 30% respectively.

            Option A: 50%
            Option B: 20%
            Option C: 30%

            These are my guesses for Blue, Green and Yellow matching to A, B and C.
            """,
            ["Blue", "Green", "Yellow"],
            [
                0.5,
                0.2,
                0.3,
            ],  # If probabilieis given as A B C, the ordering is used to match options.
        ),
    ],
)
def test_multiple_choice_extraction_success(
    reasoning: str, options: list[str], expected_probabilities: list[float]
) -> None:
    print("Starting test")
    predicted_option_list: PredictedOptionList = (
        PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, options
        )
    )
    predicted_options = predicted_option_list.predicted_options
    assert len(predicted_options) == len(options)
    for expected_option, expected_probability, predicted_option in zip(
        options, expected_probabilities, predicted_options
    ):
        assert predicted_option.option_name == expected_option
        assert predicted_option.probability == pytest.approx(
            expected_probability
        )


@pytest.mark.parametrize(
    "reasoning, options",
    [
        (
            """
            Option OnlyOne: 60
            """,
            ["Option OnlyOne", "Option Missing"],
        ),
        (
            """
            Option A: 30
            Option B: 40
            Option Yellow: 30
            """,
            ["Blue", "Green", "Yellow"],
        ),
    ],
)
def test_multiple_choice_extraction_failure(
    reasoning: str, options: list[str]
) -> None:
    with pytest.raises(ValueError):
        PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, options
        )


def create_numeric_question(
    magnitude_units: str | None = None,
) -> NumericQuestion:
    if magnitude_units is None:
        question_text = (
            "How much will the stock market be worth in 2026? (exact value)"
        )
    else:
        question_text = f"How much will the stock market be worth in 2026 in {magnitude_units}?"

    return NumericQuestion(
        question_text=question_text,
        upper_bound=1,
        lower_bound=0,
        open_upper_bound=True,
        open_lower_bound=True,
    )


@pytest.mark.parametrize(
    "gpt_response, expected_percentiles, question",
    [
        (
            """
            Percentile 20: 10
            Percentile 40: 20
            Percentile 60: 30
            """,
            [
                Percentile(value=10, percentile=0.2),
                Percentile(value=20, percentile=0.4),
                Percentile(value=30, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: 1.123
            Percentile 40: 2.123
            Percentile 60: 3.123
            """,
            [
                Percentile(value=1.123, percentile=0.2),
                Percentile(value=2.123, percentile=0.4),
                Percentile(value=3.123, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: -20
            Percentile 40: -10.45
            Percentile 60: 30
            """,
            [
                Percentile(value=-20, percentile=0.2),
                Percentile(value=-10.45, percentile=0.4),
                Percentile(value=30, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: -$20
            Percentile 40: -$10.45
            Percentile 60: $30
            """,
            [
                Percentile(value=-20, percentile=0.2),
                Percentile(value=-10.45, percentile=0.4),
                Percentile(value=30, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: -20 dollars
            Percentile 40: -10dollars
            Percentile 60: - 5.37 dollars
            """,
            [
                Percentile(value=-20, percentile=0.2),
                Percentile(value=-10, percentile=0.4),
                Percentile(value=-5.37, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: 1,000,000
            Percentile 40: 2,000,000
            Percentile 60: 3,000,000
            """,
            [
                Percentile(value=1000000, percentile=0.2),
                Percentile(value=2000000, percentile=0.4),
                Percentile(value=3000000, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: $1,000,000
            Percentile 40: $2,000,000
            Percentile 60: $3,000,000
            """,
            [
                Percentile(value=1000000, percentile=0.2),
                Percentile(value=2000000, percentile=0.4),
                Percentile(value=3000000, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: 1,000,000
            Percentile 40: 2,000,000.454
            Percentile 60: 3,000,000.00
            """,
            [
                Percentile(value=1000000, percentile=0.2),
                Percentile(value=2000000.454, percentile=0.4),
                Percentile(value=3000000.00, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Percentile 20: 1 million
            Percentile 40: 2.1m
            Percentile 60: 3,000 million
            """,
            [
                Percentile(value=1, percentile=0.2),
                Percentile(value=2.1, percentile=0.4),
                Percentile(value=3000, percentile=0.6),
            ],
            create_numeric_question(magnitude_units="millions"),
        ),
        (
            """
            Notes before hand including numbers like 3yr and 2,000 million and 70%.
            Percentile 20: 1,000,000
            Percentile 40: 2,000,000.454
            Percentile 60: 3,000,000.00
            Notes afterwards including numbers like 3yr and 2,000 million and 70%.
            More Notes including numbers like 3yr and 2,000 million and 70%.
            """,
            [
                Percentile(value=1000000, percentile=0.2),
                Percentile(value=2000000.454, percentile=0.4),
                Percentile(value=3000000.00, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (
            """
            Notes before hand including numbers like 3yr and 2,000 million and 70%.
            Percentile 10: 0.5 million
            Percentile 20: 1 million
            Percentile 40: 2.1m
            Percentile 60: 3,000 million
            Percentile 80: 4,000 million
            Percentile 90: 5,000 million
            Notes afterwards including numbers like 3yr and 2,000 million and 70%.
            More Notes including numbers like 3yr and 2,000 million and 70%.
            """,
            [
                Percentile(value=0.5, percentile=0.1),
                Percentile(value=1, percentile=0.2),
                Percentile(value=2.1, percentile=0.4),
                Percentile(value=3000, percentile=0.6),
                Percentile(value=4000, percentile=0.8),
                Percentile(value=5000, percentile=0.9),
            ],
            create_numeric_question(magnitude_units="millions"),
        ),
        # (
        #     """
        #     Percentile 20: 1,000,000
        #     Percentile 40: 2,000,000.454
        #     Percentile 60: 3,000,000.00
        #     """,
        #     [
        #         Percentile(value=1, percentile=0.2),
        #         Percentile(value=2.000000454, percentile=0.4),
        #         Percentile(value=3, percentile=0.6),
        #     ],
        #     create_numeric_question(magnitude_units="millions"),
        # ),
        # (
        #     """
        #     Percentile 20: 2.3E-2
        #     Percentile 40: 1.2e2
        #     Percentile 60: 3.1x10^2
        #     """,
        #     [
        #         Percentile(value=0.023, percentile=0.2),
        #         Percentile(value=120, percentile=0.4),
        #         Percentile(value=310, percentile=0.6),
        #     ],
        #     create_numeric_question(),
        # ),
    ],
)
def test_numeric_parsing(
    gpt_response: str,
    expected_percentiles: list[Percentile],
    question: NumericQuestion,
) -> None:
    numeric_distribution = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
        gpt_response, question
    )
    for declared_percentile, expected_percentile in zip(
        numeric_distribution.declared_percentiles, expected_percentiles
    ):
        assert declared_percentile.value == pytest.approx(
            expected_percentile.value
        )
        assert declared_percentile.percentile == pytest.approx(
            expected_percentile.percentile
        )
