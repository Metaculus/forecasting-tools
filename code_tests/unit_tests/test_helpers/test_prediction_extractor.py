import pytest

from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from forecasting_tools.helpers.prediction_extractor import PredictionExtractor


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
            Option Government_Spending: 25
            """,
            [" Financial Growth ", "Consumer Demand", "Government Spending"],
            [0.5, 0.25, 0.25],
        ),
        (
            """
            Option_Financial_Growth: 50
            Option_consumer_demand: 25%
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
            ["X", "Y", "Z"],
            [0.2, 0.3, 0.5],
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
            ["Option X", "Option Y", "Option Z"],
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
            In this forecast, we consider three options: Blue, Green, and Yellow.
            I think that Blue is 30%
            And Option Yellow is 20%

            Option Yellow: 50
            Option Blue: 20
            Option Green: 30
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
        (  # Check Probabilities are normalized (off by 0.01)
            """
            Option One: 0.33
            Option Two: 0.33
            Option Three: 0.33
            """,
            ["One", "Two", "Three"],
            [1 / 3, 1 / 3, 1 / 3],
        ),
        (  # Check Probabilities are normalized (off by 0.01)
            """
            Option One: 0.497
            Option Two: 0.247
            Option Three: 0.247
            """,
            ["One", "Two", "Three"],
            [0.501, 0.249, 0.249],
        ),
        (  # If probabilieis given as A B C, the ordering is used to match options.
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
            ],
        ),
        (
            """
            Therefore, I'll place the highest probability on '1' mention, followed closely by '2-3'. I'll retain a moderate probability for '0' reflecting the status quo possibility and a small probability for '4 or more' to account for the unexpected scenario.
            (Final Probabilities)
            0: 0.25
            1: 0.40
            2-3: 0.30
            4 or more: 0.05
            """,
            ["0", "1", "2-3", "4 or more"],
            [0.25, 0.40, 0.30, 0.05],
        ),
        (
            """
            Therefore, I'll place the highest probability on '1' mention, followed closely by '2-3'. I'll retain a moderate probability for '0' reflecting the status quo possibility and a small probability for '4 or more' to account for the unexpected scenario.
            (Final Probabilities)
            1: 0.4
            0.2: 0.3
            0.3: 0.2
            0.4: 0.1
            """,
            ["1", "0.2", "0.3", "0.4"],
            [0.4, 0.3, 0.2, 0.1],
        ),
        (
            """
            Therefore, I'll place the highest probability on '1' mention, followed closely by '2-3'. I'll retain a moderate probability for '0' reflecting the status quo possibility and a small probability for '4 or more' to account for the unexpected scenario.
            (Final Probabilities)
            Option 1: 0.4
            Option 0.2: 0.3
            Option 0.3: 0.2
            Option 0.4: 0.1
            """,
            ["1", "0.2", "0.3", "0.4"],
            [0.4, 0.3, 0.2, 0.1],
        ),
        (  # Test case insensitive
            """
            option blue: 0.3
            option green: 0.2
            option yellow: 0.5
            """,
            ["BLUE", "GREEN", "YELLOW"],
            [0.3, 0.2, 0.5],
        ),
        (  # Out of order
            """
            Option A: 30
            Option C: 30
            Option B: 40
            """,
            ["Blue", "Green", "Yellow"],
            [0.3, 0.4, 0.3],
        ),
        (  # Test bullet points
            """
            - '0': 20
            - '1': 70
            - '2': 10
            """,
            ["0", "1", "2"],
            [0.2, 0.7, 0.1],
        ),
        (  # Test special characters (ñ)
            """
            **Final Probabilities**:
            - Daemon Wayans Jr.: 15%
            - David Schwimmer: 10%
            - George Lopez: 15%
            - Jude Law: 5%
            - Sam McCarthy: 10%
            - Xolo Maridueña: 45%
            """,
            [
                "Daemon Wayans Jr.",
                "David Schwimmer",
                "George Lopez",
                "Jude Law",
                "Sam McCarthy",
                "Xolo Mariduena",
            ],
            [0.15, 0.10, 0.15, 0.05, 0.10, 0.45],
        ),
        (  # Test option names within each other
            """
            Stuff including Two and Greater than Two.

            Two: 20
            Greater than Two: 70
            Greater than: 10
            """,
            ["Two", "Greater than Two", "Greater than"],
            [0.2, 0.7, 0.1],
        ),
        (
            """
            Stuff including Two and Greater than Two.

            - Two: 20
            - Greater than Two: 70
            - Greater than: 10
            """,
            ["Two", "Greater than Two", "Greater than"],
            [0.2, 0.7, 0.1],
        ),
        (
            """
            Zero to One: 20
            Exactly Two: 70
            Greater than Two: 10
            """,
            ["Zero to One", "Exactly Two", "Greater than Two"],
            [0.2, 0.7, 0.1],
        ),
        (  # Test special characters
            """
            - '≥0 and ≤20.0': 20
            - '>20.0 and <35.0': 70
            - '≥35.0': 10
            """,
            ["≥0 and ≤20.0", ">20.0 and <35.0", "≥35.0"],
            [0.2, 0.7, 0.1],
        ),
        (  # Test special characters
            """
            - "12.00 € or under": 1
            - "Greater than 12.00 € and less than or equal to 15.00 €": 79
            - "Greater than 15.00 €": 20
            """,
            [
                "12.00 € or under",
                "Greater than 12.00 € and less than or equal to 15.00 €",
                "Greater than 15.00 €",
            ],
            [0.01, 0.79, 0.2],
        ),
        # (  # comma numbers and small percentages
        #     """
        #     - "0": 0.5%
        #     - "1": 99%
        #     - "2-100": 0.2%
        #     - "100-1,000": 0.2%
        #     - "1,001": 0.1%
        #     """,
        #     ["0", "1", "2-100", "100-1,000", "1,001"],
        #     [0.005, 0.99, 0.002, 0.002, 0.001],
        # ),
        (  # comma numbers
            """
            - "0": 2%
            - "1": 93%
            - "2-100": 2%
            - "100-1,000": 2%
            - "1,001": 1%
            """,
            ["0", "1", "2-100", "100-1,000", "1,001"],
            [0.02, 0.93, 0.02, 0.02, 0.01],
        ),
        (  # Test json format
            """
            Research and reasoning
            {
                "0": 20,
                "1": 66.98,
                "2": 9.99,
                "3": 1.01,
                "10": 2.1,
            }
            """,
            ["0", "1", "2", "3", "10"],
            [0.2, 0.6698, 0.0999, 0.0101, 0.021],
        ),
        (  # Test table format
            """
            Research and reasoning including numbers like "1": 20% and "70%".
            | Option | Probability |
            | "0"    | 20          |
            | "1"    | 70          |
            | "2"    | 10          |
            """,
            ["0", "1", "2"],
            [0.2, 0.7, 0.1],
        ),
        (  # Test table format
            """
            | Option | Probability |
            | 'Option 0'    | 20          |
            | 'Option 1'    | 70          |
            | 'Option 2'    | 10          |
            """,
            ["0", "1", "2"],
            [0.2, 0.7, 0.1],
        ),
        (
            """
            Option “0”: 20%
            Option “1”: 50%
            Option “2-3”: 25%
            Option “4 or more”: 5%
            """,
            ["0", "1", "2-3", "4 or more"],
            [0.2, 0.5, 0.25, 0.05],
        ),
    ],
)
def test_multiple_choice_extraction_success(
    reasoning: str, options: list[str], expected_probabilities: list[float]
) -> None:
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
            expected_probability, abs=0.001
        )


@pytest.mark.parametrize(
    "reasoning, options",
    [
        (  # Test missing option
            """
            Option OnlyOne: 60
            """,
            ["Option OnlyOne", "Option Missing"],
        ),
        (  # Test inconsistent option names
            """
            Option A: 30
            Option B: 40
            Option Yellow: 30
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test total probability < 1
            """
            Option Blue: 0.01
            Option Green: 0.02
            Option Yellow: 0.03
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test total probability < 1 by only a bit
            """
            Option Blue: 0.3
            Option Green: 0.2
            Option Yellow: 0.47
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test total probabiliby > 1
            """
            Option Blue: 0.5
            Option Green: 0.7
            Option Yellow: 0.8
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test total probabiliby > 1 by only a bit
            """
            Option Blue: 0.3
            Option Green: 0.2
            Option Yellow: 0.53
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test negative probabilities
            """
            Blue: -0.5
            Green: 0.3
            Yellow: 0.2
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test repeated option v1
            """
            Blue: 0.5
            Yellow: 0.2
            Green: 0.3
            Yellow: 0.2
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test repeated option v2
            """
            Blue: 0.5
            Yellow: 0.1
            Green: 0.3
            Yellow: 0.1
            """,
            ["Blue", "Green", "Yellow"],
        ),
        (  # Test repeated option v3
            """
            Blue: 0.4
            Yellow: 0.1
            Green: 0.3
            Yellow: 0.2
            """,
            ["Blue", "Green", "Yellow"],
        ),
    ],
)
def test_multiple_choice_extraction_failure(reasoning: str, options: list[str]) -> None:
    with pytest.raises(ValueError):
        PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, options
        )


def create_numeric_question(
    magnitude_units: str | None = None,
) -> NumericQuestion:
    if magnitude_units is None:
        question_text = "How much will the stock market be worth in 2026? (exact value)"
    else:
        question_text = (
            f"How much will the stock market be worth in 2026 in {magnitude_units}?"
        )

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
            Percentile 40: -£10.45
            Percentile 60: -$ 9
            Percentile 70: - $8
            Percentile 80: £ 31
            """,
            [
                Percentile(value=-20, percentile=0.2),
                Percentile(value=-10.45, percentile=0.4),
                Percentile(value=-9, percentile=0.6),
                Percentile(value=-8, percentile=0.7),
                Percentile(value=31, percentile=0.8),
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
        (
            """
            Notes before including "I think that percentile 10 should be 12, but won't list it".

            Percentile 20: 1000000
            Percentile 40: 2,000,000
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
            I think that percentile 10 should be 12, but won't list it
            The percentile for 10 should be 800
            The Percentile for 10 should be 1000
            Percentile number 10 should be 400

            Percentile 20: 1000000
            Percentile 40: 2,000,000
            Percentile 60: $3,000,000

            Finally, that is where I think this should actually be Percentile 10: 12
            """,
            [
                Percentile(value=1000000, percentile=0.2),
                Percentile(value=2000000, percentile=0.4),
                Percentile(value=3000000, percentile=0.6),
            ],
            create_numeric_question(),
        ),
        (  # testing with non breaking spaces for commas (gpt o3 uses this)
            """
            Percentile 10: 7 000
            Percentile 20: 9 000
            Percentile 40: 11 000
            Percentile 60: 12 500
            Percentile 80: 14 500
            Percentile 90: 17 000
            """,
            [
                Percentile(value=7000, percentile=0.1),
                Percentile(value=9000, percentile=0.2),
                Percentile(value=11000, percentile=0.4),
                Percentile(value=12500, percentile=0.6),
                Percentile(value=14500, percentile=0.8),
                Percentile(value=17000, percentile=0.9),
            ],
            create_numeric_question(),
        ),
        (  # Testing with regular spaces (in case o3 decides this is also a good idea)
            """
            Percentile 10: 7 000
            Percentile 20: 9 000
            Percentile 40: 11 000
            Percentile 60: 12 500
            Percentile 80: 14 500
            Percentile 90: 17 000
            """,
            [
                Percentile(value=7000, percentile=0.1),
                Percentile(value=9000, percentile=0.2),
                Percentile(value=11000, percentile=0.4),
                Percentile(value=12500, percentile=0.6),
                Percentile(value=14500, percentile=0.8),
                Percentile(value=17000, percentile=0.9),
            ],
            create_numeric_question(),
        ),
        (  # Testing complicated spaces
            """
            Percentile 10: -9 123 432
            Percentile 20: -$7 123 432
            Percentile 40: 11 000 432
            Percentile 60: 12 500 432
            Percentile 80: $14 500 432
            Percentile 90: 17 020 432.432
            """,
            [
                Percentile(value=-9123432, percentile=0.1),
                Percentile(value=-7123432, percentile=0.2),
                Percentile(value=11000432, percentile=0.4),
                Percentile(value=12500432, percentile=0.6),
                Percentile(value=14500432, percentile=0.8),
                Percentile(value=17020432.432, percentile=0.9),
            ],
            create_numeric_question(),
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
        gpt_response, question, standardize_cdf=False
    )
    assert len(numeric_distribution.declared_percentiles) == len(
        expected_percentiles
    ), f"Expected {len(expected_percentiles)} percentiles, but got {len(numeric_distribution.declared_percentiles)}"
    for declared_percentile, expected_percentile in zip(
        numeric_distribution.declared_percentiles, expected_percentiles
    ):
        assert declared_percentile.value == pytest.approx(expected_percentile.value)
        assert declared_percentile.percentile == pytest.approx(
            expected_percentile.percentile
        )


@pytest.mark.parametrize(
    "gpt_response",
    [
        """
        There are no percentiles here
        """,
        """
        Percentile 10: A
        Percentile 20: B
        Percentile 30: C
        """,
        """
        Percentile: 10 12
        Percentile: 20 14
        Percentile: 30 16
        """,
        """
        Percentile 101: 10
        Percentile 201: 14
        Percentile 301: 16
        """,
        """
        Percentile 0.1: 10
        Percentile 0.2: 14
        Percentile 0.3: 16
        """,
        """
        Percentile 10: 7 000 -432
        Percentile 20: 9 000 -432
        """,
        # """
        # Percentile 10: 7 000432
        # Percentile 20: 9 000432
        # """,
        """
        Percentile 10: 7 010432
        Percentile 20: 9 006432
        """,
        """
        Percentile 10: 7000 432
        Percentile 20: 9000 432
        """,
    ],
)
def test_numeric_parsing_failure(gpt_response: str) -> None:
    with pytest.raises(ValueError):
        PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            gpt_response, create_numeric_question()
        )


@pytest.mark.parametrize(
    "gpt_response, expected_probability",
    [
        (
            """
            Text before
            Probability: 30%
            """,
            0.3,
        ),
        (
            """
            Text before
            Probability: 30 %
            """,
            0.3,
        ),
        (
            """
            Text before
            Probability: 100%
            """,
            0.95,
        ),
        (
            """
            Text before
            Probability: 0%
            """,
            0.05,
        ),
        (
            """
            Text before
            Probability: 35.4%
            """,
            0.354,
        ),
    ],
)
def test_binary_parsing(gpt_response: str, expected_probability: list[float]) -> None:
    binary_prediction = PredictionExtractor.extract_last_percentage_value(
        gpt_response,
        max_prediction=0.95,
        min_prediction=0.05,
    )
    assert binary_prediction == pytest.approx(expected_probability, abs=0.001)


@pytest.mark.parametrize(
    "gpt_response",
    [
        """
        Research and reasoning
        Probability: 30
        """,
        """
        Text before
        Probability: 0.3
        """,
        """
        Text before
        Probability: 100.1%
        """,
        """
        Research and reasoning
        Probability: 110%
        """,
        # ( TODO: Check for negative probabilities
        #     """
        #     Research and reasoning
        #     Probability: -10%
        #     """,
        #     0.05,
        # ),
    ],
)
def test_binary_parsing_failure(gpt_response: str) -> None:
    with pytest.raises(ValueError):
        PredictionExtractor.extract_last_percentage_value(
            gpt_response,
            max_prediction=0.95,
            min_prediction=0.05,
        )
