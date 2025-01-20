import pytest

from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    Percentile,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    NumericQuestion,
    QuestionState,
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
        id_of_post=1,
        state=QuestionState.OPEN,
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
    bot = TemplateBot()
    numeric_distribution = bot._extract_forecast_from_numeric_rationale(
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
