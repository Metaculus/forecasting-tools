import textwrap
from datetime import datetime
from typing import TypeVar
from unittest.mock import Mock

from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    QuestionState,
)

T = TypeVar("T", bound=MetaculusQuestion)


class ForecastingTestManager:
    TOURNAMENT_SAFE_TO_PULL_AND_PUSH_TO = MetaculusApi.AI_WARMUP_TOURNAMENT_ID
    TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN = (
        MetaculusApi.AI_COMPETITION_ID_Q4
    )
    TOURNAMENT_WITH_MIX_OF_QUESTION_TYPES = MetaculusApi.Q4_2024_QUARTERLY_CUP
    TOURN_WITH_OPENNESS_AND_TYPE_VARIATIONS = (
        MetaculusApi.Q4_2024_QUARTERLY_CUP
    )

    @classmethod
    def get_fake_binary_questions(
        cls, community_prediction: float | None = 0.7
    ) -> BinaryQuestion:
        question = BinaryQuestion(
            question_text="Will TikTok be banned in the US?",
            id_of_post=0,
            state=QuestionState.OPEN,
            community_prediction_at_access_time=community_prediction,
        )
        return question

    @staticmethod
    def get_fake_forecast_report(
        community_prediction: float | None = 0.7, prediction: float = 0.5
    ) -> BinaryReport:
        return BinaryReport(
            question=ForecastingTestManager.get_fake_binary_questions(
                community_prediction
            ),
            prediction=prediction,
            explanation=textwrap.dedent(
                """
                # Summary
                This is a test explanation

                ## Analysis
                ### Analysis 1
                This is a test analysis

                ### Analysis 2
                This is a test analysis
                #### Analysis 2.1
                This is a test analysis
                #### Analysis 2.2
                This is a test analysis
                - Conclusion 1
                - Conclusion 2

                # Conclusion
                This is a test conclusion
                - Conclusion 1
                - Conclusion 2
                """
            ),
            other_notes=None,
        )

    @staticmethod
    def mock_forecast_bot_run_forecast(
        subclass: type[ForecastBot], mocker: Mock
    ) -> Mock:
        test_binary_question = (
            ForecastingTestManager.get_fake_binary_questions()
        )
        mock_function = mocker.patch(
            f"{subclass._run_individual_question.__module__}.{subclass._run_individual_question.__qualname__}"
        )
        assert isinstance(test_binary_question, BinaryQuestion)
        mock_function.return_value = (
            ForecastingTestManager.get_fake_forecast_report()
        )
        return mock_function

    @staticmethod
    def mock_add_forecast_report_to_database(mocker: Mock) -> Mock:
        mock_function = mocker.patch(
            f"{ForecastDatabaseManager.add_forecast_report_to_database.__module__}.{ForecastDatabaseManager.add_forecast_report_to_database.__qualname__}"
        )
        return mock_function

    @staticmethod
    def quarterly_cup_is_not_active() -> bool:
        # Quarterly cup is not active from the 1st to the 10th day of the quarter while the initial questions are being set
        current_date = datetime.now().date()
        day_of_month = current_date.day
        month = current_date.month

        is_first_month_of_quarter = month in [1, 4, 7, 10]
        is_first_10_days = day_of_month <= 10

        return is_first_month_of_quarter and is_first_10_days

    @staticmethod
    def mock_getting_benchmark_questions(mocker: Mock) -> Mock:
        mock_function = mocker.patch(
            f"{MetaculusApi.get_benchmark_questions.__module__}.{MetaculusApi.get_benchmark_questions.__qualname__}"
        )
        mock_function.return_value = [
            ForecastingTestManager.get_fake_binary_questions()
        ]
        return mock_function
