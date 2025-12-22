from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import computed_field

from forecasting_tools.data_models.conditional_models import (
    ConditionalPrediction,
    ConditionalPredictionTypes,
    PredictionAffirmed,
)
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import (
    ConditionalQuestion,
    MetaculusQuestion,
)
from forecasting_tools.util.misc import clean_indents

if TYPE_CHECKING:
    from forecasting_tools.helpers.metaculus_client import MetaculusClient


class ConditionalReport(ForecastReport):
    question: ConditionalQuestion
    prediction: ConditionalPrediction

    # TODO: separate explanations by question subtype. Should change the `explanation` type definition to allow non-string explanations
    @computed_field  # type: ignore[misc]
    @property
    def parent_report(self) -> ForecastReport:
        return self._get_question_report(
            self.question.parent, self.prediction.parent, self.explanation
        )

    @computed_field  # type: ignore[misc]
    @property
    def child_report(self) -> ForecastReport:
        return self._get_question_report(
            self.question.child, self.prediction.child, self.explanation
        )

    @computed_field  # type: ignore[misc]
    @property
    def yes_report(self) -> ForecastReport:
        return self._get_question_report(
            self.question.question_yes, self.prediction.prediction_yes, self.explanation
        )

    @computed_field  # type: ignore[misc]
    @property
    def no_report(self) -> ForecastReport:
        return self._get_question_report(
            self.question.question_no, self.prediction.prediction_no, self.explanation
        )

    @staticmethod
    def _get_question_report_type(question: MetaculusQuestion) -> type[ForecastReport]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        return DataOrganizer.get_report_type_for_question_type(type(question))

    @staticmethod
    def _get_question_report(
        question: MetaculusQuestion,
        forecast: ConditionalPredictionTypes,
        explanation: str,
    ) -> ForecastReport:

        report_type = ConditionalReport._get_question_report_type(question)
        return report_type(
            question=question, prediction=forecast, explanation=explanation
        )

    @classmethod
    async def _get_aggregate_for_affirmable_forecast(
        cls, question: MetaculusQuestion, forecasts: list[ConditionalPredictionTypes]
    ):
        forecasts_not_affirmed = [
            prediction
            for prediction in forecasts
            if not isinstance(prediction, PredictionAffirmed)
        ]
        if len(forecasts_not_affirmed) * 2 > len(forecasts):
            # TODO: Correctly aggregate affirmed forecasts later
            return await cls._get_question_report_type(question).aggregate_predictions(
                forecasts_not_affirmed, question
            )
        else:
            return PredictionAffirmed()

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[ConditionalPrediction], question: ConditionalQuestion
    ) -> ConditionalPrediction:

        parent_forecasts = [prediction.parent for prediction in predictions]
        aggregated_parent = await cls._get_aggregate_for_affirmable_forecast(
            question.parent, parent_forecasts
        )

        child_forecasts = [prediction.child for prediction in predictions]
        aggregated_child = await cls._get_aggregate_for_affirmable_forecast(
            question.child, child_forecasts
        )

        yes_forecasts = [prediction.prediction_yes for prediction in predictions]
        aggregated_yes = await cls._get_question_report_type(
            question.question_yes
        ).aggregate_predictions(yes_forecasts, question.question_yes)

        no_forecasts = [prediction.prediction_no for prediction in predictions]
        aggregated_no = await cls._get_question_report_type(
            question.question_no
        ).aggregate_predictions(no_forecasts, question.question_no)

        return ConditionalPrediction(
            parent=aggregated_parent,
            child=aggregated_child,
            prediction_yes=aggregated_yes,  # type: ignore
            prediction_no=aggregated_no,  # type: ignore
        )

    @classmethod
    def make_readable_prediction(cls, prediction: ConditionalPrediction) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        return clean_indents(
            f"""
            Parent forecast: {DataOrganizer.get_readable_prediction(prediction.parent)}
            Child forecast: {DataOrganizer.get_readable_prediction(prediction.child)}
            Yes forecast: {DataOrganizer.get_readable_prediction(prediction.prediction_yes)}
            No forecast: {DataOrganizer.get_readable_prediction(prediction.prediction_no)}
        """
        )

    async def publish_report_to_metaculus(
        self, metaculus_client: MetaculusClient | None = None
    ) -> None:
        from forecasting_tools.helpers.metaculus_client import MetaculusClient

        metaculus_client = metaculus_client or MetaculusClient()
        await self.yes_report.publish_report_to_metaculus(metaculus_client)
        await self.no_report.publish_report_to_metaculus(metaculus_client)
