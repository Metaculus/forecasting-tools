from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.conditional_models import (
    ConditionalPrediction,
    ConditionalPredictionTypes,
)
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import (
    ConditionalQuestion,
    MetaculusQuestion,
)


class ConditionalReport(ForecastReport):
    question: ConditionalQuestion
    prediction: ConditionalPrediction
    parent: ForecastReport | None = None
    child: ForecastReport | None = None
    question_yes: ForecastReport | None = None
    question_no: ForecastReport | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # TODO: separate explanations by question subtype. Should change the `explanation` type definition to allow non-string explanations
        self.parent = self._get_question_report(
            self.question.parent, self.prediction.parent, self.explanation
        )
        self.child = self._get_question_report(
            self.question.child, self.prediction.child, self.explanation
        )
        self.question_yes = self._get_question_report(
            self.question.question_yes, self.prediction.prediction_yes, self.explanation
        )
        self.question_no = self._get_question_report(
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

        parent_report_type = ConditionalReport._get_question_report_type(question)
        return parent_report_type(
            question=question, prediction=forecast, explanation=explanation
        )

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[ConditionalPrediction], question: ConditionalQuestion
    ) -> ConditionalPrediction:

        parent_forecasts = [prediction.parent for prediction in predictions]
        aggregated_parent = await cls._get_question_report_type(
            question.parent
        ).aggregate_predictions(parent_forecasts, question.parent)

        child_forecasts = [prediction.child for prediction in predictions]
        aggregated_child = await cls._get_question_report_type(
            question.child
        ).aggregate_predictions(child_forecasts, question.child)

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
        return clean_indents(
            f"""
            Parent forecast: {prediction.parent}
            Child forecast: {prediction.child}
            Yes forecast: {prediction.prediction_yes}
            No forecast: {prediction.prediction_no}
        """
        )

    async def publish_report_to_metaculus(self) -> None:
        # TODO: publish parent/child reports if necessary
        await self.question_yes.publish_report_to_metaculus()
        await self.question_no.publish_report_to_metaculus()
