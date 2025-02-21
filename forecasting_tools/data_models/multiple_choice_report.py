from pydantic import BaseModel, Field

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import MultipleChoiceQuestion
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi


class PredictedOption(BaseModel):
    option_name: str
    probability: float = Field(ge=0, le=1)


class PredictedOptionList(BaseModel):
    predicted_options: list[PredictedOption]


class MultipleChoiceReport(ForecastReport):
    question: MultipleChoiceQuestion
    prediction: PredictedOptionList

    @property
    def expected_baseline_score(self) -> float | None:
        raise NotImplementedError("Not implemented")

    @property
    def community_prediction(self) -> PredictedOptionList | None:
        raise NotImplementedError("Not implemented")

    async def publish_report_to_metaculus(self) -> None:
        if self.question.id_of_question is None:
            raise ValueError("Question ID is None")
        options_with_probabilities = {
            option.option_name: option.probability
            for option in self.prediction.predicted_options
        }
        if self.question.id_of_post is None:
            raise ValueError(
                "Publishing to Metaculus requires a post ID for the question"
            )
        MetaculusApi.post_multiple_choice_question_prediction(
            self.question.id_of_question, options_with_probabilities
        )
        MetaculusApi.post_question_comment(
            self.question.id_of_post, self.explanation
        )

    @classmethod
    async def aggregate_predictions(
        cls,
        predictions: list[PredictedOptionList],
        question: MultipleChoiceQuestion,
    ) -> PredictedOptionList:
        first_list_option_names = [
            pred_option.option_name
            for pred_option in predictions[0].predicted_options
        ]

        # Check for predicted option consistency
        for option_list in predictions:
            current_option_names = {
                option.option_name for option in option_list.predicted_options
            }
            assert current_option_names == set(
                first_list_option_names
            ), "All predictions must have the same option names"
            assert len(option_list.predicted_options) == len(
                first_list_option_names
            ), "All predictions must have the same number of options"
            for option in option_list.predicted_options:
                assert (
                    0 <= option.probability <= 1
                ), "Predictions must be between 0 and 1"

        new_predicted_options: list[PredictedOption] = []
        for current_option_name in first_list_option_names:
            probabilities_of_current_option = [
                option.probability
                for option_list in predictions
                for option in option_list.predicted_options
                if option.option_name == current_option_name
            ]

            average_probability = sum(probabilities_of_current_option) / len(
                probabilities_of_current_option
            )
            new_predicted_options.append(
                PredictedOption(
                    option_name=current_option_name,
                    probability=average_probability,
                )
            )
        return PredictedOptionList(predicted_options=new_predicted_options)

    @classmethod
    def make_readable_prediction(cls, prediction: PredictedOptionList) -> str:
        option_bullet_points = [
            f"- {option.option_name}: {round(option.probability * 100, 2)}%"
            for option in prediction.predicted_options
        ]
        combined_bullet_points = "\n".join(option_bullet_points)
        return f"\n{combined_bullet_points}\n"
