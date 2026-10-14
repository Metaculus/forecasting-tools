from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import MultipleChoiceQuestion

if TYPE_CHECKING:
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

logger = logging.getLogger(__name__)


class PredictedOption(BaseModel):
    option_name: str
    probability: float = Field(ge=0, le=1)


class PredictedOptionList(BaseModel):
    predicted_options: list[PredictedOption]

    @model_validator(mode="after")
    def validate_probability_sum(self) -> PredictedOptionList:
        sum_of_probabilities = sum(
            option.probability for option in self.predicted_options
        )
        if sum_of_probabilities > 1.01 or sum_of_probabilities < 0.99:
            raise ValueError(
                f"Sum of option probabilities {sum_of_probabilities} is "
                "too far from 1 to be confident that normalization will deliver an "
                "intended prediction."
            )
        if not math.isclose(sum_of_probabilities, 1):
            logger.warning(
                f"Sum of option probabilities {sum_of_probabilities} is not 1, but is close to 1. "
                "Normalizing the probabilities."
            )

        normalized_option_probabilities = self._normalize_with_minimum(
            [option.probability for option in self.predicted_options],
            minimum_probability=0.001,
        )

        new_sum = sum(normalized_option_probabilities)
        assert (
            abs(new_sum - 1) < 0.0001
        ), "Sum of normalized option probabilities is not 1"
        max_probability_change = 0.05
        for original_option, new_probability in zip(
            self.predicted_options, normalized_option_probabilities
        ):
            if (
                abs(new_probability - original_option.probability)
                > max_probability_change
            ):
                raise ValueError(
                    f"New probability {new_probability} differs from original probability {original_option.probability} by more than {max_probability_change} for option {original_option.option_name}"
                )

        self.predicted_options = [
            PredictedOption(option_name=option.option_name, probability=probability)
            for option, probability in zip(
                self.predicted_options, normalized_option_probabilities
            )
        ]
        return self

    @staticmethod
    def _normalize_with_minimum(
        probabilities: list[float], minimum_probability: float
    ) -> list[float]:
        """
        Normalizes to sum to 1 while keeping every option at or above the minimum.
        Options below the minimum are pinned to it and the excess is taken
        proportionally from the remaining options, so with 2+ options no option
        can exceed 1 - minimum_probability.
        """
        if minimum_probability * len(probabilities) >= 1:
            raise ValueError(
                f"Cannot give all {len(probabilities)} options a probability of at least {minimum_probability}"
            )
        total = sum(probabilities)
        unadjusted = [probability / total for probability in probabilities]
        normalized = unadjusted
        pinned_indices: set[int] = set()
        while True:
            newly_pinned_indices = {
                index
                for index, probability in enumerate(normalized)
                if probability < minimum_probability and index not in pinned_indices
            }
            if not newly_pinned_indices:
                break
            pinned_indices |= newly_pinned_indices
            unpinned_mass = 1 - minimum_probability * len(pinned_indices)
            unpinned_total = sum(
                probability
                for index, probability in enumerate(unadjusted)
                if index not in pinned_indices
            )
            normalized = [
                (
                    minimum_probability
                    if index in pinned_indices
                    else probability * unpinned_mass / unpinned_total
                )
                for index, probability in enumerate(unadjusted)
            ]

        largest_index = normalized.index(max(normalized))
        normalized[largest_index] = 1.0 - sum(
            probability
            for index, probability in enumerate(normalized)
            if index != largest_index
        )
        return normalized

    def to_dict(self) -> dict[str, float]:
        return {
            option.option_name: option.probability for option in self.predicted_options
        }


class MultipleChoiceReport(ForecastReport):
    question: MultipleChoiceQuestion
    prediction: PredictedOptionList

    @property
    def expected_baseline_score(self) -> float | None:
        raise NotImplementedError("Not implemented")

    @property
    def community_prediction(self) -> PredictedOptionList | None:
        raise NotImplementedError("Not implemented")

    async def publish_report_to_metaculus(
        self, metaculus_client: MetaculusClient | None = None
    ) -> None:
        from forecasting_tools.helpers.metaculus_client import MetaculusClient

        metaculus_client = metaculus_client or MetaculusClient()

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
        metaculus_client.post_multiple_choice_question_prediction(
            self.question.id_of_question, options_with_probabilities
        )
        metaculus_client.post_question_comment(
            self.question.id_of_post, self.explanation
        )

    @classmethod
    async def aggregate_predictions(
        cls,
        predictions: list[PredictedOptionList],
        question: MultipleChoiceQuestion,
    ) -> PredictedOptionList:
        first_list_option_names = [
            pred_option.option_name for pred_option in predictions[0].predicted_options
        ]

        # Check for predicted option consistency
        for option_list in predictions:
            current_option_names = {
                option.option_name for option in option_list.predicted_options
            }
            if current_option_names != set(first_list_option_names):
                raise ValueError(
                    f"All predictions must have the same option names, but {current_option_names} != {first_list_option_names}"
                )
            if len(option_list.predicted_options) != len(first_list_option_names):
                raise ValueError(
                    f"All predictions must have the same number of options, but {len(option_list.predicted_options)} != {len(first_list_option_names)}"
                )
            for option in option_list.predicted_options:
                if not 0 <= option.probability <= 1:
                    raise ValueError(
                        f"{option.option_name} has a probability of {option.probability}, which is not between 0 and 1"
                    )

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
