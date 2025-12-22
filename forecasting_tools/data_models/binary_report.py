from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Sequence

import numpy as np
from pydantic import AliasChoices, BaseModel, Field, field_validator

from forecasting_tools.data_models.forecast_report import ForecastReport

if TYPE_CHECKING:
    from forecasting_tools.data_models.questions import BinaryQuestion
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

logger = logging.getLogger(__name__)


class BinaryPrediction(BaseModel):
    prediction_in_decimal: float

    @field_validator("prediction_in_decimal")
    @classmethod
    def validate_prediction_range(cls, value: float) -> float:
        # Adjust the prediction when 0 or 1 since AI making this model sometimes gives 0 or 1.
        if 0.001 <= value <= 0.999:
            return value
        elif 0 <= value < 0.001:
            logger.warning(
                f"Prediction is less than 0.001, adjusting to 0.001. Value: {value}"
            )
            return 0.001
        elif 0.999 < value <= 1:
            logger.warning(
                f"Prediction is greater than 0.999, adjusting to 0.999. Value: {value}"
            )
            return 0.999
        else:
            raise ValueError("Prediction must be between 0 and 1")


class BinaryReport(ForecastReport):
    question: BinaryQuestion
    prediction: float = Field(
        validation_alias=AliasChoices("prediction_in_decimal", "prediction")
    )  # It was a poor design decision to not make prediction a BaseModel originally, but keeping it a float for backwards compatibility

    @field_validator("prediction")
    def validate_prediction(cls: BinaryReport, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Prediction must be between 0 and 1")
        return v

    async def publish_report_to_metaculus(
        self, metaculus_client: MetaculusClient | None = None
    ) -> None:
        from forecasting_tools.helpers.metaculus_client import MetaculusClient

        metaculus_client = metaculus_client or MetaculusClient()
        if self.question.id_of_question is None:
            raise ValueError("Question ID is None")
        if self.question.id_of_post is None:
            raise ValueError(
                "Publishing to Metaculus requires a post ID for the question"
            )
        metaculus_client.post_binary_question_prediction(
            self.question.id_of_question, self.prediction
        )
        metaculus_client.post_question_comment(
            self.question.id_of_post, self.explanation
        )

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[float], question: BinaryQuestion
    ) -> float:
        for prediction in predictions:
            assert 0 <= prediction <= 1, "Predictions must be between 0 and 1"
        return float(statistics.median(predictions))

    @classmethod
    def make_readable_prediction(cls, prediction: float) -> str:
        return f"{round(prediction * 100, 2)}%"

    @property
    def community_prediction(self) -> float | None:
        return self.question.community_prediction_at_access_time

    @property
    def expected_baseline_score(self) -> float | None:
        c = self.community_prediction
        p = self.prediction
        if c is None:
            return None
        return 100.0 * (c * (np.log2(p) + 1.0) + (1.0 - c) * (np.log2(1.0 - p) + 1.0))

    @property
    def inversed_expected_log_score(self) -> float | None:
        c = self.community_prediction
        p = self.prediction
        if c is None:
            return None
        expected_log_score = c * np.log(p) + (1 - c) * np.log(1 - p)
        inversed_expected_log_score = -1 * expected_log_score
        return inversed_expected_log_score

    @property
    def deviation_points(self) -> float | None:
        if self.community_prediction is None:
            return None
        return abs(self.prediction - self.community_prediction)

    @staticmethod
    def calculate_average_deviation_points(
        reports: Sequence[BinaryReport],
    ) -> float:
        assert (
            len(reports) > 0
        ), "Must have at least one report to calculate average deviation points"
        validated_deviation_points: list[float] = []
        for report in reports:
            assert report.deviation_points is not None
            validated_deviation_points.append(report.deviation_points)
        assert validated_deviation_points
        return sum(validated_deviation_points) / len(validated_deviation_points)
