from __future__ import annotations

import statistics
from typing import Sequence

import numpy as np
from pydantic import AliasChoices, Field, field_validator

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi


class BinaryReport(ForecastReport):
    question: BinaryQuestion
    prediction: float = Field(
        validation_alias=AliasChoices("prediction_in_decimal", "prediction")
    )

    @field_validator("prediction")
    def validate_prediction(cls: BinaryReport, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Prediction must be between 0 and 1")
        return v

    async def publish_report_to_metaculus(self) -> None:
        if self.question.id_of_question is None:
            raise ValueError("Question ID is None")
        if self.question.id_of_post is None:
            raise ValueError(
                "Publishing to Metaculus requires a post ID for the question"
            )
        MetaculusApi.post_binary_question_prediction(
            self.question.id_of_question, self.prediction
        )
        MetaculusApi.post_question_comment(
            self.question.id_of_post, self.explanation
        )

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[float], question: BinaryQuestion
    ) -> float:
        for prediction in predictions:
            assert 0 <= prediction <= 1, "Predictions must be between 0 and 1"
            assert isinstance(prediction, float), "Predictions must be floats"
        return statistics.median(predictions)

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
        return 100.0 * (
            c * (np.log(p) + 1.0) + (1.0 - c) * (np.log(1.0 - p) + 1.0)
        )

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
        validated_deviation_points: list[float] = []
        for report in reports:
            assert report.deviation_points is not None
            validated_deviation_points.append(report.deviation_points)
        assert validated_deviation_points
        return sum(validated_deviation_points) / len(
            validated_deviation_points
        )
