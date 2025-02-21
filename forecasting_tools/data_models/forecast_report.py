from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, field_validator

from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.data_models.report_section import ReportSection
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ReasonedPrediction(BaseModel, Generic[T]):
    prediction_value: T
    reasoning: str


class ResearchWithPredictions(BaseModel, Generic[T]):
    research_report: str
    summary_report: str
    errors: list[str] = Field(default_factory=list)
    predictions: list[ReasonedPrediction[T]]


class ForecastReport(BaseModel, Jsonable, ABC):
    question: MetaculusQuestion
    explanation: str
    other_notes: str | None = None
    price_estimate: float | None = None
    minutes_taken: float | None = None
    errors: list[str] = Field(default_factory=list)
    prediction: Any

    @field_validator("explanation")
    @classmethod
    def validate_explanation_starts_with_hash(cls, v: str) -> str:
        if not v.strip().startswith("#"):
            raise ValueError("Explanation must start with a '#' character")
        return v

    @property
    def report_sections(self) -> list[ReportSection]:
        return ReportSection.turn_markdown_into_report_sections(
            self.explanation
        )

    @property
    def summary(self) -> str:
        return self._get_section_content(index=0, expected_word="summary")

    @property
    def research(self) -> str:
        return self._get_section_content(index=1, expected_word="research")

    @property
    def forecast_rationales(self) -> str:
        return self._get_section_content(index=2, expected_word="forecast")

    @property
    def expected_baseline_score(self) -> float | None:
        """
        Uses the community prediction to calculate the expected value of the baseline score
        by assuming the community prediction is the true probability. Can be used as
        a proxy score for comparing forecasters on the same set of questions, enabling
        faster feedback loops.

        Higher is better.

        See https://www.metaculus.com/help/scores-faq/#baseline-score
        and scripts/simulate_a_tournament.ipynb for more details.
        """
        raise NotImplementedError("Not yet implemented")

    @property
    def community_prediction(self) -> Any | None:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def calculate_average_expected_baseline_score(
        reports: list[ForecastReport],
    ) -> float:
        deviation_scores: list[float | None] = [
            report.expected_baseline_score for report in reports
        ]
        validated_deviation_scores: list[float] = []
        for score in deviation_scores:
            assert score is not None
            validated_deviation_scores.append(score)
        average_deviation_score = sum(validated_deviation_scores) / len(
            validated_deviation_scores
        )
        return average_deviation_score

    @classmethod
    @abstractmethod
    async def aggregate_predictions(
        cls, predictions: list[T], question: MetaculusQuestion
    ) -> T:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @classmethod
    @abstractmethod
    def make_readable_prediction(cls, prediction: Any) -> str:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @abstractmethod
    async def publish_report_to_metaculus(self) -> None:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    def _get_section_content(self, index: int, expected_word: str) -> str:
        if len(self.report_sections) <= index:
            raise ValueError(f"Report must have at least {index + 1} sections")
        content = self.report_sections[index].text_of_section_and_subsections
        first_line = content.split("\n")[0]
        if expected_word.lower() not in first_line.lower():
            raise ValueError(
                f"Section must contain the word '{expected_word}'"
            )
        return content
