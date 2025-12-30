from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import typeguard
from pydantic import BaseModel, Field, field_validator

from forecasting_tools.data_models.markdown_tree import MarkdownTree
from forecasting_tools.util.jsonable import Jsonable

if TYPE_CHECKING:
    from forecasting_tools.data_models.questions import MetaculusQuestion
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

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
    def report_sections(self) -> list[MarkdownTree]:
        return MarkdownTree.turn_markdown_into_report_sections(self.explanation)

    @property
    def summary(self) -> str:
        return self._get_and_validate_section(
            index=0, expected_word="summary"
        ).text_of_section_and_subsections

    @property
    def research(self) -> str:
        return self._get_and_validate_section(
            index=1, expected_word="research"
        ).text_of_section_and_subsections

    @property
    def forecast_rationales(self) -> str:
        return self._get_and_validate_section(
            index=2, expected_word="forecast"
        ).text_of_section_and_subsections

    @property
    def first_rationale(self) -> str:
        return (
            self._get_and_validate_section(index=2, expected_word="forecast")
            .sub_sections[0]
            .text_of_section_and_subsections
        )

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
        reports: Sequence[ForecastReport],
    ) -> float:
        assert (
            len(reports) > 0
        ), "Must have at least one report to calculate average expected baseline score"
        try:
            scores: list[float | None] = [
                report.expected_baseline_score for report in reports
            ]
            validated_scores: list[float] = typeguard.check_type(scores, list[float])
            average_score = sum(validated_scores) / len(validated_scores)
        except Exception as e:
            raise ValueError(
                f"Error calculating average expected baseline score. {len(reports)} reports. "
                f"There were {len([score for score in scores if score is None])} None scores. Error: {e}"
            ) from e
        return average_score

    @classmethod
    @abstractmethod
    async def aggregate_predictions(
        cls, predictions: list[T], question: MetaculusQuestion
    ) -> T:
        raise NotImplementedError("Subclass must implement this abstract method")

    @classmethod
    @abstractmethod
    def make_readable_prediction(cls, prediction: Any) -> str:
        raise NotImplementedError("Subclass must implement this abstract method")

    @abstractmethod
    async def publish_report_to_metaculus(
        self, metaculus_client: MetaculusClient | None = None
    ) -> None:
        raise NotImplementedError("Subclass must implement this abstract method")

    def _get_and_validate_section(self, index: int, expected_word: str) -> MarkdownTree:
        if len(self.report_sections) <= index:
            raise ValueError(f"Report must have at least {index + 1} sections")
        section = self.report_sections[index]
        first_line = section.text_of_section_and_subsections.split("\n")[0]
        if expected_word.lower() not in first_line.lower():
            raise ValueError(
                f"The indexes for the sections are probably off. Target section title should contain the word '{expected_word}'"
            )
        return section
