from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from forecasting_tools.util.jsonable import Jsonable


class ForecastCategory(str, Enum):
    HARD_METRIC = "hard_metric"
    QUALITATIVE = "qualitative"


class HardMetricCriteria(BaseModel, Jsonable):
    agent_name: str
    item_name: str
    operator: str = ">="
    threshold: int = 0


class InterventionForecast(BaseModel, Jsonable):
    question_title: str
    question_text: str
    resolution_criteria: str
    prediction: float = Field(ge=0.0, le=1.0)
    reasoning: str
    is_conditional: bool = Field(
        default=False,
        description="True = forecast under intervention, False = forecast under status quo",
    )
    category: ForecastCategory = Field(
        default=ForecastCategory.QUALITATIVE,
        description="'hard_metric' for inventory-based forecasts, 'qualitative' for event-based",
    )
    hard_metric_criteria: HardMetricCriteria | None = Field(
        default=None,
        description="Structured criteria for auto-resolving hard metric forecasts",
    )
    resolved: bool = False
    resolution: bool | None = None
    brier_score: float | None = None


class PolicyAgentResult(BaseModel, Jsonable):
    agent_goals_analysis: str
    evaluation_criteria: list[str]
    intervention_description: str
    policy_proposal_markdown: str
    forecasts: list[InterventionForecast]

    @property
    def baseline_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if not f.is_conditional]

    @property
    def conditional_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.is_conditional]

    @property
    def hard_metric_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.category == ForecastCategory.HARD_METRIC]

    @property
    def qualitative_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.category == ForecastCategory.QUALITATIVE]


class InterventionRun(BaseModel, Jsonable):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model_name: str
    situation_name: str
    target_agent_name: str
    intervention_description: str
    policy_proposal_markdown: str
    evaluation_criteria: list[str] = Field(default_factory=list)
    warmup_steps: int
    total_steps: int
    forecasts: list[InterventionForecast] = Field(default_factory=list)
    total_cost: float = 0.0

    @property
    def hard_metric_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.category == ForecastCategory.HARD_METRIC]

    @property
    def qualitative_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.category == ForecastCategory.QUALITATIVE]

    @property
    def baseline_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if not f.is_conditional]

    @property
    def conditional_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.is_conditional]

    @property
    def resolved_forecasts(self) -> list[InterventionForecast]:
        return [f for f in self.forecasts if f.resolved]

    @property
    def average_brier_score(self) -> float | None:
        scored = [f for f in self.forecasts if f.brier_score is not None]
        if not scored:
            return None
        return sum(f.brier_score for f in scored) / len(scored)

    @property
    def average_hard_metric_brier_score(self) -> float | None:
        scored = [f for f in self.hard_metric_forecasts if f.brier_score is not None]
        if not scored:
            return None
        return sum(f.brier_score for f in scored) / len(scored)

    @property
    def average_qualitative_brier_score(self) -> float | None:
        scored = [f for f in self.qualitative_forecasts if f.brier_score is not None]
        if not scored:
            return None
        return sum(f.brier_score for f in scored) / len(scored)
