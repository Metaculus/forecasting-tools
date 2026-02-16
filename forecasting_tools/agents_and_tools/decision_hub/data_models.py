from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from forecasting_tools.util.jsonable import Jsonable


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisSession(BaseModel, Jsonable):
    session_id: str = Field(default_factory=_new_id)
    username: str
    policy_question: str
    status: Literal["in_progress", "complete"] = "in_progress"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ResearchReport(BaseModel, Jsonable):
    artifact_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    query: str
    report_markdown: str
    sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class ScenarioDriver(BaseModel, Jsonable):
    name: str
    description: str
    high_state: str
    low_state: str


class Scenario(BaseModel, Jsonable):
    name: str
    narrative: str
    driver_states: dict[str, str] = Field(default_factory=dict)
    probability: float | None = None


class ScenarioSet(BaseModel, Jsonable):
    artifact_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    drivers: list[ScenarioDriver] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    rationale: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class ForecastQuestion(BaseModel, Jsonable):
    question_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    question_text: str
    resolution_criteria: str = ""
    question_type: Literal[
        "baseline", "scenario_conditional", "proposal_conditional"
    ] = "baseline"
    conditional_on_scenario: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Forecast(BaseModel, Jsonable):
    forecast_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    question_id: str
    prediction: str
    reasoning: str = ""
    key_sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class PolicyProposal(BaseModel, Jsonable):
    artifact_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    proposal_markdown: str
    key_recommendations: list[str] = Field(default_factory=list)
    contingency_plans: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class RobustnessReport(BaseModel, Jsonable):
    artifact_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    matrix_markdown: str = ""
    robust_recommendations: str = ""
    scenario_dependent_recs: str = ""
    hedging_strategy: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class SynthesisReport(BaseModel, Jsonable):
    artifact_id: str = Field(default_factory=_new_id)
    session_id: str
    username: str
    executive_summary: str = ""
    full_report_markdown: str = ""
    blog_post: str = ""
    future_snapshot: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class RedTeamResult(BaseModel, Jsonable):
    artifact_id: str = Field(default_factory=_new_id)
    session_id: str | None = None
    username: str
    input_text: str
    mode: Literal["devils_advocate", "bias_detector"]
    critique_markdown: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
