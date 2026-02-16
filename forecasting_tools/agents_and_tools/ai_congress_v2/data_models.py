from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from forecasting_tools.util.jsonable import Jsonable

ForecastType = Literal[
    "baseline",
    "scenario_indicator",
    "scenario_conditional",
    "proposal_conditional",
    "proposal_scenario_conditional",
]


class CongressMember(BaseModel, Jsonable):
    name: str
    role: str
    political_leaning: str
    general_motivation: str
    expertise_areas: list[str]
    personality_traits: list[str]
    ai_model: str = "openrouter/anthropic/claude-sonnet-4"
    search_model: str = "openrouter/perplexity/sonar-reasoning-pro"

    @property
    def expertise_string(self) -> str:
        return ", ".join(self.expertise_areas)

    @property
    def traits_string(self) -> str:
        return ", ".join(self.personality_traits)


class ScenarioDriver(BaseModel, Jsonable):
    name: str = Field(description="Name of the external driver/uncertainty")
    description: str = Field(description="Description of the driver and why it matters")


class ScenarioCriterion(BaseModel, Jsonable):
    criterion_text: str = Field(
        description="Concrete, specific criterion that indicates this scenario"
    )
    target_date: str | None = Field(
        default=None,
        description="Date by which this criterion can be evaluated (e.g., '2027-12-31')",
    )
    resolution_criteria: str = Field(
        default="",
        description="How to determine whether this criterion has been met",
    )


class Scenario(BaseModel, Jsonable):
    name: str = Field(description="Short descriptive name for this scenario")
    narrative: str = Field(
        description="2-3 sentence narrative describing this scenario"
    )
    probability: str = Field(
        description="Estimated probability of this scenario, e.g. '30%'"
    )
    drivers: list[ScenarioDriver] = Field(
        default_factory=list,
        description="Key drivers that define this scenario",
    )
    criteria: list[ScenarioCriterion] = Field(
        default_factory=list,
        description="Concrete criteria that would indicate we are in this scenario",
    )
    is_status_quo: bool = Field(
        default=False,
        description="Whether this is the 'status quo continuation' scenario",
    )


class ForecastDescription(BaseModel, Jsonable):
    footnote_id: int = Field(description="The footnote number, e.g. 1 for [^1]")
    question_title: str = Field(description="Short title for the forecast question")
    question_text: str = Field(description="Full question text")
    resolution_criteria: str = Field(description="How this question resolves")
    prediction: str = Field(
        description=(
            "The probability or distribution, e.g. '35%' or "
            "'70% Option A, 20% Option B, 10% Option C' or "
            "'10% chance less than X units, ... ,90% chance less than Y units'"
        )
    )
    reasoning: str = Field(description="2-4 sentence summary of the reasoning")
    key_sources: list[str] = Field(
        default_factory=list,
        description="URLs or source names used. Ideally both as markdown links.",
    )
    forecast_type: ForecastType = Field(
        default="baseline",
        description=(
            "Type of forecast: 'baseline' (status quo), "
            "'scenario_indicator' (whether a scenario occurs), "
            "'scenario_conditional' (if scenario X, will Y happen), "
            "'proposal_conditional' (if proposal enacted, will Y happen), "
            "'proposal_scenario_conditional' (if proposal enacted under scenario X)"
        ),
    )
    conditional_on_scenario: str | None = Field(
        default=None,
        description="Name of the scenario this forecast is conditional on, if any",
    )
    conditional_on_proposal: bool = Field(
        default=False,
        description="Whether this forecast is conditional on the proposal being enacted",
    )

    def as_footnote_markdown(self) -> str:
        sources_str = ", ".join(self.key_sources) if self.key_sources else "N/A"
        conditional_labels: list[str] = []
        if self.conditional_on_proposal:
            conditional_labels.append("Conditional on proposal")
        if self.conditional_on_scenario:
            conditional_labels.append(f"Under scenario: {self.conditional_on_scenario}")
        label_str = (
            f" *({'; '.join(conditional_labels)})*" if conditional_labels else ""
        )
        return (
            f"[^{self.footnote_id}]: **{self.question_title}**{label_str}\n"
            f"- Question: {self.question_text}\n"
            f"- Resolution: {self.resolution_criteria}\n"
            f"- Prediction: {self.prediction}\n"
            f"- Reasoning: {self.reasoning}\n"
            f"- Sources: {sources_str}"
        )


class ProposalOption(BaseModel, Jsonable):
    name: str = Field(description="Short name for this proposal option")
    description: str = Field(description="Description of what this option entails")
    key_actions: list[str] = Field(
        default_factory=list,
        description="Specific actions this option involves",
    )


class PolicyProposal(BaseModel, Jsonable):
    member: CongressMember | None = Field(
        default=None, description="The congress member who created this proposal"
    )
    research_summary: str = Field(description="Markdown summary of background research")
    decision_criteria: list[str] = Field(
        description="Prioritized criteria for this member"
    )
    scenarios: list[Scenario] = Field(
        default_factory=list,
        description="Scenarios identified by this member",
    )
    drivers: list[ScenarioDriver] = Field(
        default_factory=list,
        description="Key external drivers identified by this member",
    )
    proposal_options: list[ProposalOption] = Field(
        default_factory=list,
        description="Distinct policy options considered by this member",
    )
    selected_proposal_name: str = Field(
        default="",
        description="Name of the proposal option the member selected",
    )
    forecasts: list[ForecastDescription] = Field(
        description="All forecasts made by this member"
    )
    proposal_markdown: str = Field(
        description="Full proposal with footnote references [^1], [^2], etc."
    )
    key_recommendations: list[str] = Field(
        description="Top 3-5 actionable recommendations"
    )
    robustness_analysis: str = Field(
        default="",
        description="Cross-scenario robustness analysis markdown",
    )
    contingency_plans: list[str] = Field(
        default_factory=list,
        description="Contingency plans like 'If X happens, do Y instead'",
    )
    price_estimate: float | None = Field(
        default=None,
        description="Estimated cost in USD for generating this proposal. If you are an AI, leave this None as you don't know the value.",
    )
    delphi_round: int = Field(
        default=1,
        description="Which Delphi round produced this proposal. None if unknown.",
    )

    @property
    def baseline_forecasts(self) -> list[ForecastDescription]:
        return [f for f in self.forecasts if f.forecast_type == "baseline"]

    @property
    def scenario_indicator_forecasts(self) -> list[ForecastDescription]:
        return [f for f in self.forecasts if f.forecast_type == "scenario_indicator"]

    @property
    def scenario_conditional_forecasts(self) -> list[ForecastDescription]:
        return [f for f in self.forecasts if f.forecast_type == "scenario_conditional"]

    @property
    def proposal_conditional_forecasts(self) -> list[ForecastDescription]:
        return [
            f
            for f in self.forecasts
            if f.forecast_type
            in ("proposal_conditional", "proposal_scenario_conditional")
        ]

    @property
    def forecasts_by_scenario(self) -> dict[str, list[ForecastDescription]]:
        result: dict[str, list[ForecastDescription]] = {}
        for f in self.forecasts:
            scenario = f.conditional_on_scenario or "General"
            result.setdefault(scenario, []).append(f)
        return result

    def get_full_markdown_with_footnotes(self) -> str:
        baseline = self.baseline_forecasts
        scenario_indicators = self.scenario_indicator_forecasts
        scenario_conditional = self.scenario_conditional_forecasts
        proposal_conditional = self.proposal_conditional_forecasts

        sections = [self.proposal_markdown]

        if baseline:
            footnotes = "\n\n".join(f.as_footnote_markdown() for f in baseline)
            sections.append(f"---\n\n## Baseline Forecast Appendix\n\n{footnotes}")

        if scenario_indicators:
            footnotes = "\n\n".join(
                f.as_footnote_markdown() for f in scenario_indicators
            )
            sections.append(
                f"---\n\n## Scenario Indicator Forecast Appendix\n\n"
                f"*These forecasts assess the likelihood of each scenario occurring.*\n\n"
                f"{footnotes}"
            )

        if scenario_conditional:
            footnotes = "\n\n".join(
                f.as_footnote_markdown() for f in scenario_conditional
            )
            sections.append(
                f"---\n\n## Scenario-Conditional Forecast Appendix\n\n"
                f"*These forecasts are conditional on specific scenarios occurring.*\n\n"
                f"{footnotes}"
            )

        if proposal_conditional:
            footnotes = "\n\n".join(
                f.as_footnote_markdown() for f in proposal_conditional
            )
            sections.append(
                f"---\n\n## Proposal-Conditional Forecast Appendix\n\n"
                f"*These forecasts are conditional on the proposed policy being implemented.*\n\n"
                f"{footnotes}"
            )

        if self.robustness_analysis:
            sections.append(
                f"---\n\n## Cross-Scenario Robustness Analysis\n\n"
                f"{self.robustness_analysis}"
            )

        return "\n\n".join(sections)


class CongressSessionInput(BaseModel, Jsonable):
    prompt: str
    member_names: list[str]
    num_delphi_rounds: int = 1


class CongressSession(BaseModel, Jsonable):
    prompt: str
    members_participating: list[CongressMember]
    proposals: list[PolicyProposal]
    aggregated_report_markdown: str
    scenario_report: str = Field(
        default="",
        description="Formal scenario planning report in the style of Shell/NIC reports",
    )
    blog_post: str = Field(default="")
    future_snapshot: str = Field(default="")
    twitter_posts: list[str] = Field(default_factory=list)
    timestamp: datetime
    errors: list[str] = Field(default_factory=list)
    total_price_estimate: float | None = Field(
        default=None, description="Total estimated cost in USD for the entire session"
    )
    num_delphi_rounds: int = Field(
        default=1, description="Number of Delphi rounds used in this session"
    )
    initial_proposals: list[PolicyProposal] = Field(
        default_factory=list,
        description="Round 1 proposals preserved when num_delphi_rounds > 1",
    )

    def get_all_forecasts(self) -> list[ForecastDescription]:
        all_forecasts = []
        for proposal in self.proposals:
            for forecast in proposal.forecasts:
                all_forecasts.append(forecast)
        return all_forecasts

    def get_all_baseline_forecasts(self) -> list[ForecastDescription]:
        return [f for f in self.get_all_forecasts() if f.forecast_type == "baseline"]

    def get_all_scenario_indicator_forecasts(self) -> list[ForecastDescription]:
        return [
            f
            for f in self.get_all_forecasts()
            if f.forecast_type == "scenario_indicator"
        ]

    def get_all_scenario_conditional_forecasts(self) -> list[ForecastDescription]:
        return [
            f
            for f in self.get_all_forecasts()
            if f.forecast_type == "scenario_conditional"
        ]

    def get_all_proposal_conditional_forecasts(self) -> list[ForecastDescription]:
        return [
            f
            for f in self.get_all_forecasts()
            if f.forecast_type
            in ("proposal_conditional", "proposal_scenario_conditional")
        ]

    def get_forecasts_by_member(self) -> dict[str, list[ForecastDescription]]:
        result: dict[str, list[ForecastDescription]] = {}
        for proposal in self.proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            result[member_name] = proposal.forecasts
        return result

    def get_all_scenarios(self) -> list[Scenario]:
        all_scenarios: list[Scenario] = []
        seen_names: set[str] = set()
        for proposal in self.proposals:
            for scenario in proposal.scenarios:
                if scenario.name not in seen_names:
                    all_scenarios.append(scenario)
                    seen_names.add(scenario.name)
        return all_scenarios

    def get_all_drivers(self) -> list[ScenarioDriver]:
        all_drivers: list[ScenarioDriver] = []
        seen_names: set[str] = set()
        for proposal in self.proposals:
            for driver in proposal.drivers:
                if driver.name not in seen_names:
                    all_drivers.append(driver)
                    seen_names.add(driver.name)
        return all_drivers
