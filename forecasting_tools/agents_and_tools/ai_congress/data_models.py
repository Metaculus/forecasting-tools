from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from forecasting_tools.util.jsonable import Jsonable


class CongressMember(BaseModel, Jsonable):
    name: str
    role: str
    political_leaning: str
    general_motivation: str
    expertise_areas: list[str]
    personality_traits: list[str]
    ai_model: str = "openrouter/anthropic/claude-sonnet-4"

    @property
    def expertise_string(self) -> str:
        return ", ".join(self.expertise_areas)

    @property
    def traits_string(self) -> str:
        return ", ".join(self.personality_traits)


class ForecastDescription(BaseModel, Jsonable):
    footnote_id: int = Field(description="The footnote number, e.g. 1 for [^1]")
    question_title: str = Field(description="Short title for the forecast question")
    question_text: str = Field(description="Full question text")
    resolution_criteria: str = Field(description="How this question resolves")
    prediction: str = Field(
        description="The probability or distribution, e.g. '35%' or '70% Option A, 20% Option B, 10% Option C' or '10% chance less than X units, ... ,90% chance less than Y units'"
    )
    reasoning: str = Field(description="2-4 sentence summary of the reasoning")
    key_sources: list[str] = Field(
        default_factory=list,
        description="URLs or source names used. Ideally both as markdown links.",
    )

    def as_footnote_markdown(self) -> str:
        sources_str = ", ".join(self.key_sources) if self.key_sources else "N/A"
        return (
            f"[^{self.footnote_id}]: **{self.question_title}**\n"
            f"- Question: {self.question_text}\n"
            f"- Resolution: {self.resolution_criteria}\n"
            f"- Prediction: {self.prediction}\n"
            f"- Reasoning: {self.reasoning}\n"
            f"- Sources: {sources_str}"
        )


class PolicyProposal(BaseModel, Jsonable):
    member: CongressMember | None = Field(
        default=None, description="The congress member who created this proposal"
    )
    research_summary: str = Field(description="Markdown summary of background research")
    decision_criteria: list[str] = Field(
        description="Prioritized criteria for this member"
    )
    forecasts: list[ForecastDescription] = Field(
        description="Extracted forecast details"
    )
    proposal_markdown: str = Field(
        description="Full proposal with footnote references [^1], [^2], etc."
    )
    key_recommendations: list[str] = Field(
        description="Top 3-5 actionable recommendations"
    )
    price_estimate: float | None = Field(
        default=None, description="Estimated cost in USD for generating this proposal"
    )

    def get_full_markdown_with_footnotes(self) -> str:
        footnotes = "\n\n".join(f.as_footnote_markdown() for f in self.forecasts)
        return f"{self.proposal_markdown}\n\n---\n\n## Forecast Appendix\n\n{footnotes}"


class CongressSessionInput(BaseModel, Jsonable):
    prompt: str
    member_names: list[str]


class CongressSession(BaseModel, Jsonable):
    prompt: str
    members_participating: list[CongressMember]
    proposals: list[PolicyProposal]
    aggregated_report_markdown: str
    blog_post: str = Field(default="")
    future_snapshot: str = Field(default="")
    twitter_posts: list[str] = Field(default_factory=list)
    timestamp: datetime
    errors: list[str] = Field(default_factory=list)
    total_price_estimate: float | None = Field(
        default=None, description="Total estimated cost in USD for the entire session"
    )

    def get_all_forecasts(self) -> list[ForecastDescription]:
        all_forecasts = []
        for proposal in self.proposals:
            for forecast in proposal.forecasts:
                all_forecasts.append(forecast)
        return all_forecasts

    def get_forecasts_by_member(self) -> dict[str, list[ForecastDescription]]:
        result: dict[str, list[ForecastDescription]] = {}
        for proposal in self.proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            result[member_name] = proposal.forecasts
        return result
