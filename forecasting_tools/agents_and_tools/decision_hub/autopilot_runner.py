from __future__ import annotations

import logging
from typing import Callable

from forecasting_tools.agents_and_tools.decision_hub.data_models import AnalysisSession
from forecasting_tools.agents_and_tools.decision_hub.database import DecisionHubDB
from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
    generate_forecast_questions,
    run_forecasts_to_completion,
)
from forecasting_tools.agents_and_tools.decision_hub.proposal_agent import (
    generate_proposal_to_completion,
)
from forecasting_tools.agents_and_tools.decision_hub.research_agent import (
    run_research_to_completion,
)
from forecasting_tools.agents_and_tools.decision_hub.robustness_agent import (
    analyze_robustness_to_completion,
)
from forecasting_tools.agents_and_tools.decision_hub.scenario_agent import (
    generate_scenarios_to_completion,
)
from forecasting_tools.agents_and_tools.decision_hub.synthesis_agent import (
    synthesize_report_to_completion,
)
from forecasting_tools.ai_models.agent_wrappers import general_trace_or_span
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)

STEP_NAMES = [
    "research",
    "scenarios",
    "forecast_questions",
    "forecasts",
    "proposal",
    "robustness",
    "synthesis",
]


async def run_full_pipeline(
    session_id: str,
    db: DecisionHubDB,
    on_step_start: Callable[[str], None] | None = None,
    on_step_complete: Callable[[str], None] | None = None,
    model: str = "openrouter/anthropic/claude-sonnet-4",
) -> AnalysisSession:
    session = db.get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    policy_question = session.policy_question
    username = session.username

    with general_trace_or_span("Decision Hub Autopilot"):
        with MonetaryCostManager():
            if on_step_start:
                on_step_start("research")
            research_report = await run_research_to_completion(
                policy_question,
                session_id,
                username,
                model=model,
            )
            db.save_artifact(
                session_id, "research", research_report.to_json(), username
            )
            if on_step_complete:
                on_step_complete("research")

            if on_step_start:
                on_step_start("scenarios")
            scenario_set = await generate_scenarios_to_completion(
                policy_question,
                research_report.report_markdown,
                session_id,
                username,
                model=model,
            )
            db.save_artifact(session_id, "scenarios", scenario_set.to_json(), username)
            if on_step_complete:
                on_step_complete("scenarios")

            if on_step_start:
                on_step_start("forecast_questions")
            forecast_questions = await generate_forecast_questions(
                policy_question,
                research_report.report_markdown,
                scenario_set,
                session_id,
                username,
                model=model,
            )
            for fq in forecast_questions:
                db.save_artifact(
                    session_id, "forecast_questions", fq.to_json(), username
                )
            if on_step_complete:
                on_step_complete("forecast_questions")

            if on_step_start:
                on_step_start("forecasts")
            forecasts = await run_forecasts_to_completion(
                forecast_questions,
                policy_question,
                research_report.report_markdown,
                scenario_set,
                session_id,
                username,
                model=model,
            )
            for f in forecasts:
                db.save_artifact(session_id, "forecasts", f.to_json(), username)
            if on_step_complete:
                on_step_complete("forecasts")

            if on_step_start:
                on_step_start("proposal")
            proposal = await generate_proposal_to_completion(
                policy_question,
                research_report.report_markdown,
                scenario_set,
                forecast_questions,
                forecasts,
                session_id,
                username,
                model=model,
            )
            db.save_artifact(session_id, "proposal", proposal.to_json(), username)
            if on_step_complete:
                on_step_complete("proposal")

            if on_step_start:
                on_step_start("robustness")
            robustness = await analyze_robustness_to_completion(
                policy_question,
                scenario_set,
                proposal,
                forecast_questions,
                forecasts,
                session_id,
                username,
                model=model,
            )
            db.save_artifact(session_id, "robustness", robustness.to_json(), username)
            if on_step_complete:
                on_step_complete("robustness")

            if on_step_start:
                on_step_start("synthesis")
            synthesis = await synthesize_report_to_completion(
                policy_question,
                [research_report],
                scenario_set,
                forecast_questions,
                forecasts,
                proposal,
                robustness,
                session_id,
                username,
                model=model,
            )
            db.save_artifact(session_id, "synthesis", synthesis.to_json(), username)
            if on_step_complete:
                on_step_complete("synthesis")

    session.status = "complete"
    db.update_session(session)
    return session
