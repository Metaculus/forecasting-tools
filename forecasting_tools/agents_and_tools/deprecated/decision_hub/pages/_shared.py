from __future__ import annotations

import logging

import streamlit as st
from agents.result import RunResultStreaming
from openai.types.responses import ResponseTextDeltaEvent

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Forecast,
    ForecastQuestion,
    PolicyProposal,
    ResearchReport,
    RobustnessReport,
    ScenarioSet,
    SynthesisReport,
)
from forecasting_tools.agents_and_tools.decision_hub.database import (
    DecisionHubDB,
    get_database,
)
from forecasting_tools.ai_models.agent_wrappers import event_to_tool_message

logger = logging.getLogger(__name__)


def require_active_session() -> bool:
    if not st.session_state.get("username"):
        st.warning("Please log in first.")
        return False
    if not st.session_state.get("active_session_id"):
        st.warning("Please select or create an analysis first.")
        return False
    return True


def get_session_info() -> tuple[str, str, str]:
    return (
        st.session_state["username"],
        st.session_state["active_session_id"],
        st.session_state["active_session"].policy_question,
    )


def get_db() -> DecisionHubDB:
    return get_database()


async def stream_agent_response(result: RunResultStreaming) -> str:
    streamed_text = ""
    placeholder = st.empty()

    with st.spinner("Generating..."):
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                streamed_text += event.data.delta
            placeholder.markdown(streamed_text)

            tool_msg = event_to_tool_message(event)
            if tool_msg:
                with st.sidebar.expander("Agent Activity", expanded=False):
                    st.write(tool_msg)

    return streamed_text


def load_research_reports(db: DecisionHubDB, session_id: str) -> list[ResearchReport]:
    artifacts = db.load_artifacts(session_id, "research")
    return [ResearchReport(**a) for a in artifacts]


def load_scenario_set(db: DecisionHubDB, session_id: str) -> ScenarioSet | None:
    artifacts = db.load_artifacts(session_id, "scenarios")
    if not artifacts:
        return None
    return ScenarioSet(**artifacts[-1])


def load_forecast_questions(
    db: DecisionHubDB, session_id: str
) -> list[ForecastQuestion]:
    artifacts = db.load_artifacts(session_id, "forecast_questions")
    return [ForecastQuestion(**a) for a in artifacts]


def load_forecasts(db: DecisionHubDB, session_id: str) -> list[Forecast]:
    artifacts = db.load_artifacts(session_id, "forecasts")
    return [Forecast(**a) for a in artifacts]


def load_proposal(db: DecisionHubDB, session_id: str) -> PolicyProposal | None:
    artifacts = db.load_artifacts(session_id, "proposal")
    if not artifacts:
        return None
    return PolicyProposal(**artifacts[-1])


def load_robustness(db: DecisionHubDB, session_id: str) -> RobustnessReport | None:
    artifacts = db.load_artifacts(session_id, "robustness")
    if not artifacts:
        return None
    return RobustnessReport(**artifacts[-1])


def load_synthesis(db: DecisionHubDB, session_id: str) -> SynthesisReport | None:
    artifacts = db.load_artifacts(session_id, "synthesis")
    if not artifacts:
        return None
    return SynthesisReport(**artifacts[-1])
