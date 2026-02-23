import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    ForecastQuestion,
    ScenarioSet,
)
from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
    generate_forecast_questions,
)
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_forecast_questions,
    load_research_reports,
    load_scenario_set,
    require_active_session,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)


async def main() -> None:
    st.title("4. Forecasting Questions")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    existing_questions = load_forecast_questions(db, session_id)
    if existing_questions:
        st.subheader(f"Current Questions ({len(existing_questions)})")
        for i, q in enumerate(existing_questions):
            cond = ""
            if q.conditional_on_scenario:
                cond = f" | Scenario: {q.conditional_on_scenario}"
            with st.expander(f"Q{i+1}: {q.question_text[:80]}..."):
                st.write(f"**Type:** {q.question_type}{cond}")
                st.write(f"**Resolution Criteria:** {q.resolution_criteria}")
                if st.button(f"Delete Q{i+1}", key=f"del_q_{q.question_id}"):
                    db.delete_artifact(q.question_id)
                    st.rerun()

    st.markdown("---")
    st.subheader("Generate AI Forecast Questions")

    scenarios = load_scenario_set(db, session_id)
    research_reports = load_research_reports(db, session_id)
    research_context = "\n\n".join(r.report_markdown for r in research_reports)

    if not scenarios:
        st.warning("No scenarios found. Go to **3. Scenarios** first.")

    if st.button("Generate Questions", key="gen_questions_btn", type="primary"):
        if not scenarios:
            scenarios = ScenarioSet(session_id=session_id, username=username)
        with st.spinner("Generating forecasting questions..."):
            with MonetaryCostManager():
                questions = await generate_forecast_questions(
                    policy_question,
                    research_context or "No prior research.",
                    scenarios,
                    session_id,
                    username,
                )
        for q in questions:
            db.save_artifact(session_id, "forecast_questions", q.to_json(), username)
        st.success(f"Generated {len(questions)} questions!")
        st.rerun()

    st.markdown("---")
    st.subheader("Add Your Own Question")

    with st.form("manual_question_form"):
        q_text = st.text_area("Question Text", key="manual_q_text")
        q_resolution = st.text_area(
            "Resolution Criteria",
            key="manual_q_resolution",
            placeholder="How would we determine if this resolved yes or no?",
        )
        q_type = st.selectbox(
            "Question Type",
            ["baseline", "scenario_conditional", "proposal_conditional"],
            key="manual_q_type",
        )
        q_scenario = st.text_input(
            "Conditional on scenario (if applicable)",
            key="manual_q_scenario",
        )
        if st.form_submit_button("Add Question"):
            if q_text.strip():
                question = ForecastQuestion(
                    session_id=session_id,
                    username=username,
                    question_text=q_text.strip(),
                    resolution_criteria=q_resolution.strip(),
                    question_type=q_type,
                    conditional_on_scenario=q_scenario.strip() or None,
                )
                db.save_artifact(
                    session_id,
                    "forecast_questions",
                    question.to_json(),
                    username,
                )
                st.success("Question added!")
                st.rerun()
            else:
                st.error("Please enter question text.")


asyncio.run(main())
