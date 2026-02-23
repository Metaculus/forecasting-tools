import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import RobustnessReport
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_forecast_questions,
    load_forecasts,
    load_proposal,
    load_robustness,
    load_scenario_set,
    require_active_session,
    stream_agent_response,
)
from forecasting_tools.agents_and_tools.decision_hub.robustness_agent import (
    analyze_robustness_streamed,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


async def main() -> None:
    st.title("7. Robustness Analysis")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    existing_robustness = load_robustness(db, session_id)
    if existing_robustness:
        st.subheader("Current Robustness Analysis")
        if existing_robustness.matrix_markdown:
            st.markdown("### Robustness Matrix")
            st.markdown(existing_robustness.matrix_markdown)
        if existing_robustness.robust_recommendations:
            st.markdown("### Robust Recommendations")
            st.markdown(existing_robustness.robust_recommendations)
        if existing_robustness.scenario_dependent_recs:
            st.markdown("### Scenario-Dependent Recommendations")
            st.markdown(existing_robustness.scenario_dependent_recs)
        if existing_robustness.hedging_strategy:
            st.markdown("### Hedging Strategy")
            st.markdown(existing_robustness.hedging_strategy)

    st.markdown("---")
    st.subheader("Generate AI Robustness Analysis")

    scenarios = load_scenario_set(db, session_id)
    proposal = load_proposal(db, session_id)
    questions = load_forecast_questions(db, session_id)
    forecasts = load_forecasts(db, session_id)

    prereq_missing = []
    if not scenarios:
        prereq_missing.append("Scenarios (Step 3)")
    if not proposal:
        prereq_missing.append("Proposal (Step 6)")

    if prereq_missing:
        st.warning(f"Missing prerequisites: {', '.join(prereq_missing)}")

    if st.button(
        "Generate Robustness Analysis",
        key="gen_robustness_btn",
        type="primary",
        disabled=not (scenarios and proposal),
    ):
        with MonetaryCostManager():
            result = analyze_robustness_streamed(
                policy_question,
                scenarios,
                proposal,
                questions,
                forecasts,
            )
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            report = await structure_output(
                streamed_text,
                RobustnessReport,
                additional_instructions=clean_indents(
                    f"""
                    Extract the robustness analysis.
                    Set session_id to "{session_id}" and username to "{username}".
                    Put the matrix table in matrix_markdown.
                    Put robust recommendations in robust_recommendations.
                    Put scenario-dependent recs in scenario_dependent_recs.
                    Put hedging strategy in hedging_strategy.
                    """
                ),
            )
            report.session_id = session_id
            report.username = username
            db.save_artifact(session_id, "robustness", report.to_json(), username)
            st.success("Robustness analysis saved!")
            st.rerun()

    st.markdown("---")
    st.subheader("Add Your Own Analysis")

    with st.form("manual_robustness_form"):
        manual_matrix = st.text_area(
            "Robustness Matrix (markdown table)",
            height=150,
            key="manual_robustness_matrix",
        )
        manual_robust = st.text_area(
            "Robust Recommendations",
            height=100,
            key="manual_robustness_robust",
        )
        manual_hedging = st.text_area(
            "Hedging Strategy",
            height=100,
            key="manual_robustness_hedging",
        )
        if st.form_submit_button("Save Analysis"):
            report = RobustnessReport(
                session_id=session_id,
                username=username,
                matrix_markdown=manual_matrix.strip(),
                robust_recommendations=manual_robust.strip(),
                hedging_strategy=manual_hedging.strip(),
            )
            db.save_artifact(session_id, "robustness", report.to_json(), username)
            st.success("Analysis saved!")
            st.rerun()


asyncio.run(main())
