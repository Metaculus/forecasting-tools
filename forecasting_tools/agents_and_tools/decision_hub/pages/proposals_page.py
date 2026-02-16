import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    PolicyProposal,
    ScenarioSet,
)
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_forecast_questions,
    load_forecasts,
    load_proposal,
    load_research_reports,
    load_scenario_set,
    require_active_session,
    stream_agent_response,
)
from forecasting_tools.agents_and_tools.decision_hub.proposal_agent import (
    generate_proposal_streamed,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


async def main() -> None:
    st.title("6. Policy Proposals")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    existing_proposal = load_proposal(db, session_id)
    if existing_proposal:
        st.subheader("Current Proposal")
        st.markdown(existing_proposal.proposal_markdown)
        if existing_proposal.key_recommendations:
            st.markdown("### Key Recommendations")
            for r in existing_proposal.key_recommendations:
                st.write(f"- {r}")
        if existing_proposal.contingency_plans:
            st.markdown("### Contingency Plans")
            for c in existing_proposal.contingency_plans:
                st.write(f"- {c}")

    st.markdown("---")
    st.subheader("Generate AI Proposal")

    scenarios = load_scenario_set(db, session_id)
    questions = load_forecast_questions(db, session_id)
    forecasts = load_forecasts(db, session_id)
    research_reports = load_research_reports(db, session_id)
    research_context = "\n\n".join(r.report_markdown for r in research_reports)

    prereq_missing = []
    if not research_context.strip():
        prereq_missing.append("Research (Step 2)")
    if not scenarios:
        prereq_missing.append("Scenarios (Step 3)")
    if not forecasts:
        prereq_missing.append("Forecasts (Step 5)")

    if prereq_missing:
        st.warning(
            f"Missing prerequisites: {', '.join(prereq_missing)}. Generation will use available data."
        )

    if st.button("Generate Proposal", key="gen_proposal_btn", type="primary"):
        with MonetaryCostManager():
            result = generate_proposal_streamed(
                policy_question,
                research_context or "No prior research.",
                scenarios or ScenarioSet(session_id=session_id, username=username),
                questions,
                forecasts,
            )
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            proposal = await structure_output(
                streamed_text,
                PolicyProposal,
                additional_instructions=clean_indents(
                    f"""
                    Extract the policy proposal.
                    Set session_id to "{session_id}" and username to "{username}".
                    Put full markdown in proposal_markdown.
                    Extract key_recommendations as list of strings.
                    Extract contingency_plans as list of strings.
                    """
                ),
            )
            proposal.session_id = session_id
            proposal.username = username
            db.save_artifact(session_id, "proposal", proposal.to_json(), username)
            st.success("Proposal saved!")
            st.rerun()

    st.markdown("---")
    st.subheader("Write Your Own Proposal")

    with st.form("manual_proposal_form"):
        manual_text = st.text_area(
            "Proposal (markdown)", height=300, key="manual_proposal_text"
        )
        manual_recs = st.text_area(
            "Key Recommendations (one per line)",
            key="manual_proposal_recs",
        )
        if st.form_submit_button("Save Proposal"):
            if manual_text.strip():
                recs = [r.strip() for r in manual_recs.strip().split("\n") if r.strip()]
                proposal = PolicyProposal(
                    session_id=session_id,
                    username=username,
                    proposal_markdown=manual_text.strip(),
                    key_recommendations=recs,
                )
                db.save_artifact(session_id, "proposal", proposal.to_json(), username)
                st.success("Proposal saved!")
                st.rerun()
            else:
                st.error("Please enter proposal text.")


asyncio.run(main())
