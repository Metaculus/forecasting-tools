import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import SynthesisReport
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_forecast_questions,
    load_forecasts,
    load_proposal,
    load_research_reports,
    load_robustness,
    load_scenario_set,
    load_synthesis,
    require_active_session,
    stream_agent_response,
)
from forecasting_tools.agents_and_tools.decision_hub.synthesis_agent import (
    synthesize_report_streamed,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


async def main() -> None:
    st.title("8. Synthesis & Report")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    existing_synthesis = load_synthesis(db, session_id)
    if existing_synthesis:
        tab_exec, tab_full, tab_blog, tab_future = st.tabs(
            ["Executive Summary", "Full Report", "Blog Post", "Picture of the Future"]
        )
        with tab_exec:
            st.markdown(existing_synthesis.executive_summary or "*Not yet generated.*")
        with tab_full:
            st.markdown(
                existing_synthesis.full_report_markdown or "*Not yet generated.*"
            )
        with tab_blog:
            st.markdown(existing_synthesis.blog_post or "*Not yet generated.*")
        with tab_future:
            st.markdown(existing_synthesis.future_snapshot or "*Not yet generated.*")

        st.download_button(
            "Download Full Report",
            data=existing_synthesis.full_report_markdown or "",
            file_name="decision_hub_report.md",
            mime="text/markdown",
        )

    st.markdown("---")
    st.subheader("Generate AI Synthesis")

    research_reports = load_research_reports(db, session_id)
    scenarios = load_scenario_set(db, session_id)
    questions = load_forecast_questions(db, session_id)
    forecasts = load_forecasts(db, session_id)
    proposal = load_proposal(db, session_id)
    robustness = load_robustness(db, session_id)

    available_steps = []
    if research_reports:
        available_steps.append("Research")
    if scenarios:
        available_steps.append("Scenarios")
    if questions:
        available_steps.append("Forecast Questions")
    if forecasts:
        available_steps.append("Forecasts")
    if proposal:
        available_steps.append("Proposal")
    if robustness:
        available_steps.append("Robustness")

    if available_steps:
        st.write(f"Available data for synthesis: {', '.join(available_steps)}")
    else:
        st.warning("No data available. Complete earlier steps first.")

    if st.button(
        "Generate Synthesis",
        key="gen_synthesis_btn",
        type="primary",
        disabled=not available_steps,
    ):
        with MonetaryCostManager():
            result = synthesize_report_streamed(
                policy_question,
                research_reports,
                scenarios,
                questions,
                forecasts,
                proposal,
                robustness,
            )
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            synthesis = await structure_output(
                streamed_text,
                SynthesisReport,
                additional_instructions=clean_indents(
                    f"""
                    Extract the synthesis report sections.
                    Set session_id to "{session_id}" and username to "{username}".
                    - executive_summary: the Executive Summary section
                    - full_report_markdown: the Full Analysis section
                    - blog_post: the Blog Post section
                    - future_snapshot: the Picture of the Future section
                    """
                ),
            )
            synthesis.session_id = session_id
            synthesis.username = username
            db.save_artifact(session_id, "synthesis", synthesis.to_json(), username)
            st.success("Synthesis saved!")
            st.rerun()

    st.markdown("---")
    st.subheader("Write Your Own Summary")

    with st.form("manual_synthesis_form"):
        manual_exec = st.text_area(
            "Executive Summary", height=150, key="manual_synthesis_exec"
        )
        manual_full = st.text_area(
            "Full Report (markdown)", height=300, key="manual_synthesis_full"
        )
        if st.form_submit_button("Save Synthesis"):
            if manual_exec.strip() or manual_full.strip():
                synthesis = SynthesisReport(
                    session_id=session_id,
                    username=username,
                    executive_summary=manual_exec.strip(),
                    full_report_markdown=manual_full.strip(),
                )
                db.save_artifact(session_id, "synthesis", synthesis.to_json(), username)
                st.success("Synthesis saved!")
                st.rerun()
            else:
                st.error("Please enter at least an executive summary or full report.")


asyncio.run(main())
