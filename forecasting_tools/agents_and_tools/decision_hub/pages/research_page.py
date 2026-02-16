import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import ResearchReport
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_research_reports,
    require_active_session,
    stream_agent_response,
)
from forecasting_tools.agents_and_tools.decision_hub.research_agent import (
    run_research_streamed,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)


async def main() -> None:
    st.title("2. Background Research")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    existing_reports = load_research_reports(db, session_id)
    if existing_reports:
        st.subheader(f"Existing Research ({len(existing_reports)} report(s))")
        for i, report in enumerate(existing_reports):
            with st.expander(
                f"Report {i+1}: {report.query[:60]}...",
                expanded=(i == len(existing_reports) - 1),
            ):
                st.markdown(report.report_markdown)
                if report.sources:
                    st.caption(f"Sources: {len(report.sources)}")

    st.markdown("---")
    st.subheader("Generate AI Research")

    custom_query = st.text_input(
        "Custom research focus (optional)",
        placeholder="e.g., Focus on economic impact analysis",
        key="research_custom_query",
    )

    if st.button("Generate Research", key="gen_research_btn", type="primary"):
        with MonetaryCostManager():
            result = run_research_streamed(
                policy_question,
                custom_query=custom_query if custom_query.strip() else None,
            )
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            report = ResearchReport(
                session_id=session_id,
                username=username,
                query=custom_query.strip() if custom_query.strip() else policy_question,
                report_markdown=streamed_text,
            )
            db.save_artifact(session_id, "research", report.to_json(), username)
            st.success("Research saved!")
            st.rerun()

    st.markdown("---")
    st.subheader("Add Your Own Research")

    with st.form("manual_research_form"):
        manual_query = st.text_input("Topic / Query", key="manual_research_query")
        manual_text = st.text_area(
            "Research text (paste or write)",
            height=200,
            key="manual_research_text",
        )
        if st.form_submit_button("Save Manual Research"):
            if manual_text.strip():
                report = ResearchReport(
                    session_id=session_id,
                    username=username,
                    query=manual_query.strip() or "Manual entry",
                    report_markdown=manual_text.strip(),
                )
                db.save_artifact(session_id, "research", report.to_json(), username)
                st.success("Manual research saved!")
                st.rerun()
            else:
                st.error("Please enter some research text.")


asyncio.run(main())
