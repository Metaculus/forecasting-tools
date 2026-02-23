import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import RedTeamResult
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    load_proposal,
    load_robustness,
    load_synthesis,
    stream_agent_response,
)
from forecasting_tools.agents_and_tools.decision_hub.red_team_agent import (
    red_team_streamed,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)


async def main() -> None:
    st.title("Red Teaming Tool")

    username = st.session_state.get("username")
    if not username:
        st.warning("Please log in first.")
        return

    st.write(
        "Paste any text (a proposal, research report, analysis, etc.) and get "
        "an adversarial critique or bias analysis."
    )

    mode = st.radio(
        "Mode",
        options=["devils_advocate", "bias_detector"],
        format_func=lambda x: {
            "devils_advocate": "Devil's Advocate (Contrarian Critique)",
            "bias_detector": "Bias Detector (Cognitive Bias Analysis)",
        }[x],
        key="red_team_mode",
        horizontal=True,
    )

    session_id = st.session_state.get("active_session_id")

    if session_id:
        st.markdown("---")
        st.subheader("Quick Load from Current Analysis")
        db = get_db()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Load Proposal", key="rt_load_proposal"):
                proposal = load_proposal(db, session_id)
                if proposal:
                    st.session_state["red_team_input"] = proposal.proposal_markdown
                    st.rerun()
                else:
                    st.warning("No proposal found.")
        with col2:
            if st.button("Load Robustness", key="rt_load_robustness"):
                robustness = load_robustness(db, session_id)
                if robustness:
                    text = f"{robustness.matrix_markdown}\n\n{robustness.robust_recommendations}\n\n{robustness.hedging_strategy}"
                    st.session_state["red_team_input"] = text
                    st.rerun()
                else:
                    st.warning("No robustness analysis found.")
        with col3:
            if st.button("Load Synthesis", key="rt_load_synthesis"):
                synthesis = load_synthesis(db, session_id)
                if synthesis:
                    st.session_state["red_team_input"] = synthesis.full_report_markdown
                    st.rerun()
                else:
                    st.warning("No synthesis found.")

    st.markdown("---")
    input_text = st.text_area(
        "Text to analyze",
        height=300,
        key="red_team_input",
        placeholder="Paste the text you want to red-team here...",
    )

    if st.button(
        "Run Red Team Analysis",
        key="run_red_team_btn",
        type="primary",
        disabled=not input_text.strip() if input_text else True,
    ):
        with MonetaryCostManager():
            result = red_team_streamed(input_text, mode)
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            red_team_result = RedTeamResult(
                session_id=session_id,
                username=username,
                input_text=input_text[:500],
                mode=mode,
                critique_markdown=streamed_text,
            )
            db = get_db()
            db.save_artifact(
                session_id or "red_team_standalone",
                "red_team",
                red_team_result.to_json(),
                username,
            )
            st.success("Red team analysis complete and saved!")


asyncio.run(main())
