import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import AnalysisSession
from forecasting_tools.agents_and_tools.decision_hub.database import get_database


def main() -> None:
    st.title("1. Policy Question")

    username = st.session_state.get("username")
    if not username:
        st.warning("Please log in first.")
        return

    active_session = st.session_state.get("active_session")
    if active_session:
        st.success(f"Active analysis: **{active_session.policy_question}**")
        st.info(
            "Navigate to the next steps in the sidebar, or create a new analysis below."
        )
        st.markdown("---")

    st.subheader("Create New Analysis")
    policy_question = st.text_area(
        "What policy question do you want to analyze?",
        height=120,
        placeholder=(
            "e.g., Should the EU impose tariffs on Chinese electric vehicles?\n"
            "e.g., Should the US adopt a universal basic income?\n"
            "e.g., What should our company's AI strategy be for the next 3 years?"
        ),
        key="new_policy_question_input",
    )

    if st.button("Create Analysis", key="create_analysis_btn", type="primary"):
        if not policy_question or not policy_question.strip():
            st.error("Please enter a policy question.")
            return

        db = get_database()
        session = AnalysisSession(
            username=username,
            policy_question=policy_question.strip(),
        )
        db.create_session(session)
        st.session_state["active_session_id"] = session.session_id
        st.session_state["active_session"] = session
        st.success("Analysis created! Navigate to **2. Research** in the sidebar.")
        st.rerun()


main()
