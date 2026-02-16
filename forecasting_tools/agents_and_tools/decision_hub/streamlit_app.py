import os
import sys

import dotenv
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(top_level_dir)

from forecasting_tools.util.custom_logger import CustomLogger


def _initialize_session_state() -> None:
    defaults = {
        "username": None,
        "active_session_id": None,
        "active_session": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _display_active_session_info() -> None:
    if st.session_state.get("active_session"):
        session = st.session_state["active_session"]
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**User:** {st.session_state['username']}")
        st.sidebar.markdown(f"**Analysis:** {session.policy_question[:60]}...")
        st.sidebar.markdown(f"**Status:** {session.status}")
        if st.sidebar.button("Switch Analysis", key="switch_analysis_btn"):
            st.session_state["active_session_id"] = None
            st.session_state["active_session"] = None
            st.rerun()
        if st.sidebar.button("Log Out", key="logout_btn"):
            st.session_state["username"] = None
            st.session_state["active_session_id"] = None
            st.session_state["active_session"] = None
            st.rerun()


def run_decision_hub() -> None:
    dotenv.load_dotenv()
    if "logger_initialized" not in st.session_state:
        CustomLogger.setup_logging()
        st.session_state["logger_initialized"] = True

    _initialize_session_state()

    login_page = st.Page(
        "pages/login_page.py",
        title="Login",
        icon=":material/person:",
        default=True,
    )
    question_page = st.Page(
        "pages/question_page.py",
        title="1. Policy Question",
        icon=":material/help:",
    )
    research_page = st.Page(
        "pages/research_page.py",
        title="2. Research",
        icon=":material/search:",
    )
    scenarios_page = st.Page(
        "pages/scenarios_page.py",
        title="3. Scenarios",
        icon=":material/account_tree:",
    )
    forecast_q_page = st.Page(
        "pages/forecast_q_page.py",
        title="4. Forecast Questions",
        icon=":material/quiz:",
    )
    forecasts_page = st.Page(
        "pages/forecasts_page.py",
        title="5. Forecasts",
        icon=":material/trending_up:",
    )
    proposals_page = st.Page(
        "pages/proposals_page.py",
        title="6. Proposals",
        icon=":material/description:",
    )
    robustness_page = st.Page(
        "pages/robustness_page.py",
        title="7. Robustness",
        icon=":material/shield:",
    )
    synthesis_page = st.Page(
        "pages/synthesis_page.py",
        title="8. Synthesis",
        icon=":material/auto_awesome:",
    )
    red_team_page = st.Page(
        "pages/red_team_page.py",
        title="Red Teaming",
        icon=":material/security:",
    )
    autopilot_page = st.Page(
        "pages/autopilot_page.py",
        title="Autopilot",
        icon=":material/rocket_launch:",
    )

    workflow_pages = [
        question_page,
        research_page,
        scenarios_page,
        forecast_q_page,
        forecasts_page,
        proposals_page,
        robustness_page,
        synthesis_page,
    ]
    tool_pages = [red_team_page, autopilot_page]

    has_user = st.session_state.get("username") is not None
    has_session = st.session_state.get("active_session_id") is not None

    if not has_user:
        nav = st.navigation({"Account": [login_page]})
    elif not has_session:
        nav = st.navigation(
            {
                "Account": [login_page],
                "Start": [question_page],
            }
        )
    else:
        nav = st.navigation(
            {
                "Account": [login_page],
                "Workflow": workflow_pages,
                "Tools": tool_pages,
            }
        )

    st.set_page_config(
        page_title="Decision Hub",
        page_icon=":material/hub:",
        layout="wide",
    )

    _display_active_session_info()
    nav.run()


if __name__ == "__main__":
    run_decision_hub()
