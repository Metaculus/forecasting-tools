import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.database import get_database


def main() -> None:
    st.title("Decision Hub")
    st.subheader("Login")

    if st.session_state.get("username"):
        st.success(f"Logged in as **{st.session_state['username']}**")
        _display_session_picker()
        return

    st.write("Enter a username to get started. No password required.")
    username = st.text_input(
        "Username",
        key="login_username_input",
        placeholder="your_username",
    )
    if st.button("Login", key="login_btn"):
        if not username or not username.strip():
            st.error("Please enter a username.")
            return
        username = username.strip()
        db = get_database()
        db.create_user(username)
        st.session_state["username"] = username
        st.rerun()


def _display_session_picker() -> None:
    username = st.session_state["username"]
    db = get_database()
    sessions = db.list_sessions(username)

    st.markdown("---")

    if sessions:
        st.subheader("Your Analyses")
        for session in sessions:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"**{session.policy_question[:80]}**")
            with col2:
                st.caption(
                    f"{session.status} | {session.created_at.strftime('%Y-%m-%d %H:%M') if hasattr(session.created_at, 'strftime') else str(session.created_at)[:16]}"
                )
            with col3:
                if st.button(
                    "Open",
                    key=f"open_session_{session.session_id}",
                ):
                    st.session_state["active_session_id"] = session.session_id
                    st.session_state["active_session"] = session
                    st.rerun()
    else:
        st.info("No analyses yet. Go to **1. Policy Question** to create one.")

    st.markdown("---")
    st.write(
        "Or navigate to **1. Policy Question** in the sidebar to create a new analysis."
    )


main()
