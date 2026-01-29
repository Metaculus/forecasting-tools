from __future__ import annotations

import json
import logging
import os

import streamlit as st

from forecasting_tools.agents_and_tools.ai_congress.congress_orchestrator import (
    CongressOrchestrator,
)
from forecasting_tools.agents_and_tools.ai_congress.data_models import (
    CongressSession,
    CongressSessionInput,
)
from forecasting_tools.agents_and_tools.ai_congress.member_profiles import (
    AVAILABLE_MEMBERS,
    get_members_by_names,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.custom_auth import CustomAuth
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer

logger = logging.getLogger(__name__)

SESSIONS_FOLDER = "temp/congress_sessions"


class CongressPage(AppPage):
    PAGE_DISPLAY_NAME: str = "🏛️ AI Forecasting Congress"
    URL_PATH: str = "/ai-congress"
    IS_DEFAULT_PAGE: bool = False

    @classmethod
    @CustomAuth.add_access_control()
    async def _async_main(cls) -> None:
        st.title("🏛️ AI Forecasting Congress")
        st.markdown(
            """
            Simulate a deliberative body of AI agents with different political
            perspectives analyzing a policy question. Each member conducts research,
            generates forecasting questions, makes quantitative predictions, and
            proposes policy recommendations.
            """
        )

        cls._display_sidebar()

        session_input = await cls._get_input()
        if session_input:
            session = await cls._run_congress(session_input)
            cls._save_session(session)
            st.session_state["latest_session"] = session

        if "latest_session" in st.session_state:
            cls._display_session(st.session_state["latest_session"])

    @classmethod
    def _display_sidebar(cls) -> None:
        with st.sidebar:
            st.header("Load Session")

            st.subheader("From File Path")
            file_path = st.text_input(
                "Enter JSON file path:",
                placeholder="temp/congress_sessions/20260129_123456.json",
                key="load_file_path",
            )
            if st.button("Load from File", key="load_file_btn"):
                if file_path:
                    session = cls._load_session_from_file(file_path)
                    if session:
                        st.session_state["latest_session"] = session
                        st.success(f"Loaded session from {file_path}")
                        st.rerun()
                else:
                    st.error("Please enter a file path.")

            st.markdown("---")
            st.subheader("From Recent Sessions")
            sessions = cls._load_previous_sessions()
            if sessions:
                session_options = [
                    f"{s.timestamp.strftime('%Y-%m-%d %H:%M')} - {s.prompt[:30]}..."
                    for s in sessions
                ]
                selected_idx = st.selectbox(
                    "Select a session:",
                    range(len(sessions)),
                    format_func=lambda i: session_options[i],
                    key="previous_session_select",
                )
                if st.button("Load Selected", key="load_selected_btn"):
                    st.session_state["latest_session"] = sessions[selected_idx]
                    st.rerun()
            else:
                st.write("No recent sessions found.")

            st.markdown("---")
            st.header("About")
            st.markdown(
                """
                **Members Available:**
                """
            )
            for member in AVAILABLE_MEMBERS:
                st.markdown(f"- **{member.name}**: {member.role}")

    EXAMPLE_PROMPTS: list[dict[str, str]] = [
        {
            "title": "AI Regulation",
            "prompt": (
                "How should the United States regulate artificial intelligence? "
                "Consider both frontier AI systems (like large language models) and "
                "narrower AI applications in areas like hiring, lending, and healthcare. "
                "What policies would balance innovation with safety and civil liberties?"
            ),
        },
        {
            "title": "Nuclear Policy",
            "prompt": (
                "What should US nuclear weapons policy be going forward? "
                "Consider modernization of the nuclear triad, arms control agreements, "
                "extended deterrence commitments to allies, and the role of tactical "
                "nuclear weapons in an era of great power competition."
            ),
        },
        {
            "title": "Climate Change",
            "prompt": (
                "What climate policies should the US adopt to meet its emissions "
                "reduction targets? Consider carbon pricing, clean energy subsidies, "
                "regulations on fossil fuels, and adaptation measures. How should costs "
                "and benefits be distributed across different communities?"
            ),
        },
        {
            "title": "Immigration Reform",
            "prompt": (
                "How should the US reform its immigration system? Consider border "
                "security, pathways to legal status, high-skilled immigration, refugee "
                "admissions, and enforcement priorities. What policies would best serve "
                "economic, humanitarian, and security interests?"
            ),
        },
        {
            "title": "Healthcare System",
            "prompt": (
                "How should the US improve its healthcare system? Consider coverage "
                "expansion, cost control, drug pricing, mental health services, and "
                "the role of public vs private insurance. What reforms would improve "
                "outcomes while managing costs?"
            ),
        },
    ]

    @classmethod
    async def _get_input(cls) -> CongressSessionInput | None:
        st.header("Start a New Session")

        with st.expander("📋 Example Policy Questions", expanded=False):
            st.markdown("Click a button to use an example prompt:")
            cols = st.columns(len(cls.EXAMPLE_PROMPTS))
            for i, example in enumerate(cls.EXAMPLE_PROMPTS):
                with cols[i]:
                    if st.button(
                        example["title"], key=f"example_{i}", use_container_width=True
                    ):
                        st.session_state["example_prompt"] = example["prompt"]
                        st.rerun()
            if st.session_state.get("example_prompt"):
                st.write(st.session_state["example_prompt"])

        default_prompt = st.session_state.pop("example_prompt", "")

        with st.form("congress_form"):
            prompt = st.text_area(
                "Policy Question",
                value=default_prompt,
                placeholder="Enter a policy question to deliberate on (e.g., 'What should US nuclear policy be?' or 'How should we regulate AI?')",
                height=100,
                key="congress_prompt",
            )

            member_names = [m.name for m in AVAILABLE_MEMBERS]
            default_members = [
                "Opus 4.5 (Anthropic)",
                "GPT 5.2 (OpenAI)",
                "Gemini 3 Pro (Google)",
                "Grok 4 (xAI)",
                "DeepSeek V3.2 (DeepSeek)",
            ]
            selected_members = st.multiselect(
                "Select Congress Members",
                options=member_names,
                default=default_members,
                key="congress_members",
            )

            st.markdown(
                """
                **Estimated Cost:** ~$3-8 per member selected
                (depends on model and research depth)
                """
            )

            submitted = st.form_submit_button("🏛️ Convene Congress")

            if submitted:
                if not prompt:
                    st.error("Please enter a policy question.")
                    return None
                if len(selected_members) < 2:
                    st.error("Please select at least 2 congress members.")
                    return None

                return CongressSessionInput(
                    prompt=prompt,
                    member_names=selected_members,
                )

        return None

    @classmethod
    async def _run_congress(
        cls, session_input: CongressSessionInput
    ) -> CongressSession:
        members = get_members_by_names(session_input.member_names)

        with st.spinner(
            f"Congress in session with {len(members)} members... "
            "This may take 5-15 minutes."
        ):
            progress_text = st.empty()
            progress_text.write("Members are researching and deliberating...")

            orchestrator = CongressOrchestrator()
            session = await orchestrator.run_session(
                prompt=session_input.prompt,
                members=members,
            )

            progress_text.write("Aggregating proposals and generating insights...")

        if session.errors:
            st.warning(
                f"⚠️ {len(session.errors)} member(s) encountered errors. "
                "Partial results shown."
            )

        return session

    @classmethod
    def _display_session(cls, session: CongressSession) -> None:
        st.header("Congress Results")

        tabs = st.tabs(
            [
                "📊 Synthesis",
                "📝 Blog Post",
                "👤 Individual Proposals",
                "🎯 Forecast Comparison",
                "🐦 Twitter Posts",
            ]
        )

        with tabs[0]:
            cls._display_synthesis_tab(session)

        with tabs[1]:
            cls._display_blog_tab(session)

        with tabs[2]:
            cls._display_proposals_tab(session)

        with tabs[3]:
            cls._display_forecasts_tab(session)

        with tabs[4]:
            cls._display_twitter_tab(session)

        cls._display_download_buttons(session)

    @classmethod
    def _display_synthesis_tab(cls, session: CongressSession) -> None:
        st.subheader("Aggregated Report")
        if session.aggregated_report_markdown:
            cleaned = ReportDisplayer.clean_markdown(session.aggregated_report_markdown)
            st.markdown(cleaned)
        else:
            st.write("No aggregated report available.")

        if session.errors:
            with st.expander("⚠️ Errors During Session"):
                for error in session.errors:
                    st.error(error)

    @classmethod
    def _display_blog_tab(cls, session: CongressSession) -> None:
        st.subheader("Blog Post")
        if session.blog_post:
            cleaned = ReportDisplayer.clean_markdown(session.blog_post)
            st.markdown(cleaned)

            st.download_button(
                label="📥 Download Blog Post (Markdown)",
                data=session.blog_post,
                file_name=f"congress_blog_{session.timestamp.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_blog",
            )
        else:
            st.write("No blog post available.")

    @classmethod
    def _display_proposals_tab(cls, session: CongressSession) -> None:
        st.subheader("Individual Member Proposals")

        if not session.proposals:
            st.write("No proposals available.")
            return

        for proposal in session.proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            member_role = proposal.member.role if proposal.member else ""

            with st.expander(f"**{member_name}** - {member_role}", expanded=False):
                st.markdown("#### Decision Criteria")
                for i, criterion in enumerate(proposal.decision_criteria, 1):
                    st.markdown(f"{i}. {criterion}")

                st.markdown("#### Key Recommendations")
                for rec in proposal.key_recommendations:
                    st.markdown(f"- {rec}")

                st.markdown("#### Full Proposal")
                cleaned = ReportDisplayer.clean_markdown(
                    proposal.get_full_markdown_with_footnotes()
                )
                st.markdown(cleaned)

    @classmethod
    def _display_forecasts_tab(cls, session: CongressSession) -> None:
        st.subheader("Forecast Comparison")

        forecasts_by_member = session.get_forecasts_by_member()

        if not forecasts_by_member:
            st.write("No forecasts available.")
            return

        all_forecasts_data = []
        for member_name, forecasts in forecasts_by_member.items():
            for f in forecasts:
                all_forecasts_data.append(
                    {
                        "Member": member_name,
                        "Question": f.question_title,
                        "Prediction": f.prediction,
                        "Reasoning (summary)": (
                            f.reasoning[:100] + "..."
                            if len(f.reasoning) > 100
                            else f.reasoning
                        ),
                    }
                )

        if all_forecasts_data:
            st.dataframe(all_forecasts_data, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Detailed Forecasts by Member")

        for member_name, forecasts in forecasts_by_member.items():
            with st.expander(f"**{member_name}** ({len(forecasts)} forecasts)"):
                for f in forecasts:
                    st.markdown(f"**[^{f.footnote_id}] {f.question_title}**")
                    st.markdown(f"- **Prediction:** {f.prediction}")
                    st.markdown(f"- **Question:** {f.question_text}")
                    st.markdown(f"- **Resolution:** {f.resolution_criteria}")
                    st.markdown(f"- **Reasoning:** {f.reasoning}")
                    if f.key_sources:
                        st.markdown(f"- **Sources:** {', '.join(f.key_sources)}")
                    st.markdown("---")

    @classmethod
    def _display_twitter_tab(cls, session: CongressSession) -> None:
        st.subheader("Twitter/X Posts")
        st.markdown(
            "These tweet-sized excerpts highlight interesting patterns from the "
            "congress session."
        )

        if not session.twitter_posts:
            st.write("No Twitter posts generated.")
            return

        for i, post in enumerate(session.twitter_posts, 1):
            st.markdown(f"**Tweet {i}** ({len(post)} chars)")
            st.info(post)
            st.button(f"📋 Copy Tweet {i}", key=f"copy_tweet_{i}")

    @classmethod
    def _display_download_buttons(cls, session: CongressSession) -> None:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            json_str = json.dumps(session.to_json(), indent=2, default=str)
            st.download_button(
                label="📥 Download Full Session (JSON)",
                data=json_str,
                file_name=f"congress_session_{session.timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        with col2:
            markdown_content = cls._session_to_markdown(session)
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=markdown_content,
                file_name=f"congress_report_{session.timestamp.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

    @classmethod
    def _session_to_markdown(cls, session: CongressSession) -> str:
        lines = [
            "# AI Forecasting Congress Report",
            "",
            f"**Policy Question:** {session.prompt}",
            "",
            f"**Date:** {session.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"**Members:** {', '.join(m.name for m in session.members_participating)}",
            "",
            "---",
            "",
            "## Synthesis Report",
            "",
            session.aggregated_report_markdown,
            "",
            "---",
            "",
            "## Individual Proposals",
            "",
        ]

        for proposal in session.proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            lines.extend(
                [
                    f"### {member_name}",
                    "",
                    proposal.get_full_markdown_with_footnotes(),
                    "",
                    "---",
                    "",
                ]
            )

        return "\n".join(lines)

    @classmethod
    def _save_session(cls, session: CongressSession) -> None:
        os.makedirs(SESSIONS_FOLDER, exist_ok=True)
        filename = f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(SESSIONS_FOLDER, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(session.to_json(), f, indent=2, default=str)
            logger.info(f"Saved session to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    @classmethod
    def _load_session_from_file(cls, file_path: str) -> CongressSession | None:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            session = CongressSession.from_json(data)
            return session
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
            return None
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            logger.error(f"Failed to load session from {file_path}: {e}")
            return None

    @classmethod
    def _load_previous_sessions(cls) -> list[CongressSession]:
        if not os.path.exists(SESSIONS_FOLDER):
            return []

        sessions = []
        for filename in sorted(os.listdir(SESSIONS_FOLDER), reverse=True)[:10]:
            if filename.endswith(".json"):
                filepath = os.path.join(SESSIONS_FOLDER, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    session = CongressSession.from_json(data)
                    sessions.append(session)
                except Exception as e:
                    logger.error(f"Failed to load session {filename}: {e}")

        return sessions


if __name__ == "__main__":
    CongressPage.main()
