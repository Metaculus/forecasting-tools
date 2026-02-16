from __future__ import annotations

import json
import logging
import os
import time

import streamlit as st

from forecasting_tools.agents_and_tools.ai_congress_v2.congress_orchestrator import (
    CongressOrchestrator,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.data_models import (
    CongressSession,
    CongressSessionInput,
    ForecastDescription,
    PolicyProposal,
    Scenario,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.member_profiles import (
    AVAILABLE_MEMBERS,
    get_members_by_names,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.custom_auth import CustomAuth
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.util.file_manipulation import (
    create_or_overwrite_file,
    load_json_file,
)

logger = logging.getLogger(__name__)

SESSIONS_FOLDER = "temp/congress_v2_sessions"
EXAMPLE_SESSION_PATH = (
    "forecasting_tools/front_end/example_outputs/congress_v2_page_example.json"
)
V2_PASSPHRASE = f"metac {CustomAuth.DEFAULT_PASSPHRASE}"


class CongressV2Page(AppPage):
    PAGE_DISPLAY_NAME: str = "🏛️ AI Congress V2 (Scenarios)"
    URL_PATH: str = "/ai-congress-v2"
    IS_DEFAULT_PAGE: bool = False

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
                "regulations on fossil fuels, and adaptation measures."
            ),
        },
        {
            "title": "Immigration Reform",
            "prompt": (
                "How should the US reform its immigration system? Consider border "
                "security, pathways to legal status, high-skilled immigration, refugee "
                "admissions, and enforcement priorities."
            ),
        },
        {
            "title": "Healthcare System",
            "prompt": (
                "How should the US improve its healthcare system? Consider coverage "
                "expansion, cost control, drug pricing, mental health services, and "
                "the role of public vs private insurance."
            ),
        },
    ]

    @classmethod
    @CustomAuth.add_access_control(allowed_passphrases=[V2_PASSPHRASE])
    async def _async_main(cls) -> None:
        st.title("🏛️ AI Forecasting Congress V2")
        st.markdown(
            """
            **Scenario-focused policy deliberation powered by AI.**

            - **Scenario Planning**: AI congress members identify plausible future scenarios,
              key drivers, and concrete scenario criteria
            - **Multi-Proposal Evaluation**: Members consider multiple policy options and
              evaluate them across scenarios
            - **Cross-Scenario Robustness**: Analysis of which policies work across all
              scenarios vs. which are fragile
            - **Contingency Plans**: "If X happens, do Y instead" recommendations
            - **Scenario Report**: A formal scenario planning document with narratives,
              early warning indicators, and strategic implications
            """
        )

        cls._display_sidebar()

        st.header("Start a New Session")
        cls._display_example_button()
        session_input = await cls._get_input()

        if session_input:
            session = await cls._run_congress(session_input)
            cls._save_session(session)
            st.session_state["latest_v2_session"] = session

        if "latest_v2_session" in st.session_state:
            cls._display_session(st.session_state["latest_v2_session"])

    @classmethod
    def _display_example_button(cls) -> None:
        with st.expander("Load Premade Example", expanded=False):
            if st.button("Load Example", key="v2_load_example_btn"):
                session = cls._load_session_from_file(EXAMPLE_SESSION_PATH)
                if session:
                    st.session_state["latest_v2_session"] = session
                    st.rerun()
                else:
                    st.error("Could not load the example session.")

    @classmethod
    def _display_sidebar(cls) -> None:
        with st.sidebar:
            st.header("Load Session")

            st.subheader("From File Path")
            file_path = st.text_input(
                "Enter JSON file path:",
                placeholder="temp/congress_v2_sessions/20260216_123456.json",
                key="v2_load_file_path",
            )
            if st.button("Load from File", key="v2_load_file_btn"):
                if file_path:
                    session = cls._load_session_from_file(file_path)
                    if session:
                        st.session_state["latest_v2_session"] = session
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
                    key="v2_previous_session_select",
                )
                if st.button("Load Selected", key="v2_load_selected_btn"):
                    st.session_state["latest_v2_session"] = sessions[selected_idx]
                    st.rerun()
            else:
                st.write("No recent sessions found.")

            st.markdown("---")
            st.header("About")
            st.markdown("**Members Available:**")
            for member in AVAILABLE_MEMBERS:
                st.markdown(f"- **{member.name}**: {member.role}")

    @classmethod
    async def _get_input(cls) -> CongressSessionInput | None:
        with st.expander("Example Prompts", expanded=False):
            st.markdown("Click a button to use an example prompt:")
            cols = st.columns(len(cls.EXAMPLE_PROMPTS))
            for i, example in enumerate(cls.EXAMPLE_PROMPTS):
                with cols[i]:
                    if st.button(
                        example["title"],
                        key=f"v2_example_{i}",
                        use_container_width=True,
                    ):
                        st.session_state["v2_example_prompt"] = example["prompt"]
                        st.rerun()
            if st.session_state.get("v2_example_prompt"):
                st.write(st.session_state["v2_example_prompt"])

        default_prompt = st.session_state.pop("v2_example_prompt", "")

        with st.form("congress_v2_form"):
            prompt = st.text_area(
                "Policy Question",
                value=default_prompt,
                placeholder=(
                    "Enter a policy question to deliberate on (e.g., "
                    "'What should US nuclear policy be?' or 'How should we regulate AI?')"
                ),
                height=100,
                key="v2_congress_prompt",
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
                key="v2_congress_members",
            )

            num_delphi_rounds = st.number_input(
                "Delphi Rounds",
                min_value=1,
                max_value=3,
                value=1,
                help=(
                    "Number of deliberation rounds. In round 1, members deliberate "
                    "independently (with scenario planning). In subsequent rounds, "
                    "each member sees others' reports and revises. More rounds "
                    "increase cost."
                ),
                key="v2_delphi_rounds",
            )

            cost_per_member = "~$1-12"
            if num_delphi_rounds > 1:
                cost_per_member = "~$1-12 for round 1 + ~$1-12 per additional round"
            st.markdown(
                f"**Estimated Cost:** {cost_per_member} per member selected "
                "(depends on model, research depth, and number of forecasts)"
            )

            submitted = st.form_submit_button("Convene Congress V2")

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
                    num_delphi_rounds=num_delphi_rounds,
                )

        return None

    @classmethod
    async def _run_congress(
        cls, session_input: CongressSessionInput
    ) -> CongressSession:
        members = get_members_by_names(session_input.member_names)
        num_rounds = session_input.num_delphi_rounds

        start_time = time.time()
        rounds_str = f" ({num_rounds} Delphi rounds)" if num_rounds > 1 else ""
        with st.spinner(
            f"Congress V2 in session with {len(members)} members{rounds_str}... "
            "This may take 10-20 minutes."
        ):
            progress_text = st.empty()
            progress_text.write(
                "Members are researching, identifying scenarios, and deliberating..."
            )

            orchestrator = CongressOrchestrator(
                num_delphi_rounds=num_rounds,
            )
            with MonetaryCostManager(50):
                session = await orchestrator.run_session(
                    prompt=session_input.prompt,
                    members=members,
                )

            progress_text.write(
                "Aggregating proposals and generating scenario report..."
            )

        elapsed_time = time.time() - start_time
        st.session_state["v2_session_generation_time"] = elapsed_time

        if session.errors:
            st.warning(
                f"{len(session.errors)} member(s) encountered errors. "
                "Partial results shown."
            )

        return session

    # =========================================================================
    # SESSION DISPLAY
    # =========================================================================

    @classmethod
    def _display_session(cls, session: CongressSession) -> None:
        st.header("Congress V2 Results")

        cls._display_cost_summary(session)

        tabs = st.tabs(
            [
                "Synthesis",
                "Scenarios",
                "Scenario Report",
                "Blog Post",
                "Picture of the Future",
                "Individual Proposals",
                "Forecast Comparison",
                "Twitter Posts",
            ]
        )

        with tabs[0]:
            cls._display_synthesis_tab(session)
        with tabs[1]:
            cls._display_scenarios_tab(session)
        with tabs[2]:
            cls._display_scenario_report_tab(session)
        with tabs[3]:
            cls._display_blog_tab(session)
        with tabs[4]:
            cls._display_future_snapshot_tab(session)
        with tabs[5]:
            cls._display_proposals_tab(session)
        with tabs[6]:
            cls._display_forecasts_tab(session)
        with tabs[7]:
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
            with st.expander("Errors During Session"):
                for error in session.errors:
                    st.error(error)

    @classmethod
    def _display_scenarios_tab(cls, session: CongressSession) -> None:
        st.subheader("Scenarios Across Members")

        if not session.proposals:
            st.write("No proposals available.")
            return

        cls._display_drivers_section(session)
        cls._display_scenarios_by_member(session)
        cls._display_scenario_indicator_table(session)

    @classmethod
    def _display_drivers_section(cls, session: CongressSession) -> None:
        all_drivers = session.get_all_drivers()
        if not all_drivers:
            return

        st.markdown("### Key Drivers Identified")
        for driver in all_drivers:
            with st.expander(f"**{driver.name}**"):
                st.markdown(driver.description)
                members_with_driver = [
                    p.member.name
                    for p in session.proposals
                    if p.member and any(d.name == driver.name for d in p.drivers)
                ]
                st.caption(f"Identified by: {', '.join(members_with_driver)}")

    @classmethod
    def _display_scenarios_by_member(cls, session: CongressSession) -> None:
        st.markdown("### Scenarios by Member")

        for proposal in session.proposals:
            if not proposal.member:
                continue

            member_name = proposal.member.name
            with st.expander(
                f"**{member_name}** — {len(proposal.scenarios)} scenarios"
            ):
                for scenario in proposal.scenarios:
                    cls._render_single_scenario(scenario)

    @staticmethod
    def _render_single_scenario(scenario: Scenario) -> None:
        status_quo_badge = " (Status Quo)" if scenario.is_status_quo else ""
        st.markdown(
            f"#### {scenario.name}{status_quo_badge} — "
            f"Probability: {scenario.probability}"
        )
        st.markdown(scenario.narrative)

        if scenario.drivers:
            st.markdown("**Drivers:**")
            for driver in scenario.drivers:
                st.markdown(f"- {driver.name}: {driver.description}")

        if scenario.criteria:
            st.markdown("**Criteria:**")
            for criterion in scenario.criteria:
                date_str = (
                    f" (by {criterion.target_date})" if criterion.target_date else ""
                )
                st.markdown(f"- {criterion.criterion_text}{date_str}")
                if criterion.resolution_criteria:
                    st.caption(f"  Resolution: {criterion.resolution_criteria}")

        st.markdown("---")

    @classmethod
    def _display_scenario_indicator_table(cls, session: CongressSession) -> None:
        scenario_indicator_forecasts = session.get_all_scenario_indicator_forecasts()
        if not scenario_indicator_forecasts:
            return

        st.markdown("### Scenario Probability Forecasts")
        scenario_data = []
        for f in scenario_indicator_forecasts:
            member_name = cls._find_forecast_member(f, session.proposals)
            scenario_data.append(
                {
                    "Member": member_name,
                    "Scenario": f.question_title,
                    "Prediction": f.prediction,
                    "Reasoning": (
                        f.reasoning[:100] + "..."
                        if len(f.reasoning) > 100
                        else f.reasoning
                    ),
                }
            )
        st.dataframe(scenario_data, use_container_width=True)

    @staticmethod
    def _find_forecast_member(
        forecast: ForecastDescription,
        proposals: list[PolicyProposal],
    ) -> str:
        for p in proposals:
            if p.member and forecast in p.forecasts:
                return p.member.name
        return "Unknown"

    @classmethod
    def _display_scenario_report_tab(cls, session: CongressSession) -> None:
        st.subheader("Scenario Report")
        st.caption(
            "A formal scenario planning document in the style of Shell/NIC reports, "
            "with scenario narratives, strategic implications, and early warning indicators."
        )

        if session.scenario_report:
            cleaned = ReportDisplayer.clean_markdown(session.scenario_report)
            st.markdown(cleaned)

            st.download_button(
                label="Download Scenario Report (Markdown)",
                data=session.scenario_report,
                file_name=(
                    f"scenario_report_"
                    f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
                ),
                mime="text/markdown",
                key="v2_download_scenario_report",
            )
        else:
            st.write("No scenario report available.")

    @classmethod
    def _display_blog_tab(cls, session: CongressSession) -> None:
        st.subheader("Blog Post")
        if session.blog_post:
            cleaned = ReportDisplayer.clean_markdown(session.blog_post)
            st.markdown(cleaned)

            st.download_button(
                label="Download Blog Post (Markdown)",
                data=session.blog_post,
                file_name=(
                    f"congress_v2_blog_"
                    f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
                ),
                mime="text/markdown",
                key="v2_download_blog",
            )
        else:
            st.write("No blog post available.")

    @classmethod
    def _display_future_snapshot_tab(cls, session: CongressSession) -> None:
        st.subheader("Picture of the Future")
        st.caption(
            "Scenario-aware newspaper articles from the future showing what might "
            "happen under different scenarios if AI recommendations were implemented "
            "or rejected."
        )

        if session.future_snapshot:
            cleaned = ReportDisplayer.clean_markdown(session.future_snapshot)
            st.markdown(cleaned)

            st.download_button(
                label="Download Future Snapshot (Markdown)",
                data=session.future_snapshot,
                file_name=(
                    f"congress_v2_future_"
                    f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
                ),
                mime="text/markdown",
                key="v2_download_future_snapshot",
            )
        else:
            st.write("No future snapshot available.")

    @classmethod
    def _display_proposals_tab(cls, session: CongressSession) -> None:
        st.subheader("Individual Member Proposals")

        if not session.proposals:
            st.write("No proposals available.")
            return

        if session.num_delphi_rounds > 1:
            st.info(
                f"This session used {session.num_delphi_rounds} Delphi rounds. "
                "Proposals shown are the final revised versions."
            )

        cls._display_proposal_list(session.proposals)

        if session.num_delphi_rounds > 1 and session.initial_proposals:
            st.markdown("---")
            st.subheader("Initial Proposals (Round 1)")
            st.caption(
                "These are the original proposals before Delphi revision rounds."
            )
            cls._display_proposal_list(session.initial_proposals, " (Round 1)")

    @classmethod
    def _display_proposal_list(
        cls,
        proposals: list[PolicyProposal],
        label_suffix: str = "",
    ) -> None:
        for proposal in proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            member_role = proposal.member.role if proposal.member else ""
            cost_str = (
                f" (${proposal.price_estimate:.2f})" if proposal.price_estimate else ""
            )
            selected_str = (
                f" | Selected: {proposal.selected_proposal_name}"
                if proposal.selected_proposal_name
                else ""
            )
            label = (
                f"**{member_name}**{label_suffix} - "
                f"{member_role}{cost_str}{selected_str}"
            )
            with st.expander(label, expanded=False):
                cls._display_single_proposal(proposal)

    @classmethod
    def _display_single_proposal(cls, proposal: PolicyProposal) -> None:
        if proposal.price_estimate:
            st.caption(f"Cost: ${proposal.price_estimate:.2f}")

        cls._display_proposal_criteria_and_scenarios(proposal)
        cls._display_proposal_options_and_recommendations(proposal)
        cls._display_proposal_body(proposal)
        cls._display_proposal_forecasts(proposal)

    @staticmethod
    def _display_proposal_criteria_and_scenarios(proposal: PolicyProposal) -> None:
        st.markdown("# Decision Criteria")
        for i, criterion in enumerate(proposal.decision_criteria, 1):
            st.markdown(f"{i}. {criterion}")

        if proposal.scenarios:
            st.markdown("# Scenarios")
            for scenario in proposal.scenarios:
                status_quo_badge = " (Status Quo)" if scenario.is_status_quo else ""
                st.markdown(
                    f"**{scenario.name}{status_quo_badge}** — "
                    f"{scenario.probability}"
                )
                st.markdown(scenario.narrative)

    @staticmethod
    def _display_proposal_options_and_recommendations(
        proposal: PolicyProposal,
    ) -> None:
        if proposal.proposal_options:
            st.markdown("# Proposal Options Considered")
            for option in proposal.proposal_options:
                st.markdown(f"**{option.name}**: {option.description}")
                for action in option.key_actions:
                    st.markdown(f"  - {action}")

        st.markdown("# Key Recommendations")
        for rec in proposal.key_recommendations:
            st.markdown(f"- {rec}")

        if proposal.contingency_plans:
            st.markdown("# Contingency Plans")
            for plan in proposal.contingency_plans:
                st.markdown(f"- {plan}")

    @staticmethod
    def _display_proposal_body(proposal: PolicyProposal) -> None:
        st.markdown("# Research Summary")
        st.markdown(proposal.research_summary)

        st.markdown("# Proposal Text")
        cleaned = ReportDisplayer.clean_markdown(
            proposal.get_full_markdown_with_footnotes()
        )
        st.markdown(cleaned)

        if proposal.robustness_analysis:
            st.markdown("# Cross-Scenario Robustness Analysis")
            st.markdown(proposal.robustness_analysis)

    @classmethod
    def _display_proposal_forecasts(cls, proposal: PolicyProposal) -> None:
        baseline = proposal.baseline_forecasts
        scenario_indicators = proposal.scenario_indicator_forecasts
        scenario_conditional = proposal.scenario_conditional_forecasts
        proposal_conditional = proposal.proposal_conditional_forecasts

        if baseline:
            st.markdown("# Baseline Forecasts (Status Quo)")
            cls._render_forecast_list(baseline)

        if scenario_indicators:
            st.markdown("# Scenario Indicator Forecasts")
            cls._render_forecast_list(scenario_indicators)

        if scenario_conditional:
            st.markdown("# Scenario-Conditional Forecasts")
            cls._render_forecast_list(scenario_conditional)

        if proposal_conditional:
            st.markdown("# Proposal-Conditional Forecasts")
            cls._render_forecast_list(proposal_conditional)

    @staticmethod
    def _render_forecast_list(forecasts: list[ForecastDescription]) -> None:
        for forecast in forecasts:
            conditional_labels: list[str] = []
            if forecast.conditional_on_proposal:
                conditional_labels.append("Conditional on proposal")
            if forecast.conditional_on_scenario:
                conditional_labels.append(
                    f"Under scenario: {forecast.conditional_on_scenario}"
                )
            label_str = (
                f" ({'; '.join(conditional_labels)})" if conditional_labels else ""
            )

            st.markdown(
                f"**[^{forecast.footnote_id}] {forecast.question_title}**{label_str}"
            )
            st.markdown(f"- **Prediction:** {forecast.prediction}")
            st.markdown(f"- **Question:** {forecast.question_text}")
            st.markdown(f"- **Resolution:** {forecast.resolution_criteria}")
            st.markdown(f"- **Reasoning:** {forecast.reasoning}")
            if forecast.key_sources:
                st.markdown(f"- **Sources:** {', '.join(forecast.key_sources)}")
            st.markdown("---")

    @classmethod
    def _display_forecasts_tab(cls, session: CongressSession) -> None:
        st.subheader("Forecast Comparison")

        forecasts_by_member = session.get_forecasts_by_member()

        if not forecasts_by_member:
            st.write("No forecasts available.")
            return

        forecast_type_filter = st.selectbox(
            "Filter by forecast type:",
            [
                "All",
                "Baseline (Status Quo)",
                "Scenario Indicator",
                "Scenario-Conditional",
                "Proposal-Conditional",
            ],
            key="v2_forecast_type_filter",
        )

        type_mapping = {
            "All": None,
            "Baseline (Status Quo)": "baseline",
            "Scenario Indicator": "scenario_indicator",
            "Scenario-Conditional": "scenario_conditional",
            "Proposal-Conditional": [
                "proposal_conditional",
                "proposal_scenario_conditional",
            ],
        }
        selected_type = type_mapping[forecast_type_filter]

        table_data = cls._build_forecast_table_data(forecasts_by_member, selected_type)

        if table_data:
            st.dataframe(table_data, use_container_width=True)
        else:
            st.write("No forecasts match the selected filter.")

        st.markdown("---")
        st.markdown("#### Detailed Forecasts by Member")

        for member_name, forecasts in forecasts_by_member.items():
            filtered = cls._filter_forecasts(forecasts, selected_type)
            if not filtered:
                continue
            with st.expander(f"**{member_name}** ({len(filtered)} forecasts)"):
                cls._render_forecast_list(filtered)

    @staticmethod
    def _filter_forecasts(
        forecasts: list[ForecastDescription],
        selected_type: str | list[str] | None,
    ) -> list[ForecastDescription]:
        if selected_type is None:
            return forecasts
        if isinstance(selected_type, list):
            return [f for f in forecasts if f.forecast_type in selected_type]
        return [f for f in forecasts if f.forecast_type == selected_type]

    @classmethod
    def _build_forecast_table_data(
        cls,
        forecasts_by_member: dict[str, list[ForecastDescription]],
        selected_type: str | list[str] | None,
    ) -> list[dict]:
        table_data = []
        for member_name, forecasts in forecasts_by_member.items():
            filtered = cls._filter_forecasts(forecasts, selected_type)
            for f in filtered:
                row = {
                    "Member": member_name,
                    "Type": f.forecast_type,
                    "Question": f.question_title,
                    "Prediction": f.prediction,
                    "Scenario": f.conditional_on_scenario or "—",
                    "Reasoning": (
                        f.reasoning[:100] + "..."
                        if len(f.reasoning) > 100
                        else f.reasoning
                    ),
                }
                table_data.append(row)
        return table_data

    @classmethod
    def _display_twitter_tab(cls, session: CongressSession) -> None:
        st.subheader("Twitter/X Posts")
        st.markdown(
            "These tweet-sized excerpts highlight interesting patterns "
            "from the scenario-focused congress session."
        )

        if not session.twitter_posts:
            st.write("No Twitter posts generated.")
            return

        for i, post in enumerate(session.twitter_posts, 1):
            st.markdown(f"**Tweet {i}** ({len(post)} chars)")
            st.info(post)

    @classmethod
    def _display_cost_summary(cls, session: CongressSession) -> None:
        total_cost = session.total_price_estimate
        generation_time = st.session_state.get("v2_session_generation_time")

        has_cost_info = total_cost is not None
        has_time_info = generation_time is not None

        if not has_cost_info and not has_time_info:
            return

        proposal_costs = [
            (p.member.name if p.member else "Unknown", p.price_estimate or 0)
            for p in session.proposals
        ]

        with st.expander("Session Stats", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                if has_time_info:
                    minutes = int(generation_time // 60)
                    seconds = int(generation_time % 60)
                    st.metric("Generation Time", f"{minutes}m {seconds}s")
                else:
                    st.metric("Generation Time", "N/A")
            with col2:
                if has_cost_info:
                    st.metric("Total Cost", f"${total_cost:.2f}")
                else:
                    st.metric("Total Cost", "N/A")
            with col3:
                st.metric("Members", len(session.proposals))

            if has_cost_info and proposal_costs:
                st.markdown("**Cost by Member:**")
                for member_name, cost in proposal_costs:
                    st.markdown(f"- {member_name}: ${cost:.2f}")

    @classmethod
    def _display_download_buttons(cls, session: CongressSession) -> None:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            json_str = json.dumps(session.to_json(), indent=2, default=str)
            st.download_button(
                label="Download Full Session (JSON)",
                data=json_str,
                file_name=(
                    f"congress_v2_session_"
                    f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                ),
                mime="application/json",
                key="v2_download_json",
            )

        with col2:
            markdown_content = cls._session_to_markdown(session)
            st.download_button(
                label="Download Report (Markdown)",
                data=markdown_content,
                file_name=(
                    f"congress_v2_report_"
                    f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
                ),
                mime="text/markdown",
                key="v2_download_markdown",
            )

    @classmethod
    def _session_to_markdown(cls, session: CongressSession) -> str:
        lines = [
            "# AI Forecasting Congress V2 Report",
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
        ]

        if session.scenario_report:
            lines.extend(
                [
                    "## Scenario Report",
                    "",
                    session.scenario_report,
                    "",
                    "---",
                    "",
                ]
            )

        lines.extend(["## Individual Proposals", ""])
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
        filename = f"{session.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(SESSIONS_FOLDER, filename)

        try:
            json_str = json.dumps(session.to_json(), indent=2, default=str)
            create_or_overwrite_file(filepath, json_str)
            logger.info(f"Saved session to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            st.error(f"Failed to save session: {e}")

    @classmethod
    def _load_session_from_file(cls, file_path: str) -> CongressSession | None:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None

        try:
            data: dict = load_json_file(file_path)[0]
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
                    data: dict = load_json_file(filepath)[0]
                    session = CongressSession.from_json(data)
                    sessions.append(session)
                except Exception as e:
                    logger.error(f"Failed to load session {filename}: {e}")

        return sessions


if __name__ == "__main__":
    CongressV2Page.main()
