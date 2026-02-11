from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import streamlit as st

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    Message,
    SimulationState,
    SimulationStep,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.simulator import Simulator
from forecasting_tools.agents_and_tools.situation_simulator.situation_generator import (
    SituationGenerator,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.custom_auth import CustomAuth

logger = logging.getLogger(__name__)

EXAMPLE_SITUATIONS_DIR = (
    "forecasting_tools/agents_and_tools/situation_simulator/example_situations"
)
SAVED_SIMULATIONS_DIR = "temp/simulations"

AGENT_COLORS = [
    "#4A90D9",
    "#E67E22",
    "#2ECC71",
    "#9B59B6",
    "#E74C3C",
    "#1ABC9C",
    "#F39C12",
    "#3498DB",
    "#E91E63",
    "#00BCD4",
    "#8BC34A",
    "#FF5722",
]


class SimulatorPage(AppPage):
    PAGE_DISPLAY_NAME: str = "🎭 Situation Simulator"
    URL_PATH: str = "/simulator"
    IS_DEFAULT_PAGE: bool = False

    @classmethod
    @CustomAuth.add_access_control()
    async def _async_main(cls) -> None:
        st.title("🎭 Situation Simulator")
        st.markdown(
            "Multi-agent simulation with Slack-like communication, "
            "inventory management, and trading."
        )

        cls._init_session_state()
        cls._display_sidebar()

        situation: Situation | None = st.session_state.get("sim_situation")
        if situation is None:
            cls._display_setup_panel()
            return

        cls._display_controls(situation)

        steps: list[SimulationStep] = st.session_state.get("sim_steps", [])
        state: SimulationState | None = st.session_state.get("sim_state")

        if not steps and state is None:
            st.info("Situation loaded. Press 'Run Step' or 'Run All' to begin.")
            cls._display_situation_summary(situation)
            return

        cls._display_main_view(situation, steps, state)

    @classmethod
    def _init_session_state(cls) -> None:
        defaults = {
            "sim_situation": None,
            "sim_state": None,
            "sim_steps": [],
            "sim_running": False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # --- Setup Panel ---

    @classmethod
    def _display_setup_panel(cls) -> None:
        st.header("Load or Generate a Situation")

        tab_example, tab_upload, tab_generate = st.tabs(
            ["Example Situations", "Upload JSON", "Generate from Prompt"]
        )

        with tab_example:
            cls._display_example_loader()

        with tab_upload:
            cls._display_json_uploader()

        with tab_generate:
            cls._display_generator()

    @classmethod
    def _display_example_loader(cls) -> None:
        example_dir = Path(EXAMPLE_SITUATIONS_DIR)
        if not example_dir.exists():
            st.warning("Example situations directory not found.")
            return

        example_files = sorted(example_dir.glob("*.json"))
        if not example_files:
            st.warning("No example files found.")
            return

        for filepath in example_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{filepath.stem}**")
            with col2:
                if st.button("Load", key=f"load_{filepath.stem}"):
                    cls._load_situation_from_file(str(filepath))
                    st.rerun()

    @classmethod
    def _display_json_uploader(cls) -> None:
        uploaded = st.file_uploader(
            "Upload a Situation JSON file",
            type=["json"],
            key="situation_upload",
        )
        if uploaded is not None and st.button("Load Uploaded File"):
            try:
                data = json.loads(uploaded.read())
                situation = Situation.model_validate(data)
                cls._set_situation(situation)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to parse situation: {e}")

    @classmethod
    def _display_generator(cls) -> None:
        prompt = st.text_area(
            "Describe the simulation you want:",
            placeholder="Simulate a startup incubator where 4 founders compete for limited investor funding...",
            height=120,
            key="gen_prompt",
        )
        if st.button("Generate Situation", key="gen_btn") and prompt:
            with st.spinner("Generating situation..."):
                generator = SituationGenerator()
                situation = asyncio.run(generator.generate(prompt))
                cls._set_situation(situation)
                st.success(f"Generated: {situation.name}")
                st.rerun()

    # --- Controls ---

    @classmethod
    def _display_controls(cls, situation: Situation) -> None:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("▶ Run Step", key="run_step_btn"):
                cls._run_one_step(situation)
                st.rerun()

        with col2:
            max_steps = st.number_input(
                "Steps", min_value=1, max_value=50, value=3, key="run_n_steps"
            )
            if st.button("⏩ Run Multiple", key="run_n_btn"):
                cls._run_n_steps(situation, int(max_steps))
                st.rerun()

        with col3:
            if st.button("🔄 Reset", key="reset_btn"):
                cls._reset_simulation()
                st.rerun()

        with col4:
            if st.button("💾 Save State", key="save_btn"):
                cls._save_simulation()

        with col5:
            if st.button("🗑 Clear Situation", key="clear_btn"):
                cls._clear_all()
                st.rerun()

    # --- Main View ---

    @classmethod
    def _display_main_view(
        cls,
        situation: Situation,
        steps: list[SimulationStep],
        state: SimulationState | None,
    ) -> None:
        tab_slack, tab_timeline, tab_inventory, tab_trades = st.tabs(
            ["💬 Slack", "📋 Timeline", "📦 Inventories", "🤝 Trades"]
        )

        with tab_slack:
            cls._display_slack_view(situation, state)

        with tab_timeline:
            cls._display_timeline(steps)

        with tab_inventory:
            cls._display_inventories(situation, state)

        with tab_trades:
            cls._display_trades(state)

    @classmethod
    def _display_slack_view(
        cls, situation: Situation, state: SimulationState | None
    ) -> None:
        if state is None:
            st.info("No messages yet.")
            return

        channel_names = [ch.name for ch in situation.communication.channels]
        all_tabs = channel_names + ["DMs"]
        if not all_tabs:
            all_tabs = ["General"]

        tabs = st.tabs(all_tabs)

        agent_color_map = cls._get_agent_color_map(situation)

        for i, tab in enumerate(tabs):
            with tab:
                if i < len(channel_names):
                    channel_name = channel_names[i]
                    channel_messages = [
                        m for m in state.message_history if m.channel == channel_name
                    ]
                    cls._render_messages(channel_messages, agent_color_map)
                else:
                    dm_messages = [
                        m for m in state.message_history if m.channel is None
                    ]
                    cls._render_messages(dm_messages, agent_color_map)

    @classmethod
    def _render_messages(
        cls,
        messages: list[Message],
        agent_color_map: dict[str, str],
    ) -> None:
        if not messages:
            st.caption("No messages in this channel yet.")
            return

        for msg in messages:
            color = agent_color_map.get(msg.sender, "#666666")
            dm_tag = ""
            if msg.channel is None:
                others = [r for r in msg.recipients if r != msg.sender]
                dm_tag = f" → {', '.join(others)}" if others else ""

            st.markdown(
                f'<div style="margin-bottom: 8px; padding: 8px; '
                f"border-left: 3px solid {color}; background: rgba(0,0,0,0.03); "
                f'border-radius: 4px;">'
                f'<strong style="color: {color};">{msg.sender}</strong>'
                f'<span style="color: #888; font-size: 0.85em;"> '
                f"Step {msg.step}{dm_tag}</span><br/>"
                f"{msg.content}</div>",
                unsafe_allow_html=True,
            )

    @classmethod
    def _display_timeline(cls, steps: list[SimulationStep]) -> None:
        if not steps:
            st.info("No steps have been run yet.")
            return

        for step in reversed(steps):
            with st.expander(
                f"Step {step.step_number} — " f"{len(step.agent_actions)} actions",
                expanded=(step == steps[-1]),
            ):
                cls._render_step_actions(step)
                cls._render_step_triggers(step)

    @classmethod
    def _render_step_actions(cls, step: SimulationStep) -> None:
        for action in step.agent_actions:
            action_text = f"**{action.agent_name}**: {action.action_name}"
            if action.parameters:
                params_str = ", ".join(f"{k}={v}" for k, v in action.parameters.items())
                action_text += f" ({params_str})"
            st.markdown(action_text)

            for msg in action.messages_to_send:
                target = f"#{msg.channel}" if msg.channel else "DM"
                st.caption(f"  💬 {target}: {msg.content[:100]}...")

    @classmethod
    def _render_step_triggers(cls, step: SimulationStep) -> None:
        if not step.triggered_effects_log:
            return
        st.markdown("**Triggered effects:**")
        for log_entry in step.triggered_effects_log:
            st.caption(f"  ⚡ {log_entry}")

    @classmethod
    def _display_inventories(
        cls, situation: Situation, state: SimulationState | None
    ) -> None:
        if state is None:
            st.info("No simulation state yet.")
            return

        st.subheader(f"Inventories at Step {state.step_number}")

        agent_color_map = cls._get_agent_color_map(situation)

        cols = st.columns(min(len(situation.agents), 4))
        for i, agent_def in enumerate(situation.agents):
            col = cols[i % len(cols)]
            with col:
                color = agent_color_map.get(agent_def.name, "#666")
                st.markdown(
                    f'<div style="border-left: 3px solid {color}; padding-left: 8px;">'
                    f"<strong>{agent_def.name}</strong></div>",
                    unsafe_allow_html=True,
                )
                inventory = state.inventories.get(agent_def.name, {})
                if inventory:
                    for item_name, qty in inventory.items():
                        st.text(f"  {item_name}: {qty}")
                else:
                    st.caption("  Empty")

        if state.environment_inventory:
            st.markdown("---")
            st.markdown("**Environment Inventory:**")
            for item_name, qty in state.environment_inventory.items():
                st.text(f"  {item_name}: {qty}")

    @classmethod
    def _display_trades(cls, state: SimulationState | None) -> None:
        if state is None:
            st.info("No simulation state yet.")
            return

        if state.pending_trades:
            st.subheader("Pending Trades")
            for trade in state.pending_trades:
                if trade.status != "pending":
                    continue
                offering = ", ".join(f"{v} {k}" for k, v in trade.offering.items())
                requesting = ", ".join(f"{v} {k}" for k, v in trade.requesting.items())
                st.markdown(
                    f"**{trade.proposer}** offers {offering} "
                    f"for {requesting} "
                    f"(expires step {trade.expires_at_step})"
                )

        if state.trade_history:
            st.subheader("Trade History")
            for record in reversed(state.trade_history[-20:]):
                st.caption(
                    f"Step {record.step}: {record.from_agent} → {record.to_agent}: "
                    f"{record.quantity} {record.item_name} (trade {record.trade_id[:8]}...)"
                )
        elif not state.pending_trades:
            st.info("No trades have occurred yet.")

    # --- Sidebar ---

    @classmethod
    def _display_sidebar(cls) -> None:
        with st.sidebar:
            situation = st.session_state.get("sim_situation")
            if situation is None:
                st.caption("No situation loaded.")
                return

            st.header(situation.name)
            st.caption(situation.description)

            state: SimulationState | None = st.session_state.get("sim_state")
            if state:
                st.metric("Current Step", state.step_number)
                st.metric("Messages", len(state.message_history))
                st.metric(
                    "Pending Trades",
                    len([t for t in state.pending_trades if t.status == "pending"]),
                )

            st.markdown("---")
            st.subheader("Agents")
            agent_color_map = cls._get_agent_color_map(situation)
            for agent_def in situation.agents:
                color = agent_color_map.get(agent_def.name, "#666")
                public_persona = [m for m in agent_def.persona if not m.hidden]
                persona_text = ", ".join(f"{m.key}: {m.value}" for m in public_persona)
                st.markdown(
                    f'<div style="margin-bottom: 6px; padding: 4px; '
                    f'border-left: 3px solid {color};">'
                    f"<strong>{agent_def.name}</strong><br/>"
                    f'<span style="font-size: 0.85em;">{persona_text}</span></div>',
                    unsafe_allow_html=True,
                )

    # --- Situation Summary ---

    @classmethod
    def _display_situation_summary(cls, situation: Situation) -> None:
        st.subheader("Situation Summary")
        st.markdown(f"**{situation.name}**: {situation.description}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Agents", len(situation.agents))
        with col2:
            st.metric("Items", len(situation.items))
        with col3:
            st.metric("Max Steps", situation.max_steps)

        with st.expander("Rules"):
            st.markdown(situation.rules_text)

        with st.expander("Items"):
            for item in situation.items:
                tradable = "✅" if item.tradable else "❌"
                st.markdown(
                    f"- **{item.name}** ({tradable} tradable): {item.description}"
                )

        with st.expander("Channels"):
            for ch in situation.communication.channels:
                members = (
                    "everyone" if ch.members == "everyone" else ", ".join(ch.members)
                )
                st.markdown(f"- **#{ch.name}**: {members}")

    # --- Simulation execution ---

    @classmethod
    def _run_one_step(cls, situation: Situation) -> None:
        state = st.session_state.get("sim_state")
        simulator = Simulator(situation)
        if state is None:
            state = simulator.create_initial_state()

        with st.spinner(f"Running step {state.step_number + 1}..."):
            step = asyncio.run(simulator.run_step(state))

        steps = st.session_state.get("sim_steps", [])
        steps.append(step)
        st.session_state["sim_steps"] = steps
        st.session_state["sim_state"] = state

    @classmethod
    def _run_n_steps(cls, situation: Situation, n: int) -> None:
        state = st.session_state.get("sim_state")
        simulator = Simulator(situation)
        if state is None:
            state = simulator.create_initial_state()

        steps = st.session_state.get("sim_steps", [])
        progress = st.progress(0)

        for i in range(n):
            progress.progress((i + 1) / n, f"Running step {state.step_number + 1}...")
            step = asyncio.run(simulator.run_step(state))
            steps.append(step)

        progress.empty()
        st.session_state["sim_steps"] = steps
        st.session_state["sim_state"] = state

    # --- State management ---

    @classmethod
    def _set_situation(cls, situation: Situation) -> None:
        st.session_state["sim_situation"] = situation
        st.session_state["sim_state"] = None
        st.session_state["sim_steps"] = []

    @classmethod
    def _reset_simulation(cls) -> None:
        st.session_state["sim_state"] = None
        st.session_state["sim_steps"] = []

    @classmethod
    def _clear_all(cls) -> None:
        st.session_state["sim_situation"] = None
        st.session_state["sim_state"] = None
        st.session_state["sim_steps"] = []

    @classmethod
    def _load_situation_from_file(cls, filepath: str) -> None:
        try:
            with open(filepath) as f:
                data = json.load(f)
            situation = Situation.model_validate(data)
            cls._set_situation(situation)
        except Exception as e:
            st.error(f"Failed to load situation: {e}")

    @classmethod
    def _save_simulation(cls) -> None:
        state = st.session_state.get("sim_state")
        situation = st.session_state.get("sim_situation")
        steps = st.session_state.get("sim_steps", [])

        if not situation or not state:
            st.warning("Nothing to save.")
            return

        os.makedirs(SAVED_SIMULATIONS_DIR, exist_ok=True)
        result_dict = {
            "situation": situation.model_dump(),
            "steps": [s.model_dump() for s in steps],
            "final_state": state.model_dump(),
        }
        filename = f"{SAVED_SIMULATIONS_DIR}/{situation.name}_{state.step_number}.json"
        with open(filename, "w") as f:
            json.dump(result_dict, f, indent=2)
        st.success(f"Saved to {filename}")

    # --- Helpers ---

    @classmethod
    def _get_agent_color_map(cls, situation: Situation) -> dict[str, str]:
        color_map: dict[str, str] = {}
        for i, agent_def in enumerate(situation.agents):
            color_map[agent_def.name] = AGENT_COLORS[i % len(AGENT_COLORS)]
        return color_map
