from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import streamlit as st

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    AgentAction,
    Message,
    SimulationState,
    SimulationStep,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.simulator import (
    DEFAULT_SIMULATIONS_DIR,
    Simulator,
    create_run_directory,
    save_full_simulation,
    save_situation_to_file,
    save_step_to_file,
)
from forecasting_tools.agents_and_tools.situation_simulator.situation_generator import (
    SituationGenerator,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.custom_auth import CustomAuth
from forecasting_tools.util import file_manipulation

logger = logging.getLogger(__name__)

EXAMPLE_SITUATIONS_DIR = (
    "forecasting_tools/agents_and_tools/situation_simulator/example_situations"
)

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
            "sim_total_cost": 0.0,
            "sim_run_dir": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # --- Setup Panel ---

    @classmethod
    def _display_setup_panel(cls) -> None:
        st.header("Load or Generate a Situation")

        tab_example, tab_upload, tab_resume, tab_generate = st.tabs(
            [
                "Example Situations",
                "Upload JSON",
                "Resume from Save",
                "Generate from Prompt",
            ]
        )

        with tab_example:
            cls._display_example_loader()

        with tab_upload:
            cls._display_json_uploader()

        with tab_resume:
            cls._display_resume_loader()

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
                with MonetaryCostManager(100):
                    situation = asyncio.run(generator.generate(prompt))
                cls._set_situation(situation)
                st.success(f"Generated: {situation.name}")
                st.rerun()

    @classmethod
    def _display_resume_loader(cls) -> None:
        st.markdown(
            "Resume a simulation from a previously saved step file or "
            "full simulation file."
        )

        browse_tab, upload_tab = st.tabs(["Browse Local Runs", "Upload Files"])

        with browse_tab:
            cls._display_local_run_browser()

        with upload_tab:
            cls._display_resume_uploader()

    @classmethod
    def _display_local_run_browser(cls) -> None:
        browse_dir = st.text_input(
            "Simulations folder",
            value=DEFAULT_SIMULATIONS_DIR,
            key="resume_browse_dir",
            help="Change this to browse a different folder for saved simulations.",
        )
        simulations_dir = (
            Path(browse_dir.strip())
            if browse_dir.strip()
            else Path(DEFAULT_SIMULATIONS_DIR)
        )

        if not simulations_dir.exists():
            st.info(f"Directory not found: `{simulations_dir}`")
            return

        run_dirs = sorted(
            [d for d in simulations_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            st.info("No simulation runs found in this directory.")
            return

        selected_run_name = st.selectbox(
            "Select a simulation run",
            options=[d.name for d in run_dirs],
            key="resume_run_select",
        )
        if selected_run_name is None:
            return

        selected_run_dir = simulations_dir / selected_run_name
        situation_path = selected_run_dir / "situation.json"
        if not situation_path.exists():
            st.warning("No situation.json found in this run directory.")
            return

        step_files = sorted(selected_run_dir.glob("step_*.json"))
        full_sim_path = selected_run_dir / "full_simulation.json"

        if not step_files and not full_sim_path.exists():
            st.warning("No step files found in this run directory.")
            return

        step_options = [f.stem for f in step_files]
        if full_sim_path.exists():
            step_options.append("full_simulation (all steps)")

        selected_step = st.selectbox(
            "Resume from step",
            options=step_options,
            index=len(step_options) - 1,
            key="resume_step_select",
        )

        if st.button("Resume Simulation", key="resume_local_btn"):
            try:
                situation_data = json.loads(situation_path.read_text())
                situation = Situation.model_validate(situation_data)

                if selected_step == "full_simulation (all steps)":
                    cls._load_from_full_simulation_file(
                        full_sim_path, run_dir=selected_run_dir
                    )
                else:
                    step_path = selected_run_dir / f"{selected_step}.json"
                    all_step_files = sorted(selected_run_dir.glob("step_*.json"))
                    earlier_steps = [
                        f for f in all_step_files if f.name <= step_path.name
                    ]
                    cls._load_from_step_files(
                        situation,
                        earlier_steps,
                        run_dir=selected_run_dir,
                    )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to resume simulation: {e}")

    @classmethod
    def _display_resume_uploader(cls) -> None:
        resume_mode = st.radio(
            "File type",
            options=["Full simulation file", "Step file + Situation file"],
            key="resume_upload_mode",
        )

        if resume_mode == "Full simulation file":
            uploaded = st.file_uploader(
                "Upload full_simulation.json",
                type=["json"],
                key="resume_full_upload",
            )
            if uploaded is not None and st.button(
                "Resume from Full Simulation", key="resume_full_btn"
            ):
                try:
                    data = json.loads(uploaded.read())
                    cls._load_from_full_simulation_data(data)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load full simulation: {e}")
        else:
            situation_file = st.file_uploader(
                "Upload situation JSON",
                type=["json"],
                key="resume_situation_upload",
            )
            step_file = st.file_uploader(
                "Upload step JSON (will resume from state_after)",
                type=["json"],
                key="resume_step_upload",
            )
            if (
                situation_file is not None
                and step_file is not None
                and st.button("Resume from Step", key="resume_step_btn")
            ):
                try:
                    situation_data = json.loads(situation_file.read())
                    situation = Situation.model_validate(situation_data)
                    step_data = json.loads(step_file.read())
                    step = SimulationStep.model_validate(step_data)
                    cls._set_situation_with_state(
                        situation,
                        state=step.state_after.deep_copy(),
                        steps=[step],
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to resume from step file: {e}")

    @classmethod
    def _load_from_full_simulation_file(
        cls, filepath: Path, run_dir: Path | None = None
    ) -> None:
        data = json.loads(filepath.read_text())
        cls._load_from_full_simulation_data(data, run_dir=run_dir)

    @classmethod
    def _load_from_full_simulation_data(
        cls,
        data: dict,
        run_dir: Path | None = None,
    ) -> None:
        situation = Situation.model_validate(data["situation"])
        steps = [SimulationStep.model_validate(s) for s in data.get("steps", [])]
        final_state = SimulationState.model_validate(data["final_state"])
        total_cost = data.get("total_cost_usd", 0.0)
        cls._set_situation_with_state(
            situation,
            state=final_state,
            steps=steps,
            total_cost=total_cost,
            run_dir=run_dir,
        )

    @classmethod
    def _load_from_step_files(
        cls,
        situation: Situation,
        step_files: list[Path],
        run_dir: Path | None = None,
    ) -> None:
        steps: list[SimulationStep] = []
        for step_file in step_files:
            step_data = json.loads(step_file.read_text())
            steps.append(SimulationStep.model_validate(step_data))

        if not steps:
            cls._set_situation(situation)
            return

        last_step = steps[-1]
        cls._set_situation_with_state(
            situation,
            state=last_step.state_after.deep_copy(),
            steps=steps,
            run_dir=run_dir,
        )

    @classmethod
    def _set_situation_with_state(
        cls,
        situation: Situation,
        state: SimulationState,
        steps: list[SimulationStep] | None = None,
        total_cost: float = 0.0,
        run_dir: Path | None = None,
    ) -> None:
        st.session_state["sim_situation"] = situation
        st.session_state["sim_state"] = state
        st.session_state["sim_steps"] = steps or []
        st.session_state["sim_total_cost"] = total_cost
        st.session_state["sim_run_dir"] = run_dir

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
        tab_slack, tab_timeline, tab_player, tab_inventory, tab_trades, tab_json = (
            st.tabs(
                [
                    "💬 Slack",
                    "📋 Timeline",
                    "👤 Player View",
                    "📦 Inventories",
                    "🤝 Trades",
                    "📄 Situation JSON",
                ]
            )
        )

        with tab_slack:
            cls._display_slack_view(situation, state)

        with tab_timeline:
            cls._display_timeline(steps)

        with tab_player:
            cls._display_player_view(situation, steps, state)

        with tab_inventory:
            cls._display_inventories(situation, state)

        with tab_trades:
            cls._display_trades(state)

        with tab_json:
            cls._display_situation_json(situation)

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
    def _display_player_view(
        cls,
        situation: Situation,
        steps: list[SimulationStep],
        state: SimulationState | None,
    ) -> None:
        if state is None:
            st.info("No simulation state yet.")
            return

        agent_names = [a.name for a in situation.agents]
        selected_agent = st.selectbox(
            "Select a player",
            options=agent_names,
            key="player_view_agent",
        )
        if selected_agent is None:
            return

        agent_color_map = cls._get_agent_color_map(situation)
        color = agent_color_map.get(selected_agent, "#666")

        agent_def = next(
            (a for a in situation.agents if a.name == selected_agent), None
        )
        if agent_def:
            public_persona = [m for m in agent_def.persona if not m.hidden]
            persona_text = ", ".join(f"{m.key}: {m.value}" for m in public_persona)
            st.markdown(
                f'<div style="padding: 8px; border-left: 3px solid {color}; '
                f'background: rgba(0,0,0,0.03); border-radius: 4px; margin-bottom: 12px;">'
                f'<strong style="color: {color};">{selected_agent}</strong><br/>'
                f'<span style="font-size: 0.85em;">{persona_text}</span></div>',
                unsafe_allow_html=True,
            )

        (
            action_tab,
            msg_tab,
            inv_tab,
            trade_tab,
        ) = st.tabs(
            [
                "🎬 Actions",
                "💬 Messages",
                "📦 Inventory History",
                "🤝 Trades",
            ]
        )

        with action_tab:
            cls._render_player_actions(selected_agent, steps, color)

        with msg_tab:
            cls._render_player_messages(selected_agent, state, agent_color_map)

        with inv_tab:
            cls._render_player_inventory_history(selected_agent, steps, state)

        with trade_tab:
            cls._render_player_trades(selected_agent, state)

    @classmethod
    def _render_player_actions(
        cls,
        agent_name: str,
        steps: list[SimulationStep],
        color: str,
    ) -> None:
        player_actions: list[tuple[int, AgentAction]] = []
        for step in steps:
            for action in step.agent_actions:
                if action.agent_name == agent_name:
                    player_actions.append((step.step_number, action))

        if not player_actions:
            st.info(f"No actions recorded for {agent_name}.")
            return

        st.markdown(f"**{len(player_actions)} actions across {len(steps)} steps**")

        for step_number, action in reversed(player_actions):
            action_label = action.action_name
            if action.parameters:
                params_str = ", ".join(f"{k}={v}" for k, v in action.parameters.items())
                action_label += f" ({params_str})"

            with st.expander(f"Step {step_number} — {action_label}"):
                st.markdown(f"**Action:** {action.action_name}")
                if action.parameters:
                    st.markdown("**Parameters:**")
                    for k, v in action.parameters.items():
                        st.text(f"  {k}: {v}")

                if action.messages_to_send:
                    st.markdown("**Messages sent:**")
                    for msg in action.messages_to_send:
                        target = f"#{msg.channel}" if msg.channel else "DM"
                        recipients = ", ".join(msg.recipients) if msg.recipients else ""
                        target_label = (
                            f"{target} → {recipients}" if recipients else target
                        )
                        st.markdown(
                            f'<div style="margin: 4px 0; padding: 6px; '
                            f"border-left: 2px solid {color}; "
                            f'background: rgba(0,0,0,0.02); border-radius: 3px;">'
                            f'<span style="color: #888; font-size: 0.8em;">'
                            f"{target_label}</span><br/>"
                            f"{msg.content}</div>",
                            unsafe_allow_html=True,
                        )

                if action.trade_proposal:
                    tp = action.trade_proposal
                    offering = ", ".join(f"{v} {k}" for k, v in tp.offering.items())
                    requesting = ", ".join(f"{v} {k}" for k, v in tp.requesting.items())
                    st.markdown(
                        f"**Trade proposed:** offering {offering} " f"for {requesting}"
                    )

                if action.trade_acceptance_id:
                    st.markdown(
                        f"**Accepted trade:** {action.trade_acceptance_id[:8]}..."
                    )

    @classmethod
    def _render_player_messages(
        cls,
        agent_name: str,
        state: SimulationState,
        agent_color_map: dict[str, str],
    ) -> None:
        sent_messages = [m for m in state.message_history if m.sender == agent_name]
        received_messages = [
            m
            for m in state.message_history
            if agent_name in m.recipients and m.sender != agent_name
        ]

        sent_tab, received_tab = st.tabs(
            [
                f"Sent ({len(sent_messages)})",
                f"Received ({len(received_messages)})",
            ]
        )

        with sent_tab:
            if not sent_messages:
                st.caption(f"No messages sent by {agent_name}.")
            else:
                cls._render_messages(sent_messages, agent_color_map)

        with received_tab:
            if not received_messages:
                st.caption(f"No messages received by {agent_name}.")
            else:
                cls._render_messages(received_messages, agent_color_map)

    @classmethod
    def _render_player_inventory_history(
        cls,
        agent_name: str,
        steps: list[SimulationStep],
        state: SimulationState,
    ) -> None:
        current_inventory = state.inventories.get(agent_name, {})
        if current_inventory:
            st.markdown(f"**Current inventory (Step {state.step_number}):**")
            for item_name, qty in current_inventory.items():
                st.text(f"  {item_name}: {qty}")
        else:
            st.caption("Current inventory is empty.")

        if not steps:
            return

        st.markdown("---")
        st.markdown("**Inventory changes by step:**")

        for step in reversed(steps):
            before = step.state_before.inventories.get(agent_name, {})
            after = step.state_after.inventories.get(agent_name, {})
            all_items = sorted(set(before.keys()) | set(after.keys()))
            changes: list[str] = []
            for item in all_items:
                qty_before = before.get(item, 0)
                qty_after = after.get(item, 0)
                diff = qty_after - qty_before
                if diff != 0:
                    sign = "+" if diff > 0 else ""
                    changes.append(f"{item}: {qty_before} → {qty_after} ({sign}{diff})")

            if changes:
                with st.expander(f"Step {step.step_number} — {len(changes)} change(s)"):
                    for change in changes:
                        st.text(f"  {change}")
            else:
                st.caption(f"Step {step.step_number} — no inventory changes")

    @classmethod
    def _render_player_trades(
        cls,
        agent_name: str,
        state: SimulationState,
    ) -> None:
        player_pending = [
            t
            for t in state.pending_trades
            if t.status == "pending"
            and (t.proposer == agent_name or agent_name in t.eligible_acceptors)
        ]
        player_trade_records = [
            r
            for r in state.trade_history
            if r.from_agent == agent_name or r.to_agent == agent_name
        ]

        if player_pending:
            st.subheader("Pending Trades")
            for trade in player_pending:
                role = "proposed" if trade.proposer == agent_name else "can accept"
                offering = ", ".join(f"{v} {k}" for k, v in trade.offering.items())
                requesting = ", ".join(f"{v} {k}" for k, v in trade.requesting.items())
                st.markdown(
                    f"**{trade.proposer}** ({role}): offers {offering} "
                    f"for {requesting} "
                    f"(expires step {trade.expires_at_step})"
                )

        if player_trade_records:
            st.subheader("Trade History")
            for record in reversed(player_trade_records[-20:]):
                direction = "sent" if record.from_agent == agent_name else "received"
                other = (
                    record.to_agent
                    if record.from_agent == agent_name
                    else record.from_agent
                )
                st.caption(
                    f"Step {record.step}: {direction} {record.quantity} "
                    f"{record.item_name} {'to' if direction == 'sent' else 'from'} "
                    f"{other} (trade {record.trade_id[:8]}...)"
                )
        elif not player_pending:
            st.info(f"No trade activity for {agent_name}.")

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
            total_cost: float = st.session_state.get("sim_total_cost", 0.0)
            if state:
                st.metric("Current Step", state.step_number)
                st.metric("Messages", len(state.message_history))
                st.metric(
                    "Pending Trades",
                    len([t for t in state.pending_trades if t.status == "pending"]),
                )
                st.metric("Est. Cost (USD)", f"${total_cost:.4f}")

            run_dir: Path | None = st.session_state.get("sim_run_dir")
            if run_dir:
                st.caption(f"Saving to: {run_dir}")

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

        with st.expander("Situation JSON"):
            cls._display_situation_json(situation)

    # --- Simulation execution ---

    @classmethod
    def _ensure_run_directory(cls, situation: Situation) -> Path:
        run_dir = st.session_state.get("sim_run_dir")
        if run_dir is None:
            run_dir = create_run_directory(situation.name)
            save_situation_to_file(run_dir, situation)
            st.session_state["sim_run_dir"] = run_dir
        return run_dir

    @classmethod
    def _run_one_step(cls, situation: Situation) -> None:
        state = st.session_state.get("sim_state")
        simulator = Simulator(situation)
        if state is None:
            state = simulator.create_initial_state()

        run_dir = cls._ensure_run_directory(situation)

        with st.spinner(f"Running step {state.step_number + 1}..."):
            with MonetaryCostManager(100) as cost_manager:
                step = asyncio.run(simulator.run_step(state))
            st.session_state["sim_total_cost"] += cost_manager.current_usage

        save_step_to_file(run_dir, step)

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

        run_dir = cls._ensure_run_directory(situation)
        steps = st.session_state.get("sim_steps", [])
        progress = st.progress(0)

        with MonetaryCostManager(100) as cost_manager:
            for i in range(n):
                progress.progress(
                    (i + 1) / n, f"Running step {state.step_number + 1}..."
                )
                step = asyncio.run(simulator.run_step(state))
                steps.append(step)
                save_step_to_file(run_dir, step)
        st.session_state["sim_total_cost"] += cost_manager.current_usage

        progress.empty()
        st.session_state["sim_steps"] = steps
        st.session_state["sim_state"] = state

    # --- State management ---

    @classmethod
    def _set_situation(cls, situation: Situation) -> None:
        st.session_state["sim_situation"] = situation
        st.session_state["sim_state"] = None
        st.session_state["sim_steps"] = []
        st.session_state["sim_total_cost"] = 0.0
        st.session_state["sim_run_dir"] = None

    @classmethod
    def _reset_simulation(cls) -> None:
        st.session_state["sim_state"] = None
        st.session_state["sim_steps"] = []
        st.session_state["sim_total_cost"] = 0.0
        st.session_state["sim_run_dir"] = None

    @classmethod
    def _clear_all(cls) -> None:
        st.session_state["sim_situation"] = None
        st.session_state["sim_state"] = None
        st.session_state["sim_steps"] = []
        st.session_state["sim_total_cost"] = 0.0
        st.session_state["sim_run_dir"] = None

    @classmethod
    def _load_situation_from_file(cls, filepath: str) -> None:
        try:
            data = file_manipulation.load_json_file(filepath)[0]
            situation = Situation.model_validate(data)
            cls._set_situation(situation)
        except Exception as e:
            st.error(f"Failed to load situation: {e}")

    @classmethod
    def _save_simulation(cls) -> None:
        state = st.session_state.get("sim_state")
        situation = st.session_state.get("sim_situation")
        steps = st.session_state.get("sim_steps", [])
        total_cost = st.session_state.get("sim_total_cost", 0.0)

        if not situation or not state:
            st.warning("Nothing to save.")
            return

        run_dir = cls._ensure_run_directory(situation)
        save_full_simulation(run_dir, situation, steps, state, total_cost)
        st.success(f"Saved full simulation to {run_dir}/full_simulation.json")

    # --- Helpers ---

    @classmethod
    def _display_situation_json(cls, situation: Situation) -> None:
        situation_json = json.dumps(situation.model_dump(), indent=2)
        st.code(situation_json, language="json")
        st.download_button(
            label="⬇ Download Situation JSON",
            data=situation_json,
            file_name=f"{situation.name}.json",
            mime="application/json",
            key=f"download_situation_{id(situation)}",
        )

    @classmethod
    def _get_agent_color_map(cls, situation: Situation) -> dict[str, str]:
        color_map: dict[str, str] = {}
        for i, agent_def in enumerate(situation.agents):
            color_map[agent_def.name] = AGENT_COLORS[i % len(AGENT_COLORS)]
        return color_map
