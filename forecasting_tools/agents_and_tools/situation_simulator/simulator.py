from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from forecasting_tools.agents_and_tools.situation_simulator.agent_runner import (
    SimulationAgentRunner,
)
from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    ActionDefinition,
    AgentAction,
    SimulationResult,
    SimulationState,
    SimulationStep,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.effect_engine import (
    EffectEngine,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util import file_manipulation

logger = logging.getLogger(__name__)

DEFAULT_SIMULATIONS_DIR = "temp/simulations"


def create_run_directory(
    situation_name: str,
    base_dir: str = DEFAULT_SIMULATIONS_DIR,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = situation_name.replace(" ", "_").lower()
    run_dir = Path(base_dir) / f"{safe_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_situation_to_file(run_dir: Path, situation: Situation) -> None:
    situation_path = run_dir / "situation.json"
    file_manipulation.write_json_file(situation_path, situation.model_dump())
    logger.info(f"Saved situation to {situation_path}")


def save_step_to_file(run_dir: Path, step: SimulationStep) -> None:
    step_path = run_dir / f"step_{step.step_number:03d}.json"
    file_manipulation.write_json_file(step_path, step.model_dump())
    logger.info(f"Saved step {step.step_number} to {step_path}")


def save_full_simulation(
    run_dir: Path,
    situation: Situation,
    steps: list[SimulationStep],
    final_state: SimulationState,
    total_cost: float,
) -> None:
    result = {
        "situation": situation.model_dump(),
        "steps": [s.model_dump() for s in steps],
        "final_state": final_state.model_dump(),
        "total_cost_usd": total_cost,
    }
    result_path = run_dir / "full_simulation.json"
    file_manipulation.write_json_file(result_path, result)
    logger.info(f"Saved full simulation to {result_path}")


class Simulator:
    def __init__(
        self,
        situation: Situation,
        agent_runner: SimulationAgentRunner | None = None,
    ) -> None:
        self.situation = situation
        self.agent_runner = agent_runner or SimulationAgentRunner()

    def create_initial_state(self) -> SimulationState:
        inventories: dict[str, dict[str, int]] = {}
        for agent_def in self.situation.agents:
            inventories[agent_def.name] = dict(agent_def.starting_inventory)

        return SimulationState(
            step_number=0,
            inventories=inventories,
            environment_inventory=dict(self.situation.environment.inventory),
            message_history=[],
            pending_trades=[],
            trade_history=[],
            action_log=[],
        )

    async def run_step(self, state: SimulationState) -> SimulationStep:
        state.step_number += 1
        state_before = state.deep_copy()

        engine = EffectEngine(state, self.situation)
        step_actions: list[AgentAction] = []
        triggered_log: list[str] = []

        for agent_def in self.situation.agents:
            try:
                action = await self.agent_runner.get_agent_action(
                    agent_def, state, self.situation
                )
            except Exception as e:
                logger.error(f"Error getting action from {agent_def.name}: {e}")
                action = AgentAction(
                    agent_name=agent_def.name,
                    action_name="no_action",
                )

            step_actions.append(action)
            state.action_log.append(action)

            action_log = self._execute_agent_action(engine, action, agent_def.name)
            triggered_log.extend(action_log)

            for msg in action.messages_to_send:
                state.message_history.append(msg)

        expire_log = engine.expire_trades()
        triggered_log.extend(expire_log)

        rule_log = engine.process_step_end_rules()
        triggered_log.extend(rule_log)

        state_after = state.deep_copy()

        return SimulationStep(
            step_number=state.step_number,
            agent_actions=step_actions,
            triggered_effects_log=triggered_log,
            state_before=state_before,
            state_after=state_after,
        )

    async def run_simulation(
        self,
        from_state: SimulationState | None = None,
        max_steps: int | None = None,
    ) -> SimulationResult:
        state = from_state or self.create_initial_state()
        steps_to_run = max_steps or self.situation.max_steps
        steps: list[SimulationStep] = []

        with MonetaryCostManager():
            for i in range(steps_to_run):
                logger.info(
                    f"Running step {state.step_number + 1} "
                    f"(iteration {i + 1}/{steps_to_run})"
                )
                step = await self.run_step(state)
                steps.append(step)
                logger.info(
                    f"Step {step.step_number} complete. "
                    f"Actions: {len(step.agent_actions)}, "
                    f"Triggers: {len(step.triggered_effects_log)}"
                )

        return SimulationResult(
            situation=self.situation,
            steps=steps,
            final_state=state.deep_copy(),
        )

    def _execute_agent_action(
        self,
        engine: EffectEngine,
        action: AgentAction,
        agent_name: str,
    ) -> list[str]:
        if action.action_name == "no_action":
            return []

        trade_handlers = {
            "trade_propose": self._handle_trade_propose,
            "trade_accept": self._handle_trade_accept,
            "trade_reject": self._handle_trade_reject,
        }
        handler = trade_handlers.get(action.action_name)
        if handler:
            return handler(engine, action, agent_name)

        return self._handle_defined_action(engine, action, agent_name)

    def _handle_trade_propose(
        self, engine: EffectEngine, action: AgentAction, agent_name: str
    ) -> list[str]:
        if action.trade_proposal:
            engine.state.pending_trades.append(action.trade_proposal)
            return [f"{agent_name} proposed a trade (ID: {action.trade_proposal.id})"]
        return []

    def _handle_trade_accept(
        self, engine: EffectEngine, action: AgentAction, agent_name: str
    ) -> list[str]:
        if action.trade_acceptance_id:
            _, msg = engine.resolve_trade(action.trade_acceptance_id, agent_name)
            return [msg]
        return []

    def _handle_trade_reject(
        self, engine: EffectEngine, action: AgentAction, agent_name: str
    ) -> list[str]:
        if action.trade_acceptance_id:
            _, msg = engine.reject_trade(action.trade_acceptance_id)
            return [msg]
        return []

    def _handle_defined_action(
        self, engine: EffectEngine, action: AgentAction, agent_name: str
    ) -> list[str]:
        action_def = self._find_action_definition(action.action_name, agent_name)
        if action_def is None:
            return [f"{agent_name} attempted unknown action '{action.action_name}'"]
        return engine.apply_effects(action_def.effects, agent_name, action.parameters)

    def _find_action_definition(
        self,
        action_name: str,
        agent_name: str,
    ) -> ActionDefinition | None:
        global_match = self._find_global_action(action_name, agent_name)
        if global_match:
            return global_match
        return self._find_special_action(action_name, agent_name)

    def _find_global_action(
        self, action_name: str, agent_name: str
    ) -> ActionDefinition | None:
        for action in self.situation.environment.global_actions:
            if action.name != action_name:
                continue
            if action.available_to == "everyone" or agent_name in action.available_to:
                return action
        return None

    def _find_special_action(
        self, action_name: str, agent_name: str
    ) -> ActionDefinition | None:
        for agent_def in self.situation.agents:
            if agent_def.name != agent_name:
                continue
            for action in agent_def.special_actions:
                if action.name == action_name:
                    return action
        return None
