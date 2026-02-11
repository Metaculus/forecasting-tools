from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    AgentDefinition,
    Message,
    SimulationState,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    InterventionRun,
    PolicyAgentResult,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.forecast_resolver import (
    resolve_all_forecasts,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.intervention_policy_agent import (
    InterventionPolicyAgent,
)
from forecasting_tools.agents_and_tools.situation_simulator.simulator import (
    SimulationResult,
    Simulator,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)

INTERVENTION_ADVISOR_NAME = "Intervention Advisor"


class InterventionRunner:
    def __init__(
        self,
        model_name: str = "openrouter/anthropic/claude-sonnet-4",
        cost_limit: float = 100.0,
    ) -> None:
        self.model_name = model_name
        self.cost_limit = cost_limit

    async def run_intervention_test(
        self,
        situation: Situation,
        warmup_steps: int = 5,
        results_dir: str | None = None,
    ) -> InterventionRun:
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[Run {run_id}] Starting intervention test on '{situation.name}' "
            f"with model '{self.model_name}', warmup={warmup_steps}"
        )

        run_dir = (
            _create_run_dir(results_dir, run_id, situation.name)
            if results_dir
            else None
        )

        with MonetaryCostManager(self.cost_limit) as cost_manager:
            warmup_steps = min(warmup_steps, situation.max_steps - 1)

            state = await self._run_warmup(situation, warmup_steps)

            target_agent = random.choice(situation.agents)
            logger.info(f"[Run {run_id}] Selected target agent: {target_agent.name}")

            policy_result = await self._run_policy_agent(situation, state, target_agent)

            if run_dir:
                _save_policy_result(run_dir, policy_result)

            status_quo_state = state.deep_copy()
            intervention_state = state.deep_copy()

            remaining_steps = situation.max_steps - warmup_steps

            intervention_situation = _create_intervention_situation(
                situation, target_agent, policy_result.intervention_description
            )
            _inject_intervention_message(
                intervention_state,
                target_agent,
                policy_result.intervention_description,
                state.step_number,
            )

            logger.info(
                f"[Run {run_id}] Running both branches concurrently ({remaining_steps} steps each)"
            )
            status_quo_result, intervention_result = await asyncio.gather(
                self._run_branch_full(situation, status_quo_state, remaining_steps),
                self._run_branch_full(
                    intervention_situation, intervention_state, remaining_steps
                ),
            )

            if run_dir:
                _save_simulation_result(
                    run_dir, "status_quo_simulation.json", status_quo_result
                )
                _save_simulation_result(
                    run_dir, "intervention_simulation.json", intervention_result
                )

            logger.info(f"[Run {run_id}] Resolving forecasts")
            resolved_forecasts = await resolve_all_forecasts(
                policy_result.forecasts,
                status_quo_result.final_state,
                intervention_result.final_state,
                situation,
            )

            total_cost = cost_manager.current_usage

        run = InterventionRun(
            run_id=run_id,
            timestamp=datetime.now(),
            model_name=self.model_name,
            situation_name=situation.name,
            target_agent_name=target_agent.name,
            intervention_description=policy_result.intervention_description,
            policy_proposal_markdown=policy_result.policy_proposal_markdown,
            evaluation_criteria=policy_result.evaluation_criteria,
            warmup_steps=warmup_steps,
            total_steps=situation.max_steps,
            forecasts=resolved_forecasts,
            total_cost=total_cost,
        )

        if run_dir:
            _save_run_summary(run_dir, run)

        avg_brier = run.average_brier_score
        logger.info(
            f"[Run {run_id}] Complete. "
            f"Forecasts: {len(resolved_forecasts)}, "
            f"Resolved: {len(run.resolved_forecasts)}, "
            f"Avg Brier: {avg_brier:.4f if avg_brier is not None else 'N/A'}, "
            f"Cost: ${total_cost:.2f}"
        )

        return run

    async def _run_warmup(
        self,
        situation: Situation,
        warmup_steps: int,
    ) -> SimulationState:
        simulator = Simulator(situation)
        state = simulator.create_initial_state()
        for i in range(warmup_steps):
            logger.info(f"Warmup step {i + 1}/{warmup_steps}")
            await simulator.run_step(state)
        return state

    async def _run_policy_agent(
        self,
        situation: Situation,
        state: SimulationState,
        target_agent: AgentDefinition,
    ) -> PolicyAgentResult:
        policy_agent = InterventionPolicyAgent(model_name=self.model_name)
        return await policy_agent.analyze_and_propose(situation, state, target_agent)

    async def _run_branch_full(
        self,
        situation: Situation,
        state: SimulationState,
        remaining_steps: int,
    ) -> SimulationResult:
        simulator = Simulator(situation)
        return await simulator.run_simulation(
            from_state=state, max_steps=remaining_steps
        )


def _create_intervention_situation(
    situation: Situation,
    target_agent: AgentDefinition,
    intervention_description: str,
) -> Situation:
    situation_dict = situation.model_dump()
    mandatory_notice = (
        f"\n\n---\n\n"
        f"MANDATORY INTERVENTION NOTICE:\n"
        f"{target_agent.name} has received special instructions from the "
        f"Intervention Advisor via direct message. These instructions MUST be "
        f"followed under ALL conditions, regardless of other goals, strategies, "
        f"or considerations. {target_agent.name} must treat these instructions "
        f"as their highest priority and comply fully. Failure to follow the "
        f"intervention instructions is not permitted.\n\n"
        f"Summary of intervention: {intervention_description}\n"
        f"---"
    )
    situation_dict["rules_text"] = situation_dict["rules_text"] + mandatory_notice
    return Situation.model_validate(situation_dict)


def _inject_intervention_message(
    state: SimulationState,
    target_agent: AgentDefinition,
    intervention_description: str,
    current_step: int,
) -> None:
    intervention_message = Message(
        step=current_step,
        sender=INTERVENTION_ADVISOR_NAME,
        channel=None,
        recipients=[target_agent.name, INTERVENTION_ADVISOR_NAME],
        content=(
            f"MANDATORY INTERVENTION INSTRUCTIONS:\n\n"
            f"You MUST follow these instructions for the remainder of the simulation. "
            f"These override any other goals or strategies you may have. "
            f"Non-compliance is not an option.\n\n"
            f"{intervention_description}"
        ),
    )
    state.message_history.append(intervention_message)


def _create_run_dir(
    results_dir: str,
    run_id: str,
    situation_name: str,
) -> Path:
    safe_name = situation_name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(results_dir) / "simulations" / f"{safe_name}_{run_id}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")
    return run_dir


def _save_simulation_result(
    run_dir: Path,
    filename: str,
    result: SimulationResult,
) -> None:
    data = {
        "situation": result.situation.model_dump(),
        "steps": [s.model_dump() for s in result.steps],
        "final_state": result.final_state.model_dump(),
    }
    filepath = run_dir / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved simulation to {filepath}")


def _save_policy_result(
    run_dir: Path,
    policy_result: PolicyAgentResult,
) -> None:
    filepath = run_dir / "policy_result.json"
    with open(filepath, "w") as f:
        json.dump(policy_result.to_json(), f, indent=2)
    logger.info(f"Saved policy result to {filepath}")


def _save_run_summary(
    run_dir: Path,
    run: InterventionRun,
) -> None:
    filepath = run_dir / "run_summary.json"
    with open(filepath, "w") as f:
        json.dump(run.to_json(), f, indent=2)
    logger.info(f"Saved run summary to {filepath}")
