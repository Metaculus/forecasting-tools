import argparse
import asyncio
import logging

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    SimulationStep,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.simulator import (
    Simulator,
    create_run_directory,
    save_full_simulation,
    save_situation_to_file,
    save_step_to_file,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


def load_situation_from_file(filepath: str) -> Situation:
    data = file_manipulation.load_json_file(filepath)
    if len(data) != 1:
        raise ValueError(
            f"Situation file must contain exactly one situation, but found {len(data)}"
        )
    return Situation.model_validate(data[0])


async def run_simulation(
    situation_path: str,
    max_steps: int | None = None,
) -> None:
    situation = load_situation_from_file(situation_path)
    logger.info(
        f"Loaded situation '{situation.name}' with "
        f"{len(situation.agents)} agents, max_steps={situation.max_steps}"
    )

    run_dir = create_run_directory(situation.name)
    logger.info(f"Saving simulation output to {run_dir}")
    save_situation_to_file(run_dir, situation)

    simulator = Simulator(situation)
    state = simulator.create_initial_state()
    steps_to_run = max_steps or situation.max_steps
    all_steps: list[SimulationStep] = []

    with MonetaryCostManager(100) as cost_manager:
        for i in range(steps_to_run):
            step_number = state.step_number + 1
            logger.info(
                f"--- Running step {step_number} "
                f"(iteration {i + 1}/{steps_to_run}) ---"
            )

            step = await simulator.run_step(state)
            all_steps.append(step)

            save_step_to_file(run_dir, step)

            logger.info(
                f"Step {step.step_number} complete. "
                f"Actions: {len(step.agent_actions)}, "
                f"Triggers: {len(step.triggered_effects_log)}, "
                f"Cost so far: ${cost_manager.current_usage:.4f}"
            )

    total_cost = cost_manager.current_usage
    save_full_simulation(run_dir, situation, all_steps, state, total_cost)

    logger.info(
        f"Simulation complete. {len(all_steps)} steps run. "
        f"Total cost: ${total_cost:.4f}. "
        f"Output: {run_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a situation simulation and save each step to disk."
    )
    parser.add_argument(
        "situation_file",
        help="Path to the situation JSON file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the max number of steps (defaults to the situation's max_steps).",
    )
    args = parser.parse_args()

    CustomLogger.setup_logging()
    asyncio.run(run_simulation(args.situation_file, args.max_steps))


if __name__ == "__main__":
    main()
