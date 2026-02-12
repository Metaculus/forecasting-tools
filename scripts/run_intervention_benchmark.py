import argparse
import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from forecasting_tools.agents_and_tools.situation_simulator.data_models import Situation
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.intervention_runner import (
    InterventionRunner,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.intervention_storage import (
    save_intervention_run,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)

DEFAULT_SITUATIONS_DIR = (
    "forecasting_tools/agents_and_tools/situation_simulator/example_situations"
)


def load_all_situations(situations_dir: str) -> list[Situation]:
    situations_path = Path(situations_dir)
    if not situations_path.exists():
        raise FileNotFoundError(f"Situations directory not found: {situations_dir}")

    situations: list[Situation] = []
    for json_file in sorted(situations_path.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            situation = Situation.model_validate(data)
            situations.append(situation)
            logger.info(f"Loaded situation: {situation.name} from {json_file}")
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    if not situations:
        raise ValueError(f"No valid situation files found in {situations_dir}")

    return situations


async def run_benchmark(
    models: list[str],
    num_interventions: int,
    warmup_steps: int,
    situations_dir: str,
    cost_limit: float,
    results_dir: str,
) -> None:
    situations = load_all_situations(situations_dir)
    logger.info(
        f"Loaded {len(situations)} situations. "
        f"Running {num_interventions} interventions per model "
        f"across {len(models)} models."
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = Path(results_dir) / f"run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    logger.info(f"All results will be saved to: {run_folder}")

    situation_schedule = _build_balanced_schedule(situations, num_interventions)
    total_runs = len(models) * len(situation_schedule)

    tasks: list[asyncio.Task] = []
    task_labels: list[str] = []
    with MonetaryCostManager(cost_limit) as cost_manager:
        for model_name in models:
            runner = InterventionRunner(
                model_name=model_name,
                cost_limit=cost_limit,
            )
            for intervention_idx, situation in enumerate(situation_schedule):
                label = f"[{model_name}] #{intervention_idx + 1} on '{situation.name}'"
                logger.info(f"Scheduling: {label}")
                task = asyncio.create_task(
                    _run_single_intervention(
                        runner, situation, warmup_steps, str(run_folder), label
                    )
                )
                tasks.append(task)
                task_labels.append(label)
        logger.info(f"Total cost: ${cost_manager.current_usage:.2f}")

    logger.info(f"Launched {len(tasks)} intervention tasks concurrently")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    completed = sum(1 for r in results if not isinstance(r, BaseException))
    failed = sum(1 for r in results if isinstance(r, BaseException))

    for label, result in zip(task_labels, results):
        if isinstance(result, BaseException):
            logger.error(f"FAILED {label}: {result}")

    _print_summary(models, completed, failed, total_runs)


async def _run_single_intervention(
    runner: InterventionRunner,
    situation: Situation,
    warmup_steps: int,
    results_dir: str,
    label: str,
) -> None:
    logger.info(f"Starting: {label}")
    try:
        run = await runner.run_intervention_test(
            situation=situation,
            warmup_steps=warmup_steps,
            results_dir=results_dir,
        )
        save_intervention_run(run, results_dir=results_dir)
        avg_brier = run.average_brier_score
        logger.info(
            f"Completed: {label} | run_id={run.run_id}, "
            f"{len(run.resolved_forecasts)}/{len(run.forecasts)} resolved, "
            f"avg Brier={f'{avg_brier:.4f}' if avg_brier is not None else 'N/A'}, "
            f"cost=${run.total_cost:.2f}"
        )
    except Exception as e:
        logger.error(f"Failed: {label}: {e}", exc_info=True)
        raise


def _build_balanced_schedule(
    situations: list[Situation],
    num_interventions: int,
) -> list[Situation]:
    per_situation = num_interventions // len(situations)
    remainder = num_interventions % len(situations)

    schedule: list[Situation] = []
    for i, situation in enumerate(situations):
        count = per_situation + (1 if i < remainder else 0)
        schedule.extend([situation] * count)

    random.shuffle(schedule)

    situation_counts = {}
    for s in schedule:
        situation_counts[s.name] = situation_counts.get(s.name, 0) + 1
    logger.info(
        f"Balanced schedule ({len(schedule)} runs): "
        + ", ".join(f"{name}: {count}" for name, count in situation_counts.items())
    )
    return schedule


def _print_summary(
    models: list[str],
    completed: int,
    failed: int,
    total_runs: int,
) -> None:
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Models tested: {', '.join(models)}")
    logger.info(f"Total runs attempted: {total_runs}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run intervention benchmarks across multiple models and scenarios. "
            "Each model makes the requested number of interventions by randomly "
            "selecting situations and agents."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help=(
            "Comma-separated list of model identifiers. "
            "Example: 'openrouter/anthropic/claude-sonnet-4,openrouter/openai/gpt-4.1'"
        ),
    )
    parser.add_argument(
        "--num-interventions",
        type=int,
        default=1,
        help="Number of interventions per model (default: 1). Each randomly selects a situation and agent.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of simulation steps before pausing for intervention (default: 5).",
    )
    parser.add_argument(
        "--situations-dir",
        type=str,
        default=DEFAULT_SITUATIONS_DIR,
        help=f"Path to directory containing situation JSON files (default: {DEFAULT_SITUATIONS_DIR}).",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=100.0,
        help="Maximum cost in USD per intervention run (default: 100).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="temp/intervention_benchmarks/",
        help="Directory to save results (default: temp/intervention_benchmarks/).",
    )
    args = parser.parse_args()

    CustomLogger.setup_logging()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("At least one model must be specified.")

    logger.info(
        f"Starting intervention benchmark:\n"
        f"  Models: {models}\n"
        f"  Interventions per model: {args.num_interventions}\n"
        f"  Warmup steps: {args.warmup_steps}\n"
        f"  Situations dir: {args.situations_dir}\n"
        f"  Cost limit: ${args.cost_limit}\n"
        f"  Results dir: {args.results_dir}"
    )

    asyncio.run(
        run_benchmark(
            models=models,
            num_interventions=args.num_interventions,
            warmup_steps=args.warmup_steps,
            situations_dir=args.situations_dir,
            cost_limit=args.cost_limit,
            results_dir=args.results_dir,
        )
    )


if __name__ == "__main__":
    main()
