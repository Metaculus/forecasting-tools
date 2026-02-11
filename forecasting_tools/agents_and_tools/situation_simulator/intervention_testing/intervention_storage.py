from __future__ import annotations

import logging
import os
from pathlib import Path

from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    InterventionRun,
)
from forecasting_tools.util.file_manipulation import add_to_jsonl_file

logger = logging.getLogger(__name__)

INTERVENTION_RESULTS_DIR = "logs/intervention_benchmarks/"


def save_intervention_run(
    run: InterventionRun,
    results_dir: str = INTERVENTION_RESULTS_DIR,
) -> str:
    safe_model_name = run.model_name.replace("/", "_")
    file_path = f"{results_dir}{safe_model_name}.jsonl"
    add_to_jsonl_file(file_path, run.to_json())
    logger.info(f"Saved intervention run {run.run_id} to {file_path}")
    return file_path


def load_all_intervention_runs(
    results_dir: str = INTERVENTION_RESULTS_DIR,
) -> list[InterventionRun]:
    all_runs: list[InterventionRun] = []
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.info(f"No intervention results directory found at {results_dir}")
        return all_runs

    jsonl_files = sorted(results_path.glob("*.jsonl"))
    for file_path in jsonl_files:
        try:
            runs = InterventionRun.load_json_from_file_path(str(file_path))
            all_runs.extend(runs)
            logger.info(f"Loaded {len(runs)} runs from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(all_runs)} total intervention runs")
    return all_runs


def load_intervention_runs_for_model(
    model_name: str,
    results_dir: str = INTERVENTION_RESULTS_DIR,
) -> list[InterventionRun]:
    safe_model_name = model_name.replace("/", "_")
    file_path = f"{results_dir}{safe_model_name}.jsonl"
    if not os.path.exists(file_path):
        logger.info(f"No results file found for model '{model_name}' at {file_path}")
        return []
    try:
        return InterventionRun.load_json_from_file_path(file_path)
    except Exception as e:
        logger.error(f"Error loading results for model '{model_name}': {e}")
        return []


def get_available_model_names(
    results_dir: str = INTERVENTION_RESULTS_DIR,
) -> list[str]:
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    model_names = []
    for file_path in sorted(results_path.glob("*.jsonl")):
        model_name = file_path.stem.replace("_", "/")
        model_names.append(model_name)
    return model_names
