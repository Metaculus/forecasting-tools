from __future__ import annotations

import logging
from pathlib import Path

from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    InterventionRun,
)
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.file_manipulation import add_to_jsonl_file

logger = logging.getLogger(__name__)

INTERVENTION_RESULTS_DIR = "temp/intervention_benchmarks/"


def save_intervention_run(
    run: InterventionRun,
    results_dir: str = INTERVENTION_RESULTS_DIR,
) -> str:
    safe_model_name = run.model_name.replace("/", "_")
    file_path = f"{results_dir.strip('/')}/{safe_model_name}.jsonl"
    add_to_jsonl_file(file_path, run.to_json())
    logger.info(f"Saved intervention run {run.run_id} to {file_path}")
    return file_path


def load_all_intervention_runs(
    results_dir: str = INTERVENTION_RESULTS_DIR,
) -> list[InterventionRun]:
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.info(f"No intervention results directory found at {results_dir}")
        return []

    all_runs: list[InterventionRun] = []
    all_runs.extend(_load_from_jsonl_files(results_path))
    all_runs.extend(_load_from_run_summary_files(results_path))

    seen_ids: set[str] = set()
    deduped: list[InterventionRun] = []
    for run in all_runs:
        if run.run_id not in seen_ids:
            seen_ids.add(run.run_id)
            deduped.append(run)

    logger.info(f"Loaded {len(deduped)} unique intervention runs from {results_dir}")
    return deduped


def list_run_folders(base_dir: str = INTERVENTION_RESULTS_DIR) -> list[str]:
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    folders = []
    for entry in sorted(base_path.iterdir()):
        if entry.is_dir():
            folders.append(str(entry))
    return folders


def _load_from_jsonl_files(results_path: Path) -> list[InterventionRun]:
    runs: list[InterventionRun] = []
    for file_path in sorted(results_path.rglob("*.jsonl")):
        try:
            loaded = InterventionRun.load_json_from_file_path(str(file_path))
            runs.extend(loaded)
            logger.info(f"Loaded {len(loaded)} runs from {file_path}")
        except Exception as e:
            logger.error(f"Error loading JSONL {file_path}: {e}")
    return runs


def _load_from_run_summary_files(results_path: Path) -> list[InterventionRun]:
    runs: list[InterventionRun] = []
    for file_path in sorted(results_path.rglob("run_summary.json")):
        try:
            data = file_manipulation.load_json_file(str(file_path))[0]
            run = InterventionRun.from_json(data)
            runs.append(run)
            logger.info(f"Loaded run {run.run_id} from {file_path}")
        except Exception as e:
            logger.error(f"Error loading run_summary {file_path}: {e}")
    return runs
