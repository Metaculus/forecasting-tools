from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_BINARY_RESOLUTIONS = {"yes", "no"}


def extract_bernoulli_observations(
    forecasts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (prediction, outcome) pairs from a forecasts DataFrame.

    Each binary forecast contributes one observation:
        (probability_yes, 1.0 if resolution=='yes' else 0.0)

    Each multiple-choice forecast contributes one observation per option:
        (p_option_i, 1.0 if resolution==option_i else 0.0)

    Rows that are not resolved binary or multiple-choice are skipped.
    Malformed rows (missing probability, unparseable JSON, length
    mismatches between options and probabilities) are also skipped.

    Returns:
        predictions: 1-D float array of predicted probabilities.
        outcomes:    1-D float array of resolved outcomes (0.0 or 1.0).

    Raises:
        ValueError: if no usable observations are found.
    """
    predictions: list[float] = []
    outcomes: list[float] = []

    for _, row in forecasts.iterrows():
        q_type = row.get("type")
        resolution = row.get("resolution")

        if pd.isna(resolution) or resolution is None:
            continue

        if q_type == "binary":
            _process_binary_row(row, resolution, predictions, outcomes)
        elif q_type == "multiple_choice":
            _process_mc_row(row, resolution, predictions, outcomes)

    if not predictions:
        raise ValueError(
            "No usable resolved binary or multiple-choice rows in DataFrame"
        )

    return np.array(predictions, dtype=np.float64), np.array(
        outcomes, dtype=np.float64
    )


def _process_binary_row(
    row: pd.Series,
    resolution: str,
    predictions: list[float],
    outcomes: list[float],
) -> None:
    if resolution not in _VALID_BINARY_RESOLUTIONS:
        return

    prob = row.get("probability_yes")
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return

    try:
        prob = float(prob)
    except (ValueError, TypeError):
        return

    if not 0.0 <= prob <= 1.0:
        return

    predictions.append(prob)
    outcomes.append(1.0 if resolution == "yes" else 0.0)


def _process_mc_row(
    row: pd.Series,
    resolution: str,
    predictions: list[float],
    outcomes: list[float],
) -> None:
    raw_probs = row.get("probability_yes_per_category")
    raw_options = row.get("options")

    if pd.isna(raw_probs) or pd.isna(raw_options):
        return

    try:
        probs = json.loads(raw_probs) if isinstance(raw_probs, str) else raw_probs
        options = json.loads(raw_options) if isinstance(raw_options, str) else raw_options
    except (json.JSONDecodeError, TypeError):
        return

    if not isinstance(probs, list) or not isinstance(options, list):
        return
    if len(probs) != len(options):
        logger.warning(
            "Skipping MC row: options length (%d) != probabilities length (%d)",
            len(options),
            len(probs),
        )
        return

    for option, prob in zip(options, probs):
        try:
            prob = float(prob)
        except (ValueError, TypeError):
            continue
        if not 0.0 <= prob <= 1.0:
            continue

        predictions.append(prob)
        outcomes.append(1.0 if resolution == option else 0.0)
