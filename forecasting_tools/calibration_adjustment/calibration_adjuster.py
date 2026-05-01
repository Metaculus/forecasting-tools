from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_BINARY_RESOLUTIONS = {"yes", "no"}


class CalibrationAdjuster(ABC):
    """Abstract base for calibration adjusters trained on a forecaster's history.

    Subclasses learn an adjustment from a DataFrame of past forecasts (plus
    their resolutions), then expose ``adjust_binary_forecast`` and
    ``adjust_multiple_choice_forecast`` to apply that adjustment to new
    forecasts.

    The training DataFrame is expected to follow the schema used by
    ``forecasting_tools.datasets`` (a subset of columns is sufficient):

    ===============  ===================  ===========================================
    Column           Type                 Notes
    ===============  ===================  ===========================================
    type             str                  ``"binary"`` or ``"multiple_choice"``
    probability_yes  float                Required for ``type=="binary"`` rows
    probability_yes_per_category                                    Required for ``type=="multiple_choice"``
                     str (JSON list)      rows
    options          str (JSON list)      Required for ``type=="multiple_choice"``
                                         rows
    resolution       str                  ``"yes"``/``"no"`` for binary; an option
                                         label for multiple-choice
    ===============  ===================  ===========================================

    Rows that cannot be interpreted as resolved binary or multiple-choice
    observations are silently skipped during training.  Callers should
    pre-filter the DataFrame (e.g. to a single forecaster) before passing it
    to :meth:`train`.

    Usage::

        adjuster = SomeAdjuster()        # construct with hyperparameters
        adjuster.train(history_df)       # fit in place; returns None
        p_adj = adjuster.adjust_binary_forecast(0.42)
        brier = adjuster.test(holdout_df)
    """

    @abstractmethod
    def train(self, forecasts: pd.DataFrame) -> None:
        """Fit the adjuster in place from past forecasts.

        Args:
            forecasts: DataFrame with the columns documented above.

        Returns:
            ``None``. Fitted state is stored on the instance.

        Raises:
            ValueError: if no usable resolved rows are found.
        """

    @abstractmethod
    def adjust_binary_forecast(self, prediction: float) -> float:
        """Adjust a single binary probability.

        Args:
            prediction: Raw forecasted probability in [0, 1].

        Returns:
            Calibrated probability in (0, 1).

        Raises:
            ValueError: if *prediction* is outside [0, 1].
        """

    @abstractmethod
    def adjust_multiple_choice_forecast(
        self, predictions: Sequence[float]
    ) -> list[float]:
        """Adjust a list of option probabilities for a multiple-choice question.

        The returned probabilities are re-normalized so they sum to 1.0.

        Args:
            predictions: Raw per-option probabilities, each in [0, 1].

        Returns:
            Calibrated and normalized per-option probabilities.

        Raises:
            ValueError: if any prediction is outside [0, 1].
        """

    def test(self, forecasts: pd.DataFrame) -> float:
        """Brier score (mean squared error) of *adjusted* forecasts.

        Each row contributes one number; the final score is the mean over rows:

        - **binary**: ``(adjust_binary_forecast(p) - y)**2`` where ``y`` is
          ``1.0`` if the resolution is ``"yes"`` else ``0.0``.
        - **multiple_choice**: ``mean_i ((adjusted_p_i - y_i)**2)`` over the
          ``N`` options, where ``y_i = 1.0`` only for the resolved option.
          The 1/N weighting puts MC rows on the same ``[0, 1]`` scale as
          binary rows.

        Rows that are not resolved binary or multiple-choice, or that are
        malformed (bad JSON, length mismatch, out-of-range probabilities,
        unknown resolution label), are skipped.

        Args:
            forecasts: DataFrame with the schema documented on the class.

        Returns:
            Mean Brier score across usable rows.

        Raises:
            ValueError: if no usable rows are found.
        """
        scores: list[float] = []

        for _, row in forecasts.iterrows():
            q_type = row.get("type")
            resolution = row.get("resolution")

            if pd.isna(resolution) or resolution is None:
                continue

            if q_type == "binary":
                score = self._score_binary_row(row, resolution)
            elif q_type == "multiple_choice":
                score = self._score_mc_row(row, resolution)
            else:
                continue

            if score is not None:
                scores.append(score)

        if not scores:
            raise ValueError(
                "No usable resolved binary or multiple-choice rows in DataFrame"
            )

        return float(np.mean(scores))

    def _score_binary_row(
        self, row: pd.Series, resolution: str
    ) -> float | None:
        if resolution not in _VALID_BINARY_RESOLUTIONS:
            return None

        prob = row.get("probability_yes")
        if prob is None or (isinstance(prob, float) and np.isnan(prob)):
            return None

        try:
            prob = float(prob)
        except (ValueError, TypeError):
            return None

        if not 0.0 <= prob <= 1.0:
            return None

        adjusted = self.adjust_binary_forecast(prob)
        y = 1.0 if resolution == "yes" else 0.0
        return float((adjusted - y) ** 2)

    def _score_mc_row(
        self, row: pd.Series, resolution: str
    ) -> float | None:
        raw_probs = row.get("probability_yes_per_category")
        raw_options = row.get("options")

        if pd.isna(raw_probs) or pd.isna(raw_options):
            return None

        try:
            probs = (
                json.loads(raw_probs) if isinstance(raw_probs, str) else raw_probs
            )
            options = (
                json.loads(raw_options)
                if isinstance(raw_options, str)
                else raw_options
            )
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(probs, list) or not isinstance(options, list):
            return None
        if len(probs) != len(options) or not options:
            return None
        if resolution not in options:
            return None

        try:
            probs_f = [float(p) for p in probs]
        except (ValueError, TypeError):
            return None
        if any(not 0.0 <= p <= 1.0 for p in probs_f):
            return None

        adjusted = self.adjust_multiple_choice_forecast(probs_f)
        if len(adjusted) != len(options):
            return None

        y = np.array(
            [1.0 if opt == resolution else 0.0 for opt in options],
            dtype=np.float64,
        )
        a = np.asarray(adjusted, dtype=np.float64)
        # 1/N weighting: each option contributes 1/N of the row's score.
        return float(np.mean((a - y) ** 2))
