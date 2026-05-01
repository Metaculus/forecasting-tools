from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting_tools.calibration_adjustments._training_data import (
    extract_bernoulli_observations,
)
from forecasting_tools.calibration_adjustments.calibration_adjuster import (
    CalibrationAdjuster,
)


class ConstantShiftAdjuster(CalibrationAdjuster):
    """Trains a single +/- shift applied uniformly to every probability.

    The shift ``n`` is the mean residual across all Bernoulli observations
    extracted from the training data::

        n = mean(outcome - prediction)

    For binary forecasts the adjusted probability is
    ``clip(prediction + n, EPS, 1 - EPS)``.

    For multiple-choice forecasts the same shift is applied to each option
    probability, each is clipped to ``[EPS, 1 - EPS]``, and the results are
    re-normalized to sum to 1.0.
    """

    EPS: float = 1e-4

    def __init__(self) -> None:
        self.shift: float | None = None

    def train(self, forecasts: pd.DataFrame) -> None:
        predictions, outcomes = extract_bernoulli_observations(forecasts)
        self.shift = float(np.mean(outcomes - predictions))

    def adjust_binary_forecast(self, prediction: float) -> float:
        if self.shift is None:
            raise RuntimeError("call train() before adjusting forecasts")
        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"prediction must be in [0, 1], got {prediction}"
            )
        return float(np.clip(prediction + self.shift, self.EPS, 1.0 - self.EPS))

    def adjust_multiple_choice_forecast(
        self, predictions: Sequence[float]
    ) -> list[float]:
        if self.shift is None:
            raise RuntimeError("call train() before adjusting forecasts")
        preds = list(predictions)
        if any(not 0.0 <= p <= 1.0 for p in preds):
            raise ValueError(
                "all predictions must be in [0, 1]"
            )
        shifted = np.array(preds, dtype=np.float64) + self.shift
        clipped = np.clip(shifted, self.EPS, 1.0 - self.EPS)
        total = clipped.sum()
        if total == 0:
            n = len(clipped)
            return [1.0 / n] * n
        normalized = clipped / total
        return normalized.tolist()
