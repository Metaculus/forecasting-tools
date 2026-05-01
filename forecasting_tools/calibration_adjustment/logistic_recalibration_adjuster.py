from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from forecasting_tools.calibration_adjustments._training_data import (
    extract_bernoulli_observations,
)
from forecasting_tools.calibration_adjustments.calibration_adjuster import (
    CalibrationAdjuster,
)


class LogisticRecalibrationAdjuster(CalibrationAdjuster):
    """Affine-in-log-odds calibration adjustment (Platt-style).

    Models the recalibration function

        mu(p) = sigmoid( (logit(p) - BIA) / CON )

    where ``BIA`` captures systematic over- or underestimation of "yes"
    and ``CON`` captures over- or underconfidence:

    - ``BIA > 0``: forecaster systematically overestimates Yes
    - ``BIA < 0``: forecaster systematically overestimates No
    - ``CON > 1``: forecaster overconfident (forecasts too extreme)
    - ``CON < 1``: forecaster underconfident (forecasts too conservative)

    Parameters are estimated by unregularized logistic regression of the
    resolved outcomes on the log-odds of the past predictions::

        sigmoid(alpha + beta * logit(p))

    with the mapping ``CON = 1 / beta`` and ``BIA = -alpha / beta``.
    """

    EPS: float = 1e-4
    LOGIT_CLIP: float = 1e-6
    MIN_OBSERVATIONS: int = 10

    def __init__(self) -> None:
        self.bias: float | None = None
        self.confidence: float | None = None

    def train(self, forecasts: pd.DataFrame) -> None:
        predictions, outcomes = extract_bernoulli_observations(forecasts)

        if len(predictions) < self.MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least MIN_OBSERVATIONS={self.MIN_OBSERVATIONS} "
                f"observations to fit, got {len(predictions)}"
            )
        if len(np.unique(outcomes)) < 2:
            raise ValueError(
                "Logistic regression requires both 'yes' and 'no' "
                "outcomes in training data"
            )

        z = self._safe_logit(predictions).reshape(-1, 1)
        model = LogisticRegression(
            penalty=None,
            fit_intercept=True,
            solver="lbfgs",
        )
        model.fit(z, outcomes.astype(int))

        alpha = float(model.intercept_.ravel()[0])
        beta = float(model.coef_.ravel()[0])

        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError(
                f"Logistic regression produced non-finite parameters "
                f"(alpha={alpha}, beta={beta})"
            )
        if beta <= 0:
            raise ValueError(
                f"Logistic regression produced non-positive slope "
                f"(beta={beta}); training data is anti-correlated with "
                f"outcomes and cannot yield a monotonic recalibration"
            )

        confidence = 1.0 / beta
        bias = -alpha / beta

        if not np.isfinite(bias):
            raise ValueError(f"bias (BIA) must be finite, got {bias}")
        if not np.isfinite(confidence) or confidence <= 0:
            raise ValueError(
                f"confidence (CON) must be finite and > 0, got {confidence}"
            )

        self.bias = bias
        self.confidence = confidence

    def _require_fitted(self) -> None:
        if self.bias is None or self.confidence is None:
            raise RuntimeError("call train() before adjusting forecasts")

    def adjust_binary_forecast(self, prediction: float) -> float:
        self._require_fitted()
        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"prediction must be in [0, 1], got {prediction}"
            )
        z = self._safe_logit(np.array([prediction]))[0]
        adjusted = float(expit((z - self.bias) / self.confidence))
        return float(np.clip(adjusted, self.EPS, 1.0 - self.EPS))

    def adjust_multiple_choice_forecast(
        self, predictions: Sequence[float]
    ) -> list[float]:
        self._require_fitted()
        preds = list(predictions)
        if any(not 0.0 <= p <= 1.0 for p in preds):
            raise ValueError("all predictions must be in [0, 1]")
        if not preds:
            return []
        z = self._safe_logit(np.array(preds, dtype=np.float64))
        adjusted = expit((z - self.bias) / self.confidence)
        clipped = np.clip(adjusted, self.EPS, 1.0 - self.EPS)
        total = clipped.sum()
        if total == 0:
            n = len(clipped)
            return [1.0 / n] * n
        return (clipped / total).tolist()

    @classmethod
    def _safe_logit(cls, p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, cls.LOGIT_CLIP, 1.0 - cls.LOGIT_CLIP)
        return logit(clipped)
