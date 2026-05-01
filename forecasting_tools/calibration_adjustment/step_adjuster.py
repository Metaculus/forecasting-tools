from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from forecasting_tools.calibration_adjustments._training_data import (
    extract_bernoulli_observations,
)
from forecasting_tools.calibration_adjustments.calibration_adjuster import (
    CalibrationAdjuster,
)


class StepAdjuster(CalibrationAdjuster):
    """Equal-width step calibration adjuster.

    Splits ``[0, 1]`` into ``N`` equal-width buckets. Each bucket stores
    an additive shift equal to the mean residual
    ``mean(outcome - prediction)`` over training observations whose
    prediction falls in that bucket. Empty buckets fall back to the
    global mean residual.

    A new prediction ``p`` is routed to bucket ``floor(p * N)`` (clamped
    to ``N - 1``), then adjusted as
    ``clip(p + shift_bucket, EPS, 1 - EPS)``.

    If ``n_buckets`` is omitted at construction time, the best ``N`` in
    ``[1, max_buckets]`` is chosen via a single train/test split inside
    the training data: for each candidate ``N`` we fit on the train
    portion and pick the ``N`` minimizing Brier score on the test
    portion. The final model is then refit on the full training data
    using the chosen ``N``.
    """

    EPS: float = 1e-4

    def __init__(
        self,
        n_buckets: Optional[int] = None,
        *,
        test_split: float = 0.3,
        max_buckets: int = 30,
        random_state: int | None = 0,
    ) -> None:
        """Construct an unfitted StepAdjuster.

        Args:
            n_buckets: Number of equal-width buckets. If ``None``, the
                best value in ``[1, max_buckets]`` is selected during
                :meth:`train` via an internal train/test split.
            test_split: Fraction of training observations held out for
                bucket-count selection (only used when
                ``n_buckets is None``). Defaults to 0.3.
            max_buckets: Upper bound for the bucket-count search
                (only used when ``n_buckets is None``).
            random_state: Seed for the internal train/test split.
        """
        if n_buckets is not None and n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
        if not 0.0 < test_split < 1.0:
            raise ValueError(
                f"test_split must be in (0, 1), got {test_split}"
            )
        if max_buckets < 1:
            raise ValueError(f"max_buckets must be >= 1, got {max_buckets}")

        self.n_buckets = n_buckets
        self.test_split = test_split
        self.max_buckets = max_buckets
        self.random_state = random_state
        self._shifts: np.ndarray | None = None
        self._global_shift: float | None = None

    def train(self, forecasts: pd.DataFrame) -> None:
        predictions, outcomes = extract_bernoulli_observations(forecasts)

        if self.n_buckets is None:
            self.n_buckets = self._select_n_buckets(predictions, outcomes)

        shifts, global_shift = self._fit_shifts(
            predictions, outcomes, self.n_buckets
        )
        self._shifts = shifts
        self._global_shift = global_shift

    @staticmethod
    def _bucket_index(
        p: np.ndarray | float, n: int
    ) -> np.ndarray:
        """Map probabilities in [0, 1] to bucket index in [0, n-1].

        ``p == 1.0`` is placed in the last bucket rather than overflowing.
        """
        arr = np.asarray(p, dtype=np.float64)
        idx = np.floor(arr * n).astype(np.int64)
        return np.minimum(idx, n - 1)

    @classmethod
    def _fit_shifts(
        cls,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, float]:
        """Compute per-bucket shifts with a global-mean fallback.

        Returns:
            (shifts, global_shift) where ``shifts`` has length ``n`` and
            ``shifts[k]`` is the mean residual within bucket ``k`` (or the
            global mean residual if bucket ``k`` is empty).
        """
        global_shift = float(np.mean(outcomes - predictions))
        shifts = np.full(n, global_shift, dtype=np.float64)
        idx = cls._bucket_index(predictions, n)
        for k in range(n):
            mask = idx == k
            if mask.any():
                shifts[k] = float(
                    np.mean(outcomes[mask] - predictions[mask])
                )
        return shifts, global_shift

    def _select_n_buckets(
        self, predictions: np.ndarray, outcomes: np.ndarray
    ) -> int:
        """Pick N minimizing Brier score on a held-out split.

        Uses a single train/test split on the Bernoulli pairs. For each
        candidate ``N`` in ``[1, max_buckets]`` we fit shifts on the train
        portion, compute the per-pair Brier score on the test portion,
        and pick the strict minimum. Falls back to ``N = 1`` if there are
        too few observations to split.
        """
        if len(predictions) < 2:
            return 1

        p_tr, p_te, y_tr, y_te = train_test_split(
            predictions,
            outcomes,
            test_size=self.test_split,
            random_state=self.random_state,
        )

        # Need at least one observation in each split for a meaningful score.
        if len(p_tr) == 0 or len(p_te) == 0:
            return 1

        best_n = 1
        best_score = math.inf
        for n in range(1, self.max_buckets + 1):
            shifts, _ = self._fit_shifts(p_tr, y_tr, n)
            idx = self._bucket_index(p_te, n)
            adjusted = np.clip(
                p_te + shifts[idx], self.EPS, 1.0 - self.EPS
            )
            score = float(np.mean((adjusted - y_te) ** 2))
            if score < best_score:
                best_score = score
                best_n = n
        return best_n

    def _require_fitted(self) -> None:
        if self._shifts is None or self.n_buckets is None:
            raise RuntimeError("call train() before adjusting forecasts")

    def adjust_binary_forecast(self, prediction: float) -> float:
        self._require_fitted()
        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"prediction must be in [0, 1], got {prediction}"
            )
        assert self._shifts is not None and self.n_buckets is not None
        k = int(self._bucket_index(prediction, self.n_buckets))
        return float(
            np.clip(
                prediction + self._shifts[k], self.EPS, 1.0 - self.EPS
            )
        )

    def adjust_multiple_choice_forecast(
        self, predictions: Sequence[float]
    ) -> list[float]:
        self._require_fitted()
        preds = list(predictions)
        if any(not 0.0 <= p <= 1.0 for p in preds):
            raise ValueError("all predictions must be in [0, 1]")
        if not preds:
            return []
        assert self._shifts is not None and self.n_buckets is not None
        arr = np.array(preds, dtype=np.float64)
        idx = self._bucket_index(arr, self.n_buckets)
        adjusted = np.clip(
            arr + self._shifts[idx], self.EPS, 1.0 - self.EPS
        )
        total = adjusted.sum()
        if total == 0:
            n = len(adjusted)
            return [1.0 / n] * n
        return (adjusted / total).tolist()
