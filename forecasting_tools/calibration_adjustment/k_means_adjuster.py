from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from forecasting_tools.calibration_adjustments._training_data import (
    extract_bernoulli_observations,
)
from forecasting_tools.calibration_adjustments.calibration_adjuster import (
    CalibrationAdjuster,
)


class KMeansAdjuster(CalibrationAdjuster):
    """K-means cluster calibration adjuster.

    Clusters past predictions in 1-D using K-means with ``K`` centers.
    Each cluster stores an additive shift equal to the mean residual
    ``mean(outcome - prediction)`` over training observations whose
    prediction was assigned to that cluster. Empty clusters fall back
    to the global mean residual.

    A new prediction ``p`` is routed to its nearest center, then
    adjusted as ``clip(p + shift_cluster, EPS, 1 - EPS)``. In 1-D this
    is equivalent to looking up ``p`` in the intervals defined by the
    midpoints between adjacent sorted centers, which is what we use at
    inference time (so we don't need to keep the fitted ``KMeans``
    object around).

    Compared to :class:`StepAdjuster`, K-means concentrates more
    clusters where prediction density is high (e.g. near 0 and near 1
    for confident forecasters), allowing finer corrections in those
    regions and broader corrections where data is sparse.

    If ``k`` is omitted at construction time, the best ``K`` in
    ``[1, max_k]`` is chosen via a single train/test split inside the
    training data: for each candidate ``K`` we fit on the train
    portion and pick the ``K`` minimizing Brier score on the test
    portion. The final model is then refit on the full training data
    using the chosen ``K``.
    """

    EPS: float = 1e-4

    def __init__(
        self,
        k: Optional[int] = None,
        *,
        test_split: float = 0.3,
        max_k: int = 30,
        random_state: int | None = None,
        n_init: int = 10,
    ) -> None:
        """Construct an unfitted KMeansAdjuster.

        Args:
            k: Number of K-means clusters. If ``None``, the best value
                in ``[1, max_k]`` is selected during :meth:`train` via
                an internal train/test split.
            test_split: Fraction of training observations held out for
                ``K`` selection (only used when ``k is None``).
                Defaults to 0.3.
            max_k: Upper bound for the ``K`` search (only used when
                ``k is None``).
            random_state: Seed forwarded to ``KMeans`` and to the
                internal train/test split. ``None`` means
                non-deterministic, matching sklearn's default.
            n_init: Number of K-means restarts. With 1-D data and
                k-means++ initialization, 10 is a reliable default.
        """
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0.0 < test_split < 1.0:
            raise ValueError(
                f"test_split must be in (0, 1), got {test_split}"
            )
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1, got {max_k}")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")

        self.k = k
        self.test_split = test_split
        self.max_k = max_k
        self.random_state = random_state
        self.n_init = n_init
        self._centers: np.ndarray | None = None  # sorted ascending
        self._shifts: np.ndarray | None = None
        self._global_shift: float | None = None

    def train(self, forecasts: pd.DataFrame) -> None:
        predictions, outcomes = extract_bernoulli_observations(forecasts)

        if self.k is None:
            self.k = self._select_k(predictions, outcomes)

        # Cap k at the number of unique predictions; K-means cannot
        # produce more clusters than distinct points.
        n_unique = int(np.unique(predictions).size)
        effective_k = max(1, min(self.k, n_unique))
        self.k = effective_k

        centers, shifts, global_shift = self._fit_clusters(
            predictions,
            outcomes,
            effective_k,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self._centers = centers
        self._shifts = shifts
        self._global_shift = global_shift

    @classmethod
    def _fit_clusters(
        cls,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        k: int,
        *,
        random_state: int | None,
        n_init: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit K-means in 1-D, then compute per-cluster mean residuals.

        Returns:
            (centers_sorted, shifts, global_shift) where
            ``centers_sorted`` has length ``k`` ascending and
            ``shifts[i]`` is the mean residual within the cluster
            whose center is ``centers_sorted[i]`` (or the global mean
            residual if that cluster is empty in training).
        """
        global_shift = float(np.mean(outcomes - predictions))

        if k == 1:
            # Degenerate case: single cluster equals ConstantShift.
            center = float(np.mean(predictions))
            return (
                np.array([center], dtype=np.float64),
                np.array([global_shift], dtype=np.float64),
                global_shift,
            )

        km = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )
        km.fit(predictions.reshape(-1, 1))

        raw_centers = km.cluster_centers_.ravel()
        raw_labels = km.labels_

        # Sort centers ascending and remap labels so that index i
        # consistently refers to the i-th smallest center.
        order = np.argsort(raw_centers)
        centers_sorted = raw_centers[order].astype(np.float64)
        # Build a remap: old_label -> new_label = position of old_label in `order`.
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        labels = remap[raw_labels]

        shifts = np.full(k, global_shift, dtype=np.float64)
        for i in range(k):
            mask = labels == i
            if mask.any():
                shifts[i] = float(
                    np.mean(outcomes[mask] - predictions[mask])
                )

        return centers_sorted, shifts, global_shift

    @staticmethod
    def _assign(
        p: np.ndarray | float, centers: np.ndarray
    ) -> np.ndarray:
        """Vectorized nearest-center assignment for sorted 1-D centers.

        Returns an integer array of cluster indices with the same shape
        as ``p`` (always at least 1-D).
        """
        arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        if centers.size == 1:
            return np.zeros(arr.shape, dtype=np.int64)
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        idx = np.searchsorted(boundaries, arr, side="right")
        return idx.astype(np.int64)

    def _select_k(
        self, predictions: np.ndarray, outcomes: np.ndarray
    ) -> int:
        """Pick K minimizing Brier score on a held-out split.

        Mirrors :meth:`StepAdjuster._select_n_buckets`: a single
        train/test split on the Bernoulli pairs; for each candidate
        ``K`` in ``[1, max_k]`` (capped at the number of unique
        training predictions) we fit on the train portion, score on
        the test portion, and take the strict minimum.
        """
        if len(predictions) < 2:
            return 1

        p_tr, p_te, y_tr, y_te = train_test_split(
            predictions,
            outcomes,
            test_size=self.test_split,
            random_state=self.random_state,
        )

        if len(p_tr) == 0 or len(p_te) == 0:
            return 1

        n_unique_tr = int(np.unique(p_tr).size)
        k_max = max(1, min(self.max_k, n_unique_tr))

        best_k = 1
        best_score = math.inf
        for k in range(1, k_max + 1):
            centers, shifts, _ = self._fit_clusters(
                p_tr,
                y_tr,
                k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            idx = self._assign(p_te, centers)
            adjusted = np.clip(
                p_te + shifts[idx], self.EPS, 1.0 - self.EPS
            )
            score = float(np.mean((adjusted - y_te) ** 2))
            if score < best_score:
                best_score = score
                best_k = k
        return best_k

    def _require_fitted(self) -> None:
        if (
            self._centers is None
            or self._shifts is None
            or self.k is None
        ):
            raise RuntimeError("call train() before adjusting forecasts")

    def adjust_binary_forecast(self, prediction: float) -> float:
        self._require_fitted()
        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"prediction must be in [0, 1], got {prediction}"
            )
        assert self._centers is not None and self._shifts is not None
        idx = int(self._assign(prediction, self._centers)[0])
        return float(
            np.clip(
                prediction + self._shifts[idx],
                self.EPS,
                1.0 - self.EPS,
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
        assert self._centers is not None and self._shifts is not None
        arr = np.array(preds, dtype=np.float64)
        idx = self._assign(arr, self._centers)
        adjusted = np.clip(
            arr + self._shifts[idx], self.EPS, 1.0 - self.EPS
        )
        total = adjusted.sum()
        if total == 0:
            n = len(adjusted)
            return [1.0 / n] * n
        return (adjusted / total).tolist()
