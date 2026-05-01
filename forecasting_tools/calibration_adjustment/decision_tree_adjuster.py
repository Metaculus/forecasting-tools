from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from forecasting_tools.calibration_adjustments._training_data import (
    extract_bernoulli_observations,
)
from forecasting_tools.calibration_adjustments.calibration_adjuster import (
    CalibrationAdjuster,
)

_TREE_UNDEFINED = -2


class DecisionTreeAdjuster(CalibrationAdjuster):
    """Adaptive-bin calibration adjuster.

    Fits a sklearn :class:`DecisionTreeRegressor` on
    ``(prediction -> outcome)`` pairs extracted from past forecasts. Each
    leaf of the trained tree acts as a bin: high-signal probability ranges
    end up split into many narrow bins, while low-signal ranges fall under
    one wide bin. This matches the intuition that a forecaster's history
    should support narrow corrections where there are many similar past
    forecasts, and only broad corrections where data is sparse.

    Each bin stores an additive adjustment (delta) equal to
    ``mean(outcome) - mean(prediction)`` for the training observations
    that fall in that bin. New forecasts are adjusted by adding the delta
    of their bin: ``adjusted = prediction + delta``, then clipped to
    ``[EPS, 1 - EPS]``. For multiple-choice forecasts, each option
    probability is adjusted independently and the result is renormalized
    to sum to 1.0.
    """

    EPS: float = 1e-4

    def __init__(
        self,
        *,
        min_samples_leaf: int = 30,
        max_leaf_nodes: int | None = 32,
        random_state: int | None = 0,
    ) -> None:
        """Construct an unfitted DecisionTreeAdjuster.

        Args:
            min_samples_leaf: Minimum number of training observations
                required in each leaf (bin). Lower values yield narrower
                bins where signal is dense, but risk overfitting in sparse
                regions.
            max_leaf_nodes: Maximum number of leaves (bins). ``None`` lets
                ``min_samples_leaf`` alone control complexity.
            random_state: Seed forwarded to the tree for deterministic
                tie-breaking.
        """
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self._tree: DecisionTreeRegressor | None = None
        self._train_predictions: np.ndarray | None = None
        self._leaf_deltas: dict[int, float] | None = None

    def train(self, forecasts: pd.DataFrame) -> None:
        """Fit the adjuster in place on the given forecasts DataFrame.

        Raises:
            ValueError: if no usable observations are found, or if there
                are fewer observations than ``min_samples_leaf``.
        """
        predictions, outcomes = extract_bernoulli_observations(forecasts)
        if len(predictions) < self.min_samples_leaf:
            raise ValueError(
                f"Need at least min_samples_leaf={self.min_samples_leaf} "
                f"observations to train, got {len(predictions)}"
            )

        tree = DecisionTreeRegressor(
            criterion="squared_error",
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
        )
        tree.fit(predictions.reshape(-1, 1), outcomes)

        # Compute per-leaf deltas: delta = mean(outcome) - mean(prediction)
        leaf_ids = tree.apply(predictions.reshape(-1, 1))
        leaf_deltas: dict[int, float] = {}
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            leaf_deltas[int(leaf_id)] = float(
                outcomes[mask].mean() - predictions[mask].mean()
            )

        self._tree = tree
        self._train_predictions = predictions
        self._leaf_deltas = leaf_deltas

    def _require_fitted(self) -> None:
        if (
            self._tree is None
            or self._train_predictions is None
            or self._leaf_deltas is None
        ):
            raise RuntimeError("call train() before adjusting forecasts")

    def adjust_binary_forecast(self, prediction: float) -> float:
        self._require_fitted()
        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"prediction must be in [0, 1], got {prediction}"
            )
        assert self._tree is not None and self._leaf_deltas is not None
        leaf_id = int(self._tree.apply(np.array([[prediction]]))[0])
        delta = self._leaf_deltas[leaf_id]
        adjusted = prediction + delta
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
        assert self._tree is not None and self._leaf_deltas is not None
        preds_arr = np.array(preds, dtype=np.float64)
        leaf_ids = self._tree.apply(preds_arr.reshape(-1, 1))
        deltas = np.array([self._leaf_deltas[int(lid)] for lid in leaf_ids])
        adjusted = preds_arr + deltas
        clipped = np.clip(adjusted, self.EPS, 1.0 - self.EPS)
        total = clipped.sum()
        if total == 0:
            n = len(clipped)
            return [1.0 / n] * n
        return (clipped / total).tolist()

    def bin_summary(self) -> list[dict[str, Any]]:
        """Return one dict per leaf bin, sorted by lower bound.

        Each dict contains:

        - ``lower``: lower bound of the bin (inclusive), clamped to ``0.0``
        - ``upper``: upper bound of the bin (exclusive), clamped to ``1.0``
        - ``width``: ``upper - lower``
        - ``n_samples``: number of training observations in the bin
        - ``mean_outcome``: empirical resolution rate in the bin
        - ``mean_prediction``: mean of the training predictions in the bin
        - ``adjustment``: the +/- delta applied to predictions falling in
          this bin (``mean_outcome - mean_prediction``)
        """
        self._require_fitted()
        assert (
            self._tree is not None
            and self._train_predictions is not None
            and self._leaf_deltas is not None
        )
        tree = self._tree.tree_
        leaves: list[dict[str, Any]] = []
        self._collect_leaves(
            tree=tree,
            node_id=0,
            lower=-math.inf,
            upper=math.inf,
            leaves=leaves,
        )

        # Compute mean_prediction per leaf by re-applying training data
        leaf_ids = self._tree.apply(self._train_predictions.reshape(-1, 1))
        leaf_id_to_mean_pred: dict[int, float] = {}
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            leaf_id_to_mean_pred[int(leaf_id)] = float(
                self._train_predictions[mask].mean()
            )

        for leaf in leaves:
            node_id = leaf["_node_id"]
            leaf["mean_prediction"] = leaf_id_to_mean_pred[node_id]
            leaf["adjustment"] = self._leaf_deltas[node_id]
            leaf["lower"] = max(0.0, leaf["lower"])
            leaf["upper"] = min(1.0, leaf["upper"])
            leaf["width"] = leaf["upper"] - leaf["lower"]
            del leaf["_node_id"]

        leaves.sort(key=lambda d: d["lower"])
        return leaves

    @staticmethod
    def _collect_leaves(
        tree: Any,
        node_id: int,
        lower: float,
        upper: float,
        leaves: list[dict[str, Any]],
    ) -> None:
        if tree.feature[node_id] == _TREE_UNDEFINED:
            leaves.append(
                {
                    "_node_id": node_id,
                    "lower": lower,
                    "upper": upper,
                    "n_samples": int(tree.n_node_samples[node_id]),
                    "mean_outcome": float(tree.value[node_id].flatten()[0]),
                }
            )
            return

        threshold = float(tree.threshold[node_id])
        DecisionTreeAdjuster._collect_leaves(
            tree=tree,
            node_id=int(tree.children_left[node_id]),
            lower=lower,
            upper=threshold,
            leaves=leaves,
        )
        DecisionTreeAdjuster._collect_leaves(
            tree=tree,
            node_id=int(tree.children_right[node_id]),
            lower=threshold,
            upper=upper,
            leaves=leaves,
        )
