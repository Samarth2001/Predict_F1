"""Ensemble model: combines base regressors with optimized non-negative weights."""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class EnsembleModel(BaseEstimator, RegressorMixin):
    """Ensemble of base regressors with non-negative weights optimized on OOF predictions."""

    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.weights_: Optional[np.ndarray] = None
        self.selected_feature_names_: Optional[List[str]] = None
        self.calibration_: Dict[str, Any] = {}

    def set_weights(self, weights: List[float]):
        self.weights_ = np.asarray(weights, dtype=float)

    def fit(self, X, y):
        for _, model in self.models.items():
            already_fit = bool(getattr(model, "fitted_", False)) or hasattr(
                model, "n_features_in_"
            )
            if already_fit:
                continue
            try:
                model.fit(X, y)
            except Exception as e:
                logger.debug(f"Skipping refit for model {type(model).__name__}: {e}")

        try:
            cols = list(X.columns)
            self.selected_feature_names_ = cols
            self.feature_names_in_ = cols
        except Exception:
            self.selected_feature_names_ = None

        if self.weights_ is None or (
            hasattr(self.weights_, "size")
            and int(getattr(self.weights_, "size", 0)) == 0
        ):
            num_models = max(1, len(self.models))
            self.weights_ = np.ones(num_models) / float(num_models)
        return self

    def predict(self, X):
        preds_matrix = np.column_stack(
            [model.predict(X) for model in self.models.values()]
        )

        # Per-segment weights (circuit type / year group) when available
        try:
            if (
                hasattr(self, "weights_by_segment_")
                and hasattr(self, "segment_keys_")
                and isinstance(X, pd.DataFrame)
                and all(k in X.columns for k in getattr(self, "segment_keys_", []))
            ):
                seg_cols: List[str] = list(getattr(self, "segment_keys_", []))
                seg_keys_df = X[seg_cols].astype(float).fillna(-1).astype(int)
                seg_tuples = list(map(tuple, seg_keys_df.values.tolist()))
                w_global = (
                    self.weights_
                    if self.weights_ is not None and len(self.weights_) == preds_matrix.shape[1]
                    else np.ones(preds_matrix.shape[1]) / preds_matrix.shape[1]
                )
                weight_map: Dict[str, List[float]] = getattr(self, "weights_by_segment_", {})
                weight_matrix = np.vstack(
                    [
                        np.asarray(weight_map.get(str(seg), w_global), dtype=float)
                        for seg in seg_tuples
                    ]
                )
                row_sums = weight_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                weight_matrix = weight_matrix / row_sums
                return np.sum(preds_matrix * weight_matrix, axis=1)
        except Exception:
            pass

        if self.weights_ is None or len(self.weights_) != preds_matrix.shape[1]:
            return np.mean(preds_matrix, axis=1)
        return preds_matrix.dot(self.weights_)

    def predict_components(self, X) -> np.ndarray:
        """Return base-model predictions matrix for uncertainty estimation."""
        return np.column_stack([model.predict(X) for model in self.models.values()])
