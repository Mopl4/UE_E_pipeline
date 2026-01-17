from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


@dataclass(frozen=True)
class HGBConfig:
    max_depth: int | None = 8
    learning_rate: float = 0.03
    max_iter: int = 300
    min_samples_leaf: int = 10
    l2_regularization: float = 5.0


class HGBTabularModel:
    name = "hgb_tabular"
    kind = "hgb"

    def __init__(
        self,
        *,
        X_tabular: np.ndarray,
        config: HGBConfig = HGBConfig(),
        sample_weight: np.ndarray | None = None,
    ):
        self.X_tabular = np.asarray(X_tabular, dtype=np.float32)
        self.config = config
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        self._pipe: Pipeline | None = None

    def _build(self) -> Pipeline:
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        )
        return Pipeline(steps=[("vt", VarianceThreshold(threshold=0.0)), ("clf", clf)])

    def fit(self, train_idx: np.ndarray, y: np.ndarray) -> "HGBTabularModel":
        self._pipe = self._build()
        if self.sample_weight is None:
            self._pipe.fit(self.X_tabular[train_idx], y[train_idx])
        else:
            sw = self.sample_weight[train_idx]
            self._pipe.fit(self.X_tabular[train_idx], y[train_idx], clf__sample_weight=sw)
        return self

    def fit_full(self, y: np.ndarray) -> "HGBTabularModel":
        self._pipe = self._build()
        if self.sample_weight is None:
            self._pipe.fit(self.X_tabular, y)
        else:
            self._pipe.fit(self.X_tabular, y, clf__sample_weight=self.sample_weight)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Model not fitted.")
        return self._pipe.predict_proba(self.X_tabular[idx]).astype(np.float32, copy=False)

    def predict_proba_full(self) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Model not fitted.")
        return self._pipe.predict_proba(self.X_tabular).astype(np.float32, copy=False)

    def clone(self) -> "HGBTabularModel":
        return HGBTabularModel(X_tabular=self.X_tabular, config=self.config, sample_weight=self.sample_weight)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipe is None:
            raise RuntimeError("Model not fitted.")
        return self._pipe
