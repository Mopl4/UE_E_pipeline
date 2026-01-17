from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


@dataclass(frozen=True)
class MetaConfig:
    C: float = 1.0
    max_iter: int = 2000


class MetaLogReg:
    name = "meta_logreg"
    kind = "meta"

    def __init__(self, config: MetaConfig = MetaConfig()) -> None:
        self.config = config
        self._model: LogisticRegression | None = None

    def fit(self, Z_oof: np.ndarray, y: np.ndarray) -> "MetaLogReg":
        self._model = LogisticRegression(
            solver="lbfgs",
            C=self.config.C,
            max_iter=self.config.max_iter,
            random_state=RANDOM_STATE,
        )
        self._model.fit(Z_oof, y)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Meta model not fitted.")
        return self._model.predict_proba(Z).astype(np.float32, copy=False)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(Z)
        return np.argmax(proba, axis=1).astype(np.int64, copy=False)

    @property
    def model(self) -> LogisticRegression:
        if self._model is None:
            raise RuntimeError("Meta model not fitted.")
        return self._model

