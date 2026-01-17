from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from sklearn.model_selection import StratifiedKFold


class IndexProbaModel(Protocol):
    name: str
    kind: str

    def fit(self, train_idx: np.ndarray, y: np.ndarray) -> "IndexProbaModel": ...

    def predict_proba(self, idx: np.ndarray) -> np.ndarray: ...

    def clone(self) -> "IndexProbaModel": ...


@dataclass(frozen=True)
class OOFResult:
    base_model_order: list[str]
    base_model_kinds: dict[str, str]
    oof_blocks: dict[str, np.ndarray]  # name -> (N, C)
    Z_oof: np.ndarray  # (N, C * n_models)
    fold_id: np.ndarray  # (N,)


def run_oof(
    *,
    base_models: list[IndexProbaModel],
    y: np.ndarray,
    n_splits: int,
    random_state: int = 42,
    timing: bool = False,
) -> OOFResult:
    y = np.asarray(y, dtype=np.int64)
    n_samples = y.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    base_model_order = [m.name for m in base_models]
    if len(set(base_model_order)) != len(base_model_order):
        raise ValueError(f"Base model names must be unique: {base_model_order}")

    base_model_kinds = {m.name: getattr(m, "kind", "unknown") for m in base_models}
    oof_blocks: dict[str, np.ndarray] = {
        m.name: np.zeros((n_samples, n_classes), dtype=np.float32) for m in base_models
    }
    fold_id = np.full(n_samples, -1, dtype=np.int16)

    dummy_X = np.zeros((n_samples, 1), dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(dummy_X, y), start=1):
        fold_id[va_idx] = fold
        print(f"[FOLD {fold}/{n_splits}] train={tr_idx.shape[0]} val={va_idx.shape[0]}")

        for model in base_models:
            print(f"  - {model.name}")
            t0 = perf_counter()
            fitted = model.clone().fit(tr_idx, y)
            t1 = perf_counter()
            proba = fitted.predict_proba(va_idx)
            t2 = perf_counter()

            if proba.shape != (va_idx.shape[0], n_classes):
                raise ValueError(
                    f"{model.name} predict_proba shape mismatch: got {proba.shape}, expected {(va_idx.shape[0], n_classes)}"
                )
            oof_blocks[model.name][va_idx] = proba.astype(np.float32, copy=False)

            if timing:
                print(f"    timing: fit {t1 - t0:.1f}s | predict {t2 - t1:.1f}s")

    Z_oof = np.concatenate([oof_blocks[name] for name in base_model_order], axis=1)
    return OOFResult(
        base_model_order=base_model_order,
        base_model_kinds=base_model_kinds,
        oof_blocks=oof_blocks,
        Z_oof=Z_oof,
        fold_id=fold_id,
    )

