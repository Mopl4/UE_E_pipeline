from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TabularDataset:
    X: np.ndarray  # (N, D)
    y: np.ndarray | None  # (N,)
    feature_names: list[str]


def load_tabular_csv(
    *,
    x_csv: str | Path,
    y_csv: str | Path | None = None,
    drop_cols: list[str] | None = None,
) -> TabularDataset:
    x_csv = Path(x_csv)
    drop_cols = drop_cols or []

    X_df = pd.read_csv(x_csv).drop(columns=drop_cols, errors="ignore")
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    y: np.ndarray | None = None
    if y_csv is not None:
        y_df = pd.read_csv(y_csv)
        # same convention as existing scripts: label is column 2 or "label"
        if "label" in y_df.columns:
            y = y_df["label"].to_numpy(dtype=np.int64)
        else:
            y = y_df.iloc[:, 1].to_numpy(dtype=np.int64)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y length mismatch: X={X.shape[0]} y={y.shape[0]}")

    return TabularDataset(X=X, y=y, feature_names=feature_names)


def load_y_csv(*, y_csv: str | Path) -> np.ndarray:
    y_df = pd.read_csv(y_csv)
    if "label" in y_df.columns:
        return y_df["label"].to_numpy(dtype=np.int64)
    return y_df.iloc[:, 1].to_numpy(dtype=np.int64)


def validate_h5_matches_y(*, x_h5: str | Path, dataset_key: str, y: np.ndarray) -> None:
    import h5py

    x_h5 = Path(x_h5)
    with h5py.File(x_h5, "r") as f:
        ds = f[dataset_key]
        if ds.shape[0] != y.shape[0]:
            raise ValueError(
                f"H5/y length mismatch: h5={ds.shape[0]} y={y.shape[0]} ({x_h5}:{dataset_key})"
            )
