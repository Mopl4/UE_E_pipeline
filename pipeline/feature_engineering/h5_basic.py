from __future__ import annotations

from pathlib import Path

import numpy as np


def load_h5_meta_features(
    *,
    h5_path: str | Path,
    dataset_key: str = "features",
    indices: np.ndarray | None = None,
    chunk_size: int = 8192,
    out_npy: str | Path | None = None,
) -> np.ndarray:
    """
    Load the 11 static/meta columns (0..10) from the H5 dataset.

    - If indices is provided, returns features in the order of sorted indices.
    - If out_npy is provided, writes a .npy memmap for caching.
    """
    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_key]
        n_samples, n_features = ds.shape
        if n_features != 1261:
            raise ValueError(f"Expected 1261 features, got {n_features}")

    if indices is not None:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if indices.size == 0:
            raise ValueError("indices is empty")
        if np.any(indices < 0) or np.any(indices >= n_samples):
            raise ValueError("indices out of bounds for this H5")
        indices = np.sort(indices)
        n_out = int(indices.shape[0])
    else:
        n_out = int(n_samples)

    out_path = Path(out_npy) if out_npy is not None else None
    X_out: np.ndarray
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        X_out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_out, 11))
    else:
        X_out = np.empty((n_out, 11), dtype=np.float32)

    try:
        from tqdm import tqdm

        it = tqdm(range(0, n_out, chunk_size), desc=f"Loading meta {h5_path.name}")
    except Exception:  # noqa: BLE001
        it = range(0, n_out, chunk_size)

    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_key]
        for start in it:
            end = min(start + chunk_size, n_out)
            if indices is None:
                batch = ds[start:end, :11]
            else:
                idx_batch = indices[start:end]
                batch = ds[idx_batch, :11]
            X_out[start:end] = np.asarray(batch, dtype=np.float32)

    if out_path is not None:
        return np.load(out_path, mmap_mode="r")
    return X_out

