from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


_Stat = Literal["mean", "std", "abs_mean", "min", "max", "energy", "entropy"]


def _entropy_from_power(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    power = np.asarray(power, dtype=np.float32)
    s = power.sum(axis=1) + eps
    return np.log(s) - (power * np.log(power + eps)).sum(axis=1) / s


def _band_stats(band: np.ndarray, stats: Iterable[_Stat]) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    out: list[np.ndarray] = []
    for st in stats:
        if st == "mean":
            out.append(band.mean(axis=1))
        elif st == "std":
            out.append(band.std(axis=1))
        elif st == "abs_mean":
            out.append(np.abs(band).mean(axis=1))
        elif st == "min":
            out.append(band.min(axis=1))
        elif st == "max":
            out.append(band.max(axis=1))
        elif st == "energy":
            out.append((band * band).mean(axis=1))
        elif st == "entropy":
            power = band * band
            out.append(_entropy_from_power(power))
        else:
            raise ValueError(f"Unknown stat: {st}")
    return np.stack(out, axis=1).astype(np.float32, copy=False)


# --- Haar DWT (from dwt_features.py) ---
@dataclass(frozen=True)
class HaarDWTConfig:
    levels: int = 7
    stats: tuple[_Stat, ...] = ("mean", "std", "abs_mean", "min", "max", "energy", "entropy")


def haar_dwt_features(signals: np.ndarray, config: HaarDWTConfig = HaarDWTConfig()) -> np.ndarray:
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2D, got {x.shape}")
    if x.shape[1] < 2**config.levels:
        raise ValueError(f"Need at least {2**config.levels} points for {config.levels} levels")

    sqrt2 = np.float32(np.sqrt(2.0))
    feats: list[np.ndarray] = []
    a = x
    levels_done = 0
    for _ in range(config.levels):
        if a.shape[1] < 2:
            break
        if a.shape[1] % 2 == 1:
            a = np.pad(a, ((0, 0), (0, 1)), mode="edge")
        a_next = (a[:, 0::2] + a[:, 1::2]) / sqrt2
        d = (a[:, 0::2] - a[:, 1::2]) / sqrt2
        feats.append(_band_stats(d, config.stats))
        a = a_next
        levels_done += 1
    if levels_done != config.levels:
        raise RuntimeError(f"Unexpected early stop: did {levels_done} levels")
    feats.append(_band_stats(a, config.stats))
    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


# --- Wavelet features + window stats (from Ethan_P/feature_engineering/dwt_max.py) ---
@dataclass(frozen=True)
class WaveletFeatureConfig:
    wavelet: str = "db4"
    level: int = 5
    stats: tuple[_Stat, ...] = ("mean", "std", "abs_mean", "min", "max", "energy", "entropy")
    n_windows: int = 4
    window_stats: tuple[_Stat, ...] = ("energy", "std", "abs_mean")


def wavelet_features(signals: np.ndarray, cfg: WaveletFeatureConfig = WaveletFeatureConfig()) -> np.ndarray:
    import pywt

    x = np.asarray(signals, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2D, got {x.shape}")

    w = pywt.Wavelet(cfg.wavelet)
    max_level = pywt.dwt_max_level(x.shape[1], w.dec_len)
    L = min(cfg.level, max_level)
    if L < 1:
        raise ValueError(f"max_level={max_level} too small for wavelet={cfg.wavelet}")

    coeffs = pywt.wavedec(x, cfg.wavelet, level=L, axis=1)  # [cA_L, cD_L, ..., cD_1]
    feats = [_band_stats(coeffs[0], cfg.stats)]
    for cD in coeffs[1:]:
        feats.append(_band_stats(cD, cfg.stats))
    dwt_feat = np.concatenate(feats, axis=1)

    # window stats on raw signal
    B, N = x.shape
    splits = np.array_split(np.arange(N), cfg.n_windows)
    win_feats = []
    for idxs in splits:
        seg = x[:, idxs]
        win_feats.append(_band_stats(seg, cfg.window_stats))
    win_feat = np.concatenate(win_feats, axis=1)

    return np.concatenate([dwt_feat, win_feat], axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DWTComboConfig:
    haar: HaarDWTConfig = HaarDWTConfig()
    wavelet: WaveletFeatureConfig = WaveletFeatureConfig()


def dwt_combo_features(signal: np.ndarray, cfg: DWTComboConfig = DWTComboConfig()) -> np.ndarray:
    """
    signal: (B, 1250)
    returns: (B, 110) with default configs (56 Haar + 54 wavelet+windows)
    """
    f1 = haar_dwt_features(signal, cfg.haar)
    f2 = wavelet_features(signal, cfg.wavelet)
    return np.concatenate([f1, f2], axis=1).astype(np.float32, copy=False)


def featurize_h5_for_hgb(
    *,
    h5_path: str | Path,
    dataset_key: str = "features",
    chunk_size: int = 4096,
    cfg: DWTComboConfig = DWTComboConfig(),
    indices: np.ndarray | None = None,
    out_npy: str | Path | None = None,
) -> np.ndarray:
    """
    Builds tabular features for boosting:
      - first 11 static features
      - + both DWT feature sets on the EEG signal (1250 points)

    Returns an array (N, 121) float32. If out_npy is provided, writes a `.npy` and loads it via memmap.
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
        # Keep deterministic order for downstream indexing (no silent reorder).
        # Caller may pass sorted indices; we sort to guarantee h5py-friendly reads.
        indices = np.sort(indices)
        n_out = int(indices.shape[0])
    else:
        n_out = int(n_samples)

    out_path = Path(out_npy) if out_npy is not None else None
    X_out: np.ndarray
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        X_out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_out, 121))
    else:
        X_out = np.empty((n_out, 121), dtype=np.float32)

    try:
        from tqdm import tqdm

        it = tqdm(range(0, n_out, chunk_size), desc=f"Featurizing {h5_path.name}")
    except Exception:  # noqa: BLE001
        it = range(0, n_out, chunk_size)

    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_key]
        for start in it:
            end = min(start + chunk_size, n_out)
            if indices is None:
                batch = ds[start:end, :]
            else:
                idx_batch = indices[start:end]
                batch = ds[idx_batch, :]
            static = batch[:, :11].astype(np.float32, copy=False)
            signal = batch[:, 11:1261].astype(np.float32, copy=False)
            feats = dwt_combo_features(signal, cfg)
            X_out[start:end, :] = np.concatenate([static, feats], axis=1)

    if out_path is not None:
        # Return a read-only memmap view
        return np.load(out_path, mmap_mode="r")
    return X_out
