from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class OptConfig:
    trials: int = 20
    seed: int = 42
    targets: tuple[str, ...] = ("meta",)

    # Budget used during optimization (keeps trials tractable)
    budget_lstm_epochs: int = 1
    budget_lstm_max_train_samples: int = 20000
    budget_cnn_epochs: int = 2
    budget_cnn_max_train_samples: int = 20000
    budget_chloe_epochs: int = 2
    budget_chloe_max_train_samples: int = 20000


def _choice(rng: np.random.Generator, vals: list[Any]) -> Any:
    return vals[int(rng.integers(0, len(vals)))]


def sample_params(
    *,
    rng: np.random.Generator,
    targets: Iterable[str],
) -> dict[str, Any]:
    """
    Random search without extra dependencies.
    Returns a flat dict of sampled parameters (namespaced by prefix).
    """
    tset = set(targets)
    out: dict[str, Any] = {}

    if "meta" in tset:
        out["meta.C"] = float(10 ** rng.uniform(-2, 2))  # 1e-2 .. 1e2

    if "hgb" in tset:
        out["hgb.max_depth"] = _choice(rng, [6, 8, 10, None])
        out["hgb.learning_rate"] = float(_choice(rng, [0.02, 0.03, 0.05, 0.08, 0.1]))
        out["hgb.max_iter"] = int(_choice(rng, [200, 300, 400, 600]))
        out["hgb.min_samples_leaf"] = int(_choice(rng, [10, 20, 50, 100]))
        out["hgb.l2_regularization"] = float(_choice(rng, [0.0, 0.1, 1.0, 5.0]))

    if "lstm" in tset:
        out["lstm.units"] = int(_choice(rng, [16, 32, 64]))
        out["lstm.dense_units"] = int(_choice(rng, [8, 16, 32]))
        out["lstm.lr"] = float(_choice(rng, [1e-3, 3e-4]))
        out["lstm.downsample"] = int(_choice(rng, [5, 10, 25]))
        out["lstm.batch_size"] = int(_choice(rng, [64, 128]))
        out["lstm.predict_batch_size"] = int(_choice(rng, [1024, 2048, 4096]))

    if "cnn" in tset:
        out["cnn.lr"] = float(_choice(rng, [1e-3, 3e-4]))
        out["cnn.downsample"] = int(_choice(rng, [1, 2, 5]))
        out["cnn.batch_size"] = int(_choice(rng, [32, 64, 128]))
        out["cnn.predict_batch_size"] = int(_choice(rng, [2048, 4096, 8192]))

    if "chloe" in tset:
        out["chloe.conv1_filters"] = int(_choice(rng, [16, 32, 48]))
        out["chloe.conv2_filters"] = int(_choice(rng, [32, 64, 96]))
        out["chloe.eeg_lstm_units"] = int(_choice(rng, [32, 64, 96]))
        out["chloe.meta_dense_units"] = int(_choice(rng, [16, 32, 64]))
        out["chloe.fusion_dense_units"] = int(_choice(rng, [16, 32, 64]))
        out["chloe.lr"] = float(_choice(rng, [1e-3, 3e-4]))
        out["chloe.batch_size"] = int(_choice(rng, [32, 64]))
        out["chloe.predict_batch_size"] = int(_choice(rng, [1024, 2048, 4096]))

    return out


def split_namespaced(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    {"hgb.max_depth": 8, "meta.C": 1.0} -> {"hgb": {"max_depth": 8}, "meta": {"C": 1.0}}
    """
    out: dict[str, dict[str, Any]] = {}
    for k, v in params.items():
        if "." not in k:
            raise ValueError(f"Expected namespaced key like 'hgb.max_depth', got: {k}")
        ns, key = k.split(".", 1)
        out.setdefault(ns, {})[key] = v
    return out
