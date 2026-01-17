from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


RANDOM_STATE = 42


@dataclass(frozen=True)
class CNNConfig:
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    downsample: int = 1
    max_train_samples: int | None = None
    predict_batch_size: int = 4096
    verbose: int = 1
    tf_cpp_min_log_level: int = 2
    load_model_path: str | None = None  # if set: load and do not train


class CNNTemporalModel:
    """
    1D CNN defined to match `stacking/Projet_deep_sleep.ipynb`:
      Input(T,1) -> Conv1D(32,7) -> MaxPool(2) -> Conv1D(64,5) -> MaxPool(2)
      -> Conv1D(128,3) -> GlobalAvgPool -> Dense(64) -> Dropout(0.3) -> Dense(n_classes, softmax)
    """

    name = "cnn_temporal"
    kind = "cnn"

    def __init__(
        self,
        *,
        x_h5_path: str,
        dataset_key: str = "features",
        config: CNNConfig = CNNConfig(),
        index_map: np.ndarray | None = None,
        class_weight: dict[int, float] | None = None,
    ) -> None:
        self.x_h5_path = str(x_h5_path)
        self.dataset_key = dataset_key
        self.config = config
        self.index_map = None if index_map is None else np.asarray(index_map, dtype=np.int64)
        self.class_weight = class_weight
        self._model = None
        self._n_classes: int | None = None

    def _ensure_tf(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(int(self.config.tf_cpp_min_log_level))
        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
        return tf

    def _load_signal_batch(self, idx: np.ndarray) -> np.ndarray:
        import h5py

        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if self.index_map is not None:
            idx = self.index_map[idx]
        order = np.argsort(idx)
        idx_sorted = idx[order]
        with h5py.File(self.x_h5_path, "r") as f:
            ds = f[self.dataset_key]
            x = ds[idx_sorted, 11:1261]
        x = np.asarray(x, dtype=np.float32)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.shape[0])
        x = x[inv]
        if self.config.downsample > 1:
            x = x[:, :: self.config.downsample]
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        return x

    def _build(self, *, n_timesteps: int, n_classes: int):
        tf = self._ensure_tf()
        from tensorflow.keras import layers, models

        tf.keras.utils.set_random_seed(RANDOM_STATE)
        model = models.Sequential(
            [
                layers.Input(shape=(n_timesteps, 1)),
                layers.Conv1D(32, 7, padding="same", activation="relu"),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 5, padding="same", activation="relu"),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, 3, padding="same", activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(n_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, train_idx: np.ndarray, y: np.ndarray) -> "CNNTemporalModel":
        self._ensure_tf()

        if self.config.load_model_path is not None:
            from tensorflow.keras.models import load_model

            self._model = load_model(self.config.load_model_path)
            out_shape = getattr(self._model, "output_shape", None)
            if out_shape is None or len(out_shape) != 2:
                raise ValueError(f"Unexpected loaded CNN output_shape={out_shape}")
            self._n_classes = int(out_shape[-1])
            return self

        if self.config.max_train_samples is not None and train_idx.shape[0] > self.config.max_train_samples:
            rng = np.random.default_rng(RANDOM_STATE)
            train_idx = rng.choice(train_idx, size=self.config.max_train_samples, replace=False)

        x0 = self._load_signal_batch(train_idx[: min(32, train_idx.shape[0])])
        n_timesteps = int(x0.shape[1])
        if n_timesteps <= 0:
            raise ValueError("Temporal signal length is zero after preprocessing.")

        n_classes = int(np.max(y) + 1)
        self._n_classes = n_classes
        self._model = self._build(n_timesteps=n_timesteps, n_classes=n_classes)

        x = self._load_signal_batch(train_idx)[..., np.newaxis]

        t0 = perf_counter()
        self._model.fit(
            x,
            y[train_idx],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose,
            shuffle=True,
            class_weight=self.class_weight,
        )
        _ = perf_counter() - t0
        return self

    def fit_full(self, y: np.ndarray) -> "CNNTemporalModel":
        idx = np.arange(y.shape[0], dtype=np.int64)
        return self.fit(idx, y)

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        if self._n_classes is None:
            out_shape = getattr(self._model, "output_shape", None)
            self._n_classes = int(out_shape[-1]) if out_shape else 3

        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros((idx.shape[0], self._n_classes), dtype=np.float32)
        bs = int(self.config.predict_batch_size)
        for start in range(0, idx.shape[0], bs):
            end = min(start + bs, idx.shape[0])
            batch_idx = idx[start:end]
            x = self._load_signal_batch(batch_idx)[..., np.newaxis]
            out[start:end] = np.asarray(self._model.predict(x, verbose=0), dtype=np.float32)
        return out

    def clone(self) -> "CNNTemporalModel":
        return CNNTemporalModel(
            x_h5_path=self.x_h5_path,
            dataset_key=self.dataset_key,
            config=self.config,
            index_map=self.index_map,
            class_weight=self.class_weight,
        )

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
