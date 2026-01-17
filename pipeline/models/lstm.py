from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


RANDOM_STATE = 42


@dataclass(frozen=True)
class LSTMConfig:
    units: int = 64
    dense_units: int = 32
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 32
    downsample: int = 5
    max_train_samples: int | None = 50_000
    predict_batch_size: int = 1024
    verbose: int = 1
    tf_cpp_min_log_level: int = 2
    load_model_path: str | None = None


class LSTMTemporalModel:
    name = "lstm_temporal"
    kind = "lstm"

    def __init__(
        self,
        *,
        x_h5_path: str,
        dataset_key: str = "features",
        config: LSTMConfig = LSTMConfig(),
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

    def _load_time_series_batch(self, idx: np.ndarray) -> np.ndarray:
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
        from tensorflow.keras.layers import Dense, Input, LSTM
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        tf.keras.utils.set_random_seed(RANDOM_STATE)
        model = Sequential()
        model.add(Input(shape=(n_timesteps, 1)))
        model.add(LSTM(self.config.units))
        model.add(Dense(self.config.dense_units, activation="relu"))
        model.add(Dense(n_classes, activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=self.config.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, train_idx: np.ndarray, y: np.ndarray) -> "LSTMTemporalModel":
        self._ensure_tf()

        if self.config.load_model_path is not None:
            from tensorflow.keras.models import load_model

            self._model = load_model(self.config.load_model_path)
            out_shape = getattr(self._model, "output_shape", None)
            if out_shape is None or len(out_shape) != 2:
                raise ValueError(f"Unexpected loaded LSTM output_shape={out_shape}")
            self._n_classes = int(out_shape[-1])
            return self

        if self.config.max_train_samples is not None and train_idx.shape[0] > self.config.max_train_samples:
            rng = np.random.default_rng(RANDOM_STATE)
            train_idx = rng.choice(train_idx, size=self.config.max_train_samples, replace=False)

        x0 = self._load_time_series_batch(train_idx[: min(32, train_idx.shape[0])])
        n_timesteps = int(x0.shape[1])
        if n_timesteps <= 0:
            raise ValueError("Temporal signal length is zero after preprocessing.")

        n_classes = int(np.max(y) + 1)
        self._n_classes = n_classes
        self._model = self._build(n_timesteps=n_timesteps, n_classes=n_classes)

        tf = self._ensure_tf()

        class _Seq(tf.keras.utils.Sequence):
            def __init__(self, parent: "LSTMTemporalModel", indices: np.ndarray, y_full: np.ndarray):
                super().__init__()
                self.parent = parent
                self.indices = np.asarray(indices, dtype=np.int64)
                self.y_full = y_full
                self.batch_size = int(parent.config.batch_size)
                self.rng = np.random.default_rng(RANDOM_STATE)
                self.order = np.arange(self.indices.shape[0], dtype=np.int64)
                self.on_epoch_end()

            def __len__(self) -> int:
                return int(np.ceil(self.indices.shape[0] / self.batch_size))

            def __getitem__(self, i: int):
                s = i * self.batch_size
                e = min((i + 1) * self.batch_size, self.indices.shape[0])
                batch_pos = self.order[s:e]
                batch_idx = self.indices[batch_pos]
                x = self.parent._load_time_series_batch(batch_idx)
                x = x.reshape((x.shape[0], x.shape[1], 1))
                return x, self.y_full[batch_idx]

            def on_epoch_end(self) -> None:
                self.rng.shuffle(self.order)

        seq = _Seq(self, train_idx, y)
        t0 = perf_counter()
        self._model.fit(
            seq,
            epochs=self.config.epochs,
            validation_split=0.0,
            verbose=self.config.verbose,
            class_weight=self.class_weight,
        )
        _ = perf_counter() - t0
        return self

    def fit_full(self, y: np.ndarray) -> "LSTMTemporalModel":
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
            x = self._load_time_series_batch(batch_idx)
            x = x.reshape((x.shape[0], x.shape[1], 1))
            out[start:end] = np.asarray(self._model.predict(x, verbose=0), dtype=np.float32)
        return out

    def clone(self) -> "LSTMTemporalModel":
        return LSTMTemporalModel(
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
