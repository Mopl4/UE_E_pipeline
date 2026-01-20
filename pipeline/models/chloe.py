from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class ChloeConfig:
    # As defined in `lstm-cnn.py` (defaults preserved)
    eeg_len: int = 1250
    conv1_filters: int = 32
    conv1_kernel: int = 10
    conv1_stride: int = 2
    pool_size: int = 2
    conv2_filters: int = 64
    conv2_kernel: int = 5
    conv2_stride: int = 2
    eeg_lstm_units: int = 64

    meta_dim: int = 14  # 11 raw + 3 engineered freqs
    meta_dense_units: int = 32
    fusion_dense_units: int = 32
    n_classes: int = 3

    optimizer: str = "adam"
    lr: float | None = None  # None -> TF default for Adam

    epochs: int = 10
    batch_size: int = 32
    predict_batch_size: int = 2048
    verbose: int = 1
    tf_cpp_min_log_level: int = 2
    load_dir: str | None = None  # if set: load model+scalers and do not train per fold
    max_train_samples: int | None = None  # optional cap per fold


class ChloeModel:
    """
    Multi-input model (EEG branch Conv1D+LSTM, Meta branch MLP) based on `lstm-cnn.py`.

    Notes:
    - Meta engineering: adds current_freq, mean_past_freq, freq_delta to the 11 meta features.
    - Scaling: StandardScaler on meta features; StandardScaler on EEG values globally (flattened), then reshape to (N, 1250, 1).
    - During OOF, scalers are fit on the train fold only (no leakage).
    - For inference, scalers must be saved/loaded with the model.
    """

    name = "chloe_model"
    kind = "chloe"

    def __init__(
        self,
        *,
        x_h5_path: str,
        dataset_key: str = "features",
        config: ChloeConfig = ChloeConfig(),
        index_map: np.ndarray | None = None,
        class_weight: dict[int, float] | None = None,
    ) -> None:
        self.x_h5_path = str(x_h5_path)
        self.dataset_key = dataset_key
        self.config = config
        self.index_map = None if index_map is None else np.asarray(index_map, dtype=np.int64)
        self.class_weight = class_weight

        self._model = None
        self._scaler_meta: StandardScaler | None = None
        self._scaler_eeg: StandardScaler | None = None

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

    def _prepare_features(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Mirrors `prepare_features` from `lstm-cnn.py`.
        batch: (B, 1261)
        returns: (X_eeg_raw (B, 1250), X_meta_enriched (B, 14))
        """
        x = np.asarray(batch, dtype=np.float32)
        X_meta_raw = x[:, :11]
        X_eeg_raw = x[:, 11:1261]

        current_freq = 1.0 / (X_meta_raw[:, 4] + 1e-6)
        mean_past_freq = 1.0 / (X_meta_raw[:, 2] + 1e-6)
        freq_delta = np.abs(current_freq - mean_past_freq)

        X_meta_enriched = np.column_stack((X_meta_raw, current_freq, mean_past_freq, freq_delta)).astype(
            np.float32, copy=False
        )
        return X_eeg_raw, X_meta_enriched

    def _iter_h5_batches(self, idx: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        import h5py

        idx = np.asarray(idx, dtype=np.int64)
        if self.index_map is not None:
            idx = self.index_map[idx]
        order = np.argsort(idx)
        idx_sorted = idx[order]

        # Keep the H5 file open across batches: opening/closing per batch is a big slowdown.
        with h5py.File(self.x_h5_path, "r") as f:
            ds = f[self.dataset_key]
            for start in range(0, idx_sorted.shape[0], batch_size):
                end = min(start + batch_size, idx_sorted.shape[0])
                sub = idx_sorted[start:end]
                batch = np.asarray(ds[sub, :], dtype=np.float32)
                yield sub, batch

    def _fit_scalers(self, train_idx: np.ndarray) -> None:
        self._scaler_meta = StandardScaler()
        self._scaler_eeg = StandardScaler()

        # Fit with partial_fit to avoid loading all in RAM.
        for _, batch in self._iter_h5_batches(train_idx, batch_size=4096):
            eeg_raw, meta_enriched = self._prepare_features(batch)
            self._scaler_meta.partial_fit(meta_enriched)
            self._scaler_eeg.partial_fit(eeg_raw.reshape(-1, 1))

    def _transform_batch(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler_meta is None or self._scaler_eeg is None:
            raise RuntimeError("Scalers not fitted/loaded.")

        eeg_raw, meta_enriched = self._prepare_features(batch)

        X_meta = self._scaler_meta.transform(meta_enriched).astype(np.float32, copy=False)
        eeg_scaled = self._scaler_eeg.transform(eeg_raw.reshape(-1, 1)).reshape(eeg_raw.shape).astype(
            np.float32, copy=False
        )
        X_eeg = eeg_scaled.reshape((eeg_scaled.shape[0], eeg_scaled.shape[1], 1))
        return X_eeg, X_meta

    def _build(self):
        tf = self._ensure_tf()
        from tensorflow.keras import layers, models

        tf.keras.utils.set_random_seed(RANDOM_STATE)

        input_eeg = layers.Input(shape=(self.config.eeg_len, 1), name="eeg")
        x = layers.Conv1D(
            self.config.conv1_filters,
            self.config.conv1_kernel,
            strides=self.config.conv1_stride,
            activation="relu",
        )(input_eeg)
        x = layers.MaxPooling1D(self.config.pool_size)(x)
        x = layers.Conv1D(
            self.config.conv2_filters,
            self.config.conv2_kernel,
            strides=self.config.conv2_stride,
            activation="relu",
        )(x)
        x = layers.LSTM(self.config.eeg_lstm_units)(x)

        input_meta = layers.Input(shape=(self.config.meta_dim,), name="meta")
        y_m = layers.Dense(self.config.meta_dense_units, activation="relu")(input_meta)

        combined = layers.concatenate([x, y_m])
        z = layers.Dense(self.config.fusion_dense_units, activation="relu")(combined)
        output = layers.Dense(self.config.n_classes, activation="softmax")(z)

        model = models.Model(inputs=[input_eeg, input_meta], outputs=output)
        if self.config.optimizer == "adam":
            if self.config.lr is None:
                opt = tf.keras.optimizers.Adam()
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=float(self.config.lr))
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, train_idx: np.ndarray, y: np.ndarray) -> "ChloeModel":
        self._ensure_tf()

        if self.config.load_dir is not None:
            model, scaler_meta, scaler_eeg = ChloeModel.load(Path(self.config.load_dir))
            self._model = model
            self._scaler_meta = scaler_meta
            self._scaler_eeg = scaler_eeg
            return self

        y = np.asarray(y, dtype=np.int64)
        if self.config.max_train_samples is not None and train_idx.shape[0] > self.config.max_train_samples:
            rng = np.random.default_rng(RANDOM_STATE)
            train_idx = rng.choice(train_idx, size=int(self.config.max_train_samples), replace=False)

        self._fit_scalers(train_idx)
        self._model = self._build()

        tf = self._ensure_tf()

        class _Seq(tf.keras.utils.Sequence):
            def __init__(self, parent: "ChloeModel", indices: np.ndarray, y_full: np.ndarray):
                super().__init__()
                import h5py

                self.parent = parent
                self.indices = np.asarray(indices, dtype=np.int64)
                self.y_full = y_full
                self.batch_size = int(parent.config.batch_size)
                self.rng = np.random.default_rng(RANDOM_STATE)
                self.order = np.arange(self.indices.shape[0], dtype=np.int64)
                # Keep file open: opening/closing per __getitem__ is extremely slow.
                self._h5 = h5py.File(self.parent.x_h5_path, "r")
                self._ds = self._h5[self.parent.dataset_key]
                self.on_epoch_end()

            def close(self) -> None:
                try:
                    self._h5.close()
                except Exception:
                    pass

            def __del__(self) -> None:
                self.close()

            def __len__(self) -> int:
                return int(np.ceil(self.indices.shape[0] / self.batch_size))

            def __getitem__(self, i: int):
                s = i * self.batch_size
                e = min((i + 1) * self.batch_size, self.indices.shape[0])
                pos = self.order[s:e]
                batch_idx = self.indices[pos]
                # Load underlying H5 rows for this batch
                idx = batch_idx
                if self.parent.index_map is not None:
                    idx = self.parent.index_map[idx]
                order = np.argsort(idx)
                idx_sorted = idx[order]
                batch = np.asarray(self._ds[idx_sorted, :], dtype=np.float32)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.shape[0])
                batch = batch[inv]

                X_eeg, X_meta = self.parent._transform_batch(batch)
                yb = self.y_full[batch_idx]
                # IMPORTANT: return tuple inputs (not list) to avoid TF/Keras
                # `from_generator` signature issues on some versions.
                return (X_eeg, X_meta), yb

            def on_epoch_end(self) -> None:
                self.rng.shuffle(self.order)

        seq = _Seq(self, train_idx, y)
        t0 = perf_counter()
        try:
            self._model.fit(
                seq,
                epochs=int(self.config.epochs),
                verbose=int(self.config.verbose),
                class_weight=self.class_weight,
            )
        finally:
            seq.close()
        _ = perf_counter() - t0
        return self

    def fit_full(self, y: np.ndarray) -> "ChloeModel":
        idx = np.arange(y.shape[0], dtype=np.int64)
        return self.fit(idx, y)

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros((idx.shape[0], self.config.n_classes), dtype=np.float32)

        import h5py

        bs = int(self.config.predict_batch_size)
        # Keep the H5 file open across all batches for speed.
        with h5py.File(self.x_h5_path, "r") as f:
            ds = f[self.dataset_key]
            for start in range(0, idx.shape[0], bs):
                end = min(start + bs, idx.shape[0])
                batch_idx = idx[start:end]

                real_idx = batch_idx
                if self.index_map is not None:
                    real_idx = self.index_map[real_idx]
                order = np.argsort(real_idx)
                idx_sorted = real_idx[order]
                batch = np.asarray(ds[idx_sorted, :], dtype=np.float32)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.shape[0])
                batch = batch[inv]

                X_eeg, X_meta = self._transform_batch(batch)
                out[start:end] = np.asarray(self._model.predict((X_eeg, X_meta), verbose=0), dtype=np.float32)
        return out

    def clone(self) -> "ChloeModel":
        return ChloeModel(
            x_h5_path=self.x_h5_path,
            dataset_key=self.dataset_key,
            config=self.config,
            index_map=self.index_map,
            class_weight=self.class_weight,
        )

    def save(self, dir_path: Path) -> None:
        if self._model is None or self._scaler_meta is None or self._scaler_eeg is None:
            raise RuntimeError("Model/scalers not fitted.")
        dir_path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(dir_path / "model.keras"))
        import joblib

        joblib.dump(self._scaler_meta, dir_path / "scaler_meta.joblib")
        joblib.dump(self._scaler_eeg, dir_path / "scaler_eeg.joblib")

    @staticmethod
    def load(dir_path: Path) -> tuple[object, StandardScaler, StandardScaler]:
        from tensorflow.keras.models import load_model
        import joblib

        model = load_model(str(dir_path / "model.keras"))
        scaler_meta = joblib.load(dir_path / "scaler_meta.joblib")
        scaler_eeg = joblib.load(dir_path / "scaler_eeg.joblib")
        return model, scaler_meta, scaler_eeg
