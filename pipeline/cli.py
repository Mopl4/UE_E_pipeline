from __future__ import annotations

import argparse
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import artifacts
from pipeline.data import h5_num_rows, load_y_csv, validate_h5_matches_y
from pipeline.metrics import compute_basic_metrics
from pipeline.models.hgb import HGBConfig, HGBTabularModel
from pipeline.models.cnn import CNNConfig, CNNTemporalModel
from pipeline.models.lstm import LSTMConfig, LSTMTemporalModel
from pipeline.models.meta import MetaConfig, MetaLogReg
from pipeline.stacking_oof import run_oof
from pipeline.feature_engineering.dwt_combo import DWTComboConfig, featurize_h5_for_hgb
from pipeline.feature_engineering.h5_basic import load_h5_meta_features
from pipeline.optimize import OptConfig, sample_params, split_namespaced


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _default_run_dir() -> Path:
    return Path("runs") / _now_stamp()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pipeline", description="Reproducible stacking pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train OOF stacking and save a run.")
    train.add_argument("--y-csv", default="data/y_train_2.csv")
    train.add_argument("--x-h5", default="data/X_train.h5")
    train.add_argument("--h5-dataset-key", default="features")
    train.add_argument("--splits", type=int, default=3)
    train.add_argument(
        "--with-hgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activer/désactiver le base model HGB (par défaut: activé).",
    )
    train.add_argument(
        "--with-lstm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activer/désactiver le base model LSTM (par défaut: désactivé).",
    )
    train.add_argument(
        "--with-cnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activer/désactiver le base model CNN (par défaut: désactivé).",
    )

    train.add_argument("--meta-C", type=float, default=1.0)
    train.add_argument("--refit-final", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--run-dir", default=None)
    train.add_argument("--timing", action="store_true")
    train.add_argument("--hgb-fe", action="store_true", help="Compute HGB features from H5 using both DWT methods.")
    train.add_argument("--hgb-fe-chunk-size", type=int, default=4096)
    train.add_argument(
        "--hgb-meta-only",
        action="store_true",
        help="Si --hgb-fe n'est pas activé, entraîne HGB uniquement sur les 11 meta features (drop du signal brut).",
    )

    # Class imbalance handling (train-time only)
    train.add_argument(
        "--undersample-balanced",
        action="store_true",
        help="Undersample pour obtenir autant d'exemples par classe (0/1/2) avant tout entraînement.",
    )
    train.add_argument("--undersample-seed", type=int, default=42, help="Seed pour l'undersampling.")
    train.add_argument(
        "--class-weights-auto",
        action="store_true",
        help="Utilise des poids de classe automatiques (inverse fréquence) pendant l'entraînement (sans supprimer de données).",
    )

    # LSTM params
    train.add_argument("--lstm-fast", action="store_true")
    train.add_argument("--lstm-downsample", type=int, default=5)
    train.add_argument("--lstm-epochs", type=int, default=3)
    train.add_argument("--lstm-max-train-samples", type=int, default=50000)
    train.add_argument("--lstm-units", type=int, default=64)
    train.add_argument("--lstm-dense-units", type=int, default=32)
    train.add_argument("--lstm-batch-size", type=int, default=32)
    train.add_argument("--lstm-predict-batch-size", type=int, default=1024)
    train.add_argument("--lstm-verbose", type=int, default=1)
    train.add_argument("--lstm-load-model", default=None)

    # CNN params
    train.add_argument("--cnn-load-model", default=str(Path("stacking") / "cnn_sleep_model.keras"))
    train.add_argument("--cnn-load-only", action="store_true", help="Load CNN model and do not retrain per fold.")
    train.add_argument("--cnn-epochs", type=int, default=20)
    train.add_argument("--cnn-batch-size", type=int, default=64)
    train.add_argument("--cnn-downsample", type=int, default=1)
    train.add_argument("--cnn-max-train-samples", type=int, default=0)
    train.add_argument("--cnn-predict-batch-size", type=int, default=4096)
    train.add_argument("--cnn-verbose", type=int, default=1)

    # Optimize (random search, no extra deps)
    train.add_argument("--optimize", action="store_true", help="Active une random search (sans dépendances).")
    train.add_argument("--opt-trials", type=int, default=20)
    train.add_argument("--opt-seed", type=int, default=42)
    train.add_argument(
        "--opt-targets",
        nargs="+",
        default=["meta"],
        help="Quels modèles optimiser: meta hgb lstm cnn (ex: --opt-targets meta hgb).",
    )
    train.add_argument("--opt-budget-lstm-epochs", type=int, default=1)
    train.add_argument("--opt-budget-lstm-max-train-samples", type=int, default=20000)
    train.add_argument("--opt-budget-cnn-epochs", type=int, default=2)
    train.add_argument("--opt-budget-cnn-max-train-samples", type=int, default=20000)

    pred = sub.add_parser("predict", help="Predict using a saved run (no training).")
    pred.add_argument("--run-dir", required=True)
    pred.add_argument("--x-h5", default=None, help="H5 (si le run attend du H5).")
    pred.add_argument("--h5-dataset-key", default="features")
    pred.add_argument("--out", required=True)

    ev = sub.add_parser("evaluate", help="Evaluate a saved run without modifying it.")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--y-csv", default=None)
    ev.add_argument("--x-h5", default=None)
    ev.add_argument("--h5-dataset-key", default="features")

    an = sub.add_parser("analyze", help="Analyze saved predictions (OOF by default) from a run.")
    an.add_argument("--run-dir", required=True)
    an.add_argument(
        "--source",
        choices=["oof"],
        default="oof",
        help="Source de prédictions à analyser (pour l'instant: oof).",
    )
    an.add_argument(
        "--print-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche le classification report (precision/recall/f1).",
    )
    an.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sauvegarde un résumé JSON dans le run (ne touche pas aux modèles).",
    )

    return p.parse_args()


def _resolve_run_dir(arg: str | None) -> Path:
    if arg is None:
        return _default_run_dir()
    return Path(arg)


def _build_lstm_config(args: argparse.Namespace) -> LSTMConfig:
    if args.lstm_fast:
        return LSTMConfig(
            units=16,
            dense_units=8,
            epochs=1,
            batch_size=128,
            downsample=25,
            max_train_samples=5000,
            predict_batch_size=4096,
            verbose=args.lstm_verbose,
            load_model_path=args.lstm_load_model,
        )
    max_train = None if int(args.lstm_max_train_samples) == 0 else int(args.lstm_max_train_samples)
    return LSTMConfig(
        units=int(args.lstm_units),
        dense_units=int(args.lstm_dense_units),
        epochs=int(args.lstm_epochs),
        batch_size=int(args.lstm_batch_size),
        downsample=int(args.lstm_downsample),
        max_train_samples=max_train,
        predict_batch_size=int(args.lstm_predict_batch_size),
        verbose=int(args.lstm_verbose),
        load_model_path=args.lstm_load_model,
    )


def _build_lstm_config_for_opt(args: argparse.Namespace, overrides: dict[str, Any], opt: OptConfig) -> LSTMConfig:
    base = _build_lstm_config(args)
    max_train = None if int(opt.budget_lstm_max_train_samples) == 0 else int(opt.budget_lstm_max_train_samples)
    return LSTMConfig(
        units=int(overrides.get("units", base.units)),
        dense_units=int(overrides.get("dense_units", base.dense_units)),
        lr=float(overrides.get("lr", base.lr)),
        epochs=int(opt.budget_lstm_epochs),
        batch_size=int(overrides.get("batch_size", base.batch_size)),
        downsample=int(overrides.get("downsample", base.downsample)),
        max_train_samples=max_train,
        predict_batch_size=int(overrides.get("predict_batch_size", base.predict_batch_size)),
        verbose=int(base.verbose),
        tf_cpp_min_log_level=int(base.tf_cpp_min_log_level),
        load_model_path=base.load_model_path,
    )


def _build_cnn_config_for_opt(args: argparse.Namespace, overrides: dict[str, Any], opt: OptConfig) -> CNNConfig:
    max_train = None if int(opt.budget_cnn_max_train_samples) == 0 else int(opt.budget_cnn_max_train_samples)
    return CNNConfig(
        lr=float(overrides.get("lr", 1e-3)),
        epochs=int(opt.budget_cnn_epochs),
        batch_size=int(overrides.get("batch_size", int(args.cnn_batch_size))),
        downsample=int(overrides.get("downsample", int(args.cnn_downsample))),
        max_train_samples=max_train,
        predict_batch_size=int(overrides.get("predict_batch_size", int(args.cnn_predict_batch_size))),
        verbose=int(args.cnn_verbose),
        tf_cpp_min_log_level=int(args.lstm_tf_log_level),
        load_model_path=args.cnn_load_model if args.cnn_load_only else None,
    )


def _build_hgb_config_for_opt(overrides: dict[str, Any]) -> HGBConfig:
    return HGBConfig(
        max_depth=overrides.get("max_depth", 8),
        learning_rate=float(overrides.get("learning_rate", 0.03)),
        max_iter=int(overrides.get("max_iter", 300)),
        min_samples_leaf=int(overrides.get("min_samples_leaf", 10)),
        l2_regularization=float(overrides.get("l2_regularization", 5.0)),
    )


def _train(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(args.run_dir)
    artifacts.ensure_dir(run_dir)

    y_full = load_y_csv(y_csv=args.y_csv)
    validate_h5_matches_y(x_h5=args.x_h5, dataset_key=args.h5_dataset_key, y=y_full)
    if args.hgb_fe and not args.with_hgb:
        raise ValueError("--hgb-fe requires HGB enabled (remove --no-with-hgb or disable --hgb-fe).")
    if args.with_hgb and not args.hgb_fe and not args.hgb_meta_only:
        raise ValueError(
            "HGB requires either --hgb-fe (feature engineering) or --hgb-meta-only (drop signal, keep 11 meta)."
        )
    if args.undersample_balanced and args.class_weights_auto:
        raise ValueError("Choose only one imbalance option: --undersample-balanced OR --class-weights-auto.")

    kept_idx: np.ndarray | None = None
    y: np.ndarray
    if args.undersample_balanced:
        rng = np.random.default_rng(int(args.undersample_seed))
        idx_by_class = {c: np.where(y_full == c)[0] for c in np.unique(y_full)}
        if any(c not in idx_by_class for c in [0, 1, 2]):
            raise ValueError(f"Expected classes 0/1/2 in y, got: {sorted(idx_by_class.keys())}")
        n_target = min(len(idx_by_class[0]), len(idx_by_class[1]), len(idx_by_class[2]))
        kept = []
        for c in [0, 1, 2]:
            if len(idx_by_class[c]) < n_target:
                raise ValueError(f"Not enough samples for class {c}: {len(idx_by_class[c])} < {n_target}")
            kept.append(rng.choice(idx_by_class[c], size=n_target, replace=False))
        kept_idx = np.sort(np.concatenate(kept).astype(np.int64, copy=False))
        y = y_full[kept_idx]

        artifacts.save_npz(
            run_dir / "balance/undersample_balanced.npz",
            kept_idx=kept_idx,
            y_full=y_full.astype(np.int64, copy=False),
        )
        artifacts.write_json(
            run_dir / "balance/stats.json",
            {
                "method": "undersample_balanced",
                "seed": int(args.undersample_seed),
                "counts_before": {str(c): int((y_full == c).sum()) for c in [0, 1, 2]},
                "counts_after": {str(c): int((y == c).sum()) for c in [0, 1, 2]},
            },
        )
        print(f"[BALANCE] undersample_balanced -> n={y.shape[0]} per_class={int(n_target)}")
    else:
        y = y_full

    class_weight: dict[int, float] | None = None
    sample_weight: np.ndarray | None = None
    if args.class_weights_auto:
        counts = np.bincount(y.astype(np.int64), minlength=3)
        if np.any(counts == 0):
            raise ValueError(f"Cannot compute class weights: missing class in y (counts={counts.tolist()})")
        n = float(y.shape[0])
        class_weight = {c: float(n / (3.0 * float(counts[c]))) for c in [0, 1, 2]}
        sample_weight = np.vectorize(class_weight.get)(y).astype(np.float32, copy=False)
        artifacts.write_json(
            run_dir / "balance/class_weights_auto.json",
            {
                "method": "class_weights_auto",
                "counts": {str(c): int(counts[c]) for c in [0, 1, 2]},
                "class_weight": {str(k): float(v) for k, v in class_weight.items()},
            },
        )
        print(f"[BALANCE] class_weights_auto={class_weight}")

    base_models: list[Any] = []

    X_tab: np.ndarray | None = None
    if args.with_hgb and args.hgb_fe:
        cache_path = run_dir / "cache/hgb_features.npy"
        X_tab = featurize_h5_for_hgb(
            h5_path=args.x_h5,
            dataset_key=args.h5_dataset_key,
            chunk_size=int(args.hgb_fe_chunk_size),
            cfg=DWTComboConfig(),
            indices=kept_idx,
            out_npy=cache_path,
        )
    elif args.with_hgb and args.hgb_meta_only:
        cache_path = run_dir / "cache/hgb_meta.npy"
        X_tab = load_h5_meta_features(
            h5_path=args.x_h5,
            dataset_key=args.h5_dataset_key,
            indices=kept_idx,
            out_npy=cache_path,
        )

    if args.with_hgb:
        if X_tab is None:
            raise RuntimeError("Internal error: HGB enabled but X_tab was not constructed.")
        base_models.append(HGBTabularModel(X_tabular=X_tab, config=HGBConfig(), sample_weight=sample_weight))

    if args.with_lstm:
        base_models.append(
            LSTMTemporalModel(
                x_h5_path=args.x_h5,
                dataset_key=args.h5_dataset_key,
                config=_build_lstm_config(args),
                index_map=kept_idx,
                class_weight=class_weight,
            )
        )
    if args.with_cnn:
        if args.cnn_load_only:
            print(f"[WARN] CNN load-only may leak if it was trained on this same dataset: {args.cnn_load_model}")
        max_train = None if int(args.cnn_max_train_samples) == 0 else int(args.cnn_max_train_samples)
        base_models.append(
            CNNTemporalModel(
                x_h5_path=args.x_h5,
                dataset_key=args.h5_dataset_key,
                config=CNNConfig(
                    epochs=int(args.cnn_epochs),
                    batch_size=int(args.cnn_batch_size),
                    downsample=int(args.cnn_downsample),
                    max_train_samples=max_train,
                    predict_batch_size=int(args.cnn_predict_batch_size),
                    verbose=int(args.cnn_verbose),
                    load_model_path=args.cnn_load_model if args.cnn_load_only else None,
                ),
                index_map=kept_idx,
                class_weight=class_weight,
            )
        )

    if not base_models:
        raise ValueError("No base models selected. Enable at least one of: --with-hgb, --with-lstm, --with-cnn.")

    best_params: dict[str, Any] | None = None
    best_score: float | None = None

    if args.optimize:
        targets = tuple(str(t) for t in args.opt_targets)
        allowed = {"meta", "hgb", "lstm", "cnn"}
        if any(t not in allowed for t in targets):
            raise ValueError(f"--opt-targets must be subset of {sorted(allowed)}, got {targets}")
        if "hgb" in targets and not args.with_hgb:
            raise ValueError("Cannot optimize HGB when --no-with-hgb is set.")
        if "lstm" in targets and not args.with_lstm:
            raise ValueError("Cannot optimize LSTM without --with-lstm.")
        if "cnn" in targets and not args.with_cnn:
            raise ValueError("Cannot optimize CNN without --with-cnn.")
        if args.cnn_load_only and "cnn" in targets:
            raise ValueError("Optimizing CNN while --cnn-load-only is enabled does not make sense (no training).")
        if args.lstm_load_model is not None and "lstm" in targets:
            raise ValueError("Optimizing LSTM while --lstm-load-model is set does not make sense (no training).")

        opt_cfg = OptConfig(
            trials=int(args.opt_trials),
            seed=int(args.opt_seed),
            targets=targets,
            budget_lstm_epochs=int(args.opt_budget_lstm_epochs),
            budget_lstm_max_train_samples=int(args.opt_budget_lstm_max_train_samples),
            budget_cnn_epochs=int(args.opt_budget_cnn_epochs),
            budget_cnn_max_train_samples=int(args.opt_budget_cnn_max_train_samples),
        )
        artifacts.write_json(run_dir / "opt/config_used.json", opt_cfg.__dict__)

        rng = np.random.default_rng(opt_cfg.seed)
        best_score = -1.0
        best_params = {}

        print(f"[OPT] targets={list(opt_cfg.targets)} trials={opt_cfg.trials}")
        for t in range(opt_cfg.trials):
            sampled = sample_params(rng=rng, targets=opt_cfg.targets)
            ns = split_namespaced(sampled)

            trial_models: list[Any] = []
            if args.with_hgb:
                hgb_cfg = HGBConfig()
                if "hgb" in opt_cfg.targets:
                    hgb_cfg = _build_hgb_config_for_opt(ns.get("hgb", {}))
                trial_models.append(HGBTabularModel(X_tabular=X_tab, config=hgb_cfg))  # type: ignore[arg-type]

            if args.with_lstm:
                lstm_cfg = _build_lstm_config_for_opt(args, ns.get("lstm", {}), opt_cfg)
                trial_models.append(
                    LSTMTemporalModel(x_h5_path=args.x_h5, dataset_key=args.h5_dataset_key, config=lstm_cfg)
                )

            if args.with_cnn:
                cnn_cfg = _build_cnn_config_for_opt(args, ns.get("cnn", {}), opt_cfg)
                trial_models.append(
                    CNNTemporalModel(x_h5_path=args.x_h5, dataset_key=args.h5_dataset_key, config=cnn_cfg)
                )

            oof_trial = run_oof(base_models=trial_models, y=y, n_splits=args.splits, timing=False)

            meta_C = float(ns.get("meta", {}).get("C", args.meta_C))
            meta_trial = MetaLogReg(MetaConfig(C=meta_C)).fit(oof_trial.Z_oof, y)
            pred_trial = meta_trial.predict(oof_trial.Z_oof)
            score = float((pred_trial == y).mean())

            artifacts.append_jsonl(
                run_dir,
                "opt/trials.jsonl",
                {
                    "trial": t,
                    "targets": list(opt_cfg.targets),
                    "sampled": sampled,
                    "budget": {
                        "lstm_epochs": opt_cfg.budget_lstm_epochs,
                        "lstm_max_train_samples": opt_cfg.budget_lstm_max_train_samples,
                        "cnn_epochs": opt_cfg.budget_cnn_epochs,
                        "cnn_max_train_samples": opt_cfg.budget_cnn_max_train_samples,
                    },
                    "oof_accuracy": score,
                },
            )

            if score > best_score:
                best_score = score
                best_params = sampled
                print(f"[OPT] best so far: acc={best_score:.4f} params={best_params}")

        artifacts.write_json(run_dir / "opt/best_params.json", best_params)
        artifacts.write_json(run_dir / "opt/best_score.json", {"oof_accuracy": float(best_score)})

        # Apply best meta_C for the final run.
        best_ns = split_namespaced(best_params)
        if "meta" in best_ns and "C" in best_ns["meta"]:
            args.meta_C = float(best_ns["meta"]["C"])

        # If HGB is optimized, rebuild HGB model with best params for the final run.
        if args.with_hgb and "hgb" in best_ns:
            base_models = [m for m in base_models if getattr(m, "name", "") != "hgb_tabular"]
            base_models.insert(
                0,
                HGBTabularModel(X_tabular=X_tab, config=_build_hgb_config_for_opt(best_ns["hgb"])),  # type: ignore[arg-type]
            )

        # Note: LSTM/CNN optimization uses budget during trials; final refit uses the training args (epochs/caps)
        # while keeping the "best" meta_C and (if selected) HGB params.

    print(f"[TRAIN] base_models={[m.name for m in base_models]} splits={args.splits}")
    oof = run_oof(base_models=base_models, y=y, n_splits=args.splits, timing=args.timing)

    meta_cfg = MetaConfig(C=float(args.meta_C))

    meta = MetaLogReg(meta_cfg).fit(oof.Z_oof, y)
    meta_proba = meta.predict_proba(oof.Z_oof)
    meta_pred = np.argmax(meta_proba, axis=1).astype(np.int64, copy=False)

    metrics = compute_basic_metrics(y_true=y, y_pred=meta_pred)
    print(f"[OOF] accuracy={metrics['accuracy']:.4f}")

    if args.save:
        # Save OOF predictions and metrics
        artifacts.save_npz(
            run_dir / "oof/oof_predictions.npz",
            idx=np.arange(y.shape[0], dtype=np.int64),
            fold_id=oof.fold_id,
            y_true=y.astype(np.int64, copy=False),
            meta_pred=meta_pred,
            meta_proba=meta_proba,
            base_names=np.array(oof.base_model_order, dtype=object),
            **{f"base_{name}_proba": oof.oof_blocks[name] for name in oof.base_model_order},
        )
        artifacts.write_json(run_dir / "metrics/oof_metrics.json", metrics)

        # Save models
        paths: dict[str, str] = {
            "meta_model": "models/meta.joblib",
        }
        artifacts.save_joblib(run_dir / paths["meta_model"], meta.model)

        if args.refit_final:
            print("[REFIT] fitting base models on full train")
            for bm in base_models:
                if hasattr(bm, "fit_full"):
                    bm.fit_full(y)  # type: ignore[attr-defined]
                else:
                    bm.fit(np.arange(y.shape[0], dtype=np.int64), y)

            for bm in base_models:
                if bm.name == "hgb_tabular":
                    paths["base_hgb_tabular"] = "models/base_hgb_tabular.joblib"
                    artifacts.save_joblib(run_dir / paths["base_hgb_tabular"], bm.pipeline)  # type: ignore[attr-defined]
                elif bm.name == "lstm_temporal":
                    paths["base_lstm_temporal"] = "models/base_lstm_temporal.keras"
                    bm.save(run_dir / paths["base_lstm_temporal"])  # type: ignore[attr-defined]
                elif bm.name == "cnn_temporal":
                    paths["base_cnn_temporal"] = "models/base_cnn_temporal.keras"
                    bm.save(run_dir / paths["base_cnn_temporal"])  # type: ignore[attr-defined]
                else:
                    raise ValueError(f"Unknown base model for saving: {bm.name}")

        manifest = artifacts.RunManifest(
            schema_version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            run_dir=str(run_dir),
            base_model_order=oof.base_model_order,
            base_model_kinds=oof.base_model_kinds,
            paths=paths,
            config_used={
                "y_csv": str(args.y_csv),
                "x_h5": str(args.x_h5),
                "h5_dataset_key": str(args.h5_dataset_key),
                "splits": int(args.splits),
                "with_lstm": bool(args.with_lstm),
                "with_hgb": bool(args.with_hgb),
                "with_cnn": bool(args.with_cnn),
                "imbalance": {
                    "undersample_balanced": bool(args.undersample_balanced),
                    "undersample_seed": int(args.undersample_seed),
                    "class_weights_auto": bool(args.class_weights_auto),
                    "kept_idx_path": "balance/undersample_balanced.npz" if args.undersample_balanced else None,
                },
                "hgb": asdict(HGBConfig()),
                "hgb_feature_engineering": "dwt_combo_h5" if args.hgb_fe else "meta_only_h5",
                "hgb_fe_chunk_size": int(args.hgb_fe_chunk_size),
                "lstm": asdict(_build_lstm_config(args)) if args.with_lstm else None,
                "cnn": {
                    "epochs": int(args.cnn_epochs),
                    "batch_size": int(args.cnn_batch_size),
                    "downsample": int(args.cnn_downsample),
                    "max_train_samples": None
                    if int(args.cnn_max_train_samples) == 0
                    else int(args.cnn_max_train_samples),
                    "predict_batch_size": int(args.cnn_predict_batch_size),
                    "verbose": int(args.cnn_verbose),
                    "load_only": bool(args.cnn_load_only),
                }
                if args.with_cnn
                else None,
                "meta": asdict(meta_cfg),
                "refit_final": bool(args.refit_final),
            },
            best_params=best_params,
            best_score=best_score,
        )
        artifacts.save_manifest(run_dir, manifest)
        print(f"[SAVE] run_dir={run_dir}")


def _predict(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = artifacts.load_manifest(run_dir)
    base_order = manifest.base_model_order

    if args.x_h5 is None:
        raise ValueError("predict now requires --x-h5 (pipeline is H5-only).")

    n = h5_num_rows(x_h5=args.x_h5, dataset_key=args.h5_dataset_key)

    idx = np.arange(n, dtype=np.int64)

    # Build HGB input if needed.
    X_tab: np.ndarray | None = None
    if "hgb_tabular" in base_order:
        fe = manifest.config_used.get("hgb_feature_engineering")
        if fe == "dwt_combo_h5":
            X_tab = featurize_h5_for_hgb(
                h5_path=args.x_h5,  # type: ignore[arg-type]
                dataset_key=args.h5_dataset_key,
                chunk_size=int(manifest.config_used.get("hgb_fe_chunk_size", 4096)),
                cfg=DWTComboConfig(),
                out_npy=None,
            )
        elif fe == "meta_only_h5":
            X_tab = load_h5_meta_features(h5_path=args.x_h5, dataset_key=args.h5_dataset_key, indices=None)
        else:
            raise ValueError(f"Unknown HGB feature mode in manifest: {fe}")

    # Load base models (final/refit required for inference)
    base_blocks: list[np.ndarray] = []
    for name in base_order:
        if name == "hgb_tabular":
            if "base_hgb_tabular" not in manifest.paths:
                raise ValueError("Run missing final base model 'base_hgb_tabular' (train with --refit-final).")
            if X_tab is None:
                raise RuntimeError("Internal error: HGB requested but X_tab is None.")
            path = run_dir / manifest.paths["base_hgb_tabular"]
            pipe = artifacts.load_joblib(path)
            base_blocks.append(pipe.predict_proba(X_tab).astype(np.float32, copy=False))
        elif name == "lstm_temporal":
            if "base_lstm_temporal" not in manifest.paths:
                raise ValueError("Run missing final base model 'base_lstm_temporal' (train with --refit-final).")
            if args.x_h5 is None:
                raise ValueError("This run includes LSTM but --x-h5 was not provided.")
            model_path = run_dir / manifest.paths["base_lstm_temporal"]
            from tensorflow.keras.models import load_model

            m = load_model(str(model_path))

            import h5py

            def load_batch(batch_idx: np.ndarray, downsample: int) -> np.ndarray:
                batch_idx = np.asarray(batch_idx, dtype=np.int64)
                order = np.argsort(batch_idx)
                sorted_idx = batch_idx[order]
                with h5py.File(args.x_h5, "r") as f:
                    ds = f[args.h5_dataset_key]
                    x = ds[sorted_idx, 11:1261]
                x = np.asarray(x, dtype=np.float32)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.shape[0])
                x = x[inv]
                if downsample > 1:
                    x = x[:, ::downsample]
                x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
                return x

            downsample = int(manifest.config_used.get("lstm", {}).get("downsample", 1))
            pred_bs = int(manifest.config_used.get("lstm", {}).get("predict_batch_size", 1024))
            n_classes = int(getattr(m, "output_shape")[-1])
            proba = np.zeros((n, n_classes), dtype=np.float32)
            for start in range(0, n, pred_bs):
                end = min(start + pred_bs, n)
                xb = load_batch(idx[start:end], downsample)
                x = xb.reshape((xb.shape[0], xb.shape[1], 1))
                proba[start:end] = np.asarray(m.predict(x, verbose=0), dtype=np.float32)
            base_blocks.append(proba)
        elif name == "cnn_temporal":
            if "base_cnn_temporal" not in manifest.paths:
                raise ValueError("Run missing final base model 'base_cnn_temporal' (train with --refit-final).")
            if args.x_h5 is None:
                raise ValueError("This run includes CNN but --x-h5 was not provided.")
            model_path = run_dir / manifest.paths["base_cnn_temporal"]
            from tensorflow.keras.models import load_model

            m = load_model(str(model_path))
            import h5py

            def load_batch(batch_idx: np.ndarray, downsample: int) -> np.ndarray:
                batch_idx = np.asarray(batch_idx, dtype=np.int64)
                order = np.argsort(batch_idx)
                sorted_idx = batch_idx[order]
                with h5py.File(args.x_h5, "r") as f:
                    ds = f[args.h5_dataset_key]
                    x = ds[sorted_idx, 11:1261]
                x = np.asarray(x, dtype=np.float32)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.shape[0])
                x = x[inv]
                if downsample > 1:
                    x = x[:, ::downsample]
                x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
                return x

            downsample = int(manifest.config_used.get("cnn", {}).get("downsample", 1))
            pred_bs = int(manifest.config_used.get("cnn", {}).get("predict_batch_size", 4096))
            n_classes = int(getattr(m, "output_shape")[-1])
            proba = np.zeros((n, n_classes), dtype=np.float32)
            for start in range(0, n, pred_bs):
                end = min(start + pred_bs, n)
                xb = load_batch(idx[start:end], downsample)[..., np.newaxis]
                proba[start:end] = np.asarray(m.predict(xb, verbose=0), dtype=np.float32)
            base_blocks.append(proba)
        else:
            raise ValueError(f"Unknown base model in manifest: {name}")

    Z = np.concatenate(base_blocks, axis=1)
    meta = artifacts.load_joblib(run_dir / manifest.paths["meta_model"])
    meta_proba = meta.predict_proba(Z)
    pred = np.argmax(meta_proba, axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "pred"] + [f"proba_{i}" for i in range(meta_proba.shape[1])]
    out = np.concatenate([idx.reshape(-1, 1), pred.reshape(-1, 1), meta_proba], axis=1)
    import pandas as pd

    pd.DataFrame(out, columns=cols).to_csv(out_path, index=False)
    print(f"[PRED] wrote {out_path}")


def _evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = artifacts.load_manifest(run_dir)

    if args.y_csv is None and args.x_h5 is None:
        # default: use OOF saved outputs
        oof = artifacts.load_npz(run_dir / "oof/oof_predictions.npz")
        y_true = oof["y_true"]
        y_pred = oof["meta_pred"]
        metrics = compute_basic_metrics(y_true=y_true, y_pred=y_pred)
        print(f"[EVAL] OOF accuracy={metrics['accuracy']:.4f}")
        print(np.array(metrics["confusion_matrix"]))
        return

    if args.x_h5 is None or args.y_csv is None:
        raise ValueError("evaluate requires both --x-h5 and --y-csv, or neither (to use saved OOF).")

    # evaluate on a provided dataset using saved final models (no training)
    tmp_out = Path(run_dir) / f"_tmp_eval_{uuid.uuid4().hex}.csv"
    _predict(
        argparse.Namespace(
            run_dir=str(run_dir),
            x_h5=args.x_h5,
            h5_dataset_key=args.h5_dataset_key,
            out=str(tmp_out),
        )
    )
    import pandas as pd

    pred_df = pd.read_csv(tmp_out)
    y_df = pd.read_csv(args.y_csv)
    if "label" in y_df.columns:
        y_true = y_df["label"].to_numpy(dtype=np.int64)
    else:
        y_true = y_df.iloc[:, 1].to_numpy(dtype=np.int64)
    y_pred = pred_df["pred"].to_numpy(dtype=np.int64)
    metrics = compute_basic_metrics(y_true=y_true, y_pred=y_pred)
    print(f"[EVAL] accuracy={metrics['accuracy']:.4f}")
    print(np.array(metrics["confusion_matrix"]))
    tmp_out.unlink(missing_ok=True)


def _analyze(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if args.source != "oof":
        raise ValueError(f"Unsupported source: {args.source}")

    manifest = artifacts.load_manifest(run_dir)
    print(f"[RUN] base_model_order={manifest.base_model_order}")
    if manifest.best_params is not None:
        print(f"[RUN] best_score={manifest.best_score} best_params={manifest.best_params}")
    else:
        meta_cfg = manifest.config_used.get("meta", {})
        print(f"[RUN] meta={meta_cfg}")
    imbalance = manifest.config_used.get("imbalance", None)
    if imbalance is not None:
        print(f"[RUN] imbalance={imbalance}")

    oof = artifacts.load_npz(run_dir / "oof/oof_predictions.npz")
    y_true = oof["y_true"]
    y_pred = oof["meta_pred"]

    metrics = compute_basic_metrics(y_true=y_true, y_pred=y_pred)
    print(f"[ANALYZE] run_dir={run_dir}")
    print(f"accuracy={metrics['accuracy']:.4f} | balanced_accuracy={metrics['balanced_accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}")
    print("\n[CONFUSION_MATRIX] rows=true cols=pred (labels 0/1/2)")
    print(np.array(metrics["confusion_matrix"]))
    print("\n[CONFUSION_MATRIX_NORMALIZED] rows sum to 1")
    print(np.array(metrics["confusion_matrix_normalized"]))
    print("\n[PER_CLASS_ERROR_RATE]")
    for c, v in metrics["per_class_error_rate"].items():  # type: ignore[union-attr]
        print(f"class {c}: error_rate={float(v):.4f}")

    if args.print_report:
        print("\n[CLASSIFICATION_REPORT]")
        print(metrics["classification_report"])

    if args.save_json:
        out = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "confusion_matrix": metrics["confusion_matrix"],
            "confusion_matrix_normalized": metrics["confusion_matrix_normalized"],
            "per_class_error_rate": metrics["per_class_error_rate"],
            "classification_report_dict": metrics["classification_report_dict"],
        }
        artifacts.write_json(run_dir / "metrics/analyze_oof.json", out)
        print(f"[SAVE] metrics/analyze_oof.json")


def main() -> None:
    args = _parse_args()
    if args.cmd == "train":
        _train(args)
    elif args.cmd == "predict":
        _predict(args)
    elif args.cmd == "evaluate":
        _evaluate(args)
    elif args.cmd == "analyze":
        _analyze(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")
