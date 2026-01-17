from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    created_at: str
    run_dir: str

    base_model_order: list[str]
    base_model_kinds: dict[str, str]  # name -> kind (hgb/lstm/cnn/...)

    paths: dict[str, str]  # logical -> relative path
    config_used: dict[str, Any]
    best_params: dict[str, Any] | None = None
    best_score: float | None = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(run_dir: Path, manifest: RunManifest) -> None:
    write_json(run_dir / "manifest.json", asdict(manifest))


def load_manifest(run_dir: Path) -> RunManifest:
    data = read_json(run_dir / "manifest.json")
    return RunManifest(**data)


def save_text(run_dir: Path, rel_path: str, text: str) -> None:
    path = run_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(run_dir: Path, rel_path: str, record: dict[str, Any]) -> None:
    path = run_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_joblib(path: Path, obj: Any) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Any:
    return np.load(path, allow_pickle=True)
