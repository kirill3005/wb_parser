"""Serialization helpers for ModelRun."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recsys.rankers.base import ModelRun


def save_model_run(run: ModelRun, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run.run_id,
        "backend": run.backend,
        "task": run.task,
        "cv_mean": run.cv_mean,
        "cv_std": run.cv_std,
        "features": run.features,
        "meta": run.meta,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    np.save(out_dir / "oof_true.npy", run.oof_true)
    np.save(out_dir / "oof_pred.npy", run.oof_pred)
    if run.test_pred.size:
        np.save(out_dir / "test_pred.npy", run.test_pred)
    if run.groups:
        with (out_dir / "groups.json").open("w", encoding="utf-8") as f:
            json.dump(run.groups, f)
    return out_dir


def load_model_run(path: Path) -> ModelRun:
    with (path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    oof_true = np.load(path / "oof_true.npy")
    oof_pred = np.load(path / "oof_pred.npy")
    test_pred = np.load(path / "test_pred.npy") if (path / "test_pred.npy").exists() else np.array([])
    groups = None
    if (path / "groups.json").exists():
        with (path / "groups.json").open("r", encoding="utf-8") as f:
            groups = json.load(f)
    return ModelRun(
        run_id=meta.get("run_id", path.name),
        backend=meta.get("backend", "unknown"),
        task=meta.get("task", "rank"),
        cv_mean=meta.get("cv_mean", 0.0),
        cv_std=meta.get("cv_std", 0.0),
        oof_true=oof_true,
        oof_pred=oof_pred,
        test_pred=test_pred,
        groups=groups,
        features=meta.get("features", []),
        artifacts_path=path,
        meta=meta.get("meta", {}),
    )


__all__ = ["save_model_run", "load_model_run"]
