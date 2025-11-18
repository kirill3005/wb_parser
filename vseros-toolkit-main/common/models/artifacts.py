from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path("artifacts/models")


def make_run_id(task: str, model: str, feat_hash: str, seed: int) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-{task}-{model}-{feat_hash}-{seed}"
    return base


def path(run_id: str) -> Path:
    p = ROOT / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}


def fold_path(run_dir: Path, k: int, ext: str) -> Path:
    return run_dir / f"model_fold_{k}{ext}"


def existing_folds(run_dir: Path, ext_candidates=(".lgb", ".cbm", ".xgb", ".joblib", ".pkl")) -> set[int]:
    out = set()
    for p in run_dir.glob("model_fold_*.*"):
        if p.suffix in ext_candidates:
            try:
                out.add(int(p.stem.split("_")[-1]))
            except Exception:
                pass
    return out


def save_array(run_dir: Path, name: str, arr) -> None:
    if name.endswith(".npy"):
        import numpy as np
        np.save(run_dir / name, arr)
    else:
        (run_dir / name).write_text(json.dumps(arr, ensure_ascii=False))
