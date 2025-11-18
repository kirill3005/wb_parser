import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy import sparse

from common.features.types import FeaturePackage

ROOT = Path("artifacts/features")


def _d(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_key(params: dict, code_fingerprint: str, data_stamp: dict) -> str:
    payload = json.dumps(
        {"p": params, "c": code_fingerprint, "d": data_stamp},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def save_feature_pkg(block: str, key: str, pkg: FeaturePackage) -> None:
    base = _d(ROOT / block / key)
    if pkg.kind == "sparse":
        sparse.save_npz(base / "train.npz", pkg.train)
        sparse.save_npz(base / "test.npz", pkg.test)
    else:
        pkg.train.to_parquet(base / "train.parquet")
        pkg.test.to_parquet(base / "test.parquet")
    (base / "meta.json").write_text(json.dumps(pkg.meta, ensure_ascii=False, indent=2))


def load_feature_pkg(block: str, key: str) -> Optional[FeaturePackage]:
    base = ROOT / block / key
    if not base.exists():
        return None
    meta = json.loads((base / "meta.json").read_text())
    if (base / "train.parquet").exists():
        tr = pd.read_parquet(base / "train.parquet")
        te = pd.read_parquet(base / "test.parquet")
        kind = "dense"
        cols = list(tr.columns)
    else:
        tr = sparse.load_npz(base / "train.npz")
        te = sparse.load_npz(base / "test.npz")
        kind = "sparse"
        cols = []
    return FeaturePackage(name=meta["name"], train=tr, test=te, kind=kind, cols=cols, meta=meta)
