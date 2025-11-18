from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib

logger = logging.getLogger(__name__)


class CacheManager:
    """Very small helper to store fitted generator artifacts."""

    def __init__(self, base_dir: str = "artifacts/recsys/candidates_cache") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def fingerprint(self, name: str, params: Dict[str, Any], cutoff_ts) -> str:
        payload = json.dumps({"name": name, "params": params, "cutoff": str(cutoff_ts)}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def load(self, name: str, params: Dict[str, Any], cutoff_ts):
        fp = self.fingerprint(name, params, cutoff_ts)
        path = self.base_dir / name / fp / "artifact.pkl"
        if path.exists():
            logger.info("Cache hit for %s", name)
            return joblib.load(path)
        logger.info("Cache miss for %s", name)
        return None

    def save(self, name: str, params: Dict[str, Any], cutoff_ts, obj) -> None:
        fp = self.fingerprint(name, params, cutoff_ts)
        path = self.base_dir / name / fp
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path / "artifact.pkl")
