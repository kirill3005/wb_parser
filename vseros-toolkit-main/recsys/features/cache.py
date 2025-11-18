from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pickle

logger = logging.getLogger(__name__)


class FeatureCache:
    """Simple file-based cache for feature block artifacts."""

    def __init__(self, root: str = "artifacts/recsys/features_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fingerprint(params: Dict[str, Any]) -> str:
        payload = json.dumps(params, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def path(self, block_name: str, params: Dict[str, Any]) -> Path:
        fp = self._fingerprint(params)
        return self.root / block_name / fp / "artifact.pkl"

    def load(self, block_name: str, params: Dict[str, Any]) -> Optional[Any]:
        path = self.path(block_name, params)
        if path.exists():
            logger.info("Cache hit for %s", block_name)
            with open(path, "rb") as f:
                return pickle.load(f)
        logger.info("Cache miss for %s", block_name)
        return None

    def save(self, block_name: str, params: Dict[str, Any], obj: Any) -> None:
        path = self.path(block_name, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved cache for %s at %s", block_name, path)
