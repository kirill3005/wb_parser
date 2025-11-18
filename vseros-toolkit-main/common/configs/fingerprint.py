from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _prune(obj: Any) -> Any:
    """Remove non-influential keys recursively."""
    if isinstance(obj, Mapping):
        pruned: dict[str, Any] = {}
        for k, v in obj.items():
            if k in {"paths", "logs", "run_tag"}:
                continue
            pruned[k] = _prune(v)
        return pruned
    if isinstance(obj, list):
        return [_prune(v) for v in obj]
    return obj


def compute_fingerprint(config: Mapping[str, Any]) -> str:
    """Compute a stable fingerprint for influential config parts."""
    payload = _prune(config)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
