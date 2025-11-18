from __future__ import annotations

import json
from typing import Dict, List


def build_catalog(block_columns: Dict[str, List[str]], timings: Dict[str, float]) -> Dict:
    catalog = []
    for block, cols in block_columns.items():
        catalog.append({"block": block, "num_features": len(cols), "columns": cols, "time_sec": timings.get(block, 0.0)})
    return {"blocks": catalog}


def save_catalog(path: str, catalog: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
