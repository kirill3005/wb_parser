"""Lightweight similarity helpers for rerankers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def jaccard_category(item_ids, items: pd.DataFrame | None) -> dict:
    if items is None or "category" not in items.columns:
        return {}
    cat_map = items.set_index("item_id")["category"].to_dict()
    sim = {}
    for i in item_ids:
        for j in item_ids:
            if i == j:
                continue
            a = cat_map.get(i)
            b = cat_map.get(j)
            if a is None or b is None:
                val = 0.0
            else:
                val = 1.0 if a == b else 0.0
            sim[(i, j)] = val
    return sim


def cosine_from_vectors(vectors: dict) -> dict:
    sim = {}
    keys = list(vectors)
    for i, ki in enumerate(keys):
        vi = np.asarray(vectors[ki])
        vi_norm = np.linalg.norm(vi) + 1e-9
        for kj in keys[i + 1 :]:
            vj = np.asarray(vectors[kj])
            score = float(np.dot(vi, vj) / (vi_norm * (np.linalg.norm(vj) + 1e-9)))
            sim[(ki, kj)] = score
            sim[(kj, ki)] = score
    return sim


__all__ = ["jaccard_category", "cosine_from_vectors"]
