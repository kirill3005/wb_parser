"""Score blending helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def blend_scores(df: pd.DataFrame, sources: list[str], mode: str = "equal", weights: dict | None = None) -> pd.Series:
    weights = weights or {}
    cols = [f"score_{s}" for s in sources]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing score columns: {missing}")
    mat = df[cols].to_numpy(dtype=float)
    if mode == "equal" or not weights:
        w = np.ones(len(cols)) / len(cols)
    else:
        w = np.array([weights.get(s, 1.0) for s in sources], dtype=float)
        w = w / (w.sum() + 1e-9)
    return pd.Series(mat.dot(w), index=df.index)


__all__ = ["blend_scores"]
