from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .utils_time import ensure_utc

logger = logging.getLogger(__name__)


NEG_STRATEGIES = {"uniform", "pop", "hard"}


def _sample_negatives(
    positives: set,
    pool: np.ndarray,
    n_neg: int,
    rng: np.random.RandomState,
    strategy: str,
    pop_map: Optional[dict] = None,
) -> list:
    candidates = [item for item in pool if item not in positives]
    if not candidates:
        return []
    if strategy == "pop" and pop_map is not None:
        weights = np.array([pop_map.get(c, 1.0) for c in candidates], dtype=float)
        weights = weights / weights.sum()
        idx = rng.choice(
            len(candidates),
            size=min(n_neg, len(candidates)),
            replace=len(candidates) < n_neg,
            p=weights,
        )
        return [candidates[i] for i in idx]
    replace = len(candidates) < n_neg
    idx = rng.choice(len(candidates), size=min(n_neg, len(candidates)), replace=replace)
    return [candidates[i] for i in idx]


def build_pairs(
    interactions: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    neg_pos_ratio: int = 10,
    neg_strategy: Literal["uniform", "pop", "hard"] = "pop",
    items: Optional[pd.DataFrame] = None,
    rng: Optional[np.random.RandomState] = None,
    scope: str = "session",
) -> pd.DataFrame:
    rng = rng or np.random.RandomState(42)
    interactions = ensure_utc(interactions, cols=["ts"])
    queries = ensure_utc(queries, cols=["ts_query"]) if "ts_query" in queries.columns else queries

    item_pool = interactions["item_id"].unique()
    pop_map = None
    if neg_strategy == "pop":
        counts = interactions["item_id"].value_counts().to_dict()
        pop_map = counts

    pairs = []
    for row in queries.itertuples():
        qid = row.query_id
        ts_query = getattr(row, "ts_query", None)
        if scope == "session" and "session_id" in interactions.columns:
            hist = interactions[interactions["session_id"] == qid]
        else:
            hist = interactions[interactions["user_id"] == qid]
        if ts_query is not None:
            hist = hist[hist["ts"] <= ts_query]
        hist = hist.sort_values("ts")
        if hist.empty:
            continue
        pos_item = hist.iloc[-1]["item_id"]
        pairs.append((qid, pos_item, ts_query, 1))
        n_neg = max(1, int(neg_pos_ratio))
        negs = _sample_negatives({pos_item}, item_pool, n_neg, rng, neg_strategy, pop_map=pop_map)
        for neg in negs:
            pairs.append((qid, neg, ts_query, 0))
    return pd.DataFrame(pairs, columns=["query_id", "item_id", "ts_query", "label"])
