"""Maximal Marginal Relevance reranker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.rerank.similarity import jaccard_category


def _similarity(item_a, item_b, sim_cache):
    if not sim_cache:
        return 0.0
    return sim_cache.get((item_a, item_b), 0.0)


def mmr(
    df: pd.DataFrame,
    items: pd.DataFrame | None,
    *,
    K: int,
    lambda_div: float,
    sim_backend: str = "cosine",
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.RandomState(0)
    rows = []
    for query_id, group in df.groupby("query_id"):
        group_sorted = group.sort_values("score_ranker", ascending=False)
        candidates = list(group_sorted.itertuples())
        selected: list[pd.Series] = []
        sim_cache = jaccard_category(group_sorted["item_id"], items) if sim_backend == "jaccard" else {}
        while candidates and len(selected) < K:
            best_idx = None
            best_score = -np.inf
            for idx, cand in enumerate(candidates):
                if not selected:
                    score = cand.score_ranker
                else:
                    sim_penalty = max(_similarity(cand.item_id, s.item_id, sim_cache) for s in selected)
                    score = lambda_div * cand.score_ranker - (1 - lambda_div) * sim_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            chosen = candidates.pop(best_idx)
            selected.append(chosen)
        for rank, cand in enumerate(selected, start=1):
            rows.append({"query_id": query_id, "item_id": cand.item_id, "score_final": cand.score_ranker, "rank_final": rank})
    return pd.DataFrame(rows)


__all__ = ["mmr"]
