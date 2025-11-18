"""Adapter to reuse recsys.eval metrics in ranker training."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from recsys.eval import metrics


def evaluate_rank_lists(
    y_true_groups: List[list[int]],
    y_pred_groups: List[list[int]],
    *,
    k: int,
) -> dict:
    """Compute recall@k and ndcg@k for grouped data.

    Parameters
    ----------
    y_true_groups: list of lists
        Each inner list contains positive item_ids for a query.
    y_pred_groups: list of lists
        Each inner list contains predicted item_ids ordered by score.
    k: int
        Cutoff.
    """

    recalls = [metrics.recall_at_k(t, p, k) for t, p in zip(y_true_groups, y_pred_groups)]
    ndcgs = [metrics.ndcg_at_k(t, p, k) for t, p in zip(y_true_groups, y_pred_groups)]
    return {"recall@k": float(np.mean(recalls)), "ndcg@k": float(np.mean(ndcgs))}


__all__ = ["evaluate_rank_lists"]
