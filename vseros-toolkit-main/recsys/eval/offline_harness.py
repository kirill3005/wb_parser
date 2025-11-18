from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import metrics
from .ranking import join_truth_and_pred, normalize_predictions
from .slices import slice_metrics
from .bootstrap import bootstrap_metrics

logger = logging.getLogger(__name__)


def evaluate_offline(
    pairs_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    items_df: Optional[pd.DataFrame] = None,
    *,
    metrics_list: Iterable[str] = ("recall@20", "ndcg@20"),
    slices: Optional[Iterable[str]] = None,
    bootstrap_iters: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    rng = rng or np.random.RandomState(42)
    result: Dict = {"overall": {}, "counts": {}}

    parsed = []
    k_values = []
    for name in metrics_list:
        m, k = name.split("@")
        k_values.append(int(k))
        parsed.append((m, int(k)))

    joined = join_truth_and_pred(pairs_df, preds_df, k=max(k_values))
    y_true = [t for t, _ in joined]
    y_pred = [p for _, p in joined]

    for metric_name, k in parsed:
        if metric_name == "recall":
            score = np.mean([metrics.recall_at_k(t, p, k) for t, p in joined])
        elif metric_name == "ndcg":
            score = np.mean([metrics.ndcg_at_k(t, p, k) for t, p in joined])
        elif metric_name == "map":
            score = np.mean([metrics.ap_at_k(t, p, k) for t, p in joined])
        elif metric_name == "mrr":
            score = np.mean([metrics.mrr_at_k(t, p, k) for t, p in joined])
        elif metric_name == "hitrate":
            score = np.mean([metrics.hitrate_at_k(t, p, k) for t, p in joined])
        else:
            continue
        result["overall"][f"{metric_name}@{k}"] = float(score)

    result["counts"] = {
        "queries": len(joined),
        "nonempty_queries": int(sum(1 for t in y_true if t)),
    }

    if slices and items_df is not None:
        result["slices"] = slice_metrics(pairs_df, preds_df, items_df, slices, k=max(k_values))

    if bootstrap_iters:
        result["bootstrap"] = bootstrap_metrics(joined, parsed, iters=bootstrap_iters, rng=rng)

    return result
