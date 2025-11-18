from __future__ import annotations

from typing import Iterable, Dict

import numpy as np
import pandas as pd

from .ranking import normalize_predictions
from .metrics import recall_at_k, ndcg_at_k


def slice_metrics(pairs: pd.DataFrame, preds: pd.DataFrame, items: pd.DataFrame, slice_cols: Iterable[str], k: int = 20) -> Dict:
    preds_norm = normalize_predictions(preds, k=k)
    pairs = pairs.merge(items, on="item_id", how="left")
    preds_norm = preds_norm.merge(items, on="item_id", how="left", suffixes=("", "_pred"))

    results: Dict = {}
    for col in slice_cols:
        if col not in pairs.columns:
            continue
        for value, grp in pairs.groupby(col):
            qids = grp["query_id"].unique()
            sub_pairs = pairs[pairs["query_id"].isin(qids)]
            truth = {qid: g[g["label"] == 1]["item_id"].tolist() for qid, g in sub_pairs.groupby("query_id")}
            pred_items = {
                qid: g.sort_values(["rank"])["item_id"].tolist()
                for qid, g in preds_norm[preds_norm["query_id"].isin(qids)].groupby("query_id")
            }
            joined = [(truth.get(qid, []), pred_items.get(qid, [])) for qid in qids]
            if not joined:
                continue
            results[f"{col}={value}"] = {
                f"recall@{k}": float(np.mean([recall_at_k(t, p, k) for t, p in joined])),
                f"ndcg@{k}": float(np.mean([ndcg_at_k(t, p, k) for t, p in joined])),
            }
    return results
