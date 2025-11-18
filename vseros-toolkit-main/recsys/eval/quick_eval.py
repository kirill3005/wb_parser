from __future__ import annotations

import pandas as pd

from recsys.eval.metrics import recall_at_k, ndcg_at_k


def quick_eval(candidates: pd.DataFrame, queries: pd.DataFrame, k: int = 20) -> dict:
    if "label" not in queries.columns:
        return {}
    metrics = {"recall@k": [], "ndcg@k": []}
    label_map = queries.set_index("query_id")["label"].to_dict()
    for qid, group in candidates.groupby("query_id"):
        truth_raw = label_map.get(qid)
        if truth_raw is None:
            continue
        truth = truth_raw if isinstance(truth_raw, list) else [truth_raw]
        pred = list(group.sort_values("score_final", ascending=False)["item_id"])
        metrics["recall@k"].append(recall_at_k(pred, truth, k))
        metrics["ndcg@k"].append(ndcg_at_k(pred, truth, k))
    if not metrics["recall@k"]:
        return {}
    return {"recall@k": float(pd.Series(metrics["recall@k"]).mean()), "ndcg@k": float(pd.Series(metrics["ndcg@k"]).mean())}
