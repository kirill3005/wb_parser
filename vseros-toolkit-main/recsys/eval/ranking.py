from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd


def normalize_predictions(preds: pd.DataFrame, *, k: int, score_col: str = "score") -> pd.DataFrame:
    """Sort predictions per query by score desc then item_id for determinism."""

    preds = preds.copy()
    preds = preds.sort_values(["query_id", score_col, "item_id"], ascending=[True, False, True])
    preds["rank"] = preds.groupby("query_id").cumcount() + 1
    return preds[preds["rank"] <= k]


def group_predictions(preds: pd.DataFrame) -> Dict:
    grouped = {}
    for qid, grp in preds.groupby("query_id"):
        grouped[qid] = grp.sort_values("rank")["item_id"].tolist()
    return grouped


def join_truth_and_pred(pairs: pd.DataFrame, preds: pd.DataFrame, k: int) -> List[Tuple[List, List]]:
    truth_grouped = {qid: grp[grp["label"] == 1]["item_id"].tolist() for qid, grp in pairs.groupby("query_id")}
    preds_norm = normalize_predictions(preds, k=k)
    pred_grouped = group_predictions(preds_norm)
    rows = []
    for qid, true_items in truth_grouped.items():
        rows.append((true_items, pred_grouped.get(qid, [])))
    return rows
