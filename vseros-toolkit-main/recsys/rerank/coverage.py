"""Simple coverage-aware filler."""
from __future__ import annotations

import pandas as pd


def apply_coverage(df: pd.DataFrame, items: pd.DataFrame | None, *, quotas: dict, K: int) -> pd.DataFrame:
    if not quotas or items is None:
        return df
    items_index = items.set_index("item_id") if not items.empty else None
    rows = []
    for q, g in df.groupby("query_id"):
        remaining = quotas.copy()
        picked = []
        for _, r in g.sort_values("score_ranker", ascending=False).iterrows():
            if len(picked) >= K:
                break
            cat = None
            if items_index is not None and r.item_id in items_index.index and "category" in items_index.columns:
                cat = items_index.loc[r.item_id].get("category")
            if cat in remaining and remaining[cat] <= 0:
                continue
            picked.append(r)
            if cat in remaining:
                remaining[cat] -= 1
        rows.extend(
            {
                "query_id": q,
                "item_id": r.item_id,
                "score_final": r.score_ranker,
                "rank_final": idx,
            }
            for idx, r in enumerate(picked, start=1)
        )
    return pd.DataFrame(rows)


__all__ = ["apply_coverage"]
