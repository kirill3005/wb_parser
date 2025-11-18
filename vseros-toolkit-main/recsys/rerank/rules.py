"""Greedy rule-based reranking helpers."""
from __future__ import annotations

import pandas as pd


def apply_rules(df: pd.DataFrame, items: pd.DataFrame | None, *, K: int, config: dict) -> pd.DataFrame:
    if items is None:
        return _topk(df, K)
    brand_limit = None
    if config.get("dedupe_brand"):
        brand_cfg = config.get("dedupe_brand")
        brand_limit = brand_cfg.get("n_per_brand", 1)
    price_bounds = config.get("price_bounds") if config else None
    rows = []
    items_index = items.set_index("item_id") if not items.empty else None
    for q, group in df.groupby("query_id"):
        taken_brand: dict[str, int] = {}
        group_sorted = group.sort_values("score_ranker", ascending=False)
        rank = 1
        for _, r in group_sorted.iterrows():
            if rank > K:
                break
            if items_index is not None and r.item_id in items_index.index:
                meta = items_index.loc[r.item_id]
                brand = meta.get("brand") if hasattr(meta, "get") else meta.get("brand")
                if brand_limit is not None:
                    cnt = taken_brand.get(brand, 0)
                    if cnt >= brand_limit:
                        continue
                if price_bounds:
                    price = meta.get("price") if hasattr(meta, "get") else meta.get("price")
                    if price_bounds.get("min") is not None and price is not None and price < price_bounds["min"]:
                        continue
                    if price_bounds.get("max") is not None and price is not None and price > price_bounds["max"]:
                        continue
                taken_brand[brand] = taken_brand.get(brand, 0) + 1
            rows.append({"query_id": q, "item_id": r.item_id, "score_final": r.score_ranker, "rank_final": rank})
            rank += 1
    return pd.DataFrame(rows)


def _topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for q, g in df.groupby("query_id"):
        top = g.sort_values("score_ranker", ascending=False).head(k)
        for idx, r in enumerate(top.itertuples(index=False), start=1):
            rows.append({"query_id": q, "item_id": r.item_id, "score_final": r.score_ranker, "rank_final": idx})
    return pd.DataFrame(rows)


__all__ = ["apply_rules"]
