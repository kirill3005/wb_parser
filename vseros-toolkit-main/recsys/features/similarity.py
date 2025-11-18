from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class SimilarityFeats(FeatureBlock):
    name = "similarity"

    def __init__(self, use_item2vec: bool = False, use_tfidf: bool = True, profile_last_k: int = 20):
        self.use_item2vec = use_item2vec
        self.use_tfidf = use_tfidf
        self.profile_last_k = profile_last_k
        self.items: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "SimilarityFeats":
        self.items = items.copy() if items is not None else pd.DataFrame()
        self.interactions = interactions.copy()
        self.interactions["ts"] = pd.to_datetime(self.interactions["ts"])
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        scope_col = "session_id" if schema.query_scope == "session" else "user_id"
        if scope_col not in self.interactions.columns:
            scope_col = "user_id"
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            hist = self.interactions[(self.interactions[scope_col] == row.query_id) & (self.interactions["ts"] <= ts_query)]
            hist = hist.sort_values("ts").tail(self.profile_last_k)
            base = {"query_id": row.query_id, "item_id": row.item_id}
            base["cat_match_share"] = 0.0
            if not self.items.empty and "category" in self.items.columns:
                try:
                    target_cat = self.items.set_index("item_id").loc[row.item_id, "category"]
                except KeyError:
                    target_cat = None
                if target_cat is not None:
                    hist_items = hist.merge(self.items[["item_id", "category"]], on="item_id", how="left")
                    denom = max(len(hist_items), 1)
                    base["cat_match_share"] = float((hist_items["category"] == target_cat).sum()) / denom
            rows.append(base)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("similarity", SimilarityFeats)
