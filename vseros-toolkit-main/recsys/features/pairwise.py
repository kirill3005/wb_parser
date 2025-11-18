from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class PairwiseCore(FeatureBlock):
    name = "pairwise_core"

    def __init__(self, windows_days: List[int] | None = None, decay_alpha: float = 0.05):
        self.windows_days = windows_days or [7, 30]
        self.decay_alpha = decay_alpha
        self.interactions: pd.DataFrame | None = None
        self.schema: Schema | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "PairwiseCore":
        self.interactions = interactions.copy()
        self.schema = schema
        self.interactions["ts"] = pd.to_datetime(self.interactions["ts"])
        return self

    def _hist(self, query_id, ts_query: pd.Timestamp) -> pd.DataFrame:
        assert self.interactions is not None
        scope_col = "session_id" if self.schema and self.schema.query_scope == "session" else "user_id"
        if scope_col not in self.interactions.columns:
            scope_col = "user_id"
        hist = self.interactions[(self.interactions[scope_col] == query_id) & (self.interactions["ts"] <= ts_query)]
        return hist

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            hist = self._hist(row.query_id, ts_query)
            recency_user = (ts_query - hist["ts"].max()).total_seconds() / 3600 if not hist.empty else np.nan
            item_hist = hist[hist["item_id"] == row.item_id]
            recency_item = (ts_query - item_hist["ts"].max()).total_seconds() / 3600 if not item_hist.empty else np.nan

            counts = {}
            for w in self.windows_days:
                start = ts_query - timedelta(days=w)
                cnt = hist[hist["ts"] >= start].shape[0]
                counts[f"user_count_{w}d"] = cnt
                item_cnt = self.interactions[
                    (self.interactions["item_id"] == row.item_id) & (self.interactions["ts"] <= ts_query) & (self.interactions["ts"] >= start)
                ].shape[0]
                counts[f"item_pop_{w}d"] = item_cnt
            seen_before = int(not item_hist.empty)
            rows.append(
                {
                    "query_id": row.query_id,
                    "item_id": row.item_id,
                    "recency_user_h": recency_user if np.isfinite(recency_user) else 0.0,
                    "recency_item_h": recency_item if np.isfinite(recency_item) else 0.0,
                    "seen_before": seen_before,
                    **counts,
                }
            )
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("pairwise_core", PairwiseCore)
