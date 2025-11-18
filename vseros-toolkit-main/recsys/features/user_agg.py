from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class UserAgg(FeatureBlock):
    name = "user_agg"

    def __init__(self, windows_days: List[int] | None = None):
        self.windows_days = windows_days or [30]
        self.interactions: pd.DataFrame | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "UserAgg":
        self.interactions = interactions.copy()
        self.interactions["ts"] = pd.to_datetime(self.interactions["ts"])
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            scope_col = "session_id" if schema.query_scope == "session" else "user_id"
            if scope_col not in self.interactions.columns:
                scope_col = "user_id"
            hist = self.interactions[(self.interactions[scope_col] == row.query_id) & (self.interactions["ts"] <= ts_query)]
            recency = (ts_query - hist["ts"].max()).total_seconds() / 3600 if not hist.empty else np.nan
            base = {"query_id": row.query_id, "item_id": row.item_id, "user_recency_h": recency if np.isfinite(recency) else 0.0}
            for w in self.windows_days:
                start = ts_query - timedelta(days=w)
                base[f"user_freq_{w}d"] = float(hist[hist["ts"] >= start].shape[0])
                base[f"user_unique_items_{w}d"] = float(hist[hist["ts"] >= start]["item_id"].nunique()) if not hist.empty else 0.0
            rows.append(base)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("user_agg", UserAgg)
