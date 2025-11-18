from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class ItemAgg(FeatureBlock):
    name = "item_agg"

    def __init__(self, windows_days: List[int] | None = None):
        self.windows_days = windows_days or [7, 30]
        self.interactions: pd.DataFrame | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "ItemAgg":
        self.interactions = interactions.copy()
        self.interactions["ts"] = pd.to_datetime(self.interactions["ts"])
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            base = {"query_id": row.query_id, "item_id": row.item_id}
            for w in self.windows_days:
                start = ts_query - timedelta(days=w)
                mask = (self.interactions["item_id"] == row.item_id) & (self.interactions["ts"] <= ts_query) & (
                    self.interactions["ts"] >= start
                )
                base[f"item_pop_{w}d"] = float(mask.sum())
            rows.append(base)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].astype(np.float32)
        return df


register("item_agg", ItemAgg)
