from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class SequenceFeats(FeatureBlock):
    name = "sequence"

    def __init__(self, last_k: List[int] | None = None, max_seq_len: int = 100):
        self.last_k = last_k or [5]
        self.max_seq_len = max_seq_len
        self.interactions: pd.DataFrame | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "SequenceFeats":
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
            hist = hist.sort_values("ts").tail(self.max_seq_len)
            base = {"query_id": row.query_id, "item_id": row.item_id}
            base["seq_len"] = float(len(hist))
            if len(hist) > 1:
                gaps = hist["ts"].diff().dt.total_seconds().dropna()
                base["gap_mean_s"] = gaps.mean()
                base["gap_std_s"] = gaps.std() if len(gaps) > 1 else 0.0
            else:
                base["gap_mean_s"] = 0.0
                base["gap_std_s"] = 0.0
            for k in self.last_k:
                recent = hist.tail(k)
                base[f"last_{k}_unique_items"] = float(recent["item_id"].nunique()) if not recent.empty else 0.0
            rows.append(base)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("sequence", SequenceFeats)
