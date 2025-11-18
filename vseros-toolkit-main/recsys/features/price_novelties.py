from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class PriceNovelties(FeatureBlock):
    name = "price_nov"

    def __init__(self):
        self.items: pd.DataFrame | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "PriceNovelties":
        self.items = items.copy() if items is not None else pd.DataFrame()
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        has_price = not self.items.empty and "price" in self.items.columns
        for row in pairs.itertuples(index=False):
            base = {"query_id": row.query_id, "item_id": row.item_id, "price": 0.0}
            if has_price:
                try:
                    base["price"] = float(self.items.set_index("item_id").loc[row.item_id, "price"])
                except KeyError:
                    base["price"] = 0.0
            rows.append(base)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].astype(np.float32)
        return df


register("price_nov", PriceNovelties)
