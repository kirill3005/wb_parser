from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class CrossFeats(FeatureBlock):
    name = "cross"

    def __init__(self, whitelist: List[str] | None = None):
        self.whitelist = whitelist or []

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "CrossFeats":
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        # Cross features will be generated later in joiner when base columns exist; return empty
        return pairs[["query_id", "item_id"]].copy()


register("cross", CrossFeats)
