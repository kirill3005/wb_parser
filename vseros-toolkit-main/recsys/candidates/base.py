from __future__ import annotations

import logging
from typing import Set

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Base interface for candidate generators."""

    name: str = "base"
    requires: Set[str] = set()

    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        cutoff_ts,
        schema: Schema,
        rng: np.random.RandomState,
    ) -> "CandidateGenerator":
        raise NotImplementedError

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        raise NotImplementedError

    def _ensure_columns(self, df: pd.DataFrame, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} for {self.name}")
