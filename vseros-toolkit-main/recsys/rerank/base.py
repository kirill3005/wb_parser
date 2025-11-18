"""Base interfaces for rerankers."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class ReRanker(Protocol):
    def apply(self, df: pd.DataFrame, items: pd.DataFrame | None, *, K: int, config: dict, rng: np.random.RandomState) -> pd.DataFrame:
        ...


__all__ = ["ReRanker"]
