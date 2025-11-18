from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .utils_time import ensure_utc

logger = logging.getLogger(__name__)


def build_queries(interactions: pd.DataFrame, *, scope: str = "session", limit: Optional[int] = None) -> pd.DataFrame:
    df = ensure_utc(interactions, cols=["ts"])
    if scope == "session" and "session_id" in df.columns:
        grp = df.groupby("session_id")
        res = grp["ts"].max().reset_index().rename(columns={"session_id": "query_id", "ts": "ts_query"})
    else:
        grp = df.groupby("user_id")
        res = grp["ts"].max().reset_index().rename(columns={"user_id": "query_id", "ts": "ts_query"})
    res = res.sort_values("ts_query", ascending=False)
    if limit:
        res = res.head(limit)
    return res.reset_index(drop=True)
