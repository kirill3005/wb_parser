from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils_time import ensure_utc

logger = logging.getLogger(__name__)


def build_sessions(
    interactions: pd.DataFrame,
    *,
    session_gap_min: int = 30,
    user_col: str = "user_id",
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Assign session ids based on time gaps per user."""

    df = interactions.copy()
    df = ensure_utc(df, cols=[ts_col])
    df = df.sort_values([user_col, ts_col]).reset_index(drop=True)
    session_ids: list[str] = []
    last_ts: dict[str, pd.Timestamp] = {}
    counters: dict[str, int] = {}

    gap = pd.Timedelta(minutes=session_gap_min)
    for row in df.itertuples():
        user = getattr(row, user_col)
        ts = getattr(row, ts_col)
        prev_ts = last_ts.get(user)
        if prev_ts is None or ts - prev_ts > gap:
            counters[user] = counters.get(user, 0) + 1
        session_ids.append(f"{user}_{counters[user]}")
        last_ts[user] = ts

    df["session_id"] = session_ids
    return df
