from __future__ import annotations

import pandas as pd


def ensure_utc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            # heuristic: seconds vs milliseconds
            max_val = series.max()
            unit = "s" if max_val < 1e11 else "ms"
            out[col] = pd.to_datetime(series, unit=unit, utc=True)
        else:
            out[col] = pd.to_datetime(series, utc=True)
    return out


def filter_by_cutoff(df: pd.DataFrame, cutoff_ts, ts_col: str = "ts") -> pd.DataFrame:
    cutoff_ts = pd.to_datetime(cutoff_ts, utc=True)
    return df[df[ts_col] <= cutoff_ts].copy()


def assert_time_safe(candidates: pd.DataFrame, queries: pd.DataFrame, *, ts_col: str = "ts_query") -> bool:
    if ts_col not in queries.columns:
        return True
    merged = candidates.merge(queries[["query_id", ts_col]], on="query_id", how="left", suffixes=("_cand", "_query"))
    mask = merged[f"{ts_col}_query"].notna()
    violating = merged[mask & (merged.get(ts_col + "_cand", merged.get("ts", pd.Timestamp.min)) > merged[f"{ts_col}_query"])]
    if not violating.empty:
        raise AssertionError("Found entries with timestamp after query")
    return True


def bucket_day(df: pd.DataFrame, ts_col: str = "ts") -> pd.Series:
    return pd.to_datetime(df[ts_col], utc=True).dt.normalize()
