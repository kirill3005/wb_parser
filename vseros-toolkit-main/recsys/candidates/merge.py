from __future__ import annotations

import pandas as pd


def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["score_norm"] = []
        return df
    min_s, max_s = df["score_raw"].min(), df["score_raw"].max()
    if max_s - min_s < 1e-9:
        df["score_norm"] = 1.0
    else:
        df["score_norm"] = (df["score_raw"] - min_s) / (max_s - min_s)
    return df


def merge_and_rank(sources: dict[str, pd.DataFrame], *, quotas: dict[str, int], weights: dict[str, float], K: int) -> pd.DataFrame:
    per_query = []
    for name, df in sources.items():
        if df is None or df.empty:
            continue
        quota = quotas.get(name, K)
        df = df.sort_values(["query_id", "score_raw"], ascending=[True, False])
        df = df.groupby("query_id").head(quota)
        df = df.copy()
        df["source"] = name
        df = _normalize_scores(df)
        df["score_weighted"] = df["score_norm"] * weights.get(name, 1.0)
        per_query.append(df)
    if not per_query:
        return pd.DataFrame(columns=["query_id", "item_id", "source", "score_raw", "rank_src", "score_norm", "score_final"])
    merged = pd.concat(per_query, ignore_index=True)
    merged = merged.sort_values(["query_id", "item_id", "score_weighted"], ascending=[True, True, False])
    merged = merged.groupby(["query_id", "item_id"], as_index=False).agg(
        {
            "score_weighted": "max",
            "score_norm": "max",
            "source": lambda s: list(set(s)),
        }
    )
    merged = merged.sort_values(["query_id", "score_weighted"], ascending=[True, False])
    merged["rank_src"] = merged.groupby("query_id").cumcount()
    merged["score_final"] = merged["score_weighted"]
    merged = merged.groupby("query_id").head(K)
    return merged
