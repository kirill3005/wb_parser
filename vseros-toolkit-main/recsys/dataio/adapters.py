from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .schema import Schema
from .utils_time import ensure_utc, filter_by_cutoff
from .indexers import Indexers

logger = logging.getLogger(__name__)


@dataclass
class AdaptedData:
    interactions: pd.DataFrame
    items: pd.DataFrame
    queries: pd.DataFrame
    schema: Schema
    indexers: Optional[Indexers] = None


REQUIRED_INTERACTIONS = {"user_id", "item_id", "ts"}


def _rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    inv = {v: k for k, v in mapping.items() if v in df.columns}
    return df.rename(columns=inv)


def _read_table(path: str) -> pd.DataFrame:
    """Read parquet or CSV depending on extension."""

    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format for {path}; use parquet or csv")


def load_interactions(path: str, schema: Schema, *, drop_dupes: bool = True, sort_by_ts: bool = True) -> pd.DataFrame:
    df = _read_table(path)
    df = _rename_columns(df, schema.interactions)
    schema.validate_required(df, section="interactions", required=REQUIRED_INTERACTIONS)
    df = ensure_utc(df, cols=["ts"])
    if drop_dupes:
        df = df.drop_duplicates()
    if sort_by_ts and "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def load_items(path: str, schema: Schema) -> pd.DataFrame:
    df = _read_table(path)
    df = _rename_columns(df, schema.items)
    return df


def load_queries(path: Optional[str], schema: Schema) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = _read_table(path)
    if schema.query_id_col in df.columns:
        df = df.rename(columns={schema.query_id_col: "query_id"})
    if "ts_query" in df.columns:
        df = ensure_utc(df, cols=["ts_query"])
    return df


def build_indexers(df_inter: pd.DataFrame, df_items: pd.DataFrame, schema: Schema, *, save_dir: Optional[str] = None) -> Indexers:
    user_ids = df_inter["user_id"].dropna().astype(str).unique()
    item_ids = df_items["item_id"].dropna().astype(str).unique() if not df_items.empty else df_inter["item_id"].dropna().astype(str).unique()
    session_ids = df_inter["session_id"].dropna().astype(str).unique() if "session_id" in df_inter.columns else np.array([], dtype=str)

    indexers = Indexers(
        user2idx={u: i for i, u in enumerate(user_ids)},
        item2idx={i: j for j, i in enumerate(item_ids)},
        session2idx={s: k for k, s in enumerate(session_ids)},
    )
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        indexers.save(save_dir)
    return indexers


def apply_indexers(df: pd.DataFrame, indexers: Indexers) -> pd.DataFrame:
    out = df.copy()
    if indexers.user2idx:
        out["user_idx"] = out["user_id"].astype(str).map(indexers.user2idx).astype("Int32")
    if indexers.item2idx:
        out["item_idx"] = out["item_id"].astype(str).map(indexers.item2idx).astype("Int32")
    if "session_id" in out.columns and indexers.session2idx:
        out["session_idx"] = out["session_id"].astype(str).map(indexers.session2idx).astype("Int32")
    return out


def load_datasets(
    *,
    schema: Schema,
    path_interactions: str,
    path_items: Optional[str] = None,
    path_queries: Optional[str] = None,
    cutoff_ts: Optional[pd.Timestamp] = None,
    save_indexers: Optional[str] = None,
) -> AdaptedData:
    """Load datasets and align column names to the canonical schema."""

    interactions = load_interactions(path_interactions, schema)
    if cutoff_ts is not None:
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")

    items = load_items(path_items, schema) if path_items else pd.DataFrame()
    queries = load_queries(path_queries, schema)

    indexers = build_indexers(interactions, items, schema, save_dir=save_indexers) if save_indexers else None

    logger.info(
        "Loaded %d interactions, %d items, %d queries", len(interactions), len(items), len(queries)
    )

    return AdaptedData(
        interactions=interactions,
        items=items,
        queries=queries,
        schema=schema,
        indexers=indexers,
    )
