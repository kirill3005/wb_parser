from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils_time import ensure_utc

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    mode: str = "global"
    train_until: Optional[str] = None
    val_until: Optional[str] = None
    test_until: Optional[str] = None
    embargo: str = "0D"


def assign_time_splits(interactions: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    df = ensure_utc(interactions, cols=["ts"])
    df = df.copy()
    if cfg.mode == "global":
        train_until = pd.to_datetime(cfg.train_until, utc=True) if cfg.train_until else df["ts"].quantile(0.6)
        val_until = pd.to_datetime(cfg.val_until, utc=True) if cfg.val_until else df["ts"].quantile(0.8)
        embargo = pd.Timedelta(cfg.embargo)
        train_mask = df["ts"] <= train_until
        val_mask = (df["ts"] >= train_until + embargo) & (df["ts"] <= val_until)
        test_mask = df["ts"] > val_until + embargo
        df.loc[train_mask, "split"] = "train"
        df.loc[val_mask, "split"] = "val"
        df.loc[test_mask, "split"] = "test"
    else:  # pragma: no cover - placeholder for rolling
        df["split"] = "train"
    return df


def save_split_report(df: pd.DataFrame, path: str) -> None:
    counts = df["split"].value_counts().to_dict()
    Path(path).write_text(json.dumps({"counts": counts}, indent=2), encoding="utf-8")
