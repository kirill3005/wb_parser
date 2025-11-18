from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def fingerprint(obj: Any) -> str:
    dumped = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()


def save_df(df: pd.DataFrame, directory: str, name: str) -> str:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def load_df(directory: str, name: str) -> pd.DataFrame:
    path = Path(directory) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
