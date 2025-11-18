import hashlib
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _select_numeric_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> List[str]:
    if columns is None:
        return df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in columns if col in df.columns]


def _impute(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "median":
        fill_values = frame.median()
    elif strategy == "mean":
        fill_values = frame.mean()
    elif strategy == "zero":
        fill_values = 0
    elif strategy is None:
        return frame
    else:
        raise ValueError(f"Unsupported impute strategy: {strategy}")
    return frame.fillna(fill_values)


def _log_transform(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = np.log1p(frame[col])
    return frame


def _scale(train: pd.DataFrame, test: pd.DataFrame, mode: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mode is None:
        return train, test
    if mode == "standard":
        scaler = StandardScaler()
    elif mode == "minmax":
        scaler = MinMaxScaler()
    elif mode == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scale mode: {mode}")

    tr_values = scaler.fit_transform(train)
    te_values = scaler.transform(test)
    tr_df = pd.DataFrame(tr_values, columns=train.columns, index=train.index)
    te_df = pd.DataFrame(te_values, columns=test.columns, index=test.index)
    return tr_df, te_df


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    num_cols: Optional[Iterable[str]] = None,
    prefix: str = "num",
    impute: Optional[str] = "median",
    log_cols: Optional[Iterable[str]] = None,
    clip_quant: Tuple[float, float] = (0.01, 0.99),
    scale: Optional[str] = None,
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """Базовая обработка числовых признаков.

    Шаги: авто-выбор числовых колонок, импутация, опциональный log1p,
    обрезка по квантилям, масштабирование и переименование с префиксом.
    Возвращает FeaturePackage(kind="dense").
    """

    columns = _select_numeric_columns(train_df, num_cols)
    log_targets = list(log_cols) if log_cols is not None else []

    params = {
        "columns": columns,
        "prefix": prefix,
        "impute": impute,
        "log_cols": log_targets,
        "clip_quant": clip_quant,
        "scale": scale,
    }
    data_stamp = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, code_fingerprint=_fingerprint(), data_stamp=data_stamp)
    if use_cache:
        cached = load_feature_pkg("num_basic", cache_key)
        if cached is not None:
            return cached

    t0 = time.time()
    train = train_df[columns].copy()
    test = test_df[columns].copy()

    train = _impute(train, impute)
    test = _impute(test, impute)

    train = _log_transform(train, log_targets)
    test = _log_transform(test, log_targets)

    if clip_quant is not None:
        lower = train.quantile(clip_quant[0])
        upper = train.quantile(clip_quant[1])
        train = train.clip(lower=lower, upper=upper, axis="columns")
        test = test.clip(lower=lower, upper=upper, axis="columns")

    train, test = _scale(train, test, scale)

    rename_map = {col: f"{prefix}__{col}" for col in train.columns}
    train = train.rename(columns=rename_map)
    test = test.rename(columns=rename_map)

    cols = list(train.columns)
    meta = {
        "name": "num_basic",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
    }

    pkg = FeaturePackage(
        name="num_basic",
        train=train,
        test=test,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("num_basic", cache_key, pkg)

    return pkg
