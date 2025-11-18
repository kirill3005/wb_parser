import hashlib
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _select_categorical_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> List[str]:
    if columns is None:
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return [col for col in columns if col in df.columns]


def _get_smoothing_value(smoothing: Union[str, float, int]) -> float:
    if isinstance(smoothing, (float, int)):
        return float(smoothing)
    if smoothing == "m-estimate":
        return 10.0
    raise ValueError(f"Unsupported smoothing strategy: {smoothing}")


def _fit_target_mapping(values: pd.Series, target: pd.Series, prior: float, m: float) -> Dict:
    grouped = pd.DataFrame({"val": values, "y": target}).groupby("val")["y"]
    stats = grouped.agg(["count", "mean"]).fillna(0)
    smoothed = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
    return smoothed.to_dict()


def _apply_mapping(series: pd.Series, mapping: Dict, default: float) -> pd.Series:
    return series.map(mapping).fillna(default)


def _target_encoding_oof(
    train_df: pd.DataFrame,
    target: pd.Series,
    test_df: pd.DataFrame,
    columns: Sequence[str],
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    smoothing: Union[str, float, int],
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = _get_smoothing_value(smoothing)
    prior = float(target.mean())

    train_features = pd.DataFrame(index=train_df.index)
    test_features = pd.DataFrame(index=test_df.index)

    for col in columns:
        oof_col = pd.Series(index=train_df.index, dtype=float)

        for train_idx, val_idx in folds:
            mapping = _fit_target_mapping(train_df.iloc[list(train_idx)][col], target.iloc[list(train_idx)], prior, m)
            oof_values = _apply_mapping(train_df.iloc[list(val_idx)][col], mapping, default=prior)
            oof_col.iloc[list(val_idx)] = oof_values

        full_mapping = _fit_target_mapping(train_df[col], target, prior, m)
        test_encoded = _apply_mapping(test_df[col], full_mapping, default=prior)

        train_features[f"{prefix}__{col}__target"] = oof_col.fillna(prior)
        test_features[f"{prefix}__{col}__target"] = test_encoded.fillna(prior)

    return train_features, test_features


def _encode_woe_ctr_stub(method: str):
    raise NotImplementedError(f"Method '{method}' is not implemented yet")


def build(
    train_df: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    *,
    cat_cols: Optional[Iterable[str]] = None,
    method: str = "target",
    smoothing: Union[str, float, int] = "m-estimate",
    prefix: str = "te",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """OOF target/WOE/CTR-кодировки для категориальных признаков."""

    columns = _select_categorical_columns(train_df, cat_cols)

    params = {
        "columns": columns,
        "method": method,
        "smoothing": smoothing,
        "prefix": prefix,
    }
    data_stamp = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "folds": [(len(tr), len(val)) for tr, val in folds],
    }
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, code_fingerprint=_fingerprint(), data_stamp=data_stamp)
    if use_cache:
        cached = load_feature_pkg("cat_te_oof", cache_key)
        if cached is not None:
            return cached

    t0 = time.time()

    if method == "target":
        train_features, test_features = _target_encoding_oof(
            train_df=train_df,
            target=y,
            test_df=test_df,
            columns=columns,
            folds=folds,
            smoothing=smoothing,
            prefix=prefix,
        )
    elif method in {"woe", "ctr"}:
        _encode_woe_ctr_stub(method)
    else:
        raise ValueError(f"Unsupported method: {method}")

    cols = list(train_features.columns)
    meta = {
        "name": "cat_te_oof",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
        "oof": True,
    }

    pkg = FeaturePackage(
        name="cat_te_oof",
        train=train_features,
        test=test_features,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("cat_te_oof", cache_key, pkg)

    return pkg
