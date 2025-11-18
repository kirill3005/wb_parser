import hashlib
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _encode_column(
    series: pd.Series,
    mapping_count: Dict,
    mapping_ratio: Dict,
    rare_values: set,
    rare_count: float,
    rare_ratio: float,
):
    def encode(val, table, rare_value):
        if val in table:
            return table[val]
        if val in rare_values:
            return rare_value
        return 0.0

    freq = series.apply(lambda x: encode(x, mapping_count, rare_count))
    ratio = series.apply(lambda x: encode(x, mapping_ratio, rare_ratio))
    return freq, ratio


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    cat_cols: Optional[Iterable[str]] = None,
    rare_threshold: float = 0.01,
    prefix: str = "catf",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """Частоты категорий с обработкой RARE/UNK без использования таргета."""

    columns = _select_categorical_columns(train_df, cat_cols)

    params = {
        "columns": columns,
        "rare_threshold": rare_threshold,
        "prefix": prefix,
    }
    data_stamp = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, code_fingerprint=_fingerprint(), data_stamp=data_stamp)
    if use_cache:
        cached = load_feature_pkg("cat_freq", cache_key)
        if cached is not None:
            return cached

    t0 = time.time()
    train_features = {}
    test_features = {}

    for col in columns:
        counts = train_df[col].value_counts(dropna=False)
        ratios = counts / len(train_df)

        rare_mask = ratios < rare_threshold
        rare_values = set(ratios[rare_mask].index)
        rare_count = counts[rare_mask].sum()
        rare_ratio = ratios[rare_mask].sum()

        mapping_count = counts[~rare_mask].to_dict()
        mapping_ratio = ratios[~rare_mask].to_dict()

        freq_train, ratio_train = _encode_column(
            train_df[col], mapping_count, mapping_ratio, rare_values, float(rare_count), float(rare_ratio)
        )
        freq_test, ratio_test = _encode_column(
            test_df[col], mapping_count, mapping_ratio, rare_values, float(rare_count), float(rare_ratio)
        )

        train_features[f"{prefix}__{col}__freq"] = freq_train
        train_features[f"{prefix}__{col}__ratio"] = ratio_train
        test_features[f"{prefix}__{col}__freq"] = freq_test
        test_features[f"{prefix}__{col}__ratio"] = ratio_test

    train_out = pd.DataFrame(train_features, index=train_df.index).fillna(0)
    test_out = pd.DataFrame(test_features, index=test_df.index).fillna(0)

    cols = list(train_out.columns)
    meta = {
        "name": "cat_freq",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
    }

    pkg = FeaturePackage(
        name="cat_freq",
        train=train_out,
        test=test_out,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("cat_freq", cache_key, pkg)

    return pkg
