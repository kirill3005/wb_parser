"""Time-based aggregations without leakage.

This module provides a single entry point :func:`build` that constructs
anti-leak rolling, lag, and exponential moving statistics over time. The
function follows the :class:`common.features.types.FeaturePackage` contract and
supports both global and out-of-fold (OOF) computation modes with optional
embargo handling.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from .types import FeaturePackage


def _fingerprint() -> str:
    return "time_agg_v1"


def _hash_timestamps(series: pd.Series) -> str:
    head = series.astype("int64").astype(str).head(100)
    return hashlib.sha1("|".join(head).encode()).hexdigest()[:12]


def _select_num_cols(
    df: pd.DataFrame, date_col: str, num_cols: Optional[Sequence[str]], group_cols: Optional[Sequence[str]]
) -> List[str]:
    if num_cols is not None:
        return list(num_cols)
    skip = {date_col}
    if group_cols:
        skip.update(group_cols)
    cols: List[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _feature_columns(
    num_cols: List[str],
    lags: Sequence[int],
    rollings_count: Sequence[int],
    rollings_time: Sequence[str],
    ewm_spans: Sequence[int],
    agg_funcs: Sequence[str],
    prefix: str,
) -> List[str]:
    cols: List[str] = []
    for col in num_cols:
        for k in lags:
            cols.append(f"{prefix}__{col}__lag{k}")
        for w in rollings_count:
            for agg in agg_funcs:
                cols.append(f"{prefix}__{col}__rollN{w}__{agg}")
        for w in rollings_time:
            for agg in agg_funcs:
                cols.append(f"{prefix}__{col}__rollT{w}__{agg}")
        for span in ewm_spans:
            cols.append(f"{prefix}__{col}__ewm{span}__mean")
            if "std" in agg_funcs:
                cols.append(f"{prefix}__{col}__ewm{span}__std")
    return cols


def _agg_block(
    df: pd.DataFrame,
    date_col: str,
    num_cols: List[str],
    lags: Sequence[int],
    rollings_count: Sequence[int],
    rollings_time: Sequence[str],
    ewm_spans: Sequence[int],
    agg_funcs: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    base = df.set_index(date_col)
    features: Dict[str, pd.Series] = {}

    for col in num_cols:
        series = base[col]
        for k in lags:
            features[f"{prefix}__{col}__lag{k}"] = series.shift(k)
        for w in rollings_count:
            rolled = series.rolling(window=w, min_periods=1, closed="left")
            agg = rolled.agg(agg_funcs)
            for agg_name in agg.columns:
                features[f"{prefix}__{col}__rollN{w}__{agg_name}"] = agg[agg_name]
        for w in rollings_time:
            rolled = series.rolling(window=w, min_periods=1, closed="left")
            agg = rolled.agg(agg_funcs)
            for agg_name in agg.columns:
                features[f"{prefix}__{col}__rollT{w}__{agg_name}"] = agg[agg_name]
        for span in ewm_spans:
            ewm = series.ewm(span=span, adjust=False)
            features[f"{prefix}__{col}__ewm{span}__mean"] = ewm.mean()
            if "std" in agg_funcs:
                features[f"{prefix}__{col}__ewm{span}__std"] = ewm.std()

    return pd.DataFrame(features, index=base.index)


def _compute_oof(
    train_df: pd.DataFrame,
    date_col: str,
    group_cols: List[str],
    folds: List[Tuple[Sequence[int], Sequence[int]]],
    num_cols: List[str],
    lags: Sequence[int],
    rollings_count: Sequence[int],
    rollings_time: Sequence[str],
    ewm_spans: Sequence[int],
    agg_funcs: Sequence[str],
    embargo: Optional[pd.Timedelta],
    prefix: str,
) -> pd.DataFrame:
    feature_cols = _feature_columns(
        num_cols=num_cols,
        lags=lags,
        rollings_count=rollings_count,
        rollings_time=rollings_time,
        ewm_spans=ewm_spans,
        agg_funcs=agg_funcs,
        prefix=prefix,
    )
    result = pd.DataFrame(index=train_df.index, columns=feature_cols, dtype="float32")
    embargo = embargo or pd.Timedelta(0)

    for train_idx, valid_idx in folds:
        train_set = train_df.loc[train_idx].copy()
        valid_set = train_df.loc[valid_idx].copy()
        if group_cols:
            train_groups = {name: grp.sort_values(date_col) for name, grp in train_set.groupby(group_cols)}
            for name, vgrp in valid_set.groupby(group_cols):
                base_train = train_groups.get(name)
                if base_train is None:
                    continue
                for idx, row in vgrp.sort_values(date_col).iterrows():
                    cutoff = row[date_col] - embargo
                    history = base_train[base_train[date_col] < cutoff]
                    if history.empty:
                        continue
                    block = _agg_block(
                        history,
                        date_col=date_col,
                        num_cols=num_cols,
                        lags=lags,
                        rollings_count=rollings_count,
                        rollings_time=rollings_time,
                        ewm_spans=ewm_spans,
                        agg_funcs=agg_funcs,
                        prefix=prefix,
                    )
                    result.loc[idx, block.columns] = block.iloc[-1]
        else:
            base_train = train_set.sort_values(date_col)
            for idx, row in valid_set.sort_values(date_col).iterrows():
                cutoff = row[date_col] - embargo
                history = base_train[base_train[date_col] < cutoff]
                if history.empty:
                    continue
                block = _agg_block(
                    history,
                    date_col=date_col,
                    num_cols=num_cols,
                    lags=lags,
                    rollings_count=rollings_count,
                    rollings_time=rollings_time,
                    ewm_spans=ewm_spans,
                    agg_funcs=agg_funcs,
                    prefix=prefix,
                )
                result.loc[idx, block.columns] = block.iloc[-1]

    return result


def _compute_global(
    df: pd.DataFrame,
    date_col: str,
    group_cols: List[str],
    num_cols: List[str],
    lags: Sequence[int],
    rollings_count: Sequence[int],
    rollings_time: Sequence[str],
    ewm_spans: Sequence[int],
    agg_funcs: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    if group_cols:
        features = []
        for _, grp in df.groupby(group_cols, sort=False):
            grp_sorted = grp.sort_values(date_col)
            feat = _agg_block(
                grp_sorted,
                date_col=date_col,
                num_cols=num_cols,
                lags=lags,
                rollings_count=rollings_count,
                rollings_time=rollings_time,
                ewm_spans=ewm_spans,
                agg_funcs=agg_funcs,
                prefix=prefix,
            )
            features.append(feat)
        return pd.concat(features).sort_index()
    grp_sorted = df.sort_values(date_col)
    return _agg_block(
        grp_sorted,
        date_col=date_col,
        num_cols=num_cols,
        lags=lags,
        rollings_count=rollings_count,
        rollings_time=rollings_time,
        ewm_spans=ewm_spans,
        agg_funcs=agg_funcs,
        prefix=prefix,
    )


def build(
    train_df: pd.DataFrame,
    date_col: str,
    group_cols: Optional[Sequence[str]] = None,
    folds: Optional[List[Tuple[Sequence[int], Sequence[int]]]] = None,
    *,
    num_cols: Optional[Sequence[str]] = None,
    lags: Sequence[int] = (1, 7),
    rollings_count: Sequence[int] = (7, 30),
    rollings_time: Sequence[str] = (),
    ewm_spans: Sequence[int] = (),
    agg_funcs: Sequence[str] = ("mean", "std", "min", "max", "sum"),
    embargo: Optional[str] = None,
    prefix: str = "time",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict[str, Any]] = None,
) -> FeaturePackage:
    """Build time-based features without target leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Input dataset to transform.
    date_col : str
        Name of the datetime column used for sorting and windowing.
    group_cols : Sequence[str] or None, optional
        Columns defining independent time series. If ``None``, treat the data as
        a single group.
    folds : list of tuple or None, optional
        Out-of-fold split represented by ``(train_idx, valid_idx)`` pairs. If
        ``None``, aggregates are computed globally.
    num_cols : Sequence[str] or None, optional
        Numerical columns to aggregate. If ``None``, they are auto-detected.
    lags : Sequence[int], default (1, 7)
        Observation shifts applied per column.
    rollings_count : Sequence[int], default (7, 30)
        Rolling windows defined by number of observations.
    rollings_time : Sequence[str], optional
        Rolling windows defined by time offsets (e.g. ``"7D"``).
    ewm_spans : Sequence[int], optional
        Exponential moving window spans.
    agg_funcs : Sequence[str], default ("mean", "std", "min", "max", "sum")
        Aggregations to apply for rolling windows.
    embargo : str or None, optional
        Pandas offset describing embargo duration for OOF mode.
    prefix : str, default "time"
        Prefix for feature names.
    use_cache : bool, default True
        Whether to read/write cached package artifacts.
    cache_key_extra : dict or None, optional
        Additional payload affecting cache key.

    Returns
    -------
    FeaturePackage
        Package with dense features for ``train_df`` and an empty test frame.
    """

    t0 = time.time()
    df = train_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    group_cols_list = list(group_cols) if group_cols else []
    num_cols_list = _select_num_cols(df, date_col, num_cols, group_cols_list)

    embargo_td = pd.Timedelta(embargo) if embargo else None

    params = {
        "date_col": date_col,
        "group_cols": group_cols_list,
        "num_cols": num_cols_list,
        "lags": list(lags),
        "rollings_count": list(rollings_count),
        "rollings_time": list(rollings_time),
        "ewm_spans": list(ewm_spans),
        "agg_funcs": list(agg_funcs),
        "embargo": embargo,
        "oof": folds is not None,
        "prefix": prefix,
    }
    if cache_key_extra:
        params["extra"] = cache_key_extra

    data_stamp = {
        "n": len(df),
        "ts": _hash_timestamps(df[date_col]),
    }
    key = make_key(params, _fingerprint(), data_stamp)

    block = "time_agg"
    if use_cache:
        cached = load_feature_pkg(block, key)
        if cached is not None:
            return cached

    if folds is not None:
        features = _compute_oof(
            df,
            date_col=date_col,
            group_cols=group_cols_list,
            folds=folds,
            num_cols=num_cols_list,
            lags=lags,
            rollings_count=rollings_count,
            rollings_time=rollings_time,
            ewm_spans=ewm_spans,
            agg_funcs=agg_funcs,
            embargo=embargo_td,
            prefix=prefix,
        )
    else:
        features = _compute_global(
            df,
            date_col=date_col,
            group_cols=group_cols_list,
            num_cols=num_cols_list,
            lags=lags,
            rollings_count=rollings_count,
            rollings_time=rollings_time,
            ewm_spans=ewm_spans,
            agg_funcs=agg_funcs,
            prefix=prefix,
        )

    features = features.sort_index()
    test_df = pd.DataFrame(columns=features.columns)

    meta = {
        "name": prefix,
        "params": params,
        "oof": folds is not None,
        "time_sec": round(time.time() - t0, 3),
        "deps": group_cols_list,
    }

    pkg = FeaturePackage(
        name=prefix,
        train=features.astype("float32"),
        test=test_df.astype("float32"),
        kind="dense",
        cols=list(features.columns),
        meta=meta,
    )

    if use_cache:
        save_feature_pkg(block, key, pkg)

    return pkg
