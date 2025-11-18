"""Feature interactions for tabular data.

This module implements dense NUM×NUM interactions and optional NUM×CAT hashed
interactions in accordance with the :class:`common.features.types.FeaturePackage`
contract.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from .types import FeaturePackage


def _fingerprint() -> str:
    return "crosses_v1"


def _select_num_cols(df: pd.DataFrame, cols: Optional[Sequence[str]]) -> List[str]:
    if cols is not None:
        return list(cols)
    return [c for c in df.columns if is_numeric_dtype(df[c])]


def _select_cat_cols(df: pd.DataFrame, cols: Optional[Sequence[str]]) -> List[str]:
    if cols is not None:
        return list(cols)
    return [c for c in df.columns if not is_numeric_dtype(df[c])]


def _generate_pairs(num_cols: List[str], whitelist: Optional[Sequence[Tuple[str, str]]], limit: int = 100) -> List[Tuple[str, str]]:
    if whitelist is not None:
        return list(whitelist)
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1 :]:
            pairs.append((a, b))
            if len(pairs) >= limit:
                return pairs
    if len(pairs) > 10000:
        raise ValueError("Too many numeric pairs; provide num_pairs_whitelist or fewer columns")
    return pairs


def _hash_timestamps(df: pd.DataFrame, cols: List[str]) -> str:
    head = df[cols].head(20).astype(str).values.ravel()
    return hashlib.sha1("|".join(head).encode()).hexdigest()[:12]


def _build_dense(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: List[str],
    pairs: List[Tuple[str, str]],
    num_num_ops: Sequence[str],
    safe_div_eps: float,
    clip_ratio: Optional[float],
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    def make_frame(df: pd.DataFrame) -> pd.DataFrame:
        feats: Dict[str, pd.Series] = {}
        for a, b in pairs:
            a_series = df[a].astype(np.float32)
            b_series = df[b].astype(np.float32)
            if "mul" in num_num_ops:
                feats[f"{prefix}__{a}__mul__{b}"] = a_series * b_series
            if "div" in num_num_ops:
                denom_ab = b_series + safe_div_eps
                denom_ba = a_series + safe_div_eps
                div_ab = a_series / denom_ab
                div_ba = b_series / denom_ba
                if clip_ratio is not None:
                    valid_ab = div_ab.dropna()
                    valid_ba = div_ba.dropna()
                    if len(valid_ab):
                        lower, upper = np.percentile(valid_ab, [clip_ratio, 100 - clip_ratio])
                        div_ab = div_ab.clip(lower, upper)
                    if len(valid_ba):
                        lower, upper = np.percentile(valid_ba, [clip_ratio, 100 - clip_ratio])
                        div_ba = div_ba.clip(lower, upper)
                feats[f"{prefix}__{a}__div__{b}"] = div_ab
                feats[f"{prefix}__{b}__div__{a}"] = div_ba
            if "sum" in num_num_ops:
                feats[f"{prefix}__{a}__sum__{b}"] = a_series + b_series
            if "diff" in num_num_ops:
                feats[f"{prefix}__{a}__diff__{b}"] = a_series - b_series
                feats[f"{prefix}__{b}__diff__{a}"] = b_series - a_series
        return pd.DataFrame(feats)

    train_feats = make_frame(train_df)
    test_feats = make_frame(test_df)
    return train_feats, test_feats, list(train_feats.columns)


def _build_sparse_hash(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    hash_buckets: int,
    prefix: str,
    as_sparse: bool,
) -> Tuple[sparse.spmatrix, sparse.spmatrix, List[str]]:
    n_train = len(train_df)
    n_test = len(test_df)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    test_rows: List[int] = []
    test_cols: List[int] = []
    test_data: List[float] = []

    for num_col in num_cols:
        for cat_col in cat_cols:
            for i, (val, cat) in enumerate(zip(train_df[num_col], train_df[cat_col])):
                bucket = hash(cat) % hash_buckets
                rows.append(i)
                cols.append(bucket)
                data.append(float(val))
            for i, (val, cat) in enumerate(zip(test_df[num_col], test_df[cat_col])):
                bucket = hash(cat) % hash_buckets
                test_rows.append(i)
                test_cols.append(bucket)
                test_data.append(float(val))

    train_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_train, hash_buckets))
    test_matrix = sparse.csr_matrix((test_data, (test_rows, test_cols)), shape=(n_test, hash_buckets))

    col_names = [f"{prefix}__h{j}" for j in range(hash_buckets)]

    if not as_sparse:
        train_df_dense = pd.DataFrame(train_matrix.toarray(), columns=col_names)
        test_df_dense = pd.DataFrame(test_matrix.toarray(), columns=col_names)
        return train_df_dense, test_df_dense, col_names

    return train_matrix, test_matrix, col_names


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    num_cols: Optional[Sequence[str]] = None,
    num_num_ops: Sequence[str] = ("mul", "div"),
    num_pairs_whitelist: Optional[Sequence[Tuple[str, str]]] = None,
    safe_div_eps: float = 1e-6,
    clip_ratio: Optional[float] = None,
    cat_cols: Optional[Sequence[str]] = None,
    num_cat: bool = False,
    hash_buckets: int = 2**20,
    as_sparse: bool = True,
    prefix: str = "x",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict[str, Any]] = None,
) -> FeaturePackage:
    """Build interaction features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    test_df : pd.DataFrame
        Test dataframe.
    num_cols : Sequence[str] or None, optional
        Numerical columns to use; detected automatically if ``None``.
    num_num_ops : Sequence[str], default ("mul", "div")
        Operations for NUM×NUM interactions.
    num_pairs_whitelist : Sequence[tuple] or None, optional
        Explicit list of numeric column pairs to process.
    safe_div_eps : float, default 1e-6
        Denominator offset for safe division.
    clip_ratio : float or None, optional
        If provided, clip division results by percentile bounds.
    cat_cols : Sequence[str] or None, optional
        Categorical columns for hashing interactions.
    num_cat : bool, default False
        Whether to include NUM×CAT hashed interactions (sparse).
    hash_buckets : int, default 2**20
        Number of hashing buckets for NUM×CAT.
    as_sparse : bool, default True
        Return sparse matrices for NUM×CAT interactions.
    prefix : str, default "x"
        Feature prefix.
    use_cache : bool, default True
        Whether to read/write cache artifacts.
    cache_key_extra : dict or None, optional
        Extra payload for cache key differentiation.

    Returns
    -------
    FeaturePackage
        Dense or sparse package depending on configuration.
    """

    t0 = time.time()
    num_cols_list = _select_num_cols(train_df, num_cols)
    cat_cols_list = _select_cat_cols(train_df, cat_cols) if num_cat else []

    if num_cat and not cat_cols_list:
        raise ValueError("num_cat=True requires cat_cols")

    if num_cat:
        potential = len(num_cols_list) * len(cat_cols_list) * hash_buckets
        if potential > 5_000_000:
            raise ValueError("NUM×CAT configuration too large; reduce hash_buckets or columns")

    params = {
        "num_cols": num_cols_list,
        "num_num_ops": list(num_num_ops),
        "num_pairs_whitelist": list(num_pairs_whitelist) if num_pairs_whitelist else None,
        "safe_div_eps": safe_div_eps,
        "clip_ratio": clip_ratio,
        "cat_cols": cat_cols_list,
        "num_cat": num_cat,
        "hash_buckets": hash_buckets,
        "as_sparse": as_sparse,
        "prefix": prefix,
    }
    if cache_key_extra:
        params["extra"] = cache_key_extra

    data_stamp = {
        "train": (len(train_df), len(train_df.columns)),
        "test": (len(test_df), len(test_df.columns)),
        "hash": _hash_timestamps(train_df, num_cols_list[:3] if num_cols_list else list(train_df.columns)[:3]),
    }
    key = make_key(params, _fingerprint(), data_stamp)

    block = "crosses_sparse" if num_cat else "crosses_dense"
    if use_cache:
        cached = load_feature_pkg(block, key)
        if cached is not None:
            return cached

    if num_cat:
        train_mat, test_mat, cols = _build_sparse_hash(
            train_df,
            test_df,
            num_cols=num_cols_list,
            cat_cols=cat_cols_list,
            hash_buckets=hash_buckets,
            prefix=prefix,
            as_sparse=as_sparse,
        )
        kind = "sparse" if as_sparse else "dense"
    else:
        pairs = _generate_pairs(num_cols_list, num_pairs_whitelist)
        train_mat, test_mat, cols = _build_dense(
            train_df,
            test_df,
            num_cols=num_cols_list,
            pairs=pairs,
            num_num_ops=num_num_ops,
            safe_div_eps=safe_div_eps,
            clip_ratio=clip_ratio,
            prefix=prefix,
        )
        kind = "dense"

    meta = {
        "name": prefix,
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "oof": False,
        "deps": [],
    }
    if num_cat and not as_sparse:
        meta["note"] = "NUMxNUM not included; NUMxCAT dense expansion"
    elif num_cat:
        meta["note"] = "NUMxNUM omitted; NUMxCAT hashed"

    pkg = FeaturePackage(
        name=prefix,
        train=train_mat,
        test=test_mat,
        kind=kind,
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg(block, key, pkg)

    return pkg
