import hashlib
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _to_bins(values: pd.Series, step_deg: float) -> pd.Series:
    bins = np.floor(values / step_deg)
    bins = pd.Series(bins, index=values.index)
    return bins.fillna(-1).astype(int)


def _meters_to_degrees(step_m: float, lat_ref: float) -> Tuple[float, float]:
    lat_deg = step_m / 111_000
    lon_deg = step_m / (111_000 * max(math.cos(math.radians(lat_ref)), 1e-6))
    return lat_deg, lon_deg


def _select_columns(df: pd.DataFrame, cols: Optional[Sequence[str]]) -> Tuple[str, str]:
    if cols is None:
        raise ValueError("Latitude and longitude columns must be provided explicitly")
    lat_col, lon_col = cols
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("lat_col or lon_col not present in DataFrame")
    return lat_col, lon_col
def _meters_to_degrees_lat(meters: float) -> float:
    return meters / 111_320.0


def _meters_to_degrees_lon(meters: float, ref_lat_deg: float) -> float:
    # avoid division by zero near poles
    denom = 111_320.0 * max(math.cos(math.radians(ref_lat_deg)), 1e-6)
    return meters / denom


def _compute_bins(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    step_lat: float,
    step_lon: float,
) -> pd.DataFrame:
    bins = pd.DataFrame(index=df.index)
    bins["lat_bin"] = np.floor(df[lat_col] / step_lat).astype(int)
    bins["lon_bin"] = np.floor(df[lon_col] / step_lon).astype(int)
    return bins


def _encode_counts(
    train_bins: pd.DataFrame,
    test_bins: pd.DataFrame,
    step_label: str,
    prefix: str,
    n_train: int,
) -> Tuple[pd.Series, pd.Series]:
    keys = list(zip(train_bins["lat_bin"], train_bins["lon_bin"]))
    counts = pd.Series(keys).value_counts()
    mapper = counts.to_dict()

    def map_counts(frame: pd.DataFrame) -> pd.Series:
        tuples = list(zip(frame["lat_bin"], frame["lon_bin"]))
        return pd.Series([mapper.get(t, 0) for t in tuples], index=frame.index)

    tr_counts = map_counts(train_bins)
    te_counts = map_counts(test_bins)

    count_col = f"{prefix}__{step_label}__count"
    ratio_col = f"{prefix}__{step_label}__ratio"

    tr_ratio = tr_counts / float(n_train)
    te_ratio = te_counts / float(n_train)

    return (
        tr_counts.rename(count_col).to_frame()[count_col],
        tr_ratio.rename(ratio_col).to_frame()[ratio_col],
    ), (
        te_counts.rename(count_col).to_frame()[count_col],
        te_ratio.rename(ratio_col).to_frame()[ratio_col],
    )


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    *,
    steps_m: Sequence[float] = (300, 1000),
    prefix: str = "geo",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """Квадратные гео-бины и частоты по ним.

    Переводит заданные метры в градусы, строит бины lat/lon и считает
    количество/долю объектов в каждой клетке. Возвращает FeaturePackage(kind="dense").
    """

    params = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "steps_m": list(steps_m),
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
        cached = load_feature_pkg("geo_grid", cache_key)
        if cached is not None:
            return cached

    if lat_col not in train_df.columns or lon_col not in train_df.columns:
        raise ValueError("lat_col or lon_col not found in train_df")
    if lat_col not in test_df.columns or lon_col not in test_df.columns:
        raise ValueError("lat_col or lon_col not found in test_df")

    t0 = time.time()

    ref_lat = float(pd.concat([train_df[lat_col], test_df[lat_col]]).mean())
    n_train = len(train_df)

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for step in steps_m:
        step_lat = _meters_to_degrees_lat(float(step))
        step_lon = _meters_to_degrees_lon(float(step), ref_lat)
        step_label = f"{int(step)}m"

        tr_bins = _compute_bins(train_df, lat_col, lon_col, step_lat, step_lon)
        te_bins = _compute_bins(test_df, lat_col, lon_col, step_lat, step_lon)

        (tr_c, tr_r), (te_c, te_r) = _encode_counts(
            tr_bins, te_bins, step_label=step_label, prefix=prefix, n_train=n_train
        )
        train_parts.extend([tr_c, tr_r])
        test_parts.extend([te_c, te_r])

    train_feat = pd.concat(train_parts, axis=1)
    test_feat = pd.concat(test_parts, axis=1)

    cols = list(train_feat.columns)
    meta = {
        "name": "geo_grid",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
    }

    pkg = FeaturePackage(
        name="geo_grid",
        train=train_feat,
        test=test_feat,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("geo_grid", cache_key, pkg)

    return pkg
