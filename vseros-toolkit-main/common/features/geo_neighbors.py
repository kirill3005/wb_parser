import hashlib
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage

try:
    from sklearn.neighbors import BallTree
except Exception:  # pragma: no cover - handled in build
    BallTree = None


EARTH_RADIUS_M = 6_371_000.0


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _meters_to_radians(distance_m: float) -> float:
    return distance_m / 6_371_000.0
def _to_radians(df: pd.DataFrame, lat_col: str, lon_col: str) -> np.ndarray:
    lat_rad = np.radians(df[lat_col].to_numpy())
    lon_rad = np.radians(df[lon_col].to_numpy())
    return np.column_stack([lat_rad, lon_rad])


def _radius_to_haversine(radius_m: float) -> float:
    return radius_m / EARTH_RADIUS_M


def _compute_neighbor_stats(
    tree: "BallTree", coords: np.ndarray, radius: float, *, subtract_self: bool
) -> Tuple[np.ndarray, np.ndarray]:
    counts = tree.query_radius(coords, r=radius, count_only=True)
    if subtract_self:
        counts = counts - 1
    counts = np.maximum(counts, 0)
    density = counts.astype(float) / (math.pi * (radius * EARTH_RADIUS_M) ** 2)
    return counts, density


def _empty_package(train_df: pd.DataFrame, test_df: pd.DataFrame, meta: Dict) -> FeaturePackage:
    train = pd.DataFrame(index=train_df.index)
    test = pd.DataFrame(index=test_df.index)
    return FeaturePackage(
        name="geo_neighbors",
        train=train,
        test=test,
        kind="dense",
        cols=[],
        meta=meta,
    )


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    *,
    radii_m: Sequence[float] = (300, 1000),
    prefix: str = "geo_nb",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """BallTree(haversine): число соседей в радиусах и плотность."""

    params = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "radii_m": list(radii_m),
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
        cached = load_feature_pkg("geo_neighbors", cache_key)
        if cached is not None:
            return cached

    if lat_col not in train_df.columns or lon_col not in train_df.columns:
        raise ValueError("lat_col or lon_col not found in train_df")
    if lat_col not in test_df.columns or lon_col not in test_df.columns:
        raise ValueError("lat_col or lon_col not found in test_df")

    t0 = time.time()
    meta = {
        "name": "geo_neighbors",
        "params": params,
        "cache_key": cache_key,
        "deps": [],
        "sklearn_available": BallTree is not None,
    }

    if BallTree is None:
        print("[geo_neighbors] sklearn is not available; returning empty package")
        meta["time_sec"] = round(time.time() - t0, 3)
        pkg = _empty_package(train_df, test_df, meta)
        if use_cache:
            save_feature_pkg("geo_neighbors", cache_key, pkg)
        return pkg

    train_coords = _to_radians(train_df, lat_col, lon_col)
    test_coords = _to_radians(test_df, lat_col, lon_col)
    tree = BallTree(train_coords, metric="haversine")

    train_parts: List[pd.Series] = []
    test_parts: List[pd.Series] = []

    for radius_m in radii_m:
        radius = _radius_to_haversine(float(radius_m))
        tr_counts, tr_density = _compute_neighbor_stats(
            tree, train_coords, radius, subtract_self=True
        )
        te_counts, te_density = _compute_neighbor_stats(
            tree, test_coords, radius, subtract_self=False
        )

        count_col = f"{prefix}__{int(radius_m)}m__count"
        dens_col = f"{prefix}__{int(radius_m)}m__density"

        train_parts.append(pd.Series(tr_counts, index=train_df.index, name=count_col))
        train_parts.append(pd.Series(tr_density, index=train_df.index, name=dens_col))
        test_parts.append(pd.Series(te_counts, index=test_df.index, name=count_col))
        test_parts.append(pd.Series(te_density, index=test_df.index, name=dens_col))

    train_feat = pd.concat(train_parts, axis=1)
    test_feat = pd.concat(test_parts, axis=1)

    cols = list(train_feat.columns)
    meta["time_sec"] = round(time.time() - t0, 3)

    pkg = FeaturePackage(
        name="geo_neighbors",
        train=train_feat,
        test=test_feat,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("geo_neighbors", cache_key, pkg)

    return pkg
