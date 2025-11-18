from __future__ import annotations

import hashlib
import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage

if importlib.util.find_spec("cv2") is not None:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
else:  # pragma: no cover - optional dependency
    cv2 = None


_DEF_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _hash_mapping(id_to_images: Dict[str, List[str]], limit: int = 200) -> str:
    items = sorted(id_to_images.items())[:limit]
    blob = "|".join(f"{k}:{v[0] if v else ''}" for k, v in items)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _compute_image_stats(path: str) -> Optional[Dict[str, float]]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            stat = ImageStat.Stat(img.convert("HSV"))
            hsv_mean = np.array(stat.mean, dtype=np.float32)
            hsv_std = np.array(stat.stddev, dtype=np.float32)
            arr = np.array(img.convert("L"))
            dark_ratio = float((arr < 30).mean())
            bright_ratio = float((arr > 225).mean())
            lap_var = None
            if cv2 is not None:
                try:
                    lap_var = float(cv2.Laplacian(arr, cv2.CV_64F).var())
                except Exception:
                    lap_var = None
            return {
                "w": float(w),
                "h": float(h),
                "aspect": float(w / h) if h != 0 else 0.0,
                "hsv_h": float(hsv_mean[0]),
                "hsv_s": float(hsv_mean[1]),
                "hsv_v": float(hsv_mean[2]),
                "hsv_h_std": float(hsv_std[0]),
                "hsv_s_std": float(hsv_std[1]),
                "hsv_v_std": float(hsv_std[2]),
                "dark_ratio": dark_ratio,
                "bright_ratio": bright_ratio,
                "lapvar": float(lap_var) if lap_var is not None else np.nan,
            }
    except Exception:
        return None


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    values = {k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float32) for k in keys}
    out: Dict[str, float] = {}
    for k, arr in values.items():
        out[f"{k}_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
        out[f"{k}_max"] = float(np.nanmax(arr)) if arr.size else np.nan
    return out


def _build_features(ids: List[str], id_to_images: Dict[str, List[str]], prefix: str) -> pd.DataFrame:
    records = []
    for sid in ids:
        imgs = id_to_images.get(str(sid), [])
        stats = [_compute_image_stats(p) for p in imgs]
        stats = [s for s in stats if s is not None]
        if stats:
            agg = _aggregate(stats)
        else:
            agg = {}
            for base_name in [
                "w",
                "h",
                "aspect",
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "hsv_h_std",
                "hsv_s_std",
                "hsv_v_std",
                "dark_ratio",
                "bright_ratio",
                "lapvar",
            ]:
                agg[f"{base_name}_mean"] = np.nan
                agg[f"{base_name}_max"] = np.nan
        records.append((sid, agg))

    df = pd.DataFrame({k: v for _, v in records}).T
    df.index = [sid for sid, _ in records]
    df = df.astype(np.float32)
    df.columns = [f"{prefix}__{c}" for c in df.columns]
    return df


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    id_to_images: Dict[str, List[str]],
    *,
    prefix: str = "imgstats",
    use_cache: bool = True,
    cache_key_extra: dict | None = None,
):
    """
    Быстрый fallback: статистики по HSV/размеру/резкости для набора изображений.

    Возвращает FeaturePackage(kind="dense").
    """

    params = {"prefix": prefix, "id_col": id_col}
    data_stamp = {
        "train_ids": len(train_df),
        "test_ids": len(test_df),
        "total_paths": sum(len(v) for v in id_to_images.values()),
        "mapping_hash": _hash_mapping(id_to_images),
    }
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, _fingerprint(), data_stamp)
    if use_cache:
        cached = load_feature_pkg("img_stats", cache_key)
        if cached is not None:
            return cached

    t0 = time.time()
    train_ids = train_df[id_col].astype(str).tolist()
    test_ids = test_df[id_col].astype(str).tolist()

    tr_by_id = _build_features(train_ids, id_to_images, prefix)
    te_by_id = _build_features(test_ids, id_to_images, prefix)

    train = tr_by_id.loc[train_ids].reset_index(drop=True)
    test = te_by_id.loc[test_ids].reset_index(drop=True)

    cols = list(train.columns)
    meta: Dict[str, Any] = {
        "name": "img_stats",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": ["img_index"],
        "cv2": bool(cv2 is not None),
    }

    pkg = FeaturePackage(
        name="img_stats",
        train=train,
        test=test,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("img_stats", cache_key, pkg)
    return pkg
