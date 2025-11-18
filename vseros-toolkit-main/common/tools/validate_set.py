#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/validate_set.py

Предполётная проверка набора фич:
- Наличие и согласованность файлов X_dense/X_sparse/y/id/folds/meta.
- Базовые ошибки (NaN/Inf в таргете, дубли id, несовпадение форм).
- Дрейф train↔test (KS, PSI, TVD), near-constant/дубликаты колонок, высокая корреляция.
- Быстрые утечки (single-column probe), mini-adversarial separability.
- Sparse-статистика (nnz/row, плотность).
- Отчёт (CSV/JSON/PNG/HTML) и итоговый validation_score (0..100).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional heavy deps importing when needed
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import sparse
    from scipy.stats import ks_2samp
except Exception:
    sparse = None
    ks_2samp = None

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Validate feature set under artifacts/sets/<tag>")

    p.add_argument("--tag", required=True, type=str, help="RUN_TAG (dataset/feature set id)")

    p.add_argument("--check-dense", action="store_true",
                   help="Принудительно проверять dense (по умолчанию: если файлы есть)")

    p.add_argument("--check-sparse", action="store_true",
                   help="Принудительно проверять sparse (по умолчанию: если файлы есть)")

    p.add_argument("--sample-rows", type=int, default=300000,
                   help="Лимит строк для детального профайлинга")

    p.add_argument("--max-features", type=int, default=20000,
                   help="Лимит числа колонок для детального профайлинга")

    p.add_argument("--drift", type=str, default="ks,psi",
                   help="Список метрик дрейфа через запятую: ks,psi,tvd")

    p.add_argument("--psi-bins", type=int, default=10, help="Число бинов для PSI (квантильные по train)")

    p.add_argument("--sparse-stats", action="store_true", help="Собирать статистику по sparse")

    p.add_argument("--fold-check", action="store_true", help="Проверить folds.pkl")

    p.add_argument("--leak-probes", action="store_true", help="Быстрые проверки утечки (1-колонные)")

    p.add_argument("--adv-mini", action="store_true", help="Mini adversarial separability (логрег)")

    p.add_argument("--html", action="store_true", help="Сохранить report.html")

    p.add_argument("--fail-on", type=str, default="error", choices=["warn", "error"],
                   help="Уровень, при котором завершаться exit!=0")

    p.add_argument("--name", type=str, default="set", help="Подпапка для результатов в artifacts/validation/<tag>/<name>")

    p.add_argument("--verbose", action="store_true")

    return p.parse_args()

# ----------------------------- FS helpers -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_json(p: Path, obj: dict):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def read_parquet_any(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return pd.read_parquet(p, engine="fastparquet")
        except Exception:
            return None

# ----------------------------- Loaders -----------------------------

def load_dense(tag: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    base = Path("artifacts") / "sets" / tag
    trp = base / "X_dense_train.parquet"
    tep = base / "X_dense_test.parquet"
    Xtr = read_parquet_any(trp) if trp.exists() else None
    Xte = read_parquet_any(tep) if tep.exists() else None
    return Xtr, Xte

def load_sparse(tag: str) -> Tuple[Optional[Any], Optional[Any]]:
    if sparse is None:
        return None, None
    base = Path("artifacts") / "sets" / tag
    trp = base / "X_sparse_train.npz"
    tep = base / "X_sparse_test.npz"
    Xtr = sparse.load_npz(trp) if trp.exists() else None
    Xte = sparse.load_npz(tep) if tep.exists() else None
    return Xtr, Xte

def load_y_ids(tag: str) -> Tuple[pd.DataFrame, pd.Series]:
    base = Path("artifacts") / "sets" / tag
    yp = base / "y_train.parquet"
    ip = base / "ids_test.parquet"
    ydf = read_parquet_any(yp)
    if ydf is None or ydf.shape[1] < 2:
        raise FileNotFoundError("Нужен y_train.parquet с колонками [id, target]")
    idsp = read_parquet_any(ip)
    if idsp is None or idsp.shape[1] < 1:
        raise FileNotFoundError("Нужен ids_test.parquet с колонкой id")
    ids_test = idsp.iloc[:, 0]
    return ydf, ids_test

def load_meta(tag: str) -> dict:
    base = Path("artifacts") / "sets" / tag
    meta = read_json(base / "meta.json") or {}
    return meta

def load_folds(tag: str) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    base = Path("artifacts") / "sets" / tag
    p = base / "folds.pkl"
    if not p.exists():
        return None
    try:
        return pickle.loads(p.read_bytes())
    except Exception:
        return None

# ----------------------------- Utils -----------------------------

def mem_gb_df(df: Optional[pd.DataFrame]) -> float:
    if df is None:
        return 0.0
    try:
        return float(df.memory_usage(deep=True).sum()) / (1024 ** 3)
    except Exception:
        return 0.0

def psi_score(train_vals: np.ndarray, test_vals: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: квантильные бины по train."""
    train_vals = np.asarray(train_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)
    if len(train_vals) == 0 or len(test_vals) == 0:
        return 0.0
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(train_vals[~np.isnan(train_vals)], qs))
    if len(cuts) < 2:
        return 0.0
    # биннинг
    def _hist(x):
        idx = np.clip(np.searchsorted(cuts, x, side="right") - 1, 0, len(cuts) - 2)
        # пропустим NaN как отдельную корзину (не учитываем в PSI)
        mask = ~np.isnan(x)
        cnt = np.bincount(idx[mask], minlength=len(cuts) - 1).astype(float)
        return cnt / max(cnt.sum(), 1.0)
    pt = _hist(train_vals)
    ps = _hist(test_vals)
    eps = 1e-12
    psi = float(np.sum((pt - ps) * np.log((pt + eps) / (ps + eps))))
    return psi

def tvd_categorical(train: Sequence, test: Sequence) -> float:
    """Total Variation Distance для категорий."""
    s1 = pd.Series(train).astype("object").value_counts(normalize=True)
    s2 = pd.Series(test).astype("object").value_counts(normalize=True)
    un = set(s1.index).union(set(s2.index))
    tv = 0.0
    for k in un:
        tv += abs(float(s1.get(k, 0.0)) - float(s2.get(k, 0.0)))
    return float(tv) / 2.0

def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None:
        return df
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)

def top_n(lst: List[Tuple[Any, float]], n: int) -> List[Tuple[Any, float]]:
    return sorted(lst, key=lambda x: -x[1])[:n]

# ----------------------------- Basic passports -----------------------------

def dataset_passport(Xd_tr, Xd_te, Xs_tr, Xs_te) -> dict:
    return {
        "dense_train_shape": None if Xd_tr is None else [int(x) for x in Xd_tr.shape],
        "dense_test_shape": None if Xd_te is None else [int(x) for x in Xd_te.shape],
        "sparse_train_shape": None if Xs_tr is None else [int(x) for x in Xs_tr.shape],
        "sparse_test_shape": None if Xs_te is None else [int(x) for x in Xs_te.shape],
        "dense_train_mem_gb": mem_gb_df(Xd_tr),
        "dense_test_mem_gb": mem_gb_df(Xd_te),
    }

def target_stats(ydf: pd.DataFrame) -> dict:
    id_col = ydf.columns[0]
    tgt_col = ydf.columns[1]
    y = ydf[tgt_col].values
    out = {"id_col": id_col, "target_col": tgt_col, "n_train": int(len(ydf))}
    try:
        y_float = pd.to_numeric(ydf[tgt_col], errors="coerce")
        uniq = pd.unique(ydf[tgt_col])
        out["unique_values"] = int(len(uniq))
        if set(pd.unique(y_float.dropna())) <= {0.0, 1.0} and len(set(y_float.dropna())) <= 2:
            pos_rate = float((y_float == 1.0).mean())
            out.update({"task_guess": "binary", "pos_rate": pos_rate})
        elif pd.api.types.is_integer_dtype(ydf[tgt_col]) and len(uniq) < 1000:
            out.update({"task_guess": "multiclass", "n_classes": int(len(uniq))})
        else:
            out.update({"task_guess": "regression"})
    except Exception:
        pass
    out["nan_in_target"] = bool(pd.isna(ydf[tgt_col]).any())
    return out

def id_checks(train_ids: pd.Series, test_ids: pd.Series) -> dict:
    res = {
        "train_id_unique": bool(train_ids.is_unique),
        "test_id_unique": bool(test_ids.is_unique),
        "train_test_intersection": int(len(set(train_ids) & set(test_ids))),
        "n_train_ids": int(len(train_ids)),
        "n_test_ids": int(len(test_ids)),
    }
    return res

# ----------------------------- Dense columns profiling -----------------------------

def profile_dense(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    ydf: pd.DataFrame,
    sample_rows: int,
    max_features: int,
    drift_metrics: List[str],
    psi_bins: int,
    verbose: bool = False,
    plots_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Возвращает таблицу schema_df со столбцами:
    name, dtype, nan_rate_tr, nan_rate_te, const_tr, const_te,
    nunique_tr, nunique_te, ks, psi, tvd, mean_tr, mean_te, std_tr, std_te,
    suspected_leak, high_corr_flag (по выборке), duplicate_hash (повторы)
    и summary словарь с агрегатами.
    """
    if Xtr is None or Xte is None:
        return pd.DataFrame(), {}

    # ограничение по строкам и колонкам
    Xtr_s = sample_df(Xtr, sample_rows)
    Xte_s = sample_df(Xte, sample_rows)
    cols = list(Xtr.columns)
    if len(cols) > max_features:
        if verbose:
            print(f"[info] truncate features for profiling: {len(cols)} -> {max_features}")
        cols = cols[:max_features]
        Xtr_s = Xtr_s[cols]
        Xte_s = Xte_s[cols]

    # duplicate columns via hash (на сэмпле для скорости)
    try:
        # хэш векторов колонок
        dup_map = {}
        col_hash = {}
        for c in cols:
            h = pd.util.hash_pandas_object(Xtr_s[c], index=False).sum()
            if h in dup_map:
                dup_map[h].append(c)
            else:
                dup_map[h] = [c]
            col_hash[c] = int(h)
        duplicate_groups = [v for v in dup_map.values() if len(v) > 1]
        duplicate_set = set([c for g in duplicate_groups for c in g])
    except Exception:
        col_hash = {c: 0 for c in cols}
        duplicate_set = set()

    # suspected leak by name
    meta_sus_names = {"id", "label", "target", "y", "date", "datetime", "timestamp"}
    suspected_name = {c: any(k in str(c).lower() for k in meta_sus_names) for c in cols}

    rows = []
    ks_values = []
    psi_values = []
    strong_drift_feats: List[Tuple[str, float]] = []
    numeric_cols = []
    # gather dtype kinds
    for c in cols:
        s_tr = Xtr_s[c]
        s_te = Xte_s[c]
        dtype = str(s_tr.dtype)

        nan_rate_tr = float(pd.isna(s_tr).mean())
        nan_rate_te = float(pd.isna(s_te).mean())

        nunq_tr = int(pd.Series(s_tr).nunique(dropna=True))
        nunq_te = int(pd.Series(s_te).nunique(dropna=True))

        const_tr = bool(nunq_tr <= 1)
        const_te = bool(nunq_te <= 1)

        mean_tr = float(pd.to_numeric(s_tr, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(s_tr) else np.nan
        mean_te = float(pd.to_numeric(s_te, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(s_te) else np.nan
        std_tr = float(pd.to_numeric(s_tr, errors="coerce").std()) if pd.api.types.is_numeric_dtype(s_tr) else np.nan
        std_te = float(pd.to_numeric(s_te, errors="coerce").std()) if pd.api.types.is_numeric_dtype(s_te) else np.nan

        ks_val = np.nan
        psi_val = np.nan
        tvd_val = np.nan

        if pd.api.types.is_numeric_dtype(s_tr):
            numeric_cols.append(c)
            if "ks" in drift_metrics and ks_2samp is not None:
                try:
                    ks_stat = ks_2samp(pd.to_numeric(s_tr, errors="coerce").dropna(),
                                       pd.to_numeric(s_te, errors="coerce").dropna()).statistic
                    ks_val = float(ks_stat)
                    ks_values.append(ks_val)
                except Exception:
                    ks_val = np.nan
            if "psi" in drift_metrics:
                try:
                    psi_val = psi_score(pd.to_numeric(s_tr, errors="coerce").values,
                                        pd.to_numeric(s_te, errors="coerce").values,
                                        bins=psi_bins)
                    psi_values.append(psi_val)
                except Exception:
                    psi_val = np.nan
        else:
            if "tvd" in drift_metrics:
                try:
                    tvd_val = tvd_categorical(s_tr.values, s_te.values)
                except Exception:
                    tvd_val = np.nan

        suspected = bool(suspected_name[c])
        rows.append(dict(
            name=c, dtype=dtype,
            nan_rate_tr=nan_rate_tr, nan_rate_te=nan_rate_te,
            const_tr=const_tr, const_te=const_te,
            nunique_tr=nunq_tr, nunique_te=nunq_te,
            mean_tr=mean_tr, mean_te=mean_te,
            std_tr=std_tr, std_te=std_te,
            ks=ks_val, psi=psi_val, tvd=tvd_val,
            suspected_leak=suspected,
            duplicate_hash=col_hash.get(c, 0)
        ))

    schema_df = pd.DataFrame(rows)

    # high correlations (numeric only) — по сэмплу, верхняя часть
    high_corr_pairs = []
    high_corr_cols = set()
    if len(numeric_cols) >= 2:
        try:
            C = Xtr_s[numeric_cols].corr().abs()
            np.fill_diagonal(C.values, 0.0)
            mask = (C >= 0.995)
            idxs = np.where(mask.values)
            for i, j in zip(idxs[0], idxs[1]):
                if i < j:
                    c1 = C.index[i]; c2 = C.columns[j]
                    high_corr_pairs.append((c1, c2, float(C.values[i, j])))
                    high_corr_cols.add(c1); high_corr_cols.add(c2)
        except Exception:
            pass

    schema_df["high_corr_flag"] = schema_df["name"].isin(high_corr_cols)

    # strong drift summary
    strong_drift = []
    if "psi" in drift_metrics:
        strong_drift = [(r["name"], r["psi"]) for _, r in schema_df.iterrows() if pd.notna(r["psi"]) and r["psi"] >= 0.25]
    if "ks" in drift_metrics:
        strong_drift += [(r["name"], r["ks"]) for _, r in schema_df.iterrows() if pd.notna(r["ks"]) and r["ks"] >= 0.5]
    strong_drift = top_n(strong_drift, 20)

    # plots for top-drift numerics
    if plots_dir is not None and plt is not None and len(strong_drift) > 0:
        ensure_dir(plots_dir)
        for feat, _v in strong_drift[:8]:
            try:
                a = pd.to_numeric(Xtr_s[feat], errors="coerce").dropna().values
                b = pd.to_numeric(Xte_s[feat], errors="coerce").dropna().values
                if len(a) == 0 or len(b) == 0:
                    continue
                fig = plt.figure()
                plt.hist(a, bins=50, alpha=0.5, density=True, label="train")
                plt.hist(b, bins=50, alpha=0.5, density=True, label="test")
                plt.title(f"Drift hist: {feat}")
                plt.legend()
                fig.savefig(plots_dir / f"drift_hist__{str(feat)[:80]}.png", bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

    # duplicates report
    duplicate_groups = {}
    for h, group in schema_df.groupby("duplicate_hash"):
        gs = list(group["name"])
        if h != 0 and len(gs) > 1:
            duplicate_groups[int(h)] = gs

    summary = {
        "n_features_profiled": int(len(schema_df)),
        "near_constant_share": float(np.mean((schema_df["nunique_tr"] <= 2) | (schema_df["nunique_te"] <= 2))) if len(schema_df) else 0.0,
        "duplicate_groups": duplicate_groups,
        "strong_drift_top": strong_drift,
        "high_corr_pairs_top": high_corr_pairs[:50],
        "psi_mean": float(np.nanmean(schema_df["psi"])) if "psi" in drift_metrics and len(schema_df) else None,
        "ks_mean": float(np.nanmean(schema_df["ks"])) if "ks" in drift_metrics and len(schema_df) else None,
        "strong_drift_share": float(np.mean(schema_df["psi"] >= 0.25)) if "psi" in drift_metrics and len(schema_df) else 0.0,
    }
    return schema_df, summary

# ----------------------------- Rows & duplicates -----------------------------

def duplicate_rows_dense(Xtr: Optional[pd.DataFrame]) -> pd.DataFrame:
    if Xtr is None or Xtr.empty:
        return pd.DataFrame()
    try:
        dup_mask = Xtr.duplicated(keep=False)
        dups = Xtr[dup_mask].copy()
        dups.insert(0, "row_index", dups.index)
        return dups
    except Exception:
        # fallback: быстрый хэш по подмножеству колонок
        cols = list(Xtr.columns)[: min(50, len(Xtr.columns))]
        sig = pd.util.hash_pandas_object(Xtr[cols].astype("object"), index=False)
        dup_mask = sig.duplicated(keep=False)
        dups = Xtr[dup_mask].copy()
        dups.insert(0, "row_index", dups.index)
        return dups

# ----------------------------- Folds -----------------------------

def validate_folds(folds: Optional[List[Tuple[np.ndarray, np.ndarray]]], n_train: int) -> Tuple[pd.DataFrame, dict, List[str]]:
    if folds is None:
        return pd.DataFrame(), {"has_folds": False}, []
    errors = []
    rows = []
    covered = np.zeros(n_train, dtype=int)
    for k, (tr, va) in enumerate(folds):
        tr = np.asarray(tr, dtype=int); va = np.asarray(va, dtype=int)
        rows.append(dict(fold=k, n_train=len(tr), n_val=len(va)))
        # пересечения валов между собой проверим позже
        covered[va] += 1
        if len(np.intersect1d(tr, va)) > 0:
            errors.append(f"fold {k}: train∩val != ∅")
        if np.any(va < 0) or np.any(va >= n_train):
            errors.append(f"fold {k}: val indices out of range")
    # покрытие
    miss = int(np.sum(covered == 0))
    multi = int(np.sum(covered > 1))
    summ = {"has_folds": True, "val_uncovered": miss, "val_multi_covered": multi}
    # проверим пересечения валов
    val_sets = [set(va.tolist()) for _, va in folds]
    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            if len(val_sets[i] & val_sets[j]) > 0:
                errors.append(f"folds {i} and {j} have intersecting val indices")
    return pd.DataFrame(rows), summ, errors

# ----------------------------- Leak probes -----------------------------

def leak_single_column_probes(Xtr: Optional[pd.DataFrame], ydf: pd.DataFrame, kfold: int = 5) -> Tuple[List[str], dict]:
    if Xtr is None or Xtr.empty:
        return [], {}
    tgt_col = ydf.columns[1]
    y = pd.to_numeric(ydf[tgt_col], errors="coerce").values
    # только числовые фичи
    num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    res = []
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    suspicious = []
    aucs = {}
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    # ограничим до 200 признаков по abs(corr)
    cors = []
    for c in num_cols:
        x = pd.to_numeric(Xtr[c], errors="coerce").fillna(0.0).values
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        cors.append((c, abs(corr)))
    cors.sort(key=lambda t: -t[1])
    probe_cols = [c for c, _ in cors[: min(200, len(cors))]]
    for c in probe_cols:
        x = pd.to_numeric(Xtr[c], errors="coerce").fillna(0.0).values.reshape(-1, 1)
        fold_scores = []
        for tr, va in kf.split(np.arange(len(y))):
            try:
                clf = LogisticRegression(max_iter=300, n_jobs=1, class_weight="balanced")
            except TypeError:
                clf = LogisticRegression(max_iter=300, class_weight="balanced")
            clf.fit(x[tr], y[tr])
            p = clf.predict_proba(x[va])[:, 1]
            score = roc_auc_score(y[va], p)
            fold_scores.append(score)
        auc = float(np.mean(fold_scores))
        aucs[c] = auc
        if auc >= 0.95:
            suspicious.append(c)
    return suspicious, {"probed": probe_cols, "aucs_top": sorted(aucs.items(), key=lambda t: -t[1])[:50]}

# ----------------------------- Adversarial mini -----------------------------

def mini_adversarial_auc(
    Xtr: Optional[pd.DataFrame],
    Xte: Optional[pd.DataFrame],
    schema_df: pd.DataFrame,
    max_feats: int = 200,
    sample_rows: int = 200000
) -> Optional[float]:
    if Xtr is None or Xte is None or Xtr.empty or Xte.empty:
        return None
    # выберем топ признаков по PSI/KS
    cand_cols = []
    if "psi" in schema_df.columns:
        cand_cols += [(r["name"], 0 if pd.isna(r["psi"]) else float(r["psi"])) for _, r in schema_df.iterrows()]
    if "ks" in schema_df.columns:
        cand_cols += [(r["name"], 0 if pd.isna(r["ks"]) else float(r["ks"])) for _, r in schema_df.iterrows()]
    if not cand_cols:
        return None
    scores = {}
    for n, v in cand_cols:
        scores[n] = max(scores.get(n, 0.0), v)
    top_cols = [k for k, _ in sorted(scores.items(), key=lambda t: -t[1])[: min(max_feats, len(scores))]]
    if not top_cols:
        return None
    A = sample_df(Xtr[top_cols], sample_rows)
    B = sample_df(Xte[top_cols], sample_rows)
    n1, n0 = len(A), len(B)
    y = np.concatenate([np.ones(n1), np.zeros(n0)])
    X = pd.concat([A, B], axis=0).fillna(0.0).values
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    try:
        clf = LogisticRegression(max_iter=300, n_jobs=1, class_weight="balanced")
    except TypeError:
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_va, p))

# ----------------------------- Sparse stats -----------------------------

def sparse_profile(Xs_tr, Xs_te) -> dict:
    if Xs_tr is None and Xs_te is None:
        return {}
    out = {}
    if Xs_tr is not None:
        nnz_row = np.asarray(Xs_tr.getnnz(axis=1)).reshape(-1)
        out["train_shape"] = [int(x) for x in Xs_tr.shape]
        out["train_density"] = float(Xs_tr.nnz / (Xs_tr.shape[0] * Xs_tr.shape[1]))
        out["train_nnz_per_row_p50"] = float(np.percentile(nnz_row, 50))
        out["train_nnz_per_row_p90"] = float(np.percentile(nnz_row, 90))
        out["train_nnz_per_row_p99"] = float(np.percentile(nnz_row, 99))
        out["train_zero_rows_share"] = float(np.mean(nnz_row == 0))
    if Xs_te is not None:
        nnz_row = np.asarray(Xs_te.getnnz(axis=1)).reshape(-1)
        out["test_shape"] = [int(x) for x in Xs_te.shape]
        out["test_density"] = float(Xs_te.nnz / (Xs_te.shape[0] * Xs_te.shape[1]))
        out["test_nnz_per_row_p50"] = float(np.percentile(nnz_row, 50))
        out["test_nnz_per_row_p90"] = float(np.percentile(nnz_row, 90))
        out["test_nnz_per_row_p99"] = float(np.percentile(nnz_row, 99))
        out["test_zero_rows_share"] = float(np.mean(nnz_row == 0))
    return out

def plot_nnz_hist(Xs, path: Path, title: str):
    if plt is None or Xs is None:
        return
    ensure_dir(path.parent)
    nnz_row = np.asarray(Xs.getnnz(axis=1)).reshape(-1)
    fig = plt.figure()
    plt.hist(nnz_row, bins=50)
    plt.title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# ----------------------------- Scoring & recommendations -----------------------------

def compute_validation_score(summary: dict) -> float:
    score = 100.0
    # ошибки и предупреждения будут привязаны в main (errors_count, warns_count)
    strong_drift_share = float(summary.get("strong_drift_share", 0.0))
    adv_auc = summary.get("adversarial_auc")
    near_const = float(summary.get("near_constant_share", 0.0))
    dup_share = float(summary.get("duplicate_cols_share", 0.0))
    fold_badness = float(summary.get("fold_badness", 0.0))

    score -= 20.0 * min(1.0, strong_drift_share)
    if adv_auc is not None:
        score -= 15.0 * max(0.0, (adv_auc - 0.5) * 2.0)  # 0.5→0 штраф, 1.0→15
    score -= 10.0 * min(1.0, near_const)
    score -= 10.0 * min(1.0, dup_share)
    score -= 15.0 * min(1.0, fold_badness)
    score = float(np.clip(score, 0.0, 100.0))
    return score

def build_recommendations(
    errors: List[str],
    warnings: List[str],
    schema_df: pd.DataFrame,
    folds_info: dict,
    summary: dict
) -> List[str]:
    recs = []
    recs += [f"FIX: {e}" for e in errors]
    # near-constant
    if not schema_df.empty:
        near_const = schema_df[(schema_df["nunique_tr"] <= 2) | (schema_df["nunique_te"] <= 2)]
        if len(near_const) > 0:
            recs.append(f"Drop near-constant features: {', '.join(list(near_const['name'].head(30)))}")
        # duplicates
        dup_groups = summary.get("duplicate_groups", {})
        for h, cols in list(dup_groups.items())[:5]:
            recs.append(f"Duplicate columns (hash={h}): {', '.join(cols[:10])}")
        # strong drift
        for feat, val in summary.get("strong_drift_top", [])[:10]:
            recs.append(f"Strong drift: {feat} (score={val:.3f}) — пересчёт блока/нормализация/регуляризация")
        # high corr
        for c1, c2, corr in summary.get("high_corr_pairs_top", [])[:10]:
            recs.append(f"High correlation: {c1} ~ {c2} (|r|={corr:.3f}) — убрать один из пары")

    # folds
    if folds_info.get("has_folds"):
        if folds_info.get("val_uncovered", 0) > 0:
            recs.append("Folds: есть строки, не попавшие ни в один вал.")
        if folds_info.get("val_multi_covered", 0) > 0:
            recs.append("Folds: перекрытия валидационных индексов между фолдами.")

    # extras
    adv_auc = summary.get("adversarial_auc")
    if adv_auc is not None and adv_auc > 0.8:
        recs.append(f"Adversarial AUC={adv_auc:.3f} — сильный дрейф train/test. Пересмотреть генерацию фич/сплиты.")
    return recs

# ----------------------------- HTML report -----------------------------

def build_html_report(out_dir: Path, summary: dict, schema_df: pd.DataFrame, folds_df: pd.DataFrame,
                      errors: List[str], warnings: List[str]) -> None:
    ensure_dir(out_dir)
    html = io.StringIO()
    html.write("<html><head><meta charset='utf-8'><title>Validation Report</title></head><body>")
    html.write("<h1>Feature Set Validation Report</h1>")

    html.write("<h2>Status</h2>")
    html.write(f"<p><b>Validation score:</b> {summary.get('validation_score', 'n/a')}</p>")
    if errors:
        html.write("<h3 style='color:red'>Errors</h3><ul>")
        for e in errors:
            html.write(f"<li>{e}</li>")
        html.write("</ul>")
    if warnings:
        html.write("<h3 style='color:orange'>Warnings</h3><ul>")
        for w in warnings:
            html.write(f"<li>{w}</li>")
        html.write("</ul>")

    html.write("<h2>Summary</h2><pre>")
    html.write(json.dumps(summary, ensure_ascii=False, indent=2))
    html.write("</pre>")

    if not schema_df.empty:
        html.write("<h2>Schema (head)</h2>")
        html.write(schema_df.head(30).to_html(index=False))

    if not folds_df.empty:
        html.write("<h2>Folds</h2>")
        html.write(folds_df.to_html(index=False))

    plots_dir = out_dir / "plots"
    if plots_dir.exists():
        html.write("<h2>Drift plots</h2>")
        for p in sorted(plots_dir.glob("drift_hist__*.png"))[:16]:
            html.write(f"<div><img src='plots/{p.name}' width='480'></div>")

    # sparse
    nnz_tr = plots_dir / "sparse_nnz_train.png"
    nnz_te = plots_dir / "sparse_nnz_test.png"
    if nnz_tr.exists() or nnz_te.exists():
        html.write("<h2>Sparse nnz/row</h2>")
        if nnz_tr.exists():
            html.write(f"<div><img src='plots/{nnz_tr.name}' width='480'></div>")
        if nnz_te.exists():
            html.write(f"<div><img src='plots/{nnz_te.name}' width='480'></div>")

    html.write("</body></html>")
    (out_dir / "report.html").write_text(html.getvalue(), encoding="utf-8")

# ----------------------------- Main orchestration -----------------------------

def main():
    args = parse_args()
    tag = args.tag
    out_dir = Path("artifacts") / "validation" / tag / args.name
    ensure_dir(out_dir)
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    # Drift metrics parse
    drift_metrics = [x.strip().lower() for x in args.drift.split(",") if x.strip()]
    drift_metrics = [m for m in drift_metrics if m in {"ks", "psi", "tvd"}]
    if not drift_metrics:
        drift_metrics = ["ks", "psi"]

    # Load artifacts
    meta = load_meta(tag)
    Xd_tr, Xd_te = load_dense(tag)
    Xs_tr, Xs_te = load_sparse(tag)
    ydf, ids_test = load_y_ids(tag)
    folds = load_folds(tag)

    # Should we check dense/sparse?
    do_dense = args.check_dense or (Xd_tr is not None and Xd_te is not None)
    do_sparse = args.check_sparse or (Xs_tr is not None and Xs_te is not None)

    errors: List[str] = []
    warnings_list: List[str] = []

    # Basic presence
    if not do_dense and not do_sparse:
        errors.append("Нет ни dense, ни sparse матриц (X_dense_* / X_sparse_*)")
    if ydf is None or ids_test is None:
        errors.append("Не найдены y_train.parquet или ids_test.parquet")

    # Basic passport
    passport = dataset_passport(Xd_tr, Xd_te, Xs_tr, Xs_te)
    save_json(out_dir / "passport.json", passport)

    # ID & target checks
    id_col = ydf.columns[0]
    tgt_col = ydf.columns[1]
    train_ids = ydf[id_col]
    id_res = id_checks(train_ids, ids_test)
    if not id_res["train_id_unique"]:
        errors.append("train id не уникальны")
    if not id_res["test_id_unique"]:
        errors.append("test id не уникальны")
    if id_res["train_test_intersection"] > 0:
        errors.append(f"Пересечение train/test id: {id_res['train_test_intersection']}")
    save_json(out_dir / "id_checks.json", id_res)

    tstats = target_stats(ydf)
    if tstats.get("nan_in_target", False):
        errors.append("В таргете есть NaN")
    if tstats.get("task_guess") == "binary":
        pr = tstats.get("pos_rate", 0.0)
        if pr < 0.01 or pr > 0.99:
            warnings_list.append(f"Сильный дисбаланс таргета: pos_rate={pr:.4f}")
    save_json(out_dir / "target_stats.json", tstats)

    # Shape checks
    if do_dense:
        if Xd_tr is None or Xd_te is None:
            errors.append("Отсутствуют dense файлы")
        else:
            if Xd_tr.shape[1] != Xd_te.shape[1]:
                errors.append(f"Dense: разное число колонок train={Xd_tr.shape[1]} vs test={Xd_te.shape[1]}")
            if list(Xd_tr.columns) != list(Xd_te.columns):
                warnings_list.append("Dense: порядок/имена колонок train/test различаются (будет приведено в assemble)")
    if do_sparse and sparse is not None:
        if Xs_tr is None or Xs_te is None:
            errors.append("Отсутствуют sparse файлы")
        else:
            if Xs_tr.shape[1] != Xs_te.shape[1]:
                errors.append(f"Sparse: разная размерность по колонкам train={Xs_tr.shape[1]} vs test={Xs_te.shape[1]}")

    # Dense profiling
    schema_df = pd.DataFrame()
    dense_summary = {}
    if do_dense and Xd_tr is not None and Xd_te is not None:
        schema_df, dense_summary = profile_dense(
            Xd_tr, Xd_te, ydf,
            sample_rows=args.sample_rows,
            max_features=args.max_features,
            drift_metrics=drift_metrics,
            psi_bins=args.psi_bins,
            verbose=args.verbose,
            plots_dir=plots_dir
        )
        if not schema_df.empty:
            schema_df.to_csv(out_dir / "schema.csv", index=False)
            # duplicate cols share
            dup_groups = dense_summary.get("duplicate_groups", {})
            total_profiled = max(1, len(schema_df))
            dup_cols = set()
            for cols in dup_groups.values():
                dup_cols.update(cols)
            dense_summary["duplicate_cols_share"] = float(len(dup_cols) / total_profiled)
        save_json(out_dir / "drift_global.json", dense_summary)

    # Duplicate rows (dense only)
    dups_df = duplicate_rows_dense(Xd_tr) if do_dense else pd.DataFrame()
    if not dups_df.empty:
        dups_df.head(2000).to_csv(out_dir / "duplicates.csv", index=False)
        warnings_list.append(f"Найдены дубликаты строк: {len(dups_df)} (sample saved)")

    # Folds
    folds_df, folds_info, fold_errs = validate_folds(folds, n_train=len(ydf))
    if not folds_df.empty:
        folds_df.to_csv(out_dir / "folds.csv", index=False)
    if fold_errs:
        errors.extend(fold_errs)

    # Leak probes
    leak_info = {}
    suspicious_cols = []
    if args.leak_probes and do_dense and Xd_tr is not None:
        try:
            suspicious_cols, leak_info = leak_single_column_probes(Xd_tr, ydf)
            if suspicious_cols:
                warnings_list.append(f"Подозрение на утечку (1-колонные AUC>=0.95): {', '.join(suspicious_cols[:10])}")
        except Exception as e:
            warnings_list.append(f"Leak probes error: {e}")
    save_json(out_dir / "leak_probes.json", {"suspicious_cols": suspicious_cols, **leak_info})

    # Adversarial mini
    adv_auc = None
    if args.adv-mini if False else args.adv_mini:  # guard for hyphen typo
        pass
    if args.adv_mini:
        try:
            adv_auc = mini_adversarial_auc(Xd_tr, Xd_te, schema_df) if do_dense else None
        except Exception as e:
            warnings_list.append(f"Adversarial mini error: {e}")

    # Sparse stats
    sparse_stats = {}
    if args.sparse_stats and do_sparse and sparse is not None:
        try:
            sparse_stats = sparse_profile(Xs_tr, Xs_te)
            save_json(out_dir / "sparse_stats.json", sparse_stats)
            if plt is not None:
                if Xs_tr is not None:
                    plot_nnz_hist(Xs_tr, plots_dir / "sparse_nnz_train.png", "Sparse nnz/row train")
                if Xs_te is not None:
                    plot_nnz_hist(Xs_te, plots_dir / "sparse_nnz_test.png", "Sparse nnz/row test")
        except Exception as e:
            warnings_list.append(f"Sparse stats error: {e}")

    # Dense stats (basic)
    dense_stats = passport
    save_json(out_dir / "dense_stats.json", dense_stats)

    # Aggregate summary
    summary = {
        "tag": tag,
        "created_at": datetime.now().isoformat(),
        "drift_metrics": drift_metrics,
        "psi_bins": args.psi_bins,
        "near_constant_share": dense_summary.get("near_constant_share", 0.0),
        "strong_drift_share": dense_summary.get("strong_drift_share", 0.0),
        "duplicate_cols_share": dense_summary.get("duplicate_cols_share", 0.0),
        "adversarial_auc": adv_auc,
        "fold_badness": float((folds_info.get("val_uncovered", 0) > 0) or (folds_info.get("val_multi_covered", 0) > 0)),
        "has_dense": bool(do_dense and Xd_tr is not None),
        "has_sparse": bool(do_sparse and Xs_tr is not None),
    }

    # Errors & warnings files
    save_json(out_dir / "errors.json", {"errors": errors})
    save_json(out_dir / "warnings.json", {"warnings": warnings_list})

    # Recommendations
    recs = build_recommendations(errors, warnings_list, schema_df, folds_info, summary)
    save_json(out_dir / "recommendations.json", {"recommendations": recs})

    # Validation score
    vscore = compute_validation_score(summary)
    summary["validation_score"] = vscore
    save_json(out_dir / "summary.json", summary)

    # HTML report
    if args.html:
        try:
            build_html_report(out_dir, summary, schema_df, folds_df, errors, warnings_list)
        except Exception as e:
            warnings_list.append(f"HTML build error: {e}")
            save_json(out_dir / "warnings.json", {"warnings": warnings_list})

    # Update index
    idx_path = Path("artifacts") / "sets" / "index.json"
    idx = read_json(idx_path) or {}
    idx_key = f"validate:{tag}:{args.name}"
    idx[idx_key] = {"path": out_dir.as_posix(), "score": vscore, "created_at": datetime.now().isoformat()}
    save_json(idx_path, idx)

    # Print summary
    print("=== VALIDATION COMPLETED ===")
    print("dir :", out_dir.as_posix())
    print("score:", vscore)
    if errors:
        print("errors:", len(errors))
    if warnings_list:
        print("warnings:", len(warnings_list))

    # Exit code by policy
    if args.fail_on == "error" and errors:
        sys.exit(1)
    if args.fail_on == "warn" and (errors or warnings_list):
        sys.exit(1)

if __name__ == "__main__":
    main()
