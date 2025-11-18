#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/stability.py

Лаборатория устойчивости моделей поверх уже посчитанных артефактов:

Входы:
- artifacts/sets/<tag>/
    y_train.parquet           # [id, target]
    folds.pkl                 # список (train_idx, val_idx) или совместимый список, опц.
    meta.json                 # опц., для срезов/каталога
    (опц.) train_extra.parquet  # дополнительные колонки для срезов (time/cat/num)
- artifacts/models/<run_id>/
    oof.npy                   # вероятности/логиты/прогнозы на train OOF
    (опц.) test_pred.npy
    (опц.) importances.csv    # feature, importance
    (опц.) metrics.json
- artifacts/adversarial/<tag>/train_weights.npy (опц.)

Выходы:
- artifacts/stability/<tag>/<name>/
  metrics.json, per_fold.csv, bootstrap_ci.csv, slices.csv, calibration.csv,
  rank_agreement.csv, feature_jaccard.csv, with_vs_without_weights.csv,
  plots/*.png, report.html

CLI примеры:
  Одиночный прогон:
    python tools/stability.py --tag s5e11 --runs lgbm_num_42 \
      --task binary --metric roc_auc --bootstrap 2000 --fold-stats \
      --slices time:row_id:bins=5,cat:state:top=10,num:amount:bins=5 \
      --use-adv-weights --save-html --verbose

  Сравнение двух прогонов + Jaccard по фичам:
    python tools/stability.py --tag s5e11 --runs lgbm_num_42,cat_te_7 \
      --task binary --metric roc_auc --bootstrap 1500 --fold-stats \
      --feature-jaccard @50,@100 --save-html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# optional SciPy
try:
    import scipy.stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# matplotlib (без seaborn)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Model stability lab over precomputed artifacts")

    p.add_argument("--tag", required=True, type=str, help="RUN_TAG (набор фич/датасет)")

    p.add_argument("--runs", required=True, type=str,
                   help="Список run_id через запятую для анализа/сравнения")

    p.add_argument("--task", type=str, default="binary",
                   choices=["binary", "multiclass", "regression"],
                   help="Тип задачи")

    p.add_argument("--metric", type=str, default="roc_auc",
                   help="Целевая метрика: binary: roc_auc|pr_auc|logloss|accuracy|f1; "
                        "multiclass: accuracy|logloss|macro_f1; regression: rmse|mae|mape|r2")

    p.add_argument("--bootstrap", type=int, default=1500,
                   help="Кол-во бутстрэпов по OOF")

    p.add_argument("--fold-stats", action="store_true",
                   help="Считать дисперсию метрики по фолдам (если есть folds.pkl)")

    p.add_argument("--slices", type=str, default="",
                   help="Срезы: time:<col>[:bins=5],cat:<col>[:top=12],num:<col>[:bins=5], через запятую")

    p.add_argument("--use-adv-weights", action="store_true",
                   help="Пересчитать метрику с adversarial-весами (если найдутся)")

    p.add_argument("--compare", action="store_true",
                   help="Включить сравнение прогонов (ранговые согласования и т.п.)")

    p.add_argument("--feature-jaccard", type=str, default="",
                   help="Напр. '@50,@100' — Jaccard по топ-k важным фичам между прогонами")

    p.add_argument("--save-html", action="store_true",
                   help="Сохранить HTML-отчёт")

    p.add_argument("--name", type=str, default=None,
                   help="Имя подпапки (по умолчанию auto: run_id или compare__...)")

    p.add_argument("--sets-dir", type=str, default=None,
                   help="Явный путь к artifacts/sets/<tag> (если не стандартный)")

    p.add_argument("--models-index", type=str, default="artifacts/models/index.json",
                   help="Путь к индексу моделей для обновления")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# ----------------------------- IO helpers -----------------------------

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
            import fastparquet  # noqa
            return pd.read_parquet(p, engine="fastparquet")
        except Exception:
            return None


def fig_to_b64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# ----------------------------- Data loading -----------------------------

@dataclass
class LoadedSet:
    y: np.ndarray
    id_vals: np.ndarray
    id_col: str
    target_col: str
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]]
    meta: dict
    extra_df: Optional[pd.DataFrame]  # для срезов (если есть)


def load_set(tag: str, sets_dir: Optional[Path], verbose=False) -> LoadedSet:
    base = sets_dir or (Path("artifacts") / "sets" / tag)
    if verbose:
        print("[info] sets_dir =", base.as_posix())

    y_path = base / "y_train.parquet"
    ydf = read_parquet_any(y_path)
    if ydf is None or ydf.shape[1] < 2:
        raise FileNotFoundError("Ожидаю y_train.parquet с [id, target] в artifacts/sets/<tag>/")

    id_col = ydf.columns[0]
    target_col = [c for c in ydf.columns if c != id_col][0]
    y = ydf[target_col].values
    id_vals = ydf[id_col].values

    meta = read_json(base / "meta.json") or {}

    folds = None
    folds_path = base / "folds.pkl"
    if folds_path.exists():
        import pickle
        folds = pickle.loads(folds_path.read_bytes())
        # ожидается список (train_idx, val_idx)
        if isinstance(folds, dict) and "folds" in folds:
            folds = folds["folds"]

    extra_df = None
    # Дополнительные колонки для срезов, если пользователь сохранил:
    for cand in ["train_extra.parquet", "train_meta.parquet"]:
        p = base / cand
        if p.exists():
            extra_df = read_parquet_any(p)
            if verbose:
                print(f"[info] extra slices source: {cand} -> {extra_df.shape}")
            break

    return LoadedSet(y=y, id_vals=id_vals, id_col=id_col, target_col=target_col,
                     folds=folds, meta=meta, extra_df=extra_df)


@dataclass
class LoadedRun:
    run_id: str
    oof: np.ndarray
    path: Path
    importances: Optional[pd.DataFrame]
    metrics: Optional[dict]


def _try_load_oof(run_dir: Path) -> Optional[np.ndarray]:
    # поддержка разных имён: oof.npy, oof_pred.npy, oof_proba.npy
    for name in ["oof.npy", "oof_pred.npy", "oof_proba.npy"]:
        p = run_dir / name
        if p.exists():
            try:
                arr = np.load(p, allow_pickle=False)
                return arr
            except Exception:
                pass
    # как крайний случай: поищем первый .npy с "oof" в имени
    for p in run_dir.glob("*.npy"):
        if "oof" in p.name.lower():
            try:
                return np.load(p, allow_pickle=False)
            except Exception:
                pass
    return None


def load_run(run_id: str, verbose=False) -> LoadedRun:
    run_dir = Path("artifacts") / "models" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Не найден путь к прогону: {run_dir}")

    oof = _try_load_oof(run_dir)
    if oof is None:
        raise FileNotFoundError(f"Не найден OOF массив в {run_dir}")

    imp = None
    imp_path = run_dir / "importances.csv"
    if imp_path.exists():
        try:
            imp = pd.read_csv(imp_path)
        except Exception:
            imp = None

    mets = read_json(run_dir / "metrics.json")

    if verbose:
        print(f"[info] run {run_id}: oof={oof.shape}, importances={'yes' if imp is not None else 'no'}")

    return LoadedRun(run_id=run_id, oof=oof.reshape(-1), path=run_dir, importances=imp, metrics=mets)


def load_adv_weights(tag: str, verbose=False) -> Optional[np.ndarray]:
    p = Path("artifacts") / "adversarial" / tag / "train_weights.npy"
    if not p.exists():
        if verbose:
            print("[info] adversarial weights not found")
        return None
    try:
        w = np.load(p, allow_pickle=False).reshape(-1)
        if verbose:
            print(f"[info] adversarial weights: {w.shape}")
        return w
    except Exception:
        return None


# ----------------------------- Metrics & scorers -----------------------------

def _safe_prob(p: np.ndarray) -> np.ndarray:
    p = p.astype(float).reshape(-1)
    # если логиты: попробуем распознать — эвристика: есть отрицательные и >1?
    if (p.min() < 0) and (p.max() > 1.0):
        p = 1.0 / (1.0 + np.exp(-p))
    # клип
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return p


def scorer_factory(task: str, metric: str):
    metric = metric.lower()

    if task == "binary":
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, log_loss,
            accuracy_score, f1_score
        )

        def roc_auc(y, p, w=None):  # w: sample_weight
            p_ = _safe_prob(p)
            return float(roc_auc_score(y, p_, sample_weight=w))

        def pr_auc(y, p, w=None):
            p_ = _safe_prob(p)
            return float(average_precision_score(y, p_, sample_weight=w))

        def logloss(y, p, w=None):
            p_ = _safe_prob(p)
            return float(log_loss(y, p_, sample_weight=w))

        def acc(y, p, w=None):
            p_ = _safe_prob(p)
            pred = (p_ >= 0.5).astype(int)
            return float(accuracy_score(y, pred, sample_weight=w))

        def f1(y, p, w=None):
            p_ = _safe_prob(p)
            pred = (p_ >= 0.5).astype(int)
            return float(f1_score(y, pred, sample_weight=w))

        mapping = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "logloss": logloss,
            "accuracy": acc,
            "f1": f1,
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported binary metric: {metric}")
        return mapping[metric]

    elif task == "multiclass":
        from sklearn.metrics import accuracy_score, log_loss, f1_score

        def _to_proba_mc(p, y):
            # ожидается (n, C) с вероятностями
            if p.ndim == 1:
                raise ValueError("Для multiclass нужен массив вероятностей формы (n, C)")
            if p.shape[0] != len(y):
                raise ValueError("Размерности OOF и y не совпадают")
            return p

        def acc(y, p, w=None):
            P = _to_proba_mc(p, y)
            pred = np.argmax(P, axis=1)
            return float(accuracy_score(y, pred, sample_weight=w))

        def logloss(y, p, w=None):
            P = _to_proba_mc(p, y)
            return float(log_loss(y, P, labels=np.unique(y), sample_weight=w))

        def macro_f1(y, p, w=None):
            P = _to_proba_mc(p, y)
            pred = np.argmax(P, axis=1)
            return float(f1_score(y, pred, average="macro", sample_weight=w))

        mapping = {
            "accuracy": acc,
            "logloss": logloss,
            "macro_f1": macro_f1,
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported multiclass metric: {metric}")
        return mapping[metric]

    elif task == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        def rmse(y, p, w=None):
            return float(np.sqrt(mean_squared_error(y, p, sample_weight=w)))

        def mae(y, p, w=None):
            return float(mean_absolute_error(y, p, sample_weight=w))

        def mape(y, p, w=None):
            y_ = np.asarray(y, float)
            p_ = np.asarray(p, float)
            eps = 1e-12
            w_ = np.ones_like(y_) if w is None else w
            return float(np.average(np.abs((y_ - p_) / np.clip(np.abs(y_), eps, None)), weights=w_) * 100.0)

        def r2(y, p, w=None):
            return float(r2_score(y, p, sample_weight=w))

        mapping = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported regression metric: {metric}")
        return mapping[metric]

    else:
        raise ValueError(f"Unsupported task: {task}")


# ----------------------------- Bootstrap & fold stats -----------------------------

def bootstrap_ci(y: np.ndarray, p: np.ndarray, scorer, B: int, seed: int, weights: Optional[np.ndarray] = None):
    """
    Перцентильный bootstrap 95% CI (2.5/97.5).
    Возвращает (metric, lo, hi, samples_df).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    vals = np.empty(B, dtype=float)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        if weights is None:
            vals[b] = scorer(y[samp], p[samp], None)
        else:
            w = weights[samp]
            vals[b] = scorer(y[samp], p[samp], w)
    metric = scorer(y, p, weights)
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    df = pd.DataFrame({"bootstrap_metric": vals})
    return float(metric), lo, hi, df


def fold_stats(y: np.ndarray, p: np.ndarray,
               folds: Optional[List[Tuple[np.ndarray, np.ndarray]]],
               scorer) -> Tuple[pd.DataFrame, dict]:
    """
    Считает метрику по вал.части каждого фолда.
    """
    if not folds:
        return pd.DataFrame(), {}
    rows = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        try:
            m = scorer(y[va_idx], p[va_idx], None)
            rows.append({"fold": i, "metric": float(m), "val_size": int(len(va_idx))})
        except Exception:
            # на случай несовпадений размеров/индексов
            pass
    df = pd.DataFrame(rows)
    summary = {}
    if len(df):
        summary = {
            "fold_mean": float(df["metric"].mean()),
            "fold_std": float(df["metric"].std(ddof=1)) if len(df) > 1 else 0.0,
            "fold_min": float(df["metric"].min()),
            "fold_max": float(df["metric"].max()),
            "fold_cov": float(df["metric"].std(ddof=1) / (df["metric"].mean() + 1e-12)) if len(df) > 1 else 0.0
        }
    return df, summary


# ----------------------------- Calibration (binary) -----------------------------

def calibration_bins_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 20):
    """
    Возвращает DataFrame с reliability-бинами и сводки (ECE, Brier).
    """
    y = y.astype(int).reshape(-1)
    p = _safe_prob(p)
    # равные по количеству бины с помощью qcut
    try:
        bins = pd.qcut(p, q=n_bins, duplicates="drop")
    except Exception:
        n_bins = max(2, min(5, np.unique(p).size))
        bins = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    g = df.groupby("bin", observed=True)
    tab = g.agg(
        n=("y", "size"),
        conf=("p", "mean"),
        freq=("y", "mean")
    ).reset_index(drop=True)
    # ECE (expected calibration error)
    ece = float(np.average(np.abs(tab["conf"] - tab["freq"]), weights=tab["n"]))
    # Brier
    brier = float(np.mean((p - y) ** 2))
    tab["abs_error"] = np.abs(tab["conf"] - tab["freq"])
    return tab, {"ece": ece, "brier": brier}


# ----------------------------- Slices -----------------------------

def parse_slices(spec: str) -> List[dict]:
    """
    time:<col>[:bins=5],cat:<col>[:top=12],num:<col>[:bins=5]
    """
    if not spec:
        return []
    out = []
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    for s in parts:
        kind, rest = s.split(":", 1) if ":" in s else (s, "")
        if ":" in rest:
            col, tail = rest.split(":", 1)
        else:
            col, tail = rest, ""
        opts = {}
        if tail:
            for kv in tail.split(":"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        opts[k] = int(v)
                    except Exception:
                        try:
                            opts[k] = float(v)
                        except Exception:
                            opts[k] = v
        out.append({"kind": kind, "col": col, "opts": opts})
    return out


def slice_stats(y: np.ndarray, p: np.ndarray,
                id_vals: np.ndarray,
                extra_df: Optional[pd.DataFrame],
                ydf: pd.DataFrame,
                slices_spec: List[dict],
                scorer) -> pd.DataFrame:
    """
    Считает метрику по срезам. Источник колонок: extra_df если есть, иначе ydf.
    Поддерживает спец. колонку row_id (порядок строк).
    """
    if not slices_spec:
        return pd.DataFrame()

    # источник
    src = None
    if extra_df is not None:
        # пытаемся сопоставить по id
        if extra_df.shape[0] == len(id_vals) and (extra_df.columns[0] == ydf.columns[0]):
            src = extra_df.copy()
        else:
            # бэкап: attach id_col для join'а
            id_col = ydf.columns[0]
            extra_df = extra_df.copy()
            if id_col in extra_df.columns:
                extra_df = extra_df.set_index(id_col).reindex(id_vals).reset_index()
                src = extra_df
    if src is None:
        src = ydf.copy()

    # row_id как индекс
    src = src.copy()
    src["__row_id__"] = np.arange(len(src))

    rows = []
    for spec in slices_spec:
        kind = spec["kind"]
        col = spec["col"]
        opts = spec.get("opts", {})

        if col == "row_id":
            col_data = src["__row_id__"]
        elif col in src.columns:
            col_data = src[col]
        else:
            # пропускаем срез, если нет колонки
            continue

        if kind == "time":
            bins = int(opts.get("bins", 5))
            # если не datetime — разобьём по порядку/рангу
            if not np.issubdtype(col_data.dtype, np.datetime64):
                # qcut по числу
                try:
                    binned = pd.qcut(col_data.rank(method="first"), q=bins, duplicates="drop")
                except Exception:
                    binned = pd.cut(col_data.rank(method="first"), bins=bins)
            else:
                # use qcut on timestamp
                try:
                    binned = pd.qcut(col_data.view("i8"), q=bins, duplicates="drop")
                except Exception:
                    binned = pd.cut(col_data.view("i8"), bins=bins)
            grp = binned

        elif kind == "num":
            bins = int(opts.get("bins", 5))
            try:
                grp = pd.qcut(col_data.astype(float), q=bins, duplicates="drop")
            except Exception:
                grp = pd.cut(col_data.astype(float), bins=bins)

        elif kind == "cat":
            top = int(opts.get("top", 12))
            vc = col_data.astype(str).value_counts()
            top_cats = set(vc.head(top).index.tolist())
            tmp = col_data.astype(str).where(col_data.astype(str).isin(top_cats), other="__OTHER__")
            grp = tmp
        else:
            continue

        df = pd.DataFrame({"y": y, "p": p, "grp": grp})
        g = df.groupby("grp", dropna=True, observed=True)
        for name, d in g:
            try:
                m = scorer(d["y"].values, d["p"].values, None)
                rows.append({"slice": f"{kind}:{col}", "bin": str(name), "n": int(len(d)), "metric": float(m)})
            except Exception:
                pass

    return pd.DataFrame(rows)


# ----------------------------- Compare runs -----------------------------

def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    # ранги и корреляция Пирсона по рангам
    ra = pd.Series(a).rank(method="average").values
    rb = pd.Series(b).rank(method="average").values
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    if HAVE_SCIPY:
        try:
            return float(sps.kendalltau(a, b, nan_policy="omit").correlation)
        except Exception:
            pass
    # упрощенная аппроксимация через Спирмена (если SciPy нет)
    return spearman_rho(a, b)


def inversion_rate(a: np.ndarray, b: np.ndarray, max_pairs: int = 100_000, seed: int = 42) -> float:
    """
    Доля инверсий порядка между двумя векторами прогнозов. Сэмплируем пары (i,j) для скорости.
    """
    n = len(a)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    m = min(max_pairs, n * (n - 1) // 2)
    cnt = 0
    inv = 0
    # выбор без повторов по индексам — сэмплируем пары через индексы
    for _ in range(m):
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        da = np.sign(a[i] - a[j])
        db = np.sign(b[i] - b[j])
        if da * db < 0:  # несогласованная пара
            inv += 1
        cnt += 1
    return float(inv / max(cnt, 1))


def rank_agreement_table(oofs: List[np.ndarray], run_ids: List[str]) -> pd.DataFrame:
    rows = []
    R = len(oofs)
    for i in range(R):
        for j in range(i + 1, R):
            a, b = oofs[i], oofs[j]
            rho = spearman_rho(a, b)
            tau = kendall_tau(a, b)
            inv = inversion_rate(a, b)
            rows.append({"run_i": run_ids[i], "run_j": run_ids[j],
                         "spearman": rho, "kendall_tau": tau, "invert_rate": inv})
    return pd.DataFrame(rows)


# ----------------------------- Feature Jaccard -----------------------------

def feature_jaccard(importances: List[Optional[pd.DataFrame]],
                    run_ids: List[str],
                    ks: List[int]) -> pd.DataFrame:
    """
    importances: список DataFrame с колонками feature, importance (или None)
    """
    rows = []
    R = len(importances)
    for k in ks:
        for i in range(R):
            if importances[i] is None:
                continue
            top_i = set(importances[i].sort_values("importance", ascending=False).head(k)["feature"].astype(str))
            for j in range(i + 1, R):
                if importances[j] is None:
                    continue
                top_j = set(importances[j].sort_values("importance", ascending=False).head(k)["feature"].astype(str))
                inter = len(top_i & top_j)
                union = len(top_i | top_j)
                jac = float(inter / max(union, 1))
                rows.append({"k": k, "run_i": run_ids[i], "run_j": run_ids[j], "jaccard": jac})
    return pd.DataFrame(rows)


# ----------------------------- Plots & HTML -----------------------------

def make_plots(out_dir: Path,
               task: str,
               y: np.ndarray,
               p: np.ndarray,
               per_fold: pd.DataFrame,
               calib_tab: Optional[pd.DataFrame],
               compare_df: Optional[pd.DataFrame],
               fj_df: Optional[pd.DataFrame]):
    ensure_dir(out_dir / "plots")

    # 1) Histogram (для бинарной: вероятности)
    if HAVE_MPL:
        plt.figure(figsize=(6, 4))
        if task == "binary":
            plt.hist(_safe_prob(p), bins=30)
            plt.title("OOF probability histogram")
            plt.xlabel("p(test=1)")
        else:
            plt.hist(p.reshape(-1), bins=30)
            plt.title("OOF predictions histogram")
            plt.xlabel("prediction")
        plt.ylabel("count")
        (out_dir / "plots" / "metric_hist.png").write_bytes(base64.b64decode(fig_to_b64().split(",")[1]))

    # 2) Fold boxplot
    if HAVE_MPL and len(per_fold):
        plt.figure(figsize=(6, 4))
        plt.boxplot(per_fold["metric"].values, vert=True)
        plt.title("Per-fold metric distribution")
        (out_dir / "plots" / "fold_boxplot.png").write_bytes(base64.b64decode(fig_to_b64().split(",")[1]))

    # 3) Reliability
    if HAVE_MPL and calib_tab is not None and len(calib_tab):
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.scatter(calib_tab["conf"].values, calib_tab["freq"].values)
        plt.title("Reliability")
        plt.xlabel("Confidence"); plt.ylabel("Empirical freq")
        (out_dir / "plots" / "reliability.png").write_bytes(base64.b64decode(fig_to_b64().split(",")[1]))

    # 4) Rank scatter для двух прогонов
    if HAVE_MPL and compare_df is not None and len(compare_df):
        # возьмём первую пару
        r0 = compare_df.iloc[0]
        # нам нужны сами векторы; тут только метрики. Скэттер построим по рангам,
        # но без хранения всех OOF обоих прогонов — пропустим скэттер,
        # оставим табличную оценку сравнения (spearman/kendall/invert_rate)

    # 5) Jaccard@k линия
    if HAVE_MPL and fj_df is not None and len(fj_df):
        # усредним по парам для каждого k
        agg = fj_df.groupby("k", as_index=False)["jaccard"].mean()
        plt.figure(figsize=(6, 4))
        plt.plot(agg["k"].values, agg["jaccard"].values, marker="o")
        plt.title("Mean Jaccard@k of feature sets")
        plt.xlabel("k"); plt.ylabel("Jaccard")
        (out_dir / "plots" / "jaccard_at_k.png").write_bytes(base64.b64decode(fig_to_b64().split(",")[1]))


def make_html(out_dir: Path,
              tag: str,
              name: str,
              summary: dict,
              tables: Dict[str, pd.DataFrame]):
    html = io.StringIO()
    def w(s=""):
        html.write(s + "\n")
    w("<html><head><meta charset='utf-8'><title>Stability Report</title></head><body>")
    w(f"<h1>Stability report — {tag} / {name}</h1>")

    w("<h2>Summary</h2>")
    w("<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>")

    # Вставим картинки, если есть
    for img in ["metric_hist.png", "fold_boxplot.png", "reliability.png", "jaccard_at_k.png"]:
        p = out_dir / "plots" / img
        if p.exists():
            b64 = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
            w(f"<h3>{img}</h3><img src='{b64}'/>")

    # Таблицы
    for title, df in tables.items():
        if df is None or not len(df):
            continue
        w(f"<h2>{title}</h2>")
        try:
            w(df.to_html(index=False))
        except Exception:
            w("<pre>table render failed</pre>")

    w("</body></html>")
    (out_dir / "report.html").write_text(html.getvalue(), encoding="utf-8")


# ----------------------------- Stability score & recommendations -----------------------------

def stability_score(summary: dict) -> float:
    """
    Интегральный скор 0..100 на основе нескольких компонент.
    """
    score = 100.0

    # ширина CI
    ci_w = summary.get("bootstrap_ci_width", 0.0)
    score -= 20.0 * float(ci_w)

    # std по фолдам
    fold_std = summary.get("fold_std", 0.0)
    score -= 15.0 * float(fold_std)

    # калибровка
    ece = summary.get("calibration_ece", 0.0)
    score -= 10.0 * max(0.0, float(ece) - 0.0)  # штраф растёт с ECE

    # робастность к дрейфу (разница с/без весов)
    diff_w = abs(summary.get("metric_weighted", summary.get("metric", 0.0)) - summary.get("metric", 0.0))
    score -= 15.0 * float(diff_w)

    # сравнение прогонов
    mean_tau = summary.get("compare_mean_tau", 1.0)
    score -= 20.0 * (1.0 - float(mean_tau))

    # стабильность фич
    mean_jacc = summary.get("feature_mean_jaccard", 1.0)
    score -= 20.0 * (1.0 - float(mean_jacc))

    return float(max(0.0, min(100.0, score)))


def recommendations(summary: dict) -> dict:
    rec = {}
    if summary.get("bootstrap_ci_width", 0) > 0.05:
        rec["ci"] = "Широкий доверительный интервал — добавь регуляризацию/упростись, проверь сплит, увеличь стабильность фич."
    if summary.get("fold_std", 0) > 0.02:
        rec["folds"] = "Большая дисперсия по фолдам — синхронизируй фолды с фичами (time/group), проверь утечки, нормализуй сложные фичи."
    if summary.get("calibration_ece", 0) > 0.03:
        rec["calibration"] = "Плохая калибровка — Platt/Isotonic по OOF; не порогуй без проверки."
    if abs(summary.get("metric_weighted", summary.get("metric", 0.0)) - summary.get("metric", 0.0)) > 0.01:
        rec["drift"] = "Метрика меняется с adv-весами — чисти дрейфовые фичи (см. adversarial.drop_features) и/или используй веса при обучении."
    if summary.get("compare_mean_tau", 1.0) < 0.9:
        rec["compare"] = "Низкое согласование между прогонами — блендуй модели с низкой корреляцией, закрепи ядро фич."
    if summary.get("feature_mean_jaccard", 1.0) < 0.6:
        rec["features"] = "Слабая стабильность важностей — пересмотри генерацию фич, снизь шум, фиксируй семена/процессы."
    return rec


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    # prepare paths
    sets_dir = Path(args.sets_dir) if args.sets_dir else (Path("artifacts") / "sets" / args.tag)
    runs = [s.strip() for s in args.runs.split(",") if s.strip()]
    if not runs:
        raise ValueError("--runs пуст")

    # output folder name
    if args.name:
        name = args.name
    elif len(runs) == 1:
        name = runs[0]
    else:
        base = "compare__" + "__".join(runs)
        if len(base) > 100:
            h = hashlib.sha1(base.encode()).hexdigest()[:8]
            name = "compare__" + "__".join(runs[:3]) + f"__{h}"
        else:
            name = base

    out_dir = Path("artifacts") / "stability" / args.tag / name
    ensure_dir(out_dir)

    # load set
    loaded = load_set(args.tag, sets_dir, verbose=args.verbose)
    y = loaded.y.reshape(-1)
    id_vals = loaded.id_vals
    ydf = pd.DataFrame({loaded.id_col: id_vals, loaded.target_col: y})

    # load runs
    loaded_runs = [load_run(r, verbose=args.verbose) for r in runs]
    oofs = [r.oof.astype(float).reshape(-1) for r in loaded_runs]

    # sanity check lengths
    n = len(y)
    for i, o in enumerate(oofs):
        if len(o) != n:
            raise ValueError(f"Размер OOF не совпадает с y для {runs[i]}: {len(o)} vs {n}")

    scorer = scorer_factory(args.task, args.metric)

    # Базовый анализ для первого прогона (или усредняем, если >1?)
    # Выберем первый как основной для детализированных отчётов
    p_main = oofs[0]

    # Bootstrap
    metric, lo, hi, boot_df = bootstrap_ci(y, p_main, scorer, args.bootstrap, args.seed, None)
    boot_df.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    # Fold stats
    per_fold_df, fold_summary = fold_stats(y, p_main, loaded.folds if args.fold_stats else None, scorer)
    if len(per_fold_df):
        per_fold_df.to_csv(out_dir / "per_fold.csv", index=False)

    # Calibration (binary only)
    calib_tab, calib_summary = (None, {})
    if args.task == "binary":
        calib_tab, calib_summary = calibration_bins_binary(y, p_main, n_bins=20)
        if len(calib_tab):
            calib_tab.to_csv(out_dir / "calibration.csv", index=False)

    # Slices
    slices_spec = parse_slices(args.slices)
    slices_df = slice_stats(y, p_main, id_vals, loaded.extra_df, ydf, slices_spec, scorer)
    if len(slices_df):
        slices_df.to_csv(out_dir / "slices.csv", index=False)

    # Robustness to adv weights
    w = load_adv_weights(args.tag, verbose=args.verbose) if args.use_adv_weights else None
    weighted_metric = None
    if w is not None and len(w) == n:
        try:
            weighted_metric = scorer(y, p_main, w)
            pd.DataFrame({"metric_unweighted": [metric], "metric_weighted": [weighted_metric]}).to_csv(
                out_dir / "with_vs_without_weights.csv", index=False
            )
        except Exception:
            weighted_metric = None

    # Compare runs (rank agreement)
    compare_df = None
    mean_tau = 1.0
    if args.compare and len(oofs) >= 2:
        compare_df = rank_agreement_table(oofs, runs)
        if len(compare_df):
            compare_df.to_csv(out_dir / "rank_agreement.csv", index=False)
            mean_tau = float(compare_df["kendall_tau"].mean())

    # Feature Jaccard
    fj_df = None
    mean_jacc = 1.0
    if args.feature_jaccard:
        ks = []
        for tok in args.feature_jaccard.split(","):
            tok = tok.strip()
            if tok.startswith("@"):
                try:
                    ks.append(int(tok[1:]))
                except Exception:
                    pass
        if ks:
            imps = [lr.importances for lr in loaded_runs]
            fj_df = feature_jaccard(imps, runs, ks)
            if len(fj_df):
                fj_df.to_csv(out_dir / "feature_jaccard.csv", index=False)
                mean_jacc = float(fj_df.groupby("k", as_index=False)["jaccard"].mean()["jaccard"].mean())

    # Plots
    try:
        make_plots(out_dir, args.task, y, p_main, per_fold_df, calib_tab, compare_df, fj_df)
    except Exception as e:
        if args.verbose:
            print("[warn] plotting failed:", e)

    # Summary
    summary = {
        "tag": args.tag,
        "name": name,
        "runs": runs,
        "task": args.task,
        "metric_name": args.metric,
        "metric": float(metric),
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "bootstrap_ci_width": float(hi - lo),
        "fold_mean": fold_summary.get("fold_mean", None),
        "fold_std": fold_summary.get("fold_std", None),
        "fold_min": fold_summary.get("fold_min", None),
        "fold_max": fold_summary.get("fold_max", None),
        "fold_cov": fold_summary.get("fold_cov", None),
        "calibration_ece": calib_summary.get("ece", None),
        "calibration_brier": calib_summary.get("brier", None),
        "metric_weighted": None if weighted_metric is None else float(weighted_metric),
        "compare_mean_tau": mean_tau,
        "feature_mean_jaccard": mean_jacc,
        "generated_at": datetime.now().isoformat()
    }

    score = stability_score(summary)
    summary["stability_score"] = score
    recs = recommendations(summary)

    save_json(out_dir / "metrics.json", summary)
    save_json(out_dir / "recommendations.json", recs)

    # HTML
    if args.save_html:
        tables = {
            "Per-fold": per_fold_df,
            "Calibration bins": calib_tab,
            "Slices": slices_df,
            "Compare runs": compare_df,
            "Feature Jaccard": fj_df
        }
        make_html(out_dir, args.tag, name, {**summary, "recommendations": recs}, tables)

    # Update models index
    models_index = Path(args.models_index)
    idx = read_json(models_index) or {}
    idx_key = f"stability:{args.tag}:{name}"
    idx[idx_key] = {
        "tag": args.tag,
        "name": name,
        "runs": runs,
        "task": args.task,
        "metric": args.metric,
        "score": float(score),
        "path": str(out_dir)
    }
    save_json(models_index, idx)

    # Final prints
    print("=== STABILITY DONE ===")
    print("tag:", args.tag)
    print("name:", name)
    print("out:", out_dir.as_posix())
    print(f"{args.metric} = {metric:.6f}  (95% CI: {lo:.6f} .. {hi:.6f})  stability_score={score:.2f}")
    if weighted_metric is not None:
        print(f"weighted {args.metric} = {weighted_metric:.6f}")
    if args.save_html:
        print("html report:", (out_dir / "report.html").as_posix())


if __name__ == "__main__":
    main()
