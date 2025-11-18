#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/adversarial.py

Диагностика train↔test drift (adversarial validation) и подготовка артефактов:
- Доменный классификатор (train=0 vs test=1) с fold-safe OOF по train-части
- Метрики AUC/AP, ROC/PR, важности (LR/SGD coef, LGBM gain), топ-фи
- PSI/KS/χ² для dense-слоя
- Вклад по фич-блокам (из meta.catalog)
- Подозрительные фичи и рекомендации drop/keep
- Importance weights для train: w = p(test|x)/(1-p(test|x))
- HTML/JSON/CSV отчёты, обновление artifacts/models/index.json

Примеры:
  Быстро по dense + HTML:
    python tools/adversarial.py --tag s5e11 --layer dense --psi-ks --block-contrib --save-html --verbose

  Sparse (TF-IDF), только доменный LR и веса:
    python tools/adversarial.py --tag s5e11_text --layer sparse --weights direct --verbose

  Бустингом:
    python tools/adversarial.py --tag geo_img_mix --layer dense --clf lgbm \
      --clf-params '{"n_estimators":400,"learning_rate":0.05,"num_leaves":63}' \
      --psi-ks --block-contrib --save-html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# SciPy (опционально)
try:
    import scipy.sparse as sp
    from scipy.stats import ks_2samp, chi2_contingency
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    class _Dummy:
        csr_matrix = None
    sp = _Dummy()

# Matplotlib (для отчёта)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# LightGBM (опционально)
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder


# -------------------------- CLI --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Adversarial validation & drift diagnostics")

    p.add_argument("--tag", required=True, type=str, help="RUN_TAG (набор фич)")

    p.add_argument("--layer", type=str, default="auto",
                   choices=["auto", "dense", "sparse"],
                   help="Какой слой фич использовать")

    p.add_argument("--sample", type=float, default=1.0,
                   help="Случайная доля строк для ускорения (0.2..1.0)")

    p.add_argument("--balance", type=str, default="auto",
                   choices=["auto", "down", "up", "none"],
                   help="Балансировка доменных классов внутри train-фолда (0/1)")

    p.add_argument("--clf", type=str, default="lr",
                   choices=["lr", "sgd", "lgbm"],
                   help="База доменного классификатора")

    p.add_argument("--clf-params", type=str, default=None,
                   help="JSON с параметрами модели")

    p.add_argument("--folds", type=str, default="foldsafe",
                   choices=["foldsafe", "simple"],
                   help="Использовать folds.pkl (fold-safe OOF) или простой StratifiedKFold по train")

    p.add_argument("--n-splits", type=int, default=5, help="Кол-во фолдов, если folds.pkl нет/не используется")
    p.add_argument("--seed", type=int, default=42, help="Сид")

    p.add_argument("--block-contrib", action="store_true", help="Считать вклад по блокам (meta.catalog)")
    p.add_argument("--psi-ks", action="store_true", help="Считать PSI/KS/χ² (только dense)")
    p.add_argument("--psi-top", type=int, default=2000, help="Максимум колонок для PSI/KS")
    p.add_argument("--drop-threshold", type=float, default=0.2, help="Порог PSI для drop-листа")

    p.add_argument("--weights", type=str, default="direct", choices=["off", "direct", "isotonic"],
                   help="Способ importance weights (обычно direct)")
    p.add_argument("--cap-weight", type=float, default=10.0, help="Кэп весов на объект")

    p.add_argument("--save-html", action="store_true", help="Сохранить HTML-отчёт")

    p.add_argument("--out-dir", type=str, default=None, help="Папка вывода (по умолчанию artifacts/adversarial/<tag>)")
    p.add_argument("--models-index", type=str, default="artifacts/models/index.json", help="Индекс моделей")
    p.add_argument("--sets-dir", type=str, default=None, help="artifacts/sets/<tag> (по умолчанию авто)")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# -------------------------- IO helpers --------------------------

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


def load_npz(path: Path):
    if not path.exists():
        return None
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy не установлен, но требуется для чтения sparse NPZ")
    return sp.load_npz(path.as_posix())


# -------------------------- Data loading --------------------------

@dataclass
class LoadedSet:
    Xtr: Union[pd.DataFrame, "sp.csr_matrix"]
    Xte: Union[pd.DataFrame, "sp.csr_matrix"]
    ydf: pd.DataFrame  # id + target
    meta: dict
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]]  # для основного таска (используем индексы вал.части)
    names: List[str]  # названия фич
    id_col: str
    target_col: str


def load_set(tag: str, layer: str, sets_dir: Optional[Path], verbose=False) -> LoadedSet:
    base = sets_dir or (Path("artifacts") / "sets" / tag)
    if verbose:
        print("[info] sets_dir =", base.as_posix())
    # y / meta / folds
    y_path = base / "y_train.parquet"
    meta_path = base / "meta.json"
    folds_path = base / "folds.pkl"

    ydf = read_parquet_any(y_path)
    if ydf is None or ydf.shape[1] < 2:
        raise FileNotFoundError("Ожидаю y_train.parquet с колонками [id, target]")
    id_col = ydf.columns[0]
    target_col = [c for c in ydf.columns if c != id_col][0]

    meta = read_json(meta_path) or {}
    folds = None
    if folds_path.exists():
        import pickle
        folds = pickle.loads(folds_path.read_bytes())

    # layer files
    Xd_tr_path = base / "X_dense_train.parquet"
    Xd_te_path = base / "X_dense_test.parquet"
    Xs_tr_path = base / "X_sparse_train.npz"
    Xs_te_path = base / "X_sparse_test.npz"

    choose_dense = (layer == "dense") or (layer == "auto" and Xd_tr_path.exists())

    if choose_dense:
        if not Xd_tr_path.exists() or not Xd_te_path.exists():
            # fallback sparse
            if layer == "dense":
                raise FileNotFoundError("Нет dense матриц")
            if not Xs_tr_path.exists() or not Xs_te_path.exists():
                raise FileNotFoundError("Нет ни dense, ни sparse матриц")
            Xtr = load_npz(Xs_tr_path)
            Xte = load_npz(Xs_te_path)
            names = [f"s_{i}" for i in range(Xtr.shape[1])]
            layer_used = "sparse"
        else:
            Xtr = read_parquet_any(Xd_tr_path)
            Xte = read_parquet_any(Xd_te_path)
            names = list(Xtr.columns)
            layer_used = "dense"
    else:
        if not Xs_tr_path.exists() or not Xs_te_path.exists():
            raise FileNotFoundError("Нет sparse матриц")
        Xtr = load_npz(Xs_tr_path)
        Xte = load_npz(Xs_te_path)
        names = [f"s_{i}" for i in range(Xtr.shape[1])]
        layer_used = "sparse"

    if verbose:
        print(f"[info] layer used: {layer_used} | train: {get_shape(Xtr)} | test: {get_shape(Xte)}")

    return LoadedSet(
        Xtr=Xtr, Xte=Xte, ydf=ydf, meta=meta, folds=folds, names=names, id_col=id_col, target_col=target_col
    )


def get_shape(X):
    try:
        return X.shape
    except Exception:
        return "?"


def maybe_sample_rows(Xtr, Xte, frac: float, seed: int):
    """Возвращает (Xtr_sub, Xte_sub, idx_tr, idx_te)."""
    if not (0 < frac < 1):
        ntr = Xtr.shape[0]
        nte = Xte.shape[0]
        return Xtr, Xte, np.arange(ntr), np.arange(nte)

    rng = np.random.default_rng(seed)
    ntr = Xtr.shape[0]
    nte = Xte.shape[0]
    tr_idx = np.sort(rng.choice(ntr, size=max(1, int(frac * ntr)), replace=False))
    te_idx = np.sort(rng.choice(nte, size=max(1, int(frac * nte)), replace=False))

    if HAVE_SCIPY and sp.issparse(Xtr):
        Xtr_sub = Xtr[tr_idx]
        Xte_sub = Xte[te_idx]
    else:
        Xtr_sub = Xtr.iloc[tr_idx] if isinstance(Xtr, pd.DataFrame) else Xtr[tr_idx]
        Xte_sub = Xte.iloc[te_idx] if isinstance(Xte, pd.DataFrame) else Xte[te_idx]
    return Xtr_sub, Xte_sub, tr_idx, te_idx


# -------------------------- Domain dataset & folds --------------------------

@dataclass
class DomainData:
    X: Union[pd.DataFrame, "sp.csr_matrix"]
    d: np.ndarray  # domain labels (0=train, 1=test)
    tr_off: int    # n_train
    tr_idx: np.ndarray  # индексы train-части в X
    te_idx: np.ndarray  # индексы test-части в X
    names: List[str]


def build_domain_matrix(
    Xtr, Xte, names: List[str]
) -> DomainData:
    ntr = Xtr.shape[0]
    nte = Xte.shape[0]
    if HAVE_SCIPY and sp.issparse(Xtr):
        X = sp.vstack([Xtr, Xte], format="csr")
    else:
        X = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
        X.columns = names
    d = np.concatenate([np.zeros(ntr, int), np.ones(nte, int)])
    tr_idx = np.arange(0, ntr)
    te_idx = np.arange(ntr, ntr + nte)
    return DomainData(X=X, d=d, tr_off=ntr, tr_idx=tr_idx, te_idx=te_idx, names=names)


def make_folds_for_domain(
    dset: DomainData,
    folds_mode: str,
    base_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Возвращает фолды для OOF по train-части: список пар (tr_idx_global, va_idx_global) в пространстве X.
    Для foldsafe: используем folds.pkl (по train-части), обучаясь на (train_fold + вся test-часть),
    валидируя на (val_fold). Это стандартная, честная схема.
    """
    ntr = dset.tr_off
    if folds_mode == "foldsafe" and base_folds:
        folds = []
        for tr, va in base_folds:
            tr_glob = np.concatenate([tr, dset.te_idx])  # обучаемся на train_tr и всех test
            va_glob = va  # валидируем только на train_va
            folds.append((tr_glob, va_glob))
        return folds

    # simple StratifiedKFold по train-части
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    y_train = dset.d[:ntr]  # все нули
    # для стратификации по нулям нельзя; добавим искусственную разметку по индексу
    idx = np.arange(ntr)
    # формально stratify нельзя по константе -> используем KFold без стратификации
    # однако для простоты — просто KFold через StratifiedKFold "обмануть" не получится
    # сделаем обычный KFold:
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(idx):
        tr_glob = np.concatenate([tr, dset.te_idx])
        folds.append((tr_glob, va))
    return folds


# -------------------------- Balancing --------------------------

def balance_indices(X, labels, idx_tr: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """
    Балансировка по доменным классам (0/1) в обучающем наборе.
    На входе idx_tr — индексы обучающей части (содержат train и test).
    Возвращаем подмножество индексов для fit.
    """
    if mode == "none":
        return idx_tr
    # выделим по классам
    y = labels[idx_tr]
    cls0 = idx_tr[y == 0]
    cls1 = idx_tr[y == 1]
    n0, n1 = len(cls0), len(cls1)

    if n0 == 0 or n1 == 0:
        return idx_tr

    rng = np.random.default_rng(seed)
    if mode == "auto":
        # downsample большинства
        if n0 > n1:
            keep0 = rng.choice(cls0, size=n1, replace=False)
            return np.concatenate([keep0, cls1])
        else:
            keep1 = rng.choice(cls1, size=n0, replace=False)
            return np.concatenate([cls0, keep1])

    if mode == "down":
        m = min(n0, n1)
        keep0 = rng.choice(cls0, size=m, replace=False)
        keep1 = rng.choice(cls1, size=m, replace=False)
        return np.concatenate([keep0, keep1])

    if mode == "up":
        m = max(n0, n1)
        rep0 = rng.choice(cls0, size=m, replace=True)
        rep1 = rng.choice(cls1, size=m, replace=True)
        return np.concatenate([rep0, rep1])

    return idx_tr


# -------------------------- Domain models --------------------------

@dataclass
class DomainResult:
    oof_train: np.ndarray         # OOF p(test|x) на train-части (n_train,)
    auc: float
    ap: float
    importances: Optional[pd.DataFrame]  # feature, importance
    final_test_scores: Optional[np.ndarray]  # p(test|x) на test-части модели, обученной на всём (для графиков)


def fit_predict_domain_lr(
    dset: DomainData,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    balance: str,
    seed: int,
    params: Optional[dict],
    verbose=False
) -> DomainResult:
    # Параметры по умолчанию
    p = dict(C=1.0, penalty="l2", solver="saga", max_iter=500, n_jobs=-1, random_state=seed)
    if params:
        p.update(params)

    ntr = dset.tr_off
    oof_train = np.zeros(ntr, dtype=float)
    for i, (tr_idx, va_idx) in enumerate(folds):
        fit_idx = balance_indices(dset.X, dset.d, tr_idx, balance, seed + i)
        X_fit = dset.X[fit_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[fit_idx]
        y_fit = dset.d[fit_idx]
        X_va = dset.X[va_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[va_idx]

        if verbose:
            print(f"[fold {i+1}] fit={len(fit_idx)}, va={len(va_idx)}")

        lr = LogisticRegression(**p)
        lr.fit(X_fit, y_fit)
        proba = lr.predict_proba(X_va)[:, 1]
        # va_idx всегда в train-части (глобальные индексы < ntr)
        oof_train[va_idx] = proba

    auc = float(roc_auc_score(np.zeros(ntr, int), oof_train))  # по конструированию d(train)=0
    ap = float(average_precision_score(np.zeros(ntr, int), oof_train))

    # финальная модель на всём
    lrF = LogisticRegression(**p)
    # обучаемся на всех данных (train+test)
    Xall = dset.X
    yall = dset.d
    if HAVE_SCIPY and sp.issparse(Xall):
        lrF.fit(Xall, yall)
        test_scores = lrF.predict_proba(dset.X[dset.te_idx])[:, 1]
    else:
        lrF.fit(Xall, yall)
        test_scores = lrF.predict_proba(dset.X.iloc[dset.te_idx])[:, 1]

    # важности — абсолютные коэфы
    try:
        coefs = np.abs(lrF.coef_.ravel())
        importances = pd.DataFrame({"feature": dset.names, "importance": coefs}).sort_values(
            "importance", ascending=False
        )
    except Exception:
        importances = None

    return DomainResult(oof_train=oof_train, auc=auc, ap=ap, importances=importances, final_test_scores=test_scores)


def fit_predict_domain_sgd(
    dset: DomainData,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    balance: str,
    seed: int,
    params: Optional[dict],
    verbose=False
) -> DomainResult:
    p = dict(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=seed, n_jobs=-1)
    if params:
        p.update(params)

    ntr = dset.tr_off
    oof_train = np.zeros(ntr, dtype=float)
    for i, (tr_idx, va_idx) in enumerate(folds):
        fit_idx = balance_indices(dset.X, dset.d, tr_idx, balance, seed + i)
        X_fit = dset.X[fit_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[fit_idx]
        y_fit = dset.d[fit_idx]
        X_va = dset.X[va_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[va_idx]

        if verbose:
            print(f"[fold {i+1}] fit={len(fit_idx)}, va={len(va_idx)}")

        sgd = SGDClassifier(**p)
        sgd.fit(X_fit, y_fit)
        # decision_function -> proba через сигмоиду
        df = sgd.decision_function(X_va)
        proba = 1.0 / (1.0 + np.exp(-df))
        oof_train[va_idx] = proba

    auc = float(roc_auc_score(np.zeros(ntr, int), oof_train))
    ap = float(average_precision_score(np.zeros(ntr, int), oof_train))

    sgdF = SGDClassifier(**p)
    Xall = dset.X
    yall = dset.d
    sgdF.fit(Xall if (HAVE_SCIPY and sp.issparse(Xall)) else Xall, yall)
    df_test = sgdF.decision_function(dset.X[dset.te_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[dset.te_idx])
    test_scores = 1.0 / (1.0 + np.exp(-df_test))

    # псевдо-важности по абсолютным коэффициентам
    try:
        coefs = np.abs(sgdF.coef_.ravel())
        importances = pd.DataFrame({"feature": dset.names, "importance": coefs}).sort_values(
            "importance", ascending=False
        )
    except Exception:
        importances = None

    return DomainResult(oof_train=oof_train, auc=auc, ap=ap, importances=importances, final_test_scores=test_scores)


def fit_predict_domain_lgbm(
    dset: DomainData,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    balance: str,
    seed: int,
    params: Optional[dict],
    verbose=False
) -> DomainResult:
    if not HAVE_LGBM:
        raise RuntimeError("LightGBM недоступен (pip install lightgbm)")

    base = dict(
        objective="binary",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1
    )
    if params:
        base.update(params)

    ntr = dset.tr_off
    oof_train = np.zeros(ntr, dtype=float)

    for i, (tr_idx, va_idx) in enumerate(folds):
        fit_idx = balance_indices(dset.X, dset.d, tr_idx, balance, seed + i)
        X_fit = dset.X[fit_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[fit_idx]
        y_fit = dset.d[fit_idx]
        X_va = dset.X[va_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[va_idx]

        if verbose:
            print(f"[fold {i+1}] fit={len(fit_idx)}, va={len(va_idx)}")

        mdl = lgb.LGBMClassifier(**base)
        mdl.fit(X_fit, y_fit)
        proba = mdl.predict_proba(X_va)[:, 1]
        oof_train[va_idx] = proba

    auc = float(roc_auc_score(np.zeros(ntr, int), oof_train))
    ap = float(average_precision_score(np.zeros(ntr, int), oof_train))

    # финал на всём
    mdlF = lgb.LGBMClassifier(**base)
    Xall = dset.X
    yall = dset.d
    mdlF.fit(Xall if (HAVE_SCIPY and sp.issparse(Xall)) else Xall, yall)
    test_scores = mdlF.predict_proba(dset.X[dset.te_idx] if HAVE_SCIPY and sp.issparse(dset.X) else dset.X.iloc[dset.te_idx])[:, 1]

    # важности
    try:
        fi = np.asarray(mdlF.booster_.feature_importance(importance_type="gain"), float)
        importances = pd.DataFrame({"feature": dset.names, "importance": fi}).sort_values(
            "importance", ascending=False
        )
    except Exception:
        importances = None

    return DomainResult(oof_train=oof_train, auc=auc, ap=ap, importances=importances, final_test_scores=test_scores)


# -------------------------- Drift measures (dense) --------------------------

def psi_single(train_vals: np.ndarray, test_vals: np.ndarray, bins: int = 10, eps=1e-6) -> float:
    """PSI на колонку (биннинг по квантилям train)."""
    q = np.linspace(0, 1, bins + 1)
    try:
        cuts = np.unique(np.quantile(train_vals[~np.isnan(train_vals)], q))
    except Exception:
        return 0.0
    if len(cuts) < 2:
        return 0.0
    # защита от одинаковых квантилей
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    def dist(x):
        cnt, _ = np.histogram(x[~np.isnan(x)], bins=cuts)
        p = cnt / max(len(x[~np.isnan(x)]), 1)
        p = np.clip(p, eps, None)
        p = p / p.sum()
        return p

    p_tr = dist(train_vals)
    p_te = dist(test_vals)
    return float(np.sum((p_tr - p_te) * np.log(p_tr / p_te)))


def ks_single(train_vals: np.ndarray, test_vals: np.ndarray) -> Tuple[float, float]:
    """KS (двухвыборочный) — если SciPy, иначе простой суррогат."""
    if HAVE_SCIPY:
        try:
            res = ks_2samp(train_vals[~np.isnan(train_vals)], test_vals[~np.isnan(test_vals)])
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass
    # суррогат: эмпирические CDF и максимум
    def ecdf(x):
        x = np.sort(x[~np.isnan(x)])
        v = np.linspace(0, 1, num=len(x), endpoint=True)
        return x, v
    xt, vt = ecdf(train_vals)
    xs, vs = ecdf(test_vals)
    if len(xt) == 0 or len(xs) == 0:
        return 0.0, 1.0
    # грубо через общий диапазон
    grid = np.unique(np.concatenate([xt, xs]))
    def cdf(x, v, g):
        # правая непрерывная
        j = np.searchsorted(x, g, side="right")
        out = np.where(len(x)>0, j/len(x), 0.0)
        return out
    Ft = cdf(xt, vt, grid)
    Fs = cdf(xs, vs, grid)
    stat = float(np.max(np.abs(Ft - Fs))) if len(grid) else 0.0
    return stat, 1.0


def chi2_categorical(train_vals: pd.Series, test_vals: pd.Series) -> Tuple[float, float, int]:
    """
    χ² для категориальных (если есть малое число уникальных).
    Возвращает (chi2, p_value, dof) либо (0,1,0).
    """
    try:
        vc_tr = train_vals.value_counts()
        vc_te = test_vals.value_counts()
        cats = list(set(vc_tr.index).union(set(vc_te.index)))
        if len(cats) < 2:  # бессмысленно
            return 0.0, 1.0, 0
        obs = np.vstack([
            np.array([vc_tr.get(c, 0) for c in cats]),
            np.array([vc_te.get(c, 0) for c in cats])
        ])
        if HAVE_SCIPY:
            chi2, p, dof, _ = chi2_contingency(obs)
            return float(chi2), float(p), int(dof)
        # суррогат: без p-value
        chi2 = float(np.sum((obs - obs.mean(axis=0))**2 / np.clip(obs.mean(axis=0), 1e-9, None)))
        return chi2, 1.0, len(cats)-1
    except Exception:
        return 0.0, 1.0, 0


def drift_table_dense(
    Xtr: pd.DataFrame, Xte: pd.DataFrame, psi_top: int, drop_threshold: float, verbose=False
) -> pd.DataFrame:
    """
    Считает PSI/KS/χ² для dense-матриц. Возвращает DataFrame:
    [feature, kind, psi, ks, p_ks, chi2, p_chi2].
    """
    cols = list(Xtr.columns)
    if psi_top and len(cols) > psi_top:
        cols = cols[:psi_top]  # можно улучшить (случайная подвыборка)
        if verbose:
            print(f"[info] PSI ограничен первыми {psi_top} столбцами")

    rows = []
    for c in cols:
        tr = Xtr[c].values
        te = Xte[c].values
        # эвристика типа
        if pd.api.types.is_numeric_dtype(Xtr[c]):
            psi = psi_single(tr, te)
            ks, pks = ks_single(tr, te)
            rows.append({"feature": c, "kind": "num", "psi": psi, "ks": ks, "p_ks": pks, "chi2": np.nan, "p_chi2": np.nan})
        else:
            # категории/строки
            chi2, p, dof = chi2_categorical(Xtr[c], Xte[c])
            # грубо зададим psi=NaN (не считаем), ks=NaN
            rows.append({"feature": c, "kind": "cat", "psi": np.nan, "ks": np.nan, "p_ks": np.nan, "chi2": chi2, "p_chi2": p})
    df = pd.DataFrame(rows)
    # упорядочим по psi у числовых
    df["psi_rank"] = df["psi"].rank(ascending=False, method="min")
    df["is_drop"] = (df["psi"].fillna(0.0) >= drop_threshold)
    return df


# -------------------------- Block contribution --------------------------

def extract_catalog(meta: dict) -> Dict[str, List[str]]:
    """
    Пытается извлечь mapping: block -> list(features).
    meta['catalog'] может быть разного формата; поддержим несколько.
    """
    cat = meta.get("catalog")
    if cat is None:
        return {}
    # вариант 1: словарь пакет -> {cols:[...]}
    if isinstance(cat, dict):
        out = {}
        for k, v in cat.items():
            if isinstance(v, dict) and "cols" in v and isinstance(v["cols"], list):
                out[k] = list(v["cols"])
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = list(v)
        return out
    # вариант 2: список объектов с полями name/cols
    if isinstance(cat, list):
        out = {}
        for item in cat:
            if isinstance(item, dict) and "name" in item and "cols" in item:
                out[item["name"]] = list(item["cols"])
        return out
    return {}


def aggregate_by_block(importances: Optional[pd.DataFrame],
                       drift_df: Optional[pd.DataFrame],
                       catalog: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
    if importances is None and drift_df is None:
        return None
    rows = []
    for block, cols in catalog.items():
        imp = None
        if importances is not None and "feature" in importances.columns:
            imp = importances[importances["feature"].isin(cols)]["importance"]
        psi_mean = None
        if drift_df is not None and "feature" in drift_df.columns:
            psi_mean = drift_df.loc[drift_df["feature"].isin(cols), "psi"].mean()
        rows.append({
            "block": block,
            "n_cols": len(cols),
            "total_importance": None if imp is None else float(np.nansum(imp.values)),
            "mean_importance": None if imp is None else float(np.nanmean(imp.values)),
            "psi_mean": None if psi_mean is None or np.isnan(psi_mean) else float(psi_mean),
        })
    return pd.DataFrame(rows).sort_values(
        ["total_importance", "psi_mean"], ascending=[False, False]
    )


# -------------------------- Suspicious features --------------------------

def suspicious_features(importances: Optional[pd.DataFrame], drift_df: Optional[pd.DataFrame],
                        top_k_imp: int = 200, psi_thr: float = 0.3) -> pd.DataFrame:
    """
    Простая эвристика:
      - в топ-K по важности доменного
      - либо PSI >= psi_thr (числовые)
    """
    feats = set()
    rows = []
    if importances is not None:
        imp_top = importances.head(top_k_imp)
        for _, r in imp_top.iterrows():
            feats.add(r["feature"])
            rows.append({"feature": r["feature"], "reason": "high_importance", "score": float(r["importance"])})
    if drift_df is not None and "psi" in drift_df.columns:
        bad = drift_df[drift_df["psi"] >= psi_thr]
        for _, r in bad.iterrows():
            feats.add(r["feature"])
            rows.append({"feature": r["feature"], "reason": "high_psi", "score": float(r["psi"])})
    df = pd.DataFrame(rows).drop_duplicates(subset=["feature"])
    return df.sort_values("score", ascending=False)


# -------------------------- Importance weights --------------------------

def compute_importance_weights(p: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """
    p = p(test|x) ∈ (0,1). w = p/(1-p), нормировка до mean=1, капирование.
    """
    eps = 1e-6
    p = np.clip(p.reshape(-1), eps, 1 - eps)
    w = p / (1.0 - p)
    w = w / (np.mean(w) + 1e-12)
    if cap is not None and cap > 0:
        w = np.clip(w, 0.0, cap)
    return w


# -------------------------- Plots & HTML --------------------------

def fig_to_b64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def make_plots(out_dir: Path,
               y_true: np.ndarray,
               oof_scores: np.ndarray,
               importances: Optional[pd.DataFrame],
               drift_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    ensure_dir(out_dir / "plots")
    out = {}

    # ROC
    fpr, tpr, _ = roc_curve(y_true, oof_scores)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Domain ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    out["roc"] = fig_to_b64()
    (out_dir / "plots" / "roc.png").write_bytes(base64.b64decode(out["roc"].split(",")[1]))

    # PR
    pr, rc, _ = precision_recall_curve(y_true, oof_scores)
    plt.figure(figsize=(6, 4))
    plt.plot(rc, pr)
    plt.title("Domain PR")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    out["pr"] = fig_to_b64()
    (out_dir / "plots" / "pr.png").write_bytes(base64.b64decode(out["pr"].split(",")[1]))

    # Importances (top 30)
    if importances is not None and len(importances):
        top = importances.head(30).iloc[::-1]
        plt.figure(figsize=(7, 10))
        plt.barh(top["feature"].astype(str).values, top["importance"].values)
        plt.title("Top feature importances (domain)")
        out["importances"] = fig_to_b64()
        (out_dir / "plots" / "top_importances.png").write_bytes(base64.b64decode(out["importances"].split(",")[1]))

    # PSI top (числовые)
    if drift_df is not None and "psi" in drift_df.columns and drift_df["psi"].notna().any():
        dnum = drift_df.dropna(subset=["psi"]).sort_values("psi", ascending=False).head(30).iloc[::-1]
        if len(dnum):
            plt.figure(figsize=(7, 10))
            plt.barh(dnum["feature"].astype(str).values, dnum["psi"].values)
            plt.title("Top PSI (numeric)")
            out["psi_top"] = fig_to_b64()
            (out_dir / "plots" / "psi_top.png").write_bytes(base64.bdecode(out["psi_top"].split(",")[1]) if False else base64.b64decode(out["psi_top"].split(",")[1]))

    return out


def make_html_report(out_dir: Path,
                     tag: str,
                     domain_auc: float,
                     domain_ap: float,
                     plots_b64: Dict[str, str],
                     importances: Optional[pd.DataFrame],
                     drift_df: Optional[pd.DataFrame],
                     block_df: Optional[pd.DataFrame]):
    html = io.StringIO()
    def w(s=""):
        html.write(s + "\n")

    w("<html><head><meta charset='utf-8'><title>Adversarial Report</title></head><body>")
    w(f"<h1>Adversarial report — {tag}</h1>")
    w(f"<p><b>Domain AUC:</b> {domain_auc:.6f} &nbsp; <b>AP:</b> {domain_ap:.6f}</p>")

    if "roc" in plots_b64:
        w("<h2>ROC</h2>")
        w(f"<img src='{plots_b64['roc']}'/>")
    if "pr" in plots_b64:
        w("<h2>PR</h2>")
        w(f"<img src='{plots_b64['pr']}'/>")

    if "importances" in plots_b64:
        w("<h2>Top importances</h2>")
        w(f"<img src='{plots_b64['importances']}'/>")

    if "psi_top" in plots_b64:
        w("<h2>Top PSI</h2>")
        w(f"<img src='{plots_b64['psi_top']}'/>")

    if importances is not None and len(importances):
        w("<h2>Importances (head)</h2>")
        w(importances.head(50).to_html(index=False))

    if drift_df is not None and len(drift_df):
        w("<h2>PSI/KS/χ² (head)</h2>")
        w(drift_df.head(50).to_html(index=False))

    if block_df is not None and len(block_df):
        w("<h2>Block contribution</h2>")
        w(block_df.to_html(index=False))

    w("</body></html>")
    (out_dir / "report.html").write_text(html.getvalue(), encoding="utf-8")


# -------------------------- Main --------------------------

def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    sets_dir = Path(args.sets_dir) if args.sets_dir else (Path("artifacts") / "sets" / args.tag)
    out_dir = Path(args.out_dir) if args.out_dir else (Path("artifacts") / "adversarial" / args.tag)
    ensure_dir(out_dir)

    # Загрузка
    loaded = load_set(args.tag, args.layer, sets_dir, verbose=args.verbose)

    # Сэмпл
    Xtr0, Xte0, idx_tr_sub, idx_te_sub = maybe_sample_rows(loaded.Xtr, loaded.Xte, args.sample, args.seed)
    names = list(loaded.names)
    if isinstance(Xtr0, pd.DataFrame):
        names = list(Xtr0.columns)

    # Сборка domain-матрицы
    dset = build_domain_matrix(Xtr0, Xte0, names)

    # Фолды
    folds = make_folds_for_domain(
        dset=dset,
        folds_mode=args.folds,
        base_folds=loaded.folds,
        n_splits=args.n_splits,
        seed=args.seed
    )

    # Модель доменного классификатора
    clf_params = json.loads(args.clf_params) if args.clf_params else None
    if args.clf == "lr":
        res = fit_predict_domain_lr(dset, folds, args.balance, args.seed, clf_params, verbose=args.verbose)
    elif args.clf == "sgd":
        res = fit_predict_domain_sgd(dset, folds, args.balance, args.seed, clf_params, verbose=args.verbose)
    elif args.clf == "lgbm":
        res = fit_predict_domain_lgbm(dset, folds, args.balance, args.seed, clf_params, verbose=args.verbose)
    else:
        raise ValueError(args.clf)

    if args.verbose:
        print(f"[done] domain AUC={res.auc:.6f} AP={res.ap:.6f}")

    # Сохраним доменные метрики/скоры/важности
    (out_dir / "domain_auc.txt").write_text(f"{res.auc:.8f}")
    report = {
        "tag": args.tag,
        "layer": args.layer,
        "sample_frac": args.sample,
        "clf": args.clf,
        "clf_params": clf_params,
        "balance": args.balance,
        "folds": args.folds,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "domain_auc": float(res.auc),
        "domain_ap": float(res.ap),
        "n_train": int(dset.tr_off),
        "n_test": int(len(dset.te_idx)),
        "generated_at": datetime.now().isoformat()
    }

    # Важности
    if res.importances is not None:
        res.importances.to_csv(out_dir / "importances.csv", index=False)

    # PSI/KS/χ²
    drift_df = None
    if args.psi_ks and isinstance(Xtr0, pd.DataFrame):
        drift_df = drift_table_dense(Xtr0, Xte0, args.psi_top, args.drop_threshold, verbose=args.verbose)
        drift_df.to_csv(out_dir / "psi_ks.csv", index=False)

    # Блоки
    block_df = None
    if args.block_contrib:
        catalog = extract_catalog(loaded.meta)
        if len(catalog):
            block_df = aggregate_by_block(res.importances, drift_df, catalog)
            if block_df is not None and len(block_df):
                block_df.to_csv(out_dir / "block_contrib.csv", index=False)

    # Подозрительные + drop/keep
    susp = suspicious_features(res.importances, drift_df, top_k_imp=200, psi_thr=max(args.drop_threshold, 0.3))
    if len(susp):
        susp.to_csv(out_dir / "suspicious.csv", index=False)

    drop_list = []
    if drift_df is not None:
        drop_list = list(drift_df.loc[drift_df["psi"].fillna(0.0) >= args.drop_threshold, "feature"].astype(str).values)
    # добавим подозрительные по важности
    drop_list = sorted(set(drop_list).union(set(susp["feature"].values if len(susp) else [])))
    (out_dir / "drop_features.txt").write_text("\n".join(drop_list), encoding="utf-8")

    # keep — комплимент (если dense и есть колонки)
    keep_list = []
    if isinstance(Xtr0, pd.DataFrame):
        keep_list = [c for c in Xtr0.columns if c not in drop_list]
        (out_dir / "keep_features.txt").write_text("\n".join(keep_list), encoding="utf-8")

    # Importance weights для train (по OOF)
    # Важно: oof_train соотносится с train-подвыборкой (после sample). Нужно промапить обратно на полную выборку.
    # Если был сэмпл — создадим массив weights длиной исходного train, остальное=1.0.
    weights_full = None
    if args.weights != "off":
        oof_p = res.oof_train  # длина = len(Xtr0)
        w = compute_importance_weights(oof_p, cap=args.cap_weight)
        # map обратно к исходным индексам train
        weights_full = np.ones(loaded.Xtr.shape[0], dtype=float)
        weights_full[idx_tr_sub] = w
        np.save(out_dir / "train_weights.npy", weights_full)

        # также parquet (id + weight)
        wdf = pd.DataFrame({loaded.id_col: loaded.ydf.iloc[idx_tr_sub][loaded.id_col].values, "weight": w})
        wdf.to_parquet(out_dir / "train_weights.parquet", index=False)

    # сохраняем OOF-доменные скоры по train (в порядке исходной матрицы train)
    oof_full = np.zeros(loaded.Xtr.shape[0], dtype=float)
    oof_full[idx_tr_sub] = res.oof_train
    np.save(out_dir / "oof_domain.npy", oof_full)

    # ПЛОТЫ и HTML
    plots_b64 = {}
    if HAVE_MPL:
        # для ROC/PR используем y_true=0 на train (домен 0), но нам нужны 0/1 по train?
        # OOF рассчитаны для train-части, где "истина" — 0; для ROC нужен набор 0/1 — возьмём y=[0]*n и скоры p(test|x)
        y_true = np.zeros_like(res.oof_train, dtype=int)
        plots_b64 = make_plots(out_dir, y_true, res.oof_train, res.importances, drift_df)

    # JSON-отчёт
    save_json(out_dir / "report.json", report)
    if args.save_html and HAVE_MPL:
        make_html_report(out_dir, args.tag, res.auc, res.ap, plots_b64, res.importances, drift_df, block_df)

    # Обновления индекса моделей
    models_index = Path(args.models_index)
    idx = read_json(models_index) or {}
    idx_key = f"adversarial:{args.tag}"
    idx[idx_key] = {
        "tag": args.tag,
        "cand": "adversarial",
        "task": "diagnostic",
        "metric": "domain_auc",
        "cv_mean": float(res.auc),
        "cv_std": 0.0,
        "path": str(out_dir),
        "blend": False
    }
    save_json(models_index, idx)

    # Финальный принт
    print("=== ADVERSARIAL DONE ===")
    print("tag:", args.tag)
    print("out:", out_dir.as_posix())
    print(f"domain AUC={res.auc:.6f} AP={res.ap:.6f}")
    if weights_full is not None:
        print("train_weights:", (out_dir / "train_weights.npy").as_posix())
    print("drop_features:", (out_dir / "drop_features.txt").as_posix())
    if args.save_html and HAVE_MPL:
        print("html report:", (out_dir / "report.html").as_posix())


if __name__ == "__main__":
    main()
