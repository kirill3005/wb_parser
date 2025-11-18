#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/run_blend.py

Оркестратор блендинга/стакинга готовых прогонов (run_id) из artifacts/models/*.
Поддерживает:
- режимы: equal | dirichlet | nnls | coord | level2
- пространства бленда: proba | logit | rank
- fold-safe подбор весов (--cv-weights)
- калибровка (Platt/Isotonic) и подбор τ (binary)
- auto-topk по index.json
- сохранение артефактов бленда и обновление index.json

Примеры:
  Быстрый AUC-бленд (rank+dirichlet) для топ-3 запусков:
    python tools/run_blend.py --tag s5e11 \
      --auto-topk 3 --by cv_mean \
      --mode dirichlet --blend-space rank \
      --metric roc_auc --task auto \
      --dirichlet-samples 4000 --cv-weights \
      --save-test --name auc_rank_dir

  Fold-safe level-2 (binary) + Platt + τ(F1):
    python tools/run_blend.py --tag s5e11 \
      --members runA,runB,runC \
      --mode level2 --metric roc_auc --task binary \
      --calibrate platt --threshold f1 \
      --save-test --name l2_platt_f1
"""

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional deps
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        log_loss,
        accuracy_score,
        f1_score,
        mean_squared_error,
        mean_absolute_error,
    )
    from sklearn.preprocessing import label_binarize
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.isotonic import IsotonicRegression
except Exception as e:
    raise RuntimeError("Нужно: scikit-learn (pip install scikit-learn)") from e

# SciPy NNLS (не обязателен)
try:
    from scipy.optimize import nnls as scipy_nnls
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ------------------------ IO helpers ------------------------

def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def read_parquet_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        import fastparquet  # noqa: F401
        return pd.read_parquet(path, engine="fastparquet")


# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Blend/stack existing runs")

    p.add_argument("--tag", required=True, type=str, help="RUN_TAG (набор фич)")

    group_members = p.add_mutually_exclusive_group(required=True)
    group_members.add_argument("--members", type=str, help="Список run_id через запятую")
    group_members.add_argument("--auto-topk", type=int, help="Выбрать топ-K по index.json")
    p.add_argument("--by", type=str, default="cv_mean", help="Поле сортировки для --auto-topk (default: cv_mean)")

    p.add_argument("--mode", type=str, required=True,
                   choices=["equal", "dirichlet", "nnls", "coord", "level2"],
                   help="Режим бленда")

    p.add_argument("--task", type=str, default="auto",
                   choices=["auto", "binary", "multiclass", "regression", "multilabel"],
                   help="Тип задачи (auto пытается угадать по y)")

    p.add_argument("--metric", type=str, default="roc_auc",
                   choices=["roc_auc", "pr_auc", "logloss", "accuracy", "f1", "rmse", "mae", "mape"],
                   help="Целевая метрика для поиска весов")

    p.add_argument("--blend-space", type=str, default="proba",
                   choices=["proba", "logit", "rank"],
                   help="Пространство бленда")

    p.add_argument("--dirichlet-samples", type=int, default=4000, help="Кол-во сэмплов в dirichlet")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--nnls", action="store_true", help="Использовать NNLS как тёплый старт (для coord/regression)")
    p.add_argument("--coord-iters", type=int, default=2000, help="Итерации coordinate search")
    p.add_argument("--coord-step", type=float, default=0.01, help="Шаг изменения веса в coord")
    p.add_argument("--nonneg", action="store_true", help="Ограничить веса w>=0")
    p.add_argument("--sum-to-one", action="store_true", help="Проецировать веса на симплекс (сумма=1)")

    p.add_argument("--cv-weights", action="store_true",
                   help="Fold-safe подбор весов: на train-OOF каждого фолда подбираем веса, на вал-фолде применяем")
    p.add_argument("--reopt-after-cv", action="store_true",
                   help="После fold-safe можно дооптимизировать веса на полном OOF (осторожно с утечкой)")

    p.add_argument("--calibrate", type=str, default="off", choices=["off", "platt", "isotonic"], help="Калибровка (binary)")
    p.add_argument("--threshold", type=str, default="off",
                   help="off | f1 | youden | custom:0.42  (только binary)")

    p.add_argument("--sets-dir", type=str, default=None, help="Директория с сетом (по умолчанию artifacts/sets/<tag>)")
    p.add_argument("--models-index", type=str, default="artifacts/models/index.json", help="Путь к index.json")
    p.add_argument("--out-dir", type=str, default="artifacts/models/blends", help="Базовый каталог для блендов")
    p.add_argument("--name", type=str, default=None, help="Имя бленда (часть id)")

    p.add_argument("--analyze-only", action="store_true", help="Только анализ без сохранения")
    p.add_argument("--save-test", action="store_true", help="Сохранять test_pred.npy (если доступно)")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# ------------------------ Task & metric ------------------------

def detect_task(y: np.ndarray) -> str:
    # Если целевая колонка целочисленная и имеет <= 2 уникальных — binary
    # Если целевая целочисленная с >2 классами — multiclass
    # Иначе — regression
    y_unique = np.unique(y)
    if np.issubdtype(y.dtype, np.integer):
        if len(y_unique) <= 2:
            return "binary"
        else:
            return "multiclass"
    # иногда читается как float, но по факту 0/1
    if len(y_unique) <= 2 and set(np.unique(y)).issubset({0.0, 1.0}):
        return "binary"
    return "regression"


def get_metric_fn(task: str, metric: str):
    """
    Возвращает:
      scorer(y_true, y_pred) -> float  (чем больше, тем лучше)
      pretty_name (для сохранения)
      minimize_original (bool) — нужно ли сохранять "оригинал" как минимизируемую метрику (RMSE/MAE)
    """
    m = metric.lower()

    if task in ("binary", "multiclass", "multilabel"):
        if m in ("roc_auc", "auc"):
            def _f(y_true, y_pred):
                if task == "binary":
                    return roc_auc_score(y_true, y_pred.reshape(-1))
                elif task == "multiclass":
                    classes = np.unique(y_true)
                    Y = label_binarize(y_true, classes=classes)
                    return roc_auc_score(Y, y_pred, average="macro", multi_class="ovr")
                else:
                    # multilabel
                    return roc_auc_score(y_true, y_pred, average="macro")
            return _f, "roc_auc", False

        if m in ("pr_auc", "ap", "average_precision"):
            def _f(y_true, y_pred):
                if task == "binary":
                    return average_precision_score(y_true, y_pred.reshape(-1))
                elif task == "multiclass":
                    classes = np.unique(y_true)
                    Y = label_binarize(y_true, classes=classes)
                    return average_precision_score(Y, y_pred, average="macro")
                else:
                    return average_precision_score(y_true, y_pred, average="macro")
            return _f, "pr_auc", False

        if m == "logloss":
            def _f(y_true, y_pred):
                if task == "binary":
                    p = np.clip(y_pred.reshape(-1), 1e-15, 1 - 1e-15)
                    P = np.vstack([1 - p, p]).T
                    return -log_loss(y_true, P, labels=[0, 1])  # maximize -> отрицательный logloss
                elif task == "multiclass":
                    return -log_loss(y_true, y_pred)
                else:
                    raise ValueError("logloss неприменим к multilabel")
            return _f, "neg_logloss", False

        if m in ("accuracy", "acc"):
            def _f(y_true, y_pred):
                if task == "binary":
                    return accuracy_score(y_true, (y_pred.reshape(-1) >= 0.5).astype(int))
                elif task == "multiclass":
                    return accuracy_score(y_true, np.argmax(y_pred, axis=1))
                else:
                    raise ValueError
            return _f, "accuracy", False

        if m in ("f1", "macro_f1"):
            def _f(y_true, y_pred):
                if task == "binary":
                    return f1_score(y_true, (y_pred.reshape(-1) >= 0.5).astype(int))
                elif task == "multiclass":
                    return f1_score(y_true, np.argmax(y_pred, axis=1), average="macro")
                else:
                    raise ValueError
            return _f, "f1", False

    # Regression
    if m == "rmse":
        def _f(y_true, y_pred):
            return -math.sqrt(mean_squared_error(y_true, y_pred))  # maximize -> отрицательный RMSE
        return _f, "neg_rmse", True
    if m == "mae":
        def _f(y_true, y_pred):
            return -mean_absolute_error(y_true, y_pred)
        return _f, "neg_mae", True
    if m == "mape":
        def _f(y_true, y_pred):
            y_true = np.asarray(y_true, float)
            y_pred = np.asarray(y_pred, float)
            eps = 1e-9
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0
            return -mape
        return _f, "neg_mape", True

    raise ValueError(f"Unknown metric: {metric}")


# ------------------------ Load set & runs ------------------------

def load_set(sets_dir: Path) -> Tuple[np.ndarray, Optional[List[Tuple[np.ndarray, np.ndarray]]], pd.DataFrame, str]:
    y_path = sets_dir / "y_train.parquet"
    ids_test_path = sets_dir / "ids_test.parquet"
    folds_path = sets_dir / "folds.pkl"

    if not y_path.exists():
        raise FileNotFoundError(f"{y_path} not found")
    ydf = read_parquet_any(y_path)
    # id + target
    if ydf.shape[1] < 2:
        raise RuntimeError("y_train.parquet должен содержать id и target")
    id_col = ydf.columns[0]
    target_col = [c for c in ydf.columns if c != id_col][0]
    y = ydf[target_col].to_numpy()

    ids_test = read_parquet_any(ids_test_path) if ids_test_path.exists() else None

    folds = None
    if folds_path.exists():
        import pickle
        folds = pickle.loads(folds_path.read_bytes())

    return y, folds, ids_test, id_col


def load_index(index_path: Path) -> Dict[str, dict]:
    return load_json(index_path) or {}


def choose_auto_topk(index: Dict[str, dict], tag: str, by: str, k: int) -> List[str]:
    rows = []
    for rid, rec in index.items():
        if rec.get("tag") == tag and not rec.get("blend", False):
            val = rec.get(by)
            if val is None:
                continue
            try:
                rows.append((float(val), rid))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0], reverse=True)
    return [rid for _, rid in rows[:k]]


def load_member_run(index: Dict[str, dict], run_id: str) -> Dict[str, Any]:
    rec = index.get(run_id)
    if rec is None:
        raise KeyError(f"run_id not found in index.json: {run_id}")
    path = Path(rec.get("path", ""))
    if not path.exists():
        # fallback: artifacts/models/<run_id>
        alt = Path("artifacts") / "models" / run_id
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Model path not found for {run_id}")
    oof_p = path / "oof.npy"
    test_p = path / "test_pred.npy"
    if not oof_p.exists():
        raise FileNotFoundError(f"oof.npy missing for {run_id}")
    oof = np.load(oof_p)
    test = np.load(test_p) if test_p.exists() else None
    return {"run_id": run_id, "path": path, "oof": oof, "test": test, "meta": rec}


# ------------------------ Normalization / spaces ------------------------

def normalize_preds_task(task: str, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if task == "binary":
        if a.ndim == 1:
            return a
        if a.ndim == 2:
            if a.shape[1] == 1:
                return a.reshape(-1)
            if a.shape[1] == 2:
                return a[:, 1]
            # если >2 колонок — возьмём max proba (худший случай)
            return a.max(axis=1)
        return a.reshape(-1)
    elif task == "multiclass":
        if a.ndim == 1:
            raise ValueError("multiclass predictions must be (n, C)")
        return a
    else:
        # regression
        return a.reshape(-1)


def clip_proba(p, eps=1e-6):
    return np.clip(p, eps, 1 - eps)


def to_blend_space(task: str, X_list: List[np.ndarray], space: str) -> List[np.ndarray]:
    """
    Преобразовать список предсказаний в нужное пространство для бленда.
    - proba: как есть (для binary клип для логитов)
    - logit: log(p/(1-p)) для binary; multiclass — по каждому классу; regression — как есть
    - rank: ранги/квантили по колонке
    """
    out = []
    space = space.lower()
    for X in X_list:
        if space == "proba":
            if task == "binary":
                out.append(clip_proba(X))
            elif task == "multiclass":
                # клип по всем классам
                out.append(np.clip(X, 1e-8, 1 - 1e-8))
            else:
                out.append(X)
        elif space == "logit":
            if task == "binary":
                p = clip_proba(X)
                out.append(np.log(p / (1 - p)))
            elif task == "multiclass":
                P = np.clip(X, 1e-8, 1 - 1e-8)
                # логиты по одному классу относительно класса 0 (необязательно), но для усреднения норм ок
                out.append(np.log(P / np.clip(1 - P, 1e-8, None)))
            else:
                out.append(X)
        elif space == "rank":
            if task in ("binary", "regression"):
                r = pd.Series(X).rank(method="average").to_numpy()
                r = (r - 1) / max(len(r) - 1, 1)
                out.append(r)
            elif task == "multiclass":
                R = np.zeros_like(X, dtype=float)
                for c in range(X.shape[1]):
                    rc = pd.Series(X[:, c]).rank(method="average").to_numpy()
                    rc = (rc - 1) / max(len(rc) - 1, 1)
                    R[:, c] = rc
                out.append(R)
            else:
                out.append(X)
        else:
            raise ValueError(f"unknown blend-space: {space}")
    return out


def from_blend_space(task: str, y_blend: np.ndarray, space: str) -> np.ndarray:
    space = space.lower()
    if space == "logit":
        # обратная сигмоида только для binary; для multiclass – оставляем как есть (нормировка ниже)
        if task == "binary":
            return 1.0 / (1.0 + np.exp(-y_blend))
        elif task == "multiclass":
            # при стакинге логитов для многокласса лучше нормировать в proba
            P = 1.0 / (1.0 + np.exp(-y_blend))
            # нормировка по строке
            P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-8, None)
            return P
        return y_blend
    elif space == "rank":
        # ранги уже [0..1], ок
        return y_blend
    return y_blend


# ------------------------ Weight search ------------------------

def equal_weights(m: int) -> np.ndarray:
    return np.ones(m, dtype=float) / float(m)


def dirichlet_search(Y: np.ndarray, y: np.ndarray, scorer, n_samples: int, seed: int) -> np.ndarray:
    """
    Y: (n, m) бинарный/регрессионный стек или (n, m, C) в мультиклассе
    """
    rng = np.random.default_rng(seed)
    m = Y.shape[1]
    best_w, best_s = None, -1e18
    for _ in range(n_samples):
        w = rng.dirichlet([1.0] * m)
        if Y.ndim == 3:
            # (n, m, C) -> (n, C)
            y_hat = np.tensordot(Y, w, axes=(1, 0))  # суммирование по m
        else:
            y_hat = (Y @ w).reshape(-1)
        s = scorer(y, y_hat)
        if s > best_s:
            best_s, best_w = s, w
    return best_w


def nnls_weights(Y: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    NNLS по MSE: только для regression/aux-инициализации. Возвращает w>=0.
    """
    n, m = Y.shape
    if HAS_SCIPY:
        w, _ = scipy_nnls(Y, y)
        s = w.sum()
        if s > 0:
            w = w / s
        return w
    # fallback: псевдорешение + усечение
    w, *_ = np.linalg.lstsq(Y, y, rcond=None)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = equal_weights(m)
    return w


def project_simplex(w: np.ndarray) -> np.ndarray:
    # Euclidean projection onto simplex {w: w>=0, sum=1}
    # Algorithm: Wang & Carreira-Perpinan (2013)
    v = np.sort(w)[::-1]
    cssv = np.cumsum(v)
    rho = np.nonzero(v * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(w - theta, 0)
    return w


def coord_search(
    Y: np.ndarray,
    y: np.ndarray,
    scorer,
    iters: int = 2000,
    step: float = 0.01,
    nonneg: bool = True,
    sum_to_one: bool = True,
    init_w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Простой coordinate/greedy поиск весов, улучшая метрику.
    """
    n, m = Y.shape[:2]
    if init_w is None:
        w = equal_weights(m)
    else:
        w = init_w.copy().astype(float)

    def regularize_weights(w):
        if nonneg:
            w = np.clip(w, 0, None)
        if sum_to_one:
            s = w.sum()
            if s <= 0:
                w = equal_weights(m)
            else:
                w = w / s
        return w

    def combine(w):
        if Y.ndim == 3:
            return np.tensordot(Y, w, axes=(1, 0))
        return (Y @ w).reshape(-1)

    y_hat = combine(w)
    best_s = scorer(y, y_hat)
    rng = np.random.default_rng(123)

    for _ in range(iters):
        j = int(rng.integers(0, m))
        # пробуем +step/-step
        trial_up = w.copy()
        trial_up[j] += step
        trial_up = regularize_weights(trial_up)
        s_up = scorer(y, combine(trial_up))

        trial_dn = w.copy()
        trial_dn[j] = max(0.0, trial_dn[j] - step) if nonneg else trial_dn[j] - step
        trial_dn = regularize_weights(trial_dn)
        s_dn = scorer(y, combine(trial_dn))

        if s_up > best_s or s_dn > best_s:
            if s_up >= s_dn:
                w, best_s = trial_up, s_up
            else:
                w, best_s = trial_dn, s_dn
    return w


def fold_safe_weights(
    Y: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    mode: str,
    scorer,
    dirichlet_samples: int,
    seed: int,
    coord_iters: int,
    coord_step: float,
    nonneg: bool,
    sum_to_one: bool,
    use_nnls: bool,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[np.ndarray]]:
    """
    Для каждого фолда подбираем веса на train-OOF, предсказываем вал.
    Возвращает:
      oof_blend (n,),
      test_weights_avg (усреднение весов по фолдам),
      fold_scores,
      fold_weights (список весов по фолдам)
    """
    n = Y.shape[0]
    m = Y.shape[1]
    oof_bl = np.zeros((n, Y.shape[2])) if Y.ndim == 3 else np.zeros(n)
    fold_scores = []
    fold_ws = []
    rng = np.random.default_rng(seed)

    for (tr_idx, va_idx) in folds:
        if Y.ndim == 3:
            Y_tr = Y[tr_idx, :, :]
            Y_va = Y[va_idx, :, :]
        else:
            Y_tr = Y[tr_idx, :]
            Y_va = Y[va_idx, :]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # warm start
        w0 = None
        if mode == "coord" and use_nnls and Y.ndim == 2:
            w0 = nnls_weights(Y_tr, y_tr)

        if mode == "equal":
            w = equal_weights(m)
        elif mode == "dirichlet":
            w = dirichlet_search(Y_tr, y_tr, scorer, dirichlet_samples, int(rng.integers(0, 10**9)))
        elif mode == "nnls":
            if Y.ndim != 2:
                w = equal_weights(m)
            else:
                w = nnls_weights(Y_tr, y_tr)
        elif mode == "coord":
            if w0 is None:
                w0 = equal_weights(m)
            w = coord_search(
                Y_tr, y_tr, scorer,
                iters=coord_iters, step=coord_step,
                nonneg=nonneg, sum_to_one=sum_to_one, init_w=w0
            )
        else:
            raise ValueError(f"fold-safe не применим к mode={mode}")

        fold_ws.append(w)
        # применяем на вал
        if Y.ndim == 3:
            yhat_va = np.tensordot(Y_va, w, axes=(1, 0))
            oof_bl[va_idx, :] = yhat_va
            s = scorer(y_va, yhat_va)
        else:
            yhat_va = (Y_va @ w).reshape(-1)
            oof_bl[va_idx] = yhat_va
            s = scorer(y_va, yhat_va)
        fold_scores.append(float(s))

    # усредняем веса по фолдам
    W = np.vstack(fold_ws)
    w_avg = W.mean(axis=0)
    if sum_to_one:
        s = w_avg.sum()
        if s > 0:
            w_avg = w_avg / s
        else:
            w_avg = equal_weights(m)
    if nonneg:
        w_avg = np.clip(w_avg, 0, None)
        if sum_to_one:
            w_avg = w_avg / max(w_avg.sum(), 1e-9)
    return oof_bl, w_avg, fold_scores, fold_ws


# ------------------------ Level-2 (stacking) ------------------------

def build_stack_features(task: str, preds_list: List[np.ndarray]) -> np.ndarray:
    if task in ("binary", "regression"):
        X = np.column_stack([p.reshape(-1) for p in preds_list])
        return X
    elif task == "multiclass":
        # конкатенируем proba по классам
        return np.hstack(preds_list)
    else:
        raise ValueError("multilabel stacking не реализован")


def fold_safe_level2(
    task: str,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_oof: np.ndarray,
    X_test: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
    """
    Возвращает (oof_blend, test_blend, fitted_model)
    """
    n = X_oof.shape[0]
    if task == "binary":
        oof_bl = np.zeros(n, dtype=float)
        for tr_idx, va_idx in folds:
            mdl = LogisticRegression(max_iter=1000)
            mdl.fit(X_oof[tr_idx], y[tr_idx].astype(int))
            oof_bl[va_idx] = mdl.predict_proba(X_oof[va_idx])[:, 1]
        # финальная модель на всех для test
        mdl = LogisticRegression(max_iter=1000)
        mdl.fit(X_oof, y.astype(int))
        test_bl = None if X_test is None else mdl.predict_proba(X_test)[:, 1]
        return oof_bl, test_bl, mdl

    elif task == "multiclass":
        # Ridge one-vs-rest через многомерную целевую
        classes = np.unique(y)
        Ybin = pd.get_dummies(y).values  # (n, C)
        oof_bl = np.zeros_like(Ybin, dtype=float)
        for tr_idx, va_idx in folds:
            mdl = Ridge(alpha=1.0)
            mdl.fit(X_oof[tr_idx], Ybin[tr_idx])
            P = mdl.predict(X_oof[va_idx])
            P = np.clip(P, 1e-8, None)
            P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-8, None)
            oof_bl[va_idx] = P
        mdl = Ridge(alpha=1.0)
        mdl.fit(X_oof, Ybin)
        if X_test is not None:
            T = mdl.predict(X_test)
            T = np.clip(T, 1e-8, None)
            T = T / np.clip(T.sum(axis=1, keepdims=True), 1e-8, None)
        else:
            T = None
        return oof_bl, T, mdl

    else:  # regression
        oof_bl = np.zeros(n, dtype=float)
        for tr_idx, va_idx in folds:
            mdl = Ridge(alpha=1.0)
            mdl.fit(X_oof[tr_idx], y[tr_idx])
            oof_bl[va_idx] = mdl.predict(X_oof[va_idx])
        mdl = Ridge(alpha=1.0)
        mdl.fit(X_oof, y)
        test_bl = None if X_test is None else mdl.predict(X_test)
        return oof_bl, test_bl, mdl


# ------------------------ Calibration & threshold ------------------------

def fit_calibrator(method: str, y: np.ndarray, p: np.ndarray):
    method = method.lower()
    if method in ("off", "none", ""):
        return None, (lambda z: z)
    if method == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p.reshape(-1, 1), y.astype(int))
        return ("platt", lr), (lambda z: lr.predict_proba(z.reshape(-1, 1))[:, 1])
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p.reshape(-1), y.astype(int))
        return ("isotonic", ir), (lambda z: ir.predict(z.reshape(-1)))
    raise ValueError(f"Unknown calibrator: {method}")


def find_tau(strategy: str, y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    strategy = strategy.lower()
    if strategy.startswith("custom:"):
        t = float(strategy.split(":", 1)[1])
        # F1 для отчёта
        return t, f1_score(y, (p >= t).astype(int))
    if strategy == "f1":
        best, best_t = -1.0, 0.5
        for t in np.linspace(0, 1, 401):
            s = f1_score(y, (p >= t).astype(int))
            if s > best:
                best, best_t = s, t
        return best_t, best
    if strategy == "youden":
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y, p)
        j = tpr - fpr
        i = int(np.argmax(j))
        return float(thr[i]), float(j[i])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


# ------------------------ Main ------------------------

def main():
    args = parse_args()

    tag = args.tag
    sets_dir = Path(args.sets_dir) if args.sets_dir else (Path("artifacts") / "sets" / tag)
    index_path = Path(args.models_index)
    blends_root = Path(args.out_dir)

    # Загрузка сетов
    y, folds, ids_test, id_col = load_set(sets_dir)
    task = args.task if args.task != "auto" else detect_task(y)
    scorer, metric_name, minimize_orig = get_metric_fn(task, args.metric)

    # Загрузка индекса и выбор участников
    index = load_index(index_path)
    if args.members:
        members = [m.strip() for m in args.members.split(",") if m.strip()]
    else:
        members = choose_auto_topk(index, tag, args.by, args.auto_topk)
        if not members:
            raise RuntimeError("auto-topk не нашёл кандидатов")

    if args.verbose:
        print(f"[info] tag={tag} task={task} metric={metric_name} members={members}")

    # Загрузка run_id предсказаний
    loaded = [load_member_run(index, rid) for rid in members]

    # Нормализация форм под task
    oof_list = [normalize_preds_task(task, d["oof"]) for d in loaded]
    test_list = [None if d["test"] is None else normalize_preds_task(task, d["test"]) for d in loaded]

    # Согласование размеров
    n = len(y)
    for i, p in enumerate(oof_list):
        if task == "multiclass":
            if p.shape[0] != n:
                raise ValueError(f"OOF length mismatch for {members[i]}")
        else:
            if p.shape[0] != n:
                raise ValueError(f"OOF length mismatch for {members[i]}")

    # Blend space преобразование
    oof_space = to_blend_space(task, oof_list, args.blend_space)
    test_space = None
    if args.save_test and all([t is not None for t in test_list]):
        test_space = to_blend_space(task, test_list, args.blend_space)

    # Стеки
    if task == "multiclass":
        # (n, m, C)
        Y_oof = np.stack(oof_space, axis=1)
        Y_test = None if test_space is None else np.stack(test_space, axis=1)
    else:
        # (n, m)
        Y_oof = np.column_stack(oof_space)
        Y_test = None if test_space is None else np.column_stack(test_space)

    # Блендинг
    mode = args.mode.lower()
    rng = np.random.default_rng(args.seed)

    weights = None
    fold_scores = None
    fold_weights = None

    if mode in ("equal", "dirichlet", "nnls", "coord"):
        if args.cv_weights and folds is not None:
            # fold-safe подбор весов
            oof_bl, w_avg, fold_scores, fold_weights = fold_safe_weights(
                Y=Y_oof,
                y=y,
                folds=folds,
                mode=mode,
                scorer=scorer,
                dirichlet_samples=args.dirichlet_samples,
                seed=args.seed,
                coord_iters=args.coord_iters,
                coord_step=args.coord_step,
                nonneg=args.nonneg,
                sum_to_one=args.sum_to_one,
                use_nnls=args.nnls
            )
            weights = w_avg
            if args.reopt_after_cv:
                # дооптимизировать на полном OOF (осторожно с утечкой!)
                if mode == "dirichlet":
                    w0 = dirichlet_search(Y_oof, y, scorer, args.dirichlet_samples, args.seed)
                elif mode == "coord":
                    init_w = weights if weights is not None else equal_weights(Y_oof.shape[1])
                    w0 = coord_search(
                        Y_oof, y, scorer,
                        iters=args.coord_iters, step=args.coord_step,
                        nonneg=args.nonneg, sum_to_one=args.sum_to_one, init_w=init_w
                    )
                elif mode == "nnls" and Y_oof.ndim == 2:
                    w0 = nnls_weights(Y_oof, y)
                else:
                    w0 = equal_weights(Y_oof.shape[1])
                weights = w0
                # пересобираем oof_bl (без fold-safe)
                if Y_oof.ndim == 3:
                    oof_bl = np.tensordot(Y_oof, weights, axes=(1, 0))
                else:
                    oof_bl = (Y_oof @ weights).reshape(-1)
        else:
            # глобальный подбор
            if mode == "equal":
                weights = equal_weights(Y_oof.shape[1])
            elif mode == "dirichlet":
                weights = dirichlet_search(Y_oof, y, scorer, args.dirichlet_samples, args.seed)
            elif mode == "nnls":
                if Y_oof.ndim == 3:
                    # не определено — fallback на equal
                    weights = equal_weights(Y_oof.shape[1])
                else:
                    weights = nnls_weights(Y_oof, y)
            elif mode == "coord":
                init_w = None
                if args.nnls and Y_oof.ndim == 2:
                    init_w = nnls_weights(Y_oof, y)
                weights = coord_search(
                    Y_oof, y, scorer,
                    iters=args.coord_iters, step=args.coord_step,
                    nonneg=args.nonneg, sum_to_one=args.sum_to_one, init_w=init_w
                )
            else:
                raise ValueError
            # собрать oof_bl
            if Y_oof.ndim == 3:
                oof_bl = np.tensordot(Y_oof, weights, axes=(1, 0))
            else:
                oof_bl = (Y_oof @ weights).reshape(-1)

        # собрать test_bl
        if Y_test is not None and weights is not None:
            if Y_test.ndim == 3:
                test_bl = np.tensordot(Y_test, weights, axes=(1, 0))
            else:
                test_bl = (Y_test @ weights).reshape(-1)
        else:
            test_bl = None

    elif mode == "level2":
        # Стэкинг в исходном (proba/rank/logit) пространстве уже учтён в X
        X_oof = build_stack_features(task, oof_space)
        X_test = None if test_space is None else build_stack_features(task, test_space)
        if folds is None:
            raise RuntimeError("Для level2 нужен folds.pkl (fold-safe)")
        oof_bl, test_bl, model = fold_safe_level2(task, y, folds, X_oof, X_test)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Обратно из пространства бленда → в вероятности/скоры
    if mode != "level2":
        oof_bl = from_blend_space(task, oof_bl, args.blend_space)
        test_bl = None if test_bl is None else from_blend_space(task, test_bl, args.blend_space)

    # Метрики на OOF
    score = float(get_metric_fn(task, args.metric)[0](y, oof_bl))
    # Для отчёта: "положительная" метрика
    report_metric = score
    # и оригинальная (положительная) для RMSE/MAE/MAPE
    if args.metric in ("rmse", "mae", "mape"):
        report_metric = float(-score)

    # Калибровка / τ — только binary
    calib_obj = None
    tau = None
    if task == "binary":
        cal_mode = args.calibrate.lower()
        if cal_mode != "off":
            calib_obj, apply_fn = fit_calibrator(cal_mode, y, oof_bl.reshape(-1))
            oof_cal = apply_fn(oof_bl.reshape(-1))
            score_cal = float(get_metric_fn(task, args.metric)[0](y, oof_cal))
            report_metric_cal = score_cal if args.metric not in ("rmse", "mae", "mape") else float(-score_cal)
            if args.verbose:
                print(f"[calibration] {cal_mode}: oof_metric={report_metric_cal:.6f} (base={report_metric:.6f})")
            oof_bl = oof_cal
            if test_bl is not None:
                test_bl = apply_fn(test_bl.reshape(-1))
            # обновим report_metric на калиброванный
            report_metric = report_metric_cal

        thr_mode = args.threshold.lower()
        if thr_mode != "off":
            tau, tscore = find_tau(thr_mode, y, oof_bl.reshape(-1))
            if args.verbose:
                print(f"[threshold] {thr_mode}: tau={tau:.6f} score={tscore:.6f}")

    # Analyze-only
    if args.analyze_only:
        print("=== ANALYZE-ONLY ===")
        print("members:", members)
        print("mode:", mode, "| space:", args.blend_space)
        if weights is not None:
            print("weights:", np.round(weights, 6).tolist())
        if fold_scores is not None:
            print("fold_scores:", [float(x) for x in fold_scores])
        print(f"oof_metric({args.metric}) = {report_metric:.6f}")
        return

    # Сохранение артефактов
    blend_name = args.name or f"{mode}_{args.blend_space}"
    blend_id = f"blend_{blend_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = (blends_root / blend_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # OOF/Test
    np.save(out_dir / "oof.npy", oof_bl)
    if args.save_test and test_bl is not None:
        np.save(out_dir / "test_pred.npy", test_bl)

    # members.json
    members_map = {d["run_id"]: str(d["path"]) for d in loaded}
    save_json(out_dir / "members.json", members_map)

    # weights.json
    if weights is not None:
        save_json(out_dir / "weights.json", {rid: float(w) for rid, w in zip(members, weights)})

    # fold_weights.json
    if fold_weights is not None:
        fw = [{rid: float(wi) for rid, wi in zip(members, w)} for w in fold_weights]
        save_json(out_dir / "fold_weights.json", {"fold_weights": fw})

    # model/cali/τ
    if mode == "level2":
        try:
            import joblib
            joblib.dump(model, out_dir / "level2.joblib")
        except Exception:
            pass

    if calib_obj is not None:
        try:
            import joblib
            save_json(out_dir / "calibrator.json", {"type": calib_obj[0]})
            joblib.dump(calib_obj[1], out_dir / "calibrator.joblib")
        except Exception:
            pass

    if tau is not None:
        save_json(out_dir / "thresholds.json", {"tau": float(tau)})

    # metrics.json
    metrics_out = {
        "oof_metric_name": args.metric,
        "oof_metric": float(report_metric),
        "fold_scores": None if fold_scores is None else [float(x) for x in fold_scores],
        "cv_weights": bool(args.cv_weights),
        "blend_space": args.blend_space,
        "mode": mode,
        "calibration": args.calibrate,
        "threshold": args.threshold,
        "members": members
    }
    save_json(out_dir / "metrics.json", metrics_out)

    # config.json
    cfg = {
        "tag": tag,
        "task": task,
        "metric": args.metric,
        "blend_space": args.blend_space,
        "mode": mode,
        "params": {
            "dirichlet_samples": args.dirichlet_samples,
            "coord_iters": args.coord_iters,
            "coord_step": args.coord_step,
            "nonneg": args.nonneg,
            "sum_to_one": args.sum_to_one,
            "cv_weights": args.cv_weights,
            "reopt_after_cv": args.reopt_after_cv,
            "nnls": args.nnls,
        },
        "calibration": args.calibrate,
        "threshold": args.threshold,
        "seed": args.seed,
        "members": members
    }
    save_json(out_dir / "config.json", cfg)

    # Обновление index.json
    index.setdefault(blend_id, {})
    index[blend_id].update({
        "tag": tag,
        "cand": f"blend:{mode}",
        "task": task,
        "metric": args.metric,
        "cv_mean": float(report_metric),
        "cv_std": 0.0,
        "path": str(out_dir),
        "blend": True
    })
    save_json(index_path, index)

    print("=== BLEND SAVED ===")
    print("id:", blend_id)
    print("dir:", out_dir.as_posix())
    print(f"oof_metric({args.metric}) = {report_metric:.6f}")
    if weights is not None:
        print("weights:", np.round(weights, 6).tolist())
    if fold_scores is not None:
        print("fold_scores:", [float(x) for x in fold_scores])


if __name__ == "__main__":
    main()
