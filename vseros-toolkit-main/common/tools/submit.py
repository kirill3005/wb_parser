#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/submit.py

Сборщик сабмита из артефактов моделей/блендов.

Входы:
- artifacts/sets/<tag>/ids_test.parquet
- artifacts/sets/<tag>/meta.json  (опц., submit-схема)
- artifacts/models/<run_id>/test_pred.npy  (или test_post.npy)
- artifacts/models/<run_id>/calibrator.json|calibrator.joblib (опц.)
- artifacts/models/<run_id>/thresholds.json (опц.)
- artifacts/models/<run_id>/oof.npy (для auto threshold, опц.)
- artifacts/models/<run_id>/metrics.json (опц.)
- artifacts/adversarial/<tag>/train_weights.npy (опц., для подбора τ)

Выходы:
- artifacts/submissions/<tag>/<name>/submission.csv
- preview_head.csv, manifest.json, kaggle_push.log (опц.)
- обновление artifacts/models/index.json секцией submit:<tag>:<name>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Submission builder over precomputed model artifacts")

    p.add_argument("--tag", required=True, type=str, help="RUN_TAG (dataset/feature set id)")

    p.add_argument("--runs", required=True, type=str,
                   help="Список run_id через запятую, опционально с весами: run1=0.6,run2=0.4")

    p.add_argument("--task", type=str, required=True,
                   choices=["binary", "multiclass", "multilabel", "regression"])

    p.add_argument("--metric", type=str, default="roc_auc",
                   help="Целевая метрика (используется как подсказка и для подбора τ/Top-K в auto): "
                        "binary: roc_auc|pr_auc|logloss|accuracy|f1; "
                        "multiclass: logloss|accuracy|macro_f1; regression: rmse|mae|mape|r2")

    p.add_argument("--id-col", type=str, default=None, help="Имя id-колонки (если нет в meta/sample)")

    p.add_argument("--target-cols", type=str, default=None,
                   help="Список целевых колонок через запятую (для multiclass/multilabel). "
                        "Для binary/regression одна колонка; если не задано, попытаемся взять из meta/sample.")

    p.add_argument("--use-calibrator", type=str, default="auto", choices=["auto", "on", "off"],
                   help="Применять калибратор Platt/Isotonic к финальным предсказаниям")

    p.add_argument("--threshold", type=str, default="auto",
                   help="Пороги: auto|<float>|per-class:<json>|topk:<int>. "
                        "Для AUC/PR/logloss по умолчанию не порогует.")

    p.add_argument("--threshold-metric", type=str, default=None,
                   help="Метрика для автоподбора τ/Top-K по OOF (если threshold=auto). "
                        "По умолчанию = --metric.")

    p.add_argument("--weights", type=str, default="none", choices=["adv", "none"],
                   help="Учитывать adversarial-веса при автоподборе τ/Top-K (если есть).")

    p.add_argument("--clip", type=str, default=None,
                   help="Клип предсказаний: '0,1' для вероятностей; по умолчанию для binary/multi=0,1.")

    p.add_argument("--round", type=int, default=6, help="Округление при записи CSV")

    p.add_argument("--blend-mode", type=str, default="mean", choices=["mean", "median", "wmean", "rank"],
                   help="Способ блендинга run'ов")

    p.add_argument("--sample-sub", type=str, default=None,
                   help="Путь к sample_submission.csv для приведения формата")

    p.add_argument("--name", type=str, default=None, help="Имя подпапки сабмита (иначе авто)")

    p.add_argument("--kaggle-compet", type=str, default=None, help="Слаг соревнования для kaggle CLI submit")

    p.add_argument("--kaggle-message", type=str, default="", help="Комментарий при сабмите в Kaggle")

    p.add_argument("--dry-run", action="store_true", help="Не писать файлы и не пушить в Kaggle, только валидировать")

    p.add_argument("--models-index", type=str, default="artifacts/models/index.json",
                   help="Индекс моделей, куда добавится запись о сабмите")

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
            import fastparquet  # noqa
            return pd.read_parquet(p, engine="fastparquet")
        except Exception:
            return None


# ----------------------------- Schema & IDs -----------------------------

@dataclass
class SubmitSchema:
    id_col: str
    target_cols: List[str]
    sample: Optional[pd.DataFrame]


def load_schema(tag: str, id_col_cli: Optional[str], target_cols_cli: Optional[str],
                sample_path: Optional[str], verbose=False) -> SubmitSchema:
    sets_dir = Path("artifacts") / "sets" / tag
    meta = read_json(sets_dir / "meta.json") or {}
    submit_meta = meta.get("submit", {}) if isinstance(meta, dict) else {}

    # 1) sample
    sample = None
    if sample_path:
        sp = Path(sample_path)
        if sp.exists():
            try:
                sample = pd.read_csv(sp)
                if verbose:
                    print(f"[info] loaded sample_submission.csv: {sp.as_posix()} shape={sample.shape}")
            except Exception:
                sample = None

    # 2) id_col
    id_col = id_col_cli or submit_meta.get("id_col")
    if not id_col and sample is not None:
        id_col = sample.columns[0]
        if verbose:
            print(f"[info] id_col from sample: {id_col}")
    if not id_col:
        id_col = "id"  # дефолт
        if verbose:
            print("[warn] id_col not provided; defaulting to 'id'")

    # 3) target_cols
    target_cols = None
    if target_cols_cli:
        target_cols = [c.strip() for c in target_cols_cli.split(",") if c.strip()]
    elif isinstance(submit_meta, dict) and "target_cols" in submit_meta:
        target_cols = list(submit_meta["target_cols"])
    elif sample is not None:
        target_cols = list(sample.columns[1:])

    if target_cols is None:
        # для binary/regression допустимо 1 колонка по умолчанию
        target_cols = ["target"]
        if verbose:
            print("[warn] target_cols not provided; defaulting to ['target']")

    return SubmitSchema(id_col=id_col, target_cols=target_cols, sample=sample)


def load_ids_test(tag: str, id_col: str, sample: Optional[pd.DataFrame], verbose=False) -> pd.Series:
    sets_dir = Path("artifacts") / "sets" / tag
    p = sets_dir / "ids_test.parquet"
    if p.exists():
        df = read_parquet_any(p)
        if df is None or id_col not in df.columns:
            raise FileNotFoundError(f"ids_test.parquet не найден или не содержит колонку {id_col}")
        if verbose:
            print("[info] ids_test.parquet:", df.shape)
        return df[id_col].astype(df[id_col].dtype)
    # fallback: из sample
    if sample is not None:
        if verbose:
            print("[warn] ids_test.parquet отсутствует — берём id из sample_submission.csv")
        return sample.iloc[:, 0]
    raise FileNotFoundError("Не найден ids_test.parquet и не передан sample_submission.csv")


# ----------------------------- Runs & predictions -----------------------------

@dataclass
class RunArtifacts:
    run_id: str
    test_pred: np.ndarray
    oof: Optional[np.ndarray]
    calibrator: Optional[dict]  # или joblib-объект (см. ниже)
    thresholds: Optional[dict]
    metrics: Optional[dict]
    path: Path


def _load_npy_any(dir_path: Path, names: List[str]) -> Optional[np.ndarray]:
    for n in names:
        p = dir_path / n
        if p.exists():
            try:
                return np.load(p, allow_pickle=False)
            except Exception:
                pass
    # поищем первый *.npy с ключевым словом
    for n in names:
        key = n.replace(".npy", "").lower()
        for p in dir_path.glob("*.npy"):
            if key in p.name.lower():
                try:
                    return np.load(p, allow_pickle=False)
                except Exception:
                    pass
    return None


def _load_calibrator(run_dir: Path, verbose=False):
    # варианты: calibrator.json (наш формат), calibrator.joblib (sklearn Isotonic/CalibratedClassifier)
    jj = run_dir / "calibrator.json"
    if jj.exists():
        try:
            obj = json.loads(jj.read_text(encoding="utf-8"))
            if verbose:
                print("[info] calibrator.json loaded")
            return obj
        except Exception as e:
            if verbose:
                print("[warn] failed to read calibrator.json:", e)
    jb = run_dir / "calibrator.joblib"
    if jb.exists():
        try:
            import joblib
            obj = joblib.load(jb)
            if verbose:
                print("[info] calibrator.joblib loaded")
            return obj
        except Exception as e:
            if verbose:
                print("[warn] failed to load calibrator.joblib:", e)
    return None


def load_run(run_id: str, verbose=False) -> RunArtifacts:
    run_dir = Path("artifacts") / "models" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_dir}")

    test_pred = _load_npy_any(run_dir, ["test_post.npy", "test_pred.npy", "test.npy"])
    if test_pred is None:
        raise FileNotFoundError(f"test_pred not found in {run_dir}")

    oof = _load_npy_any(run_dir, ["oof.npy", "oof_pred.npy", "oof_proba.npy"])
    thresholds = read_json(run_dir / "thresholds.json")
    metrics = read_json(run_dir / "metrics.json")
    calibrator = _load_calibrator(run_dir, verbose=verbose)

    if verbose:
        shp = test_pred.shape
        print(f"[info] run {run_id}: test_pred shape={shp}, oof={'yes' if oof is not None else 'no'}, "
              f"calib={'yes' if calibrator is not None else 'no'}, thr={'yes' if thresholds else 'no'}")

    return RunArtifacts(run_id=run_id, test_pred=test_pred, oof=oof,
                        calibrator=calibrator, thresholds=thresholds, metrics=metrics, path=run_dir)


def parse_runs_weights(runs_str: str) -> Tuple[List[str], List[float]]:
    items = [s.strip() for s in runs_str.split(",") if s.strip()]
    run_ids, ws = [], []
    for it in items:
        if "=" in it:
            rid, w = it.split("=", 1)
            run_ids.append(rid.strip())
            ws.append(float(w))
        else:
            run_ids.append(it)
            ws.append(1.0)
    # нормализация весов для wmean (для mean/median/rank игнорим)
    total = sum(ws)
    if total <= 0:
        ws = [1.0 / max(len(ws), 1)] * len(ws)
    else:
        ws = [w / total for w in ws]
    return run_ids, ws


# ----------------------------- Probas & blending -----------------------------

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def to_proba_binary(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    # если выходит за [0,1], считаем логитами
    if (x.min() < -1e-6) or (x.max() > 1.0 + 1e-6):
        x = _sigmoid(x)
    # клип
    eps = 1e-12
    return np.clip(x, eps, 1.0 - eps)


def _softmax_rowwise(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = P - P.max(axis=1, keepdims=True)
    np.exp(P, out=P)
    P /= (P.sum(axis=1, keepdims=True) + 1e-12)
    return P


def to_proba_multiclass(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError("Для multiclass ожидается матрица (n, C)")
    # эвристика: если значения не в [0,1] или строки не суммируются ≈ 1 — применим softmax
    row_sums = P.sum(axis=1)
    if (P.min() < -1e-6) or (P.max() > 1.0 + 1e-6) or (np.any(np.abs(row_sums - 1.0) > 1e-3)):
        P = _softmax_rowwise(P)
    P = np.clip(P, 1e-12, 1.0)
    P /= (P.sum(axis=1, keepdims=True) + 1e-12)
    return P


def to_proba_multilabel(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    # если выходит за [0,1], считаем логитами и применяем сигмоиду поэлементно
    if (P.min() < -1e-6) or (P.max() > 1.0 + 1e-6):
        P = _sigmoid(P)
    return np.clip(P, 1e-12, 1.0 - 1e-12)


def blend_preds(preds: List[np.ndarray], mode: str, weights: List[float]) -> np.ndarray:
    if len(preds) == 1:
        return preds[0]
    mode = mode.lower()
    # приведем формы
    shapes = [p.shape for p in preds]
    if len(set(shapes)) != 1:
        raise ValueError(f"Нельзя блендить предсказания с разными формами: {shapes}")

    X = np.stack(preds, axis=0)  # (R, n) или (R, n, C/L)
    if mode == "mean":
        return X.mean(axis=0)
    elif mode == "median":
        return np.median(X, axis=0)
    elif mode == "wmean":
        w = np.array(weights, dtype=float).reshape(-1, 1, *([1] * (X.ndim - 2)))
        return (X * w).sum(axis=0)
    elif mode == "rank":
        # усреднение рангов (устойчивее к масштабу)
        if X.ndim == 2:
            # (R, n)
            ranks = np.empty_like(X)
            for i in range(X.shape[0]):
                ranks[i] = pd.Series(X[i]).rank(method="average").values
            return ranks.mean(axis=0)
        else:
            # (R, n, C)
            R, n, C = X.shape
            out = np.empty((n, C), dtype=float)
            for c in range(C):
                rc = np.empty((R, n), dtype=float)
                for i in range(R):
                    rc[i] = pd.Series(X[i, :, c]).rank(method="average").values
                out[:, c] = rc.mean(axis=0)
            return out
    else:
        raise ValueError(f"Unknown blend-mode: {mode}")


# ----------------------------- Calibration -----------------------------

def apply_calibrator_binary(p: np.ndarray, calibrator: Any) -> np.ndarray:
    """
    Поддержаны варианты:
    - dict {"method":"platt","a":..,"b":..}  → p := sigmoid(a*logit(p)+b)
    - dict {"method":"isotonic","x":[...],"y":[...]}  → кусочно-линейная интерполяция
    - sklearn объект с методом predict или transform (joblib), применяем к вероятностям
    """
    if calibrator is None:
        return p
    p = to_proba_binary(p)
    try:
        # dict формат
        if isinstance(calibrator, dict):
            method = str(calibrator.get("method", "")).lower()
            if method == "platt":
                a = float(calibrator.get("a", 1.0))
                b = float(calibrator.get("b", 0.0))
                # переведём p в логиты, затем линейная трансформация
                logits = np.log(p / (1.0 - p + 1e-12) + 1e-12)
                return to_proba_binary(a * logits + b)
            elif method == "isotonic":
                xs = np.array(calibrator.get("x", []), dtype=float)
                ys = np.array(calibrator.get("y", []), dtype=float)
                if len(xs) >= 2 and len(xs) == len(ys):
                    return np.interp(p, xs, ys, left=ys[0], right=ys[-1])
                return p
            else:
                return p
        # sklearn объект
        if hasattr(calibrator, "predict"):
            out = calibrator.predict(p.reshape(-1, 1))
            return to_proba_binary(out.reshape(-1))
        if hasattr(calibrator, "transform"):
            out = calibrator.transform(p.reshape(-1, 1))
            return to_proba_binary(out.reshape(-1))
    except Exception:
        pass
    return p


# ----------------------------- Thresholds -----------------------------

def scorer_factory(task: str, metric: str):
    metric = metric.lower()
    if task == "binary":
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, log_loss,
            accuracy_score, f1_score
        )
        def roc_auc(y, p, w=None):
            return float(roc_auc_score(y, to_proba_binary(p), sample_weight=w))
        def pr_auc(y, p, w=None):
            return float(average_precision_score(y, to_proba_binary(p), sample_weight=w))
        def logloss(y, p, w=None):
            return float(log_loss(y, to_proba_binary(p), sample_weight=w))
        def acc(y, p, w=None):
            pred = (to_proba_binary(p) >= 0.5).astype(int)
            return float(accuracy_score(y, pred, sample_weight=w))
        def f1(y, p, w=None):
            pred = (to_proba_binary(p) >= 0.5).astype(int)
            return float(f1_score(y, pred, sample_weight=w))
        mapping = {
            "roc_auc": roc_auc, "pr_auc": pr_auc, "logloss": logloss,
            "accuracy": acc, "f1": f1
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported metric for binary: {metric}")
        return mapping[metric]

    elif task == "multiclass":
        from sklearn.metrics import accuracy_score, log_loss, f1_score
        def acc(y, P, w=None):
            P = to_proba_multiclass(P)
            pred = np.argmax(P, axis=1)
            return float(accuracy_score(y, pred, sample_weight=w))
        def logloss(y, P, w=None):
            P = to_proba_multiclass(P)
            labels = np.unique(y)
            return float(log_loss(y, P, labels=labels, sample_weight=w))
        def macro_f1(y, P, w=None):
            from sklearn.metrics import f1_score
            P = to_proba_multiclass(P)
            pred = np.argmax(P, axis=1)
            return float(f1_score(y, pred, average="macro", sample_weight=w))
        mapping = {"accuracy": acc, "logloss": logloss, "macro_f1": macro_f1}
        if metric not in mapping:
            raise ValueError(f"Unsupported metric for multiclass: {metric}")
        return mapping[metric]

    elif task == "multilabel":
        from sklearn.metrics import f1_score
        def macro_f1(y, P, w=None):
            P = to_proba_multilabel(P)
            pred = (P >= 0.5).astype(int)
            return float(f1_score(y, pred, average="macro", sample_weight=w))
        mapping = {"macro_f1": macro_f1}
        if metric not in mapping:
            raise ValueError(f"Unsupported metric for multilabel: {metric}")
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
            return float(np.mean(np.abs((y_ - p_) / np.clip(np.abs(y_), eps, None))) * 100.0)
        def r2(y, p, w=None):
            return float(r2_score(y, p, sample_weight=w))
        mapping = {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
        if metric not in mapping:
            raise ValueError(f"Unsupported metric for regression: {metric}")
        return mapping[metric]
    else:
        raise ValueError(f"Unsupported task: {task}")


def load_adv_weights(tag: str) -> Optional[np.ndarray]:
    p = Path("artifacts") / "adversarial" / tag / "train_weights.npy"
    if p.exists():
        try:
            return np.load(p, allow_pickle=False).reshape(-1)
        except Exception:
            return None
    return None


def find_threshold_binary(y: np.ndarray, p: np.ndarray, metric: str,
                          weights: Optional[np.ndarray]) -> float:
    """
    Ищем глобальный τ по сетке. Для AUC/PR/logloss смысла мало — вернём 0.5, а вызов предупредим раньше.
    """
    metric = metric.lower()
    if metric in {"roc_auc", "pr_auc", "logloss"}:
        return 0.5
    from sklearn.metrics import f1_score, accuracy_score
    p = to_proba_binary(p)
    best_tau, best_m = 0.5, -1.0
    for tau in np.linspace(0.01, 0.99, 199):
        pred = (p >= tau).astype(int)
        if metric == "f1":
            m = f1_score(y, pred, sample_weight=weights)
        elif metric == "accuracy":
            m = accuracy_score(y, pred, sample_weight=weights)
        else:
            # fallback f1
            m = f1_score(y, pred, sample_weight=weights)
        if m > best_m:
            best_m, best_tau = m, tau
    return float(best_tau)


def apply_thresholds(task: str, preds: np.ndarray, threshold_spec: dict) -> np.ndarray:
    """
    threshold_spec может быть:
    - {"mode":"none"}
    - {"mode":"global","value":0.42}  # binary
    - {"mode":"per-class","values":[...]}  # multilabel / multiclass one-vs-rest
    - {"mode":"topk","k":3}  # multilabel/multiclass
    """
    mode = threshold_spec.get("mode", "none")
    if mode == "none":
        return preds

    if task == "binary":
        if mode == "global":
            tau = float(threshold_spec["value"])
            p = to_proba_binary(preds)
            return (p >= tau).astype(int)
        else:
            return preds

    if task == "multilabel":
        P = to_proba_multilabel(preds)
        if mode == "per-class":
            vals = np.array(threshold_spec["values"], dtype=float).reshape(1, -1)
            return (P >= vals).astype(int)
        elif mode == "topk":
            k = int(threshold_spec["k"])
            out = np.zeros_like(P, dtype=int)
            topk_idx = np.argpartition(-P, kth=min(k, P.shape[1]-1), axis=1)[:, :k]
            rows = np.arange(P.shape[0])[:, None]
            out[rows, topk_idx] = 1
            return out
        else:
            return P

    if task == "multiclass":
        # Обычно сабмит требует вероятности. Если явный top-k → one-hot top-k (редко)
        P = to_proba_multiclass(preds)
        if mode == "topk":
            k = int(threshold_spec["k"])
            out = np.zeros_like(P, dtype=int)
            topk_idx = np.argpartition(-P, kth=min(k, P.shape[1]-1), axis=1)[:, :k]
            rows = np.arange(P.shape[0])[:, None]
            out[rows, topk_idx] = 1
            return out
        return P

    # regression — пороги не применяются
    return preds


# ----------------------------- Validation & formatting -----------------------------

def align_with_sample(df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    # привести порядок колонок и форму под sample
    cols = list(sample.columns)
    # если нет каких-то столбцов — добавим нули
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]
    return df


def validate_submission(df: pd.DataFrame, id_col: str, target_cols: List[str], task: str):
    if id_col not in df.columns:
        raise ValueError(f"submission: нет колонки id '{id_col}'")
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"submission: нет целевой колонки '{c}'")
    if df[id_col].isna().any():
        raise ValueError("submission: id содержит NaN")
    if df[target_cols].isna().any().any():
        raise ValueError("submission: target содержит NaN")
    if task in ("binary", "multiclass", "multilabel"):
        # вероятности/0-1: проверка диапазона
        vals = df[target_cols].values
        if (vals.min() < -1e-8) or (vals.max() > 1.0 + 1e-8):
            raise ValueError("submission: значения вне диапазона [0,1] для вероятностной задачи")


# ----------------------------- Kaggle CLI -----------------------------

def kaggle_submit(csv_path: Path, slug: str, message: str, log_path: Path) -> Tuple[bool, str]:
    cmd = ["kaggle", "competitions", "submit", "-c", slug, "-f", str(csv_path), "-m", message]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        out = proc.stdout
        ensure_dir(log_path.parent)
        log_path.write_text(out, encoding="utf-8")
        ok = proc.returncode == 0
        return ok, out
    except Exception as e:
        return False, f"Exception while running kaggle CLI: {e}"


# ----------------------------- Orchestration -----------------------------

def build_submission(args):
    warnings.filterwarnings("ignore")
    tag = args.tag

    # schema & ids
    schema = load_schema(tag, args.id_col, args.target_cols, args.sample_sub, verbose=args.verbose)
    id_col, target_cols, sample = schema.id_col, schema.target_cols, schema.sample
    ids_test = load_ids_test(tag, id_col, sample, verbose=args.verbose)
    n_test = len(ids_test)

    # parse runs & load
    run_ids, weights = parse_runs_weights(args.runs)
    runs = [load_run(r, verbose=args.verbose) for r in run_ids]

    # name
    if args.name:
        name = args.name
    else:
        base = f"{args.blend_mode}__" + "__".join(run_ids)
        if len(base) > 100:
            h = hashlib.sha1(base.encode()).hexdigest()[:8]
            name = f"{args.blend_mode}__{run_ids[0]}__{h}"
        else:
            name = base

    out_dir = Path("artifacts") / "submissions" / tag / name
    ensure_dir(out_dir)

    # prepare predictions list in correct probability space
    preds_list = []
    for r in runs:
        P = r.test_pred
        if args.task == "binary":
            P = to_proba_binary(P)
        elif args.task == "multiclass":
            P = to_proba_multiclass(P)
        elif args.task == "multilabel":
            P = to_proba_multilabel(P)
        elif args.task == "regression":
            P = np.asarray(P, dtype=float).reshape(n_test, -1)
            if P.shape[1] != 1:
                # если модель вернула много столбцов — возьмём первый
                P = P[:, [0]]
        preds_list.append(P)

        # sanity on length
        if P.shape[0] != n_test:
            raise ValueError(f"Размер test_pred {r.run_id} != n_test: {P.shape[0]} vs {n_test}")

    # blend
    final_pred = blend_preds(preds_list, args.blend_mode, weights)

    # calibrate (binary only, чаще всего)
    calib_source = None
    if args.use_calibrator in ("on", "auto") and args.task == "binary":
        # правило: если блендим вероятности — применяем один калибратор «первого» прогона
        first_cal = runs[0].calibrator
        if first_cal is not None:
            final_pred = apply_calibrator_binary(final_pred.reshape(-1), first_cal).reshape(-1)
            calib_source = runs[0].run_id
        elif args.use_calibrator == "on":
            print("[warn] calibrator requested but not found; skipping")

    # auto threshold logic
    thr_spec = {"mode": "none"}
    user_thr = args.threshold.strip().lower() if isinstance(args.threshold, str) else "auto"
    thr_metric = (args.threshold_metric or args.metric).lower()

    is_auc_like = (args.task == "binary" and thr_metric in {"roc_auc", "pr_auc", "logloss"})
    if user_thr == "auto":
        if is_auc_like:
            # совет: не пороговать для AUC/PR/LogLoss
            print("[info] AUC/PR/LogLoss task — оставляем вероятности (no threshold). "
                  "Если нужно пороговать, укажи --threshold <float>.")
            thr_spec = {"mode": "none"}
        else:
            # попробуем взять thresholds.json из первого подходящего прогона
            used_thr = None
            for r in runs:
                if r.thresholds:
                    used_thr = r.thresholds
                    break
            if used_thr is not None:
                # нормализуем формат
                if "tau" in used_thr:
                    thr_spec = {"mode": "global", "value": float(used_thr["tau"])}
                elif "per_class" in used_thr:
                    thr_spec = {"mode": "per-class", "values": list(map(float, used_thr["per_class"]))}
                elif "top_k" in used_thr:
                    thr_spec = {"mode": "topk", "k": int(used_thr["top_k"])}
                else:
                    thr_spec = {"mode": "none"}
            else:
                # подберём по OOF первого прогона (если есть)
                if runs[0].oof is not None:
                    oof = runs[0].oof
                    adv_w = load_adv_weights(tag) if args.weights == "adv" else None
                    if args.task == "binary":
                        tau = find_threshold_binary(oof_true_from_meta(tag), oof, thr_metric, adv_w)
                        # ↑ oof_true_from_meta: подтянем y из sets/<tag>/y_train.parquet
                        # но эту функцию нужно реализовать тут (делаем ниже)
                    else:
                        # для multilabel/multiclass — в этом MVP оставим без автоподбора (можно расширить)
                        tau = None
                    if tau is not None:
                        thr_spec = {"mode": "global", "value": float(tau)}
                else:
                    thr_spec = {"mode": "none"}
    elif user_thr.startswith("per-class:"):
        try:
            js = user_thr[len("per-class:"):].strip()
            vals = json.loads(js)
            thr_spec = {"mode": "per-class", "values": list(map(float, vals))}
        except Exception:
            raise ValueError("Не удалось распарсить per-class:<json>")
    elif user_thr.startswith("topk:"):
        k = int(user_thr[len("topk:"):])
        thr_spec = {"mode": "topk", "k": k}
    else:
        # глобальный float
        try:
            tau = float(user_thr)
            thr_spec = {"mode": "global", "value": tau}
        except Exception:
            raise ValueError("Неверный формат --threshold")

    # clip
    if args.clip:
        a, b = args.clip.split(",")
        lo, hi = float(a), float(b)
    else:
        if args.task in ("binary", "multiclass", "multilabel"):
            lo, hi = 0.0, 1.0
        else:
            lo, hi = -np.inf, np.inf

    def do_clip_round(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        arr = np.clip(arr, lo, hi)
        if args.round is not None:
            arr = np.round(arr, args.round)
        return arr

    # apply thresholds (если нужны дискретные метки)
    applied_thresholds = False
    if thr_spec.get("mode") != "none":
        final_pred = apply_thresholds(args.task, final_pred, thr_spec)
        applied_thresholds = True

    # pack into DataFrame (wide format by default)
    if args.task in ("binary", "regression"):
        colname = target_cols[0] if len(target_cols) else "target"
        arr = final_pred.reshape(-1, 1)
        arr = do_clip_round(arr)
        sub = pd.DataFrame({schema.id_col: ids_test})
        sub[colname] = arr[:, 0]

    elif args.task in ("multiclass", "multilabel"):
        P = np.asarray(final_pred, dtype=float)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        if P.shape[1] != len(target_cols):
            raise ValueError(f"Число столбцов предсказаний {P.shape[1]} != len(target_cols) {len(target_cols)}")
        P = do_clip_round(P)
        sub = pd.DataFrame({schema.id_col: ids_test})
        for j, c in enumerate(target_cols):
            sub[c] = P[:, j]
        # для multiclass «класс-метка» (long) обычно не требуется; если нужно — пользователь задаст отдельно

    else:
        raise ValueError("Unsupported task")

    # align to sample if present
    if schema.sample is not None:
        sub = align_with_sample(sub, schema.sample)

    # validate
    validate_submission(sub, schema.id_col, target_cols, args.task)
    if sub.shape[0] != n_test:
        raise ValueError(f"submission rows mismatch: {sub.shape[0]} vs n_test {n_test}")
    if sub[schema.id_col].duplicated().any():
        raise ValueError("submission: duplicate ids")

    # output filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_hint = ""
    if runs[0].metrics and "cv_mean" in runs[0].metrics:
        try:
            metric_hint = f"__cv{float(runs[0].metrics['cv_mean']):.5f}"
        except Exception:
            metric_hint = ""
    csv_name = f"{ts}__{name}{metric_hint}.csv"
    csv_path = out_dir / "submission.csv"  # фиксированное «боевое» имя
    ver_csv_path = out_dir / csv_name      # версия с timestamp

    manifest = {
        "tag": tag,
        "runs": [{"id": r.run_id} for r in runs],
        "weights": weights,
        "blend_mode": args.blend_mode,
        "use_calibrator": args.use_calibrator,
        "calibrator_source": calib_source,
        "threshold": thr_spec,
        "clip": [lo, hi],
        "round": args.round,
        "task": args.task,
        "metric": args.metric,
        "id_col": schema.id_col,
        "target_cols": target_cols,
        "n_test": int(n_test),
        "created_at": datetime.now().isoformat(),
        "source_dirs": [r.path.as_posix() for r in runs],
        "sample_used": schema.sample is not None
    }

    if args.dry_run:
        print("[dry-run] submission is valid; nothing is written.")
        return {"out_dir": out_dir, "csv_path": None, "manifest": manifest}

    # save files
    ensure_dir(out_dir)
    sub.to_csv(csv_path, index=False)
    sub.head(20).to_csv(out_dir / "preview_head.csv", index=False)
    save_json(out_dir / "manifest.json", manifest)
    # версионированная копия
    try:
        sub.to_csv(ver_csv_path, index=False)
    except Exception:
        pass

    # Kaggle push
    if args.kaggle_compet:
        ok, log = kaggle_submit(csv_path, args.kaggle_compet, args.kaggle_message, out_dir / "kaggle_push.log")
        print("[kaggle]", "OK" if ok else "FAIL")
        if not ok:
            print(log[:500])

    # Update models index
    idx_path = Path(args.models_index)
    idx = read_json(idx_path) or {}
    idx_key = f"submit:{tag}:{name}"
    idx[idx_key] = {
        "tag": tag,
        "name": name,
        "runs": [r.run_id for r in runs],
        "path": out_dir.as_posix(),
        "file": csv_path.as_posix(),
        "created_at": datetime.now().isoformat(),
        "kaggle_compet": args.kaggle_compet,
        "message": args.kaggle_message,
    }
    save_json(idx_path, idx)

    print("=== SUBMISSION SAVED ===")
    print("dir :", out_dir.as_posix())
    print("file:", csv_path.as_posix())
    return {"out_dir": out_dir, "csv_path": csv_path, "manifest": manifest}


# ----------------------------- Helpers for auto-threshold (y_true) -----------------------------

def oof_true_from_meta(tag: str) -> np.ndarray:
    """
    Достаёт y_true из artifacts/sets/<tag>/y_train.parquet
    """
    sets_dir = Path("artifacts") / "sets" / tag
    y_path = sets_dir / "y_train.parquet"
    ydf = read_parquet_any(y_path)
    if ydf is None or ydf.shape[1] < 2:
        raise FileNotFoundError("Нужен y_train.parquet с колонками [id, target] для автоподбора порога")
    target_col = [c for c in ydf.columns if c != ydf.columns[0]][0]
    return ydf[target_col].values.reshape(-1)


# ----------------------------- Entrypoint -----------------------------

def main():
    args = parse_args()

    # дружеское предупреждение про пороги на AUC/PR/LogLoss
    if (args.task == "binary") and (args.threshold == "auto"):
        m = (args.threshold_metric or args.metric or "").lower()
        if m in {"roc_auc", "pr_auc", "logloss"}:
            print("[note] Для AUC/PR/LogLoss обычно отправляют СЫРЫЕ вероятности без порогования.")

    try:
        build_submission(args)
    except Exception as e:
        print("[error]", e)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
