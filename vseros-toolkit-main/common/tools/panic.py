#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/panic.py

Паник-кнопка на последние 10–40 минут:
- Источник фич: artifacts/sets/<tag> (если есть) или быстрый сбор с нуля из CSV.
- Кандидаты:
  A) GBDT (dense) + Linear (sparse) → equal-weight → Platt (опц.) → τ (binary)
  B) Только GBDT (или только Linear — если нет dense)
  C) Экстренно: сырые CSV с LabelEncoder + один CatBoost/LightGBM
- Жёсткое управление временем: --time-budget-min
- Отказоустойчивость: при сбое падаем на следующий профиль, но сабмит всё равно делаем.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy import sparse

# --- ML libs (optional) ---
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import roc_auc_score, f1_score, log_loss, r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser("panic: быстрый сабмит за ограниченное время")
    # источники
    p.add_argument("--tag", type=str, default=None, help="Тег сета в artifacts/sets/<tag>")
    p.add_argument("--data-dir", type=str, default="data", help="Каталог с train.csv/test.csv, если нет tag")
    p.add_argument("--sets-dir", type=str, default="artifacts/sets")
    p.add_argument("--models-dir", type=str, default="artifacts/models")
    p.add_argument("--submissions-dir", type=str, default="artifacts/submissions")

    # колонки/задача
    p.add_argument("--id-col", type=str, default="id")
    p.add_argument("--target-col", type=str, default=None)
    p.add_argument("--task", type=str, default=None, choices=[None, "binary", "multiclass", "regression"])
    p.add_argument("--submit-col", type=str, default="prediction", help="Имя колонки предсказания в submission.csv")

    # бюджет/фолды
    p.add_argument("--time-budget-min", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gbdt-lib", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "catboost"])
    p.add_argument("--no-text", action="store_true", help="Игнорировать текстовые столбцы при сборе фич с нуля")
    p.add_argument("--no-calibration", action="store_true", help="Отключить Platt-калибровку")
    p.add_argument("--name", type=str, default="panic")

    # логирование/диагностика
    p.add_argument("--log-every", type=int, default=5, help="Периодичность heartbeat, сек")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()

# -------------------- FS utils & logging --------------------

ROOT = Path(".").resolve()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class Logger:
    file: Optional[Path]
    to_stdout: bool = True
    last_heartbeat: float = time.time()
    period: int = 5

    def write(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if self.to_stdout:
            print(line, flush=True)
        if self.file:
            try:
                with self.file.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def heartbeat(self, text="..."):
        now = time.time()
        if now - self.last_heartbeat >= self.period:
            self.write(text)
            self.last_heartbeat = now

# -------------------- Timer/Watchdog --------------------

class PhaseTimer:
    def __init__(self, total_min: int, log: Logger):
        self.total = total_min * 60.0
        self.start_ts = time.time()
        self.log = log
        self.checkpoints: List[Tuple[str, float, float]] = []  # (name, start_ts, end_ts)

    def time_left_sec(self) -> float:
        return max(0.0, self.total - (time.time() - self.start_ts))

    def phase(self, name: str):
        return _PhaseCtx(self, name)

class _PhaseCtx:
    def __init__(self, timer: PhaseTimer, name: str):
        self.timer = timer
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        self.timer.log.write(f"→ START phase: {self.name} | time left ~{self.timer.time_left_sec():.1f}s")
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.time()
        self.timer.checkpoints.append((self.name, self.t0, t1))
        self.timer.log.write(f"← END   phase: {self.name} | took {t1 - self.t0:.1f}s | left ~{self.timer.time_left_sec():.1f}s")

# -------------------- I/O helpers --------------------

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def load_set(tag: str, sets_dir: Path, log: Logger):
    base = sets_dir / tag
    if not base.exists():
        raise FileNotFoundError(f"Set tag not found: {base}")

    X_dense_tr = X_dense_te = None
    X_sparse_tr = X_sparse_te = None
    y = ids_test = None
    folds = None

    if (base / "X_dense_train.parquet").exists():
        X_dense_tr = pd.read_parquet(base / "X_dense_train.parquet")
        X_dense_te = pd.read_parquet(base / "X_dense_test.parquet")
        log.write(f"[set] dense: {X_dense_tr.shape} / {X_dense_te.shape}")

    if (base / "X_sparse_train.npz").exists():
        X_sparse_tr = sparse.load_npz(base / "X_sparse_train.npz")
        X_sparse_te = sparse.load_npz(base / "X_sparse_test.npz")
        log.write(f"[set] sparse: {X_sparse_tr.shape} / {X_sparse_te.shape}")

    if (base / "y_train.parquet").exists():
        y = pd.read_parquet(base / "y_train.parquet")["y"].to_numpy()

    if (base / "ids_test.parquet").exists():
        ids_test = pd.read_parquet(base / "ids_test.parquet")["id"].astype(str).to_numpy()

    if (base / "folds.pkl").exists():
        import pickle
        folds = pickle.loads((base / "folds.pkl").read_bytes())
        log.write(f"[set] folds: {len(folds)}")

    meta = {}
    if (base / "meta.json").exists():
        meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))

    return X_dense_tr, X_dense_te, X_sparse_tr, X_sparse_te, y, ids_test, folds, meta

# -------------------- Quick features from CSV --------------------

@dataclass
class QuickFeaturesResult:
    X_dense_tr: Optional[pd.DataFrame]
    X_dense_te: Optional[pd.DataFrame]
    X_sparse_tr: Optional[sparse.csr_matrix]
    X_sparse_te: Optional[sparse.csr_matrix]
    y: Optional[np.ndarray]
    ids_test: np.ndarray
    cat_cols: List[str]
    num_cols: List[str]
    text_cols: List[str]

def quick_build_features_from_csv(
    data_dir: Path,
    id_col: str,
    target_col: Optional[str],
    no_text: bool,
    log: Logger,
) -> QuickFeaturesResult:
    train_path = data_dir / "train.csv"
    test_path  = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing CSVs in {data_dir}: need train.csv and test.csv")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    if id_col not in train.columns or id_col not in test.columns:
        raise ValueError(f"id_col '{id_col}' not found in CSVs")

    y = None
    if target_col and target_col in train.columns:
        y = train[target_col].to_numpy()
        train = train.drop(columns=[target_col])

    # унификация столбцов
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test  = test[common_cols]

    # авто-детект
    num_cols, cat_cols, text_cols = [], [], []
    for c in common_cols:
        if c == id_col:
            continue
        if pd.api.types.is_numeric_dtype(train[c]):
            num_cols.append(c)
        elif pd.api.types.is_string_dtype(train[c]):
            # простой критерий текста
            avg_len = train[c].dropna().astype(str).map(len).mean()
            if not no_text and avg_len and avg_len >= 30:
                text_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            cat_cols.append(c)

    log.write(f"[auto] num={len(num_cols)} cat={len(cat_cols)} text={len(text_cols)}")

    # ---- Dense: числа + частоты категорий ----
    Xd_tr = pd.DataFrame(index=train.index)
    Xd_te = pd.DataFrame(index=test.index)

    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        Xn_tr = pd.DataFrame(imp_num.fit_transform(train[num_cols]), columns=[f"num__{c}" for c in num_cols])
        Xn_te = pd.DataFrame(imp_num.transform(test[num_cols]), columns=[f"num__{c}" for c in num_cols])
        # лёгкий клиппинг хвостов
        q01 = np.nanpercentile(Xn_tr, 1, axis=0)
        q99 = np.nanpercentile(Xn_tr, 99, axis=0)
        Xn_tr = np.clip(Xn_tr, q01, q99)
        Xn_te = np.clip(Xn_te, q01, q99)
        Xd_tr = pd.concat([Xd_tr, Xn_tr], axis=1)
        Xd_te = pd.concat([Xd_te, Xn_te], axis=1)

    if cat_cols:
        # частоты категорий как dense
        for c in cat_cols:
            vc = train[c].value_counts(dropna=False, normalize=True)
            Xd_tr[f"catf__{c}"] = train[c].map(vc).fillna(0).to_numpy()
            Xd_te[f"catf__{c}"] = test[c].map(vc).fillna(0).to_numpy()

    if Xd_tr.shape[1] == 0:
        Xd_tr = None
        Xd_te = None

    # ---- Sparse: OHE для cat + TF-IDF для текста ----
    Xs_tr_list = []
    Xs_te_list = []

    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        Xc_tr = ohe.fit_transform(train[cat_cols].astype(str))
        Xc_te = ohe.transform(test[cat_cols].astype(str))
        Xs_tr_list.append(Xc_tr)
        Xs_te_list.append(Xc_te)

    if text_cols:
        for c in text_cols:
            tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 2))
            Xtxt_tr = tfidf.fit_transform(train[c].fillna("").astype(str))
            Xtxt_te = tfidf.transform(test[c].fillna("").astype(str))
            Xs_tr_list.append(Xtxt_tr)
            Xs_te_list.append(Xtxt_te)

    if Xs_tr_list:
        Xs_tr = sparse.hstack(Xs_tr_list).tocsr()
        Xs_te = sparse.hstack(Xs_te_list).tocsr()
    else:
        Xs_tr = None
        Xs_te = None

    ids_test = test[id_col].astype(str).to_numpy()

    return QuickFeaturesResult(
        X_dense_tr=Xd_tr,
        X_dense_te=Xd_te,
        X_sparse_tr=Xs_tr,
        X_sparse_te=Xs_te,
        y=y,
        ids_test=ids_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        text_cols=text_cols,
    )

# -------------------- Folds/Task utils --------------------

def infer_task(y: Optional[np.ndarray], task: Optional[str]) -> str:
    if task in ("binary", "multiclass", "regression"):
        return task
    if y is None:
        # без таргета — считаем binary по умолчанию (часто требуется prob)
        return "binary"
    y_unique = np.unique(y[~pd.isna(y)])
    if y.dtype.kind in "ifu" and len(y_unique) > 20:
        return "regression"
    if len(y_unique) == 2:
        return "binary"
    return "multiclass"

def make_folds(y: Optional[np.ndarray], task: str, n_splits: int, seed: int):
    if y is None:
        # заглушка: фолды по индексу
        idx = np.arange(1000)
        kf = KFold(n_splits=min(5, n_splits), shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in kf.split(idx)]
    n = len(y)
    idx = np.arange(n)
    if task in ("binary", "multiclass"):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in skf.split(idx, y)]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in kf.split(idx)]

# -------------------- Metrics --------------------

def compute_metric(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if task == "binary":
            # y_pred — вероятности класса 1
            return float(roc_auc_score(y_true, y_pred))
        elif task == "multiclass":
            # y_pred shape (n, C) — macro AUC не всегда устойчив; fallback на -logloss
            try:
                # пробуем macro-ovr AUC, если метки от 0..C-1
                classes = np.unique(y_true)
                if y_pred.ndim == 2 and len(classes) == y_pred.shape[1]:
                    return float(roc_auc_score(y_true, y_pred, multi_class="ovr"))
            except Exception:
                pass
            return float(-log_loss(y_true, y_pred, labels=np.unique(y_true)))
        else:
            # regression — минус RMSE (больше — лучше)
            return float(-np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float("nan")

def find_tau_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # быстрый поиск τ по F1
    grid = np.linspace(0.05, 0.95, 37)
    best, best_tau = -1.0, 0.5
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best:
            best, best_tau = f1, t
    return float(best_tau)

# -------------------- Model Run (artifacts I/O) --------------------

@dataclass
class ModelRun:
    name: str
    run_id: str
    path: Path
    task: str
    oof_true: Optional[np.ndarray]
    oof_pred: Optional[np.ndarray]
    test_pred: np.ndarray
    cv_mean: Optional[float]
    cv_std: Optional[float]
    lib: str
    params: Dict[str, Any]

def save_model_run(run: ModelRun, model_objects: List[Any]):
    d = run.path
    ensure_dir(d)
    # preds
    if run.oof_true is not None:
        np.save(d / "oof_true.npy", run.oof_true)
    if run.oof_pred is not None:
        np.save(d / "oof_pred.npy", run.oof_pred)
    np.save(d / "test_pred.npy", run.test_pred)
    # metrics
    meta = {
        "cv_mean": run.cv_mean,
        "cv_std": run.cv_std,
        "task": run.task,
        "lib": run.lib,
        "params": run.params,
        "created_at": datetime.now().isoformat(),
    }
    (d / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # models
    try:
        import joblib
        for i, m in enumerate(model_objects):
            joblib.dump(m, d / f"model_{i}.joblib")
    except Exception:
        pass

# -------------------- Training: GBDT --------------------

def train_gbdt_cv(
    X_tr: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    X_te: Union[pd.DataFrame, np.ndarray],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    task: str,
    lib: str,
    params: Dict[str, Any],
    seed: int,
    models_dir: Path,
    log: Logger,
) -> ModelRun:
    n = len(y)
    oof = np.zeros((n,), dtype=float) if task != "multiclass" else np.zeros((n, len(np.unique(y))), dtype=float)
    test_pred = None
    models = []

    if lib == "lightgbm" and lgb is None:
        log.write("[warn] lightgbm недоступен, переключаюсь на xgboost/catboost fallback")
        lib = "xgboost" if xgb is not None else "catboost"

    if lib == "xgboost" and xgb is None:
        log.write("[warn] xgboost недоступен, переключаюсь на lightgbm/catboost fallback")
        lib = "lightgbm" if lgb is not None else "catboost"

    if lib == "catboost" and (CatBoostClassifier is None or CatBoostRegressor is None):
        log.write("[warn] catboost недоступен, переключаюсь на lightgbm/xgboost fallback")
        lib = "lightgbm" if lgb is not None else "xgboost"

    def _pred_shape_k(te_pred):
        # нормализация формы предсказаний теста
        if task == "multiclass":
            return te_pred
        else:
            return te_pred.reshape(-1)

    for k, (tr_idx, va_idx) in enumerate(folds):
        Xtr = X_tr.iloc[tr_idx] if isinstance(X_tr, pd.DataFrame) else X_tr[tr_idx]
        Xva = X_tr.iloc[va_idx] if isinstance(X_tr, pd.DataFrame) else X_tr[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        if lib == "lightgbm":
            if task == "regression":
                obj = "regression"
                metric = "rmse"
            elif task == "multiclass":
                obj = "multiclass"
                metric = "multi_logloss"
                num_class = len(np.unique(y))
            else:
                obj = "binary"
                metric = "auc"
                num_class = None

            lgb_params = dict(
                objective=obj,
                metric=metric,
                learning_rate=params.get("learning_rate", 0.05),
                num_leaves=params.get("num_leaves", 2 ** params.get("max_depth", 7)),
                max_depth=params.get("max_depth", 7),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                min_data_in_leaf=params.get("min_data_in_leaf", 32),
                reg_lambda=params.get("reg_lambda", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                verbose=-1,
                random_state=seed + k,
            )
            if task == "multiclass":
                lgb_params["num_class"] = num_class

            dtr = lgb.Dataset(Xtr, label=ytr)
            dva = lgb.Dataset(Xva, label=yva)
            booster = lgb.train(
                lgb_params,
                dtr,
                valid_sets=[dva],
                num_boost_round=params.get("n_estimators", 300),
                early_stopping_rounds=50,
                verbose_eval=False
            )
            models.append(booster)

            if task == "multiclass":
                oof[va_idx, :] = booster.predict(Xva, num_iteration=booster.best_iteration)
                te_pred = booster.predict(X_te, num_iteration=booster.best_iteration)
            elif task == "binary":
                oof[va_idx] = booster.predict(Xva, num_iteration=booster.best_iteration)
                te_pred = booster.predict(X_te, num_iteration=booster.best_iteration)
            else:
                oof[va_idx] = booster.predict(Xva, num_iteration=booster.best_iteration)
                te_pred = booster.predict(X_te, num_iteration=booster.best_iteration)

            te_pred = _pred_shape_k(te_pred)
            test_pred = te_pred if test_pred is None else test_pred + te_pred

        elif lib == "xgboost":
            if task == "regression":
                obj = "reg:squarederror"
            elif task == "multiclass":
                obj = "multi:softprob"
                num_class = len(np.unique(y))
            else:
                obj = "binary:logistic"

            xgb_params = dict(
                objective=obj,
                eta=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 7),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                reg_lambda=params.get("reg_lambda", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                random_state=seed + k,
                eval_metric="auc" if task == "binary" else ("mlogloss" if task == "multiclass" else "rmse"),
            )
            if task == "multiclass":
                xgb_params["num_class"] = num_class

            dtr = xgb.DMatrix(Xtr, label=ytr)
            dva = xgb.DMatrix(Xva, label=yva)
            dte = xgb.DMatrix(X_te)
            booster = xgb.train(
                xgb_params,
                dtr,
                num_boost_round=params.get("n_estimators", 300),
                evals=[(dva, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            models.append(booster)
            if task == "multiclass":
                oof[va_idx, :] = booster.predict(xgb.DMatrix(Xva), ntree_limit=booster.best_ntree_limit)
                te_pred = booster.predict(dte, ntree_limit=booster.best_ntree_limit)
            else:
                oof[va_idx] = booster.predict(xgb.DMatrix(Xva), ntree_limit=booster.best_ntree_limit)
                te_pred = booster.predict(dte, ntree_limit=booster.best_ntree_limit)
            te_pred = _pred_shape_k(te_pred)
            test_pred = te_pred if test_pred is None else test_pred + te_pred

        else:  # catboost
            if task == "regression":
                model = CatBoostRegressor(
                    depth=params.get("max_depth", 7),
                    learning_rate=params.get("learning_rate", 0.05),
                    n_estimators=params.get("n_estimators", 300),
                    loss_function="RMSE",
                    random_seed=seed + k,
                    verbose=False
                )
            elif task == "multiclass":
                model = CatBoostClassifier(
                    depth=params.get("max_depth", 7),
                    learning_rate=params.get("learning_rate", 0.05),
                    n_estimators=params.get("n_estimators", 300),
                    loss_function="MultiClass",
                    random_seed=seed + k,
                    verbose=False,
                    auto_class_weights="Balanced"
                )
            else:
                model = CatBoostClassifier(
                    depth=params.get("max_depth", 7),
                    learning_rate=params.get("learning_rate", 0.05),
                    n_estimators=params.get("n_estimators", 300),
                    loss_function="Logloss",
                    random_seed=seed + k,
                    verbose=False,
                    auto_class_weights="Balanced"
                )
            model.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
            models.append(model)

            if task == "multiclass":
                oof[va_idx, :] = model.predict_proba(Xva)
                te_pred = model.predict_proba(X_te)
            elif task == "binary":
                oof[va_idx] = model.predict_proba(Xva)[:, 1]
                te_pred = model.predict_proba(X_te)[:, 1]
            else:
                oof[va_idx] = model.predict(Xva).reshape(-1)
                te_pred = model.predict(X_te).reshape(-1)

            te_pred = _pred_shape_k(te_pred)
            test_pred = te_pred if test_pred is None else test_pred + te_pred

    test_pred = test_pred / max(1, len(folds))

    # cv
    fold_scores = []
    for _, va_idx in folds:
        yp = oof[va_idx]
        yt = y[va_idx]
        score = compute_metric(task, yt, yp)
        if not np.isnan(score):
            fold_scores.append(score)
    cv_mean = float(np.mean(fold_scores)) if fold_scores else None
    cv_std = float(np.std(fold_scores)) if fold_scores else None

    # build run
    feat_sig = f"dense_{X_tr.shape if hasattr(X_tr,'shape') else 'na'}"
    run_id = f"{lib}-{short_hash(str(feat_sig))}-{datetime.now().strftime('%m%d_%H%M%S')}"
    run_path = models_dir / run_id
    run = ModelRun(
        name=f"{lib}_gbdt",
        run_id=run_id,
        path=run_path,
        task=task,
        oof_true=y,
        oof_pred=oof,
        test_pred=test_pred,
        cv_mean=cv_mean,
        cv_std=cv_std,
        lib=lib,
        params=params
    )
    save_model_run(run, models)
    return run

# -------------------- Training: Linear (sparse) --------------------

def train_linear_cv(
    X_tr: sparse.csr_matrix,
    y: np.ndarray,
    X_te: sparse.csr_matrix,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    task: str,
    seed: int,
    models_dir: Path,
    log: Logger,
) -> ModelRun:
    n = len(y)
    classes = np.unique(y) if y is not None else None
    if task == "multiclass":
        C = len(classes)
        oof = np.zeros((n, C), dtype=float)
    else:
        oof = np.zeros((n,), dtype=float)
    test_pred = None
    models = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        if task == "regression":
            model = Ridge(alpha=1.0, random_state=seed + k)
            model.fit(Xtr, ytr)
            pred_va = model.predict(Xva).reshape(-1)
            pred_te = model.predict(X_te).reshape(-1)
        elif task == "multiclass":
            model = LogisticRegression(
                C=2.0, max_iter=200, n_jobs=-1, solver="lbfgs", multi_class="auto", random_state=seed + k
            )
            model.fit(Xtr, ytr)
            pred_va = model.predict_proba(Xva)
            pred_te = model.predict_proba(X_te)
        else:
            # binary
            pos_rate = np.mean(ytr)
            class_weight = "balanced" if pos_rate < 0.2 or pos_rate > 0.8 else None
            model = LogisticRegression(
                C=2.0, max_iter=200, n_jobs=-1, solver="lbfgs", random_state=seed + k, class_weight=class_weight
            )
            model.fit(Xtr, ytr)
            pred_va = model.predict_proba(Xva)[:, 1]
            pred_te = model.predict_proba(X_te)[:, 1]

        models.append(model)
        if task == "multiclass":
            oof[va_idx, :] = pred_va
            test_pred = pred_te if test_pred is None else test_pred + pred_te
        else:
            oof[va_idx] = pred_va
            test_pred = pred_te if test_pred is None else test_pred + pred_te

    test_pred = test_pred / max(1, len(folds))

    fold_scores = []
    for _, va_idx in folds:
        yt, yp = y[va_idx], oof[va_idx]
        score = compute_metric(task, yt, yp)
        if not np.isnan(score):
            fold_scores.append(score)
    cv_mean = float(np.mean(fold_scores)) if fold_scores else None
    cv_std = float(np.std(fold_scores)) if fold_scores else None

    feat_sig = f"sparse_{X_tr.shape}"
    run_id = f"linear-{short_hash(str(feat_sig))}-{datetime.now().strftime('%m%d_%H%M%S')}"
    run_path = models_dir / run_id
    run = ModelRun(
        name="linear_sparse",
        run_id=run_id,
        path=run_path,
        task=task,
        oof_true=y,
        oof_pred=oof,
        test_pred=test_pred,
        cv_mean=cv_mean,
        cv_std=cv_std,
        lib="sklearn",
        params={"algo": "logreg/ridge"}
    )
    save_model_run(run, models)
    return run

# -------------------- Blending/Calibration --------------------

def equal_weight(runs: List[ModelRun], task: str) -> ModelRun:
    assert len(runs) >= 2
    # OOF
    y_true = runs[0].oof_true
    oofs = [r.oof_pred for r in runs if r.oof_pred is not None]
    tests = [r.test_pred for r in runs]

    oof_bl = None
    for o in oofs:
        if o is None:
            continue
        oof_bl = o if oof_bl is None else oof_bl + o
    if oof_bl is not None:
        oof_bl = oof_bl / len(oofs)

    test_bl = None
    for t in tests:
        test_bl = t if test_bl is None else test_bl + t
    test_bl = test_bl / len(tests)

    # cv
    cv_mean = None
    cv_std = None
    if y_true is not None and oof_bl is not None:
        # восстановим фолды из первого
        # (точно совпадают, так как мы их передавали одинаковыми)
        # Здесь просто вычислим единственную метрику «целиком», без пофолдовой
        cv_mean = compute_metric(task, y_true, oof_bl)
        cv_std = 0.0

    run_id = f"blend-eq-{short_hash(','.join([r.run_id for r in runs]))}-{datetime.now().strftime('%m%d_%H%M%S')}"
    path = (runs[0].path.parent) / run_id
    run = ModelRun(
        name="blend_eq",
        run_id=run_id,
        path=path,
        task=task,
        oof_true=y_true,
        oof_pred=oof_bl,
        test_pred=test_bl,
        cv_mean=cv_mean,
        cv_std=cv_std,
        lib="blend",
        params={"weights": "equal"}
    )
    save_model_run(run, [])
    return run

# Platt (binary)
@dataclass
class PlattCalibrator:
    a: float
    b: float

def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[PlattCalibrator]:
    # логистическая регрессия на одном признаке с bias
    try:
        y_true = y_true.astype(int)
        x = y_prob.reshape(-1, 1)
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(x, y_true)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])
        return PlattCalibrator(a=a, b=b)
    except Exception:
        return None

def apply_platt(cal: Optional[PlattCalibrator], y_prob: np.ndarray) -> np.ndarray:
    if cal is None:
        return y_prob
    z = cal.a * y_prob + cal.b
    return 1.0 / (1.0 + np.exp(-z))

# -------------------- Submission & Manifest --------------------

def write_submission(out_dir: Path, ids: np.ndarray, preds: np.ndarray, id_col: str, submit_col: str):
    ensure_dir(out_dir)
    df = pd.DataFrame({id_col: ids, submit_col: preds})
    df.to_csv(out_dir / "submission.csv", index=False)

def write_manifest(out_dir: Path, manifest: Dict[str, Any]):
    ensure_dir(out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# -------------------- Orchestration --------------------

def choose_profile(has_dense, has_sparse, time_left_sec) -> str:
    # простая эвристика: если есть оба и времени >= 15 мин — профиль A
    if has_dense and has_sparse and time_left_sec >= 15 * 60:
        return "A"
    if (has_dense or has_sparse) and time_left_sec >= 7 * 60:
        return "B"
    return "C"

def run_profile_A(
    X_dense_tr, y, X_dense_te,
    X_sparse_tr, X_sparse_te,
    folds, task, lib, models_dir, seed, log
) -> Tuple[ModelRun, List[ModelRun]]:
    gbdt_params = dict(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.9, colsample_bytree=0.8)
    runs = []
    if X_dense_tr is not None:
        run_g = train_gbdt_cv(X_dense_tr, y, X_dense_te, folds, task, lib, gbdt_params, seed, models_dir, log)
        runs.append(run_g)
    if X_sparse_tr is not None:
        run_l = train_linear_cv(X_sparse_tr, y, X_sparse_te, folds, task, seed, models_dir, log)
        runs.append(run_l)

    if len(runs) >= 2:
        bl = equal_weight(runs, task)
        return bl, runs
    else:
        return runs[0], runs

def run_profile_B(
    X_dense_tr, y, X_dense_te,
    X_sparse_tr, X_sparse_te,
    folds, task, lib, models_dir, seed, log
) -> Tuple[ModelRun, List[ModelRun]]:
    if X_dense_tr is not None:
        params = dict(n_estimators=250, learning_rate=0.07, max_depth=6, subsample=0.9, colsample_bytree=0.8)
        run_g = train_gbdt_cv(X_dense_tr, y, X_dense_te, folds, task, lib, params, seed, models_dir, log)
        return run_g, [run_g]
    else:
        run_l = train_linear_cv(X_sparse_tr, y, X_sparse_te, folds, task, seed, models_dir, log)
        return run_l, [run_l]

def run_profile_C_raw(
    train_csv: Path, test_csv: Path, id_col: str, target_col: Optional[str],
    task: str, lib: str, models_dir: Path, seed: int, log: Logger
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], ModelRun]:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    ids_test = test[id_col].astype(str).to_numpy()

    y = None
    if target_col and target_col in train.columns:
        y = train[target_col].to_numpy()
        X = train.drop(columns=[target_col])
    else:
        X = train.copy()

    # простая очистка
    common_cols = [c for c in X.columns if c in test.columns]
    X = X[common_cols]
    T = test[common_cols]

    # авто-детект
    num_cols, cat_cols = [], []
    for c in common_cols:
        if c == id_col:
            continue
        (num_cols if pd.api.types.is_numeric_dtype(X[c]) else cat_cols).append(c)

    # импутация
    if num_cols:
        imp = SimpleImputer(strategy="median")
        X.loc[:, num_cols] = imp.fit_transform(X[num_cols])
        T.loc[:, num_cols] = imp.transform(T[num_cols])

    # LabelEncoder (через OrdinalEncoder) для категорий
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X.loc[:, cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
        T.loc[:, cat_cols] = enc.transform(T[cat_cols].astype(str))

    # модель: catboost предпочтительно, иначе lightgbm/xgb
    model = None
    lib_used = None
    if task == "regression":
        if CatBoostRegressor is not None:
            model = CatBoostRegressor(depth=6, learning_rate=0.07, n_estimators=300, random_seed=seed, verbose=False)
            lib_used = "catboost"
        elif lgb is not None:
            params = dict(objective="regression", metric="rmse", learning_rate=0.07, max_depth=6)
            model = ("lgb", params)
            lib_used = "lightgbm"
        else:
            # fallback simple ridge
            model = Ridge(alpha=1.0, random_state=seed)
            lib_used = "sklearn"
    else:
        # классификация
        if CatBoostClassifier is not None:
            model = CatBoostClassifier(depth=6, learning_rate=0.07, n_estimators=300,
                                       loss_function="Logloss" if task == "binary" else "MultiClass",
                                       auto_class_weights="Balanced", random_seed=seed, verbose=False)
            lib_used = "catboost"
        elif lgb is not None:
            params = dict(objective="binary" if task == "binary" else "multiclass",
                          metric="auc" if task == "binary" else "multi_logloss",
                          learning_rate=0.07, max_depth=6)
            model = ("lgb", params)
            lib_used = "lightgbm"
        elif xgb is not None:
            params = dict(objective="binary:logistic" if task == "binary" else "multi:softprob",
                          eta=0.07, max_depth=6)
            model = ("xgb", params)
            lib_used = "xgboost"
        else:
            # fallback логрег
            model = LogisticRegression(C=2.0, max_iter=200, n_jobs=-1)
            lib_used = "sklearn"

    # fit/predict
    if isinstance(model, tuple) and model[0] == "lgb":
        params = model[1]
        if task == "multiclass":
            params["num_class"] = len(np.unique(y)) if y is not None else 2
        dtr = lgb.Dataset(X.drop(columns=[id_col]), label=y) if y is not None else lgb.Dataset(X.drop(columns=[id_col]))
        booster = lgb.train(params, dtr, num_boost_round=300, verbose_eval=False)
        if task == "multiclass":
            te_pred = booster.predict(T.drop(columns=[id_col]))
        elif task == "binary":
            te_pred = booster.predict(T.drop(columns=[id_col]))
        else:
            te_pred = booster.predict(T.drop(columns=[id_col]))
        yprob = None
        run_models = [booster]
    elif isinstance(model, tuple) and model[0] == "xgb":
        params = model[1]
        if task == "multiclass":
            params["num_class"] = len(np.unique(y)) if y is not None else 2
        dtr = xgb.DMatrix(X.drop(columns=[id_col]), label=y)
        dte = xgb.DMatrix(T.drop(columns=[id_col]))
        booster = xgb.train(params, dtr, num_boost_round=300, verbose_eval=False)
        te_pred = booster.predict(dte)
        yprob = None
        run_models = [booster]
    else:
        model.fit(X.drop(columns=[id_col]), y) if y is not None else model.fit(X.drop(columns=[id_col]), np.zeros(len(X)))
        if task == "multiclass":
            te_pred = model.predict_proba(T.drop(columns=[id_col]))
        elif task == "binary":
            try:
                te_pred = model.predict_proba(T.drop(columns=[id_col]))[:, 1]
            except Exception:
                te_pred = model.predict(T.drop(columns=[id_col]))
        else:
            te_pred = model.predict(T.drop(columns=[id_col]))
        yprob = None
        run_models = [model]

    run_id = f"raw-{lib_used}-{short_hash(str(X.shape))}-{datetime.now().strftime('%m%d_%H%M%S')}"
    run_path = models_dir / run_id
    run = ModelRun(
        name="raw_single",
        run_id=run_id,
        path=run_path,
        task=task,
        oof_true=y,
        oof_pred=yprob,
        test_pred=np.asarray(te_pred),
        cv_mean=None,
        cv_std=None,
        lib=lib_used,
        params={"profile": "C_raw"}
    )
    save_model_run(run, run_models)
    return ids_test, y, common_cols, run

# -------------------- Main --------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # prepare output dirs
    sets_dir = Path(args.sets_dir)
    models_dir = Path(args.models_dir)
    subs_dir = Path(args.submissions_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_tag = args.tag if args.tag else f"csv_{ts}"
    out_dir = subs_dir / submit_tag / args.name
    ensure_dir(out_dir)
    logger = Logger(file=out_dir / "console.log", to_stdout=True, period=args.log_every)

    timer = PhaseTimer(args.time_budget_min, logger)

    # 1) Загрузка/сбор фич
    with timer.phase("features"):
        X_dense_tr = X_dense_te = None
        X_sparse_tr = X_sparse_te = None
        y = None
        ids_test = None
        folds = None
        set_meta = {}

        if args.tag:
            try:
                X_dense_tr, X_dense_te, X_sparse_tr, X_sparse_te, y, ids_test, folds, set_meta = load_set(args.tag, sets_dir, logger)
            except Exception as e:
                logger.write(f"[warn] не удалось загрузить сет '{args.tag}': {e}. Перехожу к сбору с нуля.")
                args.tag = None  # будем собирать

        if args.tag is None:
            qf = quick_build_features_from_csv(Path(args.data_dir), args.id_col, args.target_col, args.no_text, logger)
            X_dense_tr, X_dense_te = qf.X_dense_tr, qf.X_dense_te
            X_sparse_tr, X_sparse_te = qf.X_sparse_tr, qf.X_sparse_te
            y, ids_test = qf.y, qf.ids_test

    # 2) Задача/фолды
    task = infer_task(y, args.task)
    if folds is None:
        ns = args.n_splits
        # автоматическое снижение фолдов при маленьком бюджете
        if timer.time_left_sec() < 20 * 60:
            ns = min(ns, 3)
        folds = make_folds(y, task, ns, args.seed)

    has_dense = X_dense_tr is not None and X_dense_te is not None
    has_sparse = X_sparse_tr is not None and X_sparse_te is not None
    profile = choose_profile(has_dense, has_sparse, timer.time_left_sec())
    logger.write(f"[info] task={task} | profile={profile} | dense={has_dense} | sparse={has_sparse} | folds={len(folds)}")

    # 3) Тренировка
    best_run: Optional[ModelRun] = None
    used_runs: List[str] = []
    cand_runs: List[ModelRun] = []

    with timer.phase("training"):
        try:
            if profile == "A" and (y is not None):
                best_run, cand_runs = run_profile_A(
                    X_dense_tr, y, X_dense_te, X_sparse_tr, X_sparse_te,
                    folds, task, args.gbdt_lib, models_dir, args.seed, logger
                )
                used_runs = [r.run_id for r in cand_runs] + ([best_run.run_id] if best_run.run_id not in [r.run_id for r in cand_runs] else [])
            elif profile == "B" and (y is not None):
                best_run, cand_runs = run_profile_B(
                    X_dense_tr, y, X_dense_te, X_sparse_tr, X_sparse_te,
                    folds, task, args.gbdt_lib, models_dir, args.seed, logger
                )
                used_runs = [r.run_id for r in cand_runs]
            else:
                # C: на сырых CSV
                ids_test, y_raw, cols, run = run_profile_C_raw(
                    Path(args.data_dir) / "train.csv",
                    Path(args.data_dir) / "test.csv",
                    args.id_col, args.target_col, task, args.gbdt_lib, models_dir, args.seed, logger
                )
                best_run = run
                used_runs = [run.run_id]
                cand_runs = [run]
        except Exception as e:
            logger.write(f"[error] training failed: {e}. Fallback to profile C.")
            ids_test, y_raw, cols, run = run_profile_C_raw(
                Path(args.data_dir) / "train.csv",
                Path(args.data_dir) / "test.csv",
                args.id_col, args.target_col, task, args.gbdt_lib, models_dir, args.seed, logger
            )
            best_run = run
            used_runs = [run.run_id]
            cand_runs = [run]

    # 4) Калибровка/τ
    calib = None
    tau = None

    with timer.phase("calibration"):
        if (not args.no_calibration) and task == "binary" and best_run.oof_true is not None and best_run.oof_pred is not None:
            try:
                calib = fit_platt(best_run.oof_true, best_run.oof_pred)
                logger.write(f"[calib] platt={'ok' if calib is not None else 'skip'}")
                if calib is not None:
                    best_run.test_pred = apply_platt(calib, best_run.test_pred)
                    # сохраним калибратор
                    try:
                        (best_run.path / "calibrator.json").write_text(json.dumps({"a": calib.a, "b": calib.b}), encoding="utf-8")
                    except Exception:
                        pass
                # τ
                tau = find_tau_by_f1(best_run.oof_true, best_run.oof_pred if calib is None else apply_platt(calib, best_run.oof_pred))
                (best_run.path / "thresholds.json").write_text(json.dumps({"tau": float(tau)}, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.write(f"[calib] tau={tau:.4f}")
            except Exception as e:
                logger.write(f"[warn] calibration skipped: {e}")

    # 5) Сабмит
    with timer.phase("submission"):
        if ids_test is None:
            # IDs не были загружены из сета — попробуем из CSV
            test_csv = Path(args.data_dir) / "test.csv"
            if test_csv.exists():
                ids_test = pd.read_csv(test_csv)[args.id_col].astype(str).to_numpy()
            else:
                # экстренно создадим индексы
                ids_test = np.arange(best_run.test_pred.shape[0]).astype(str)

        preds = best_run.test_pred
        # бинаризация только если требуется метка, но обычно на Kaggle нужны вероятности.
        # Оставляем вероятности. Если нужен класс, можно включить ниже:
        # if task == "binary" and tau is not None:
        #     preds = (preds >= tau).astype(int)

        write_submission(out_dir, ids_test, preds, args.id_col, args.submit_col)

        passport = {
            "profile": profile,
            "task": task,
            "runs": used_runs,
            "best_run": best_run.run_id,
            "cv_mean": best_run.cv_mean,
            "cv_std": best_run.cv_std,
            "tau": None if tau is None else float(tau),
            "calibration": calib is not None,
            "time_budget_min": args.time_budget_min,
            "time_left_sec": timer.time_left_sec(),
            "phases": [
                {"name": n, "took_sec": t1 - t0} for (n, t0, t1) in timer.checkpoints
            ],
            "set_tag": args.tag,
            "set_meta": set_meta if args.tag else {},
            "id_col": args.id_col,
            "submit_col": args.submit_col,
            "created_at": datetime.now().isoformat(),
        }
        (out_dir / "panic_passport.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest = {
            "runs": used_runs,
            "best_run": best_run.run_id,
            "set_tag": args.tag,
            "profile": profile,
            "metric_cv": best_run.cv_mean,
            "tau": None if tau is None else float(tau),
            "created_at": datetime.now().isoformat(),
        }
        write_manifest(out_dir, manifest)

    logger.write(f"✓ DONE | submission → {out_dir.as_posix()}/submission.csv")

if __name__ == "__main__":
    main()
