#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/run_model.py

Единый оркестратор обучения кандидатов на готовом фич-наборе:
- загрузка X_dense/X_sparse/y/folds,
- пофолдовое обучение с OOF и усреднением test-предсказаний,
- метрики, best_iter, фич-важности,
- опциональная калибровка (binary),
- резюмирование и таймауты.

Зависимости (опционально/по моделям):
  pyyaml, numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost, joblib
"""

from __future__ import annotations
import argparse, json, os, sys, time, math, pickle, uuid, hashlib, platform, subprocess, signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    import joblib
except Exception:
    joblib = None

# sparse
try:
    from scipy import sparse as sp
except Exception:
    sp = None

# sklearn metrics & models
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, accuracy_score,
    f1_score, mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import SGDClassifier

# libs
def _try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def _try_import_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def _try_import_catboost():
    try:
        import catboost as cb
        return cb
    except Exception:
        return None


# -------------------- utils & io --------------------
def setup_logging():
    # простой stdout-логгер
    class Logger:
        def __init__(self): pass
        def info(self, *a): print(*a, flush=True)
        def warn(self, *a): print(*a, flush=True)
        def error(self, *a): print(*a, flush=True, file=sys.stderr)
    return Logger()

LOG = setup_logging()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def env_info() -> Dict[str, Any]:
    def _ver(pkg):
        try:
            m = __import__(pkg)
            return getattr(m, "__version__", None)
        except Exception:
            return None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "scipy": _ver("scipy"),
        "sklearn": _ver("sklearn"),
        "lightgbm": _ver("lightgbm"),
        "xgboost": _ver("xgboost"),
        "catboost": _ver("catboost"),
    }


def get_git_commit(base: Union[str, Path]) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        return res.stdout.strip()
    except Exception:
        return None


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML не установлен (pip install pyyaml).")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML не найден: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def try_read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        return pd.read_parquet(p)
    except Exception:
        # fallback на fastparquet
        import fastparquet  # noqa: F401
        return pd.read_parquet(p, engine="fastparquet")


# -------------------- metrics --------------------
def resolve_task_from_y(y: np.ndarray) -> str:
    t = type_of_target(y)
    # 'binary', 'multiclass', 'continuous', 'multilabel-indicator', 'continuous-multioutput'
    if t == "binary":
        return "binary"
    if t == "multiclass":
        return "multiclass"
    if t == "multilabel-indicator":
        return "multilabel"
    return "regression"


def metric_fn(task: str, metric: str):
    m = metric.lower()
    if task in ("binary", "multiclass", "multilabel"):
        if m in ("roc_auc", "auc"):
            def _f(y_true, y_pred):
                # y_pred: proba (binary: (n,) or (n,1); multiclass: (n,C))
                if task == "binary":
                    p = y_pred.reshape(-1)
                    return roc_auc_score(y_true, p)
                elif task == "multiclass":
                    # one-vs-rest macro AUC
                    classes = np.unique(y_true)
                    Y = label_binarize(y_true, classes=classes)
                    return roc_auc_score(Y, y_pred, average="macro", multi_class="ovr")
                else:  # multilabel
                    return roc_auc_score(y_true, y_pred, average="macro")
            return _f
        if m in ("pr_auc", "ap", "average_precision"):
            def _f(y_true, y_pred):
                if task == "binary":
                    p = y_pred.reshape(-1)
                    return average_precision_score(y_true, p)
                elif task == "multiclass":
                    classes = np.unique(y_true)
                    Y = label_binarize(y_true, classes=classes)
                    # macro-AP
                    return average_precision_score(Y, y_pred, average="macro")
                else:
                    return average_precision_score(y_true, y_pred, average="macro")
            return _f
        if m == "logloss":
            def _f(y_true, y_pred):
                # y_pred: probs
                eps = 1e-15
                if task == "binary":
                    p = y_pred.reshape(-1)
                    p = np.clip(p, eps, 1 - eps)
                    P = np.vstack([1 - p, p]).T
                    return log_loss(y_true, P, labels=[0, 1])
                elif task == "multiclass":
                    return log_loss(y_true, y_pred)
                else:
                    raise ValueError("logloss не для multilabel")
            return _f
        if m in ("accuracy", "acc"):
            def _f(y_true, y_pred):
                if task == "binary":
                    pred = (y_pred.reshape(-1) >= 0.5).astype(int)
                elif task == "multiclass":
                    pred = np.argmax(y_pred, axis=1)
                else:
                    raise ValueError("accuracy не для multilabel")
                return accuracy_score(y_true, pred)
            return _f
        if m in ("f1", "macro_f1"):
            def _f(y_true, y_pred):
                if task == "binary":
                    pred = (y_pred.reshape(-1) >= 0.5).astype(int)
                    return f1_score(y_true, pred)
                elif task == "multiclass":
                    pred = np.argmax(y_pred, axis=1)
                    return f1_score(y_true, pred, average="macro")
                else:
                    raise ValueError("f1 не для multilabel")
            return _f
    # regression
    if m == "rmse":
        def _f(y_true, y_pred):
            return math.sqrt(mean_squared_error(y_true, y_pred))
        return _f
    if m == "mae":
        def _f(y_true, y_pred):
            return mean_absolute_error(y_true, y_pred)
        return _f
    if m == "mape":
        def _f(y_true, y_pred):
            # ручной MAPE во избежание зависимости от версии sklearn
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            eps = 1e-9
            return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0
        return _f

    raise ValueError(f"Неизвестная метрика: {metric} для задачи {task}")


def native_eval_name(lib: str, task: str, metric: str) -> Optional[str]:
    """Мэппинг целевой метрики на нативное имя для ранней остановки.
    Возвращаем максимально подходящий вариант; если нет прямого — None.
    """
    lib = (lib or "").lower()
    m = metric.lower()
    if lib == "lightgbm":
        if m in ("roc_auc", "auc"): return "auc"
        if m in ("pr_auc", "average_precision", "ap"): return "auc"  # LGB не умеет PR AUC нативно → возьмём AUC
        if m == "logloss":
            return "binary_logloss" if task == "binary" else "multi_logloss"
        if m == "rmse": return "rmse"
        if m == "mae": return "l1"
        return None
    if lib == "xgboost":
        if m in ("roc_auc", "auc"): return "auc"
        if m in ("pr_auc", "average_precision", "ap"): return "aucpr"
        if m == "logloss": return "logloss"
        if m == "rmse": return "rmse"
        if m == "mae": return "mae"
        return None
    if lib == "catboost":
        if m in ("roc_auc", "auc"): return "AUC"
        if m == "logloss": return "Logloss"
        if m == "rmse": return "RMSE"
        if m == "mae": return "MAE"
        return None
    return None


# -------------------- calibration (binary) --------------------
def calibrate_binary(method: str, y_true: np.ndarray, oof_pred: np.ndarray):
    """Возвращает объект-калибратор и функцию apply(pred)."""
    method = (method or "off").lower()
    if method in ("off", "none", ""):
        return None, lambda x: x
    p = oof_pred.reshape(-1)
    if method == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p.reshape(-1, 1), y_true.astype(int))
        def apply_fn(x):
            return lr.predict_proba(x.reshape(-1, 1))[:, 1]
        return ("platt", lr), apply_fn
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y_true.astype(int))
        def apply_fn(x):
            return ir.predict(x.reshape(-1))
        return ("isotonic", ir), apply_fn
    raise ValueError(f"Неизвестный метод калибровки: {method}")


def save_obj(path: Path, obj: Any):
    try:
        if joblib is not None:
            joblib.dump(obj, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
    except Exception as e:
        LOG.warn(f"[warn] Не удалось сохранить объект {path.name}: {e}")


def load_obj(path: Path) -> Any:
    try:
        if joblib is not None:
            return joblib.load(path)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None


# -------------------- data loading --------------------
def load_set(set_dir: Union[str, Path], require_dense: bool, require_sparse: bool):
    set_dir = Path(set_dir)
    y_path = set_dir/"y_train.parquet"
    ids_tr = set_dir/"ids_train.parquet"
    ids_te = set_dir/"ids_test.parquet"
    folds_pkl = set_dir/"folds.pkl"

    if not y_path.exists() or not ids_tr.exists() or not ids_te.exists() or not folds_pkl.exists():
        raise FileNotFoundError("Отсутствуют базовые файлы сета (y/ids/folds). Проверь artifacts/sets/<tag>")

    ydf = try_read_parquet(y_path)
    id_tr = try_read_parquet(ids_tr)
    id_te = try_read_parquet(ids_te)
    with open(folds_pkl, "rb") as f:
        folds = pickle.load(f)

    # dense
    Xd_tr = Xd_te = None
    if (set_dir/"X_dense_train.parquet").exists() and (set_dir/"X_dense_test.parquet").exists():
        Xd_tr = try_read_parquet(set_dir/"X_dense_train.parquet")
        Xd_te = try_read_parquet(set_dir/"X_dense_test.parquet")
    elif require_dense:
        raise FileNotFoundError("Нет dense матриц, а они требуются.")

    # sparse
    Xs_tr = Xs_te = None
    if (set_dir/"X_sparse_train.npz").exists() and (set_dir/"X_sparse_test.npz").exists():
        if sp is None:
            raise RuntimeError("SciPy не установлен, не могу читать sparse.")
        Xs_tr = sp.load_npz(set_dir/"X_sparse_train.npz")
        Xs_te = sp.load_npz(set_dir/"X_sparse_test.npz")
    elif require_sparse:
        raise FileNotFoundError("Нет sparse матриц, а они требуются.")

    # target
    tgt_cols = [c for c in ydf.columns if c != ydf.columns[0]]
    if not tgt_cols:
        raise RuntimeError("Файл y_train.parquet должен содержать id и target.")
    target_col = tgt_cols[0]
    y = ydf[target_col].to_numpy()

    return {
        "y": y,
        "target_col": target_col,
        "ids_train": id_tr.iloc[:, 0].to_numpy(),
        "ids_test": id_te.iloc[:, 0].to_numpy(),
        "folds": folds,
        "Xd_tr": Xd_tr, "Xd_te": Xd_te,
        "Xs_tr": Xs_tr, "Xs_te": Xs_te
    }


# -------------------- candidate config --------------------
def load_cand_yaml(base: Union[str, Path], name: str) -> Dict[str, Any]:
    p = Path(base)/"configs"/"models"/f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Кандидат {name} не найден: {p}")
    return read_yaml(p)


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


# -------------------- training backends --------------------
def set_threads_env(threads: int):
    if threads is None or threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)


def train_one_fold_lightgbm(params: Dict[str, Any], fit_args: Dict[str, Any],
                            X_tr, y_tr, X_va, y_va, X_te, task: str, device: str):
    lgb = _try_import_lightgbm()
    if lgb is None:
        raise RuntimeError("lightgbm не установлен.")
    # device
    p = dict(params)
    if device == "gpu":
        p["device_type"] = "gpu"
    if "n_jobs" in p and p["n_jobs"] is None:
        p.pop("n_jobs")
    model = None
    if task in ("binary", "multiclass"):
        model = lgb.LGBMClassifier(**p)
    elif task == "regression":
        model = lgb.LGBMRegressor(**p)
    else:
        raise ValueError(f"LightGBM не поддерживает task={task}")
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], **fit_args)
    best_iter = getattr(model, "best_iteration_", None)
    if task == "binary":
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
    elif task == "multiclass":
        p_va = model.predict_proba(X_va)
        p_te = model.predict_proba(X_te)
    else:
        p_va = model.predict(X_va)
        p_te = model.predict(X_te)
    return model, p_va, p_te, best_iter


def train_one_fold_xgboost(params: Dict[str, Any], fit_args: Dict[str, Any],
                           X_tr, y_tr, X_va, y_va, X_te, task: str, device: str):
    xgb = _try_import_xgboost()
    if xgb is None:
        raise RuntimeError("xgboost не установлен.")
    p = dict(params)
    # device policy
    # xgboost>=2.0 поддерживает param device='cuda'
    if device == "gpu":
        p.setdefault("tree_method", "hist")
        p["device"] = "cuda"
    else:
        p.setdefault("tree_method", "hist")
        p["device"] = "cpu"
    # model
    if task == "binary":
        model = xgb.XGBClassifier(**p)
    elif task == "multiclass":
        model = xgb.XGBClassifier(**p)
    elif task == "regression":
        model = xgb.XGBRegressor(**p)
    else:
        raise ValueError(f"XGBoost не поддерживает task={task}")

    eval_metric = fit_args.pop("eval_metric", None)
    early_stopping_rounds = fit_args.pop("early_stopping_rounds", None)
    verbose = fit_args.pop("verbose", False)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        **fit_args
    )
    best_iter = getattr(model, "best_iteration", None)
    if task == "binary":
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
    elif task == "multiclass":
        p_va = model.predict_proba(X_va)
        p_te = model.predict_proba(X_te)
    else:
        p_va = model.predict(X_va)
        p_te = model.predict(X_te)
    return model, p_va, p_te, best_iter


def train_one_fold_catboost(params: Dict[str, Any], fit_args: Dict[str, Any],
                            X_tr, y_tr, X_va, y_va, X_te, task: str, device: str):
    cb = _try_import_catboost()
    if cb is None:
        raise RuntimeError("catboost не установлен.")
    p = dict(params)
    # device
    if device == "gpu":
        p["task_type"] = "GPU"
    # model
    if task == "binary":
        model = cb.CatBoostClassifier(**p)
    elif task == "multiclass":
        model = cb.CatBoostClassifier(**p)  # auto-detect classes
    elif task == "regression":
        model = cb.CatBoostRegressor(**p)
    else:
        raise ValueError(f"CatBoost не поддерживает task={task}")

    verbose = fit_args.pop("verbose", False)
    es = fit_args.pop("early_stopping_rounds", None)
    eval_metric = fit_args.pop("eval_metric", None)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=verbose, early_stopping_rounds=es)
    try:
        best_iter = int(model.get_best_iteration())
    except Exception:
        best_iter = None

    if task in ("binary", "multiclass"):
        p_va = model.predict_proba(X_va)
        p_te = model.predict_proba(X_te)
        if task == "binary":
            p_va = p_va[:, 1]
            p_te = p_te[:, 1]
    else:
        p_va = model.predict(X_va)
        p_te = model.predict(X_te)
    return model, p_va, p_te, best_iter


def train_one_fold_linear(algo: str, params: Dict[str, Any],
                          X_tr, y_tr, X_va, y_va, X_te, task: str):
    algo = (algo or "lr").lower()
    if task in ("binary", "multiclass"):
        if algo == "lr":
            # multinomial для многокласса, liblinear для бинарной
            kwargs = dict(params)
            if task == "multiclass":
                kwargs.setdefault("multi_class", "auto")
                kwargs.setdefault("solver", "saga")
                kwargs.setdefault("max_iter", 1000)
            else:
                kwargs.setdefault("solver", "liblinear")
                kwargs.setdefault("max_iter", 1000)
            model = LogisticRegression(**kwargs)
            model.fit(X_tr, y_tr)
            p_va = model.predict_proba(X_va)
            p_te = model.predict_proba(X_te)
            if task == "binary":
                p_va = p_va[:, 1]
                p_te = p_te[:, 1]
            best_iter = None
            return model, p_va, p_te, best_iter
        elif algo == "sgd":
            # быстрая линейка с log_loss
            kwargs = dict(params)
            kwargs.setdefault("loss", "log_loss")
            kwargs.setdefault("max_iter", 1000)
            kwargs.setdefault("tol", 1e-3)
            model = SGDClassifier(**kwargs)
            model.fit(X_tr, y_tr)
            try:
                p_va = model.predict_proba(X_va)
                p_te = model.predict_proba(X_te)
                if task == "binary":
                    p_va = p_va[:, 1]
                    p_te = p_te[:, 1]
            except Exception:
                # если нет probas (некоторые конфиги) — fallback на decision_function -> sigmoid
                def _sigmoid(z): return 1 / (1 + np.exp(-z))
                d_va = model.decision_function(X_va)
                d_te = model.decision_function(X_te)
                if task == "binary":
                    p_va = _sigmoid(d_va)
                    p_te = _sigmoid(d_te)
                else:
                    # one-vs-rest для многокласса
                    e_va = np.exp(d_va)
                    p_va = e_va / np.sum(e_va, axis=1, keepdims=True)
                    e_te = np.exp(d_te)
                    p_te = e_te / np.sum(e_te, axis=1, keepdims=True)
            best_iter = None
            return model, p_va, p_te, best_iter
        else:
            raise ValueError(f"linear algo={algo} для классификации не поддержан (используй lr или sgd)")
    elif task == "regression":
        if algo == "ridge":
            model = Ridge(**params)
        elif algo == "lasso":
            model = Lasso(**params)
        else:
            raise ValueError("Для регрессии поддержаны ridge|lasso")
        model.fit(X_tr, y_tr)
        p_va = model.predict(X_va)
        p_te = model.predict(X_te)
        best_iter = None
        return model, p_va, p_te, best_iter
    else:
        raise ValueError(f"linear не поддерживает task={task}")


def feature_importance_table(model, feature_names: List[str], lib: str) -> Optional[pd.DataFrame]:
    lib = (lib or "").lower()
    try:
        if lib == "lightgbm":
            imp = model.feature_importances_
            return pd.DataFrame({"feature": feature_names, "gain": imp})
        if lib == "xgboost":
            # по gain через booster
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            rows = []
            for i, f in enumerate(feature_names):
                key = f"f{i}"
                rows.append({"feature": f, "gain": float(score.get(key, 0.0))})
            return pd.DataFrame(rows)
        if lib == "catboost":
            imp = model.get_feature_importance(type="FeatureImportance")
            return pd.DataFrame({"feature": feature_names, "gain": imp})
    except Exception:
        return None
    return None


# -------------------- main train loop --------------------
def train_candidate(
    run_dir: Path, set_dir: Path, cand_name: str, cand_cfg: Dict[str, Any],
    task: str, metric: str, data: Dict[str, Any],
    device: str, threads: int, resume: bool, timeout_min: int
) -> Dict[str, Any]:

    set_threads_env(threads)

    # resolve matrix
    matrix_pref = (cand_cfg.get("matrix") or "auto").lower()
    lib = (cand_cfg.get("lib") or "").lower()
    algo = (cand_cfg.get("algo") or None)

    Xd_tr, Xd_te, Xs_tr, Xs_te = data["Xd_tr"], data["Xd_te"], data["Xs_tr"], data["Xs_te"]
    use_sparse = use_dense = False
    if matrix_pref == "dense":
        use_dense = True
    elif matrix_pref == "sparse":
        use_sparse = True
    else:  # auto
        if lib == "linear" and (Xs_tr is not None):
            use_sparse = True
        elif Xd_tr is not None:
            use_dense = True
        elif Xs_tr is not None:
            use_sparse = True
        else:
            raise RuntimeError("Нет доступных матриц (ни dense, ни sparse).")

    if use_dense and Xd_tr is None:
        raise FileNotFoundError("Кандидат требует dense-матрицы, но она отсутствует.")
    if use_sparse and Xs_tr is None:
        raise FileNotFoundError("Кандидат требует sparse-матрицы, но она отсутствует.")

    # feature names (для импортансов)
    if use_dense:
        feat_names = list(Xd_tr.columns)
    else:
        # без имён — f_{j}
        feat_names = [f"f_{j}" for j in range(Xs_tr.shape[1])]

    y = data["y"]
    folds: List[Tuple[np.ndarray, np.ndarray]] = data["folds"]
    n = len(y)
    n_te = len(data["ids_test"])
    # classes для многокласса
    classes = np.unique(y) if task == "multiclass" else None
    C = (len(classes) if task == "multiclass" else 1)

    # allocate OOF/test
    oof = np.zeros((n, C), dtype=float)
    test_pred_folds = np.zeros((len(folds), n_te, C), dtype=float)
    per_fold = []
    best_iter_per_fold = []
    times_per_fold = []
    truncated = False

    # params & fit args
    params = cand_cfg.get("params", {}) or {}
    fit_args = cand_cfg.get("fit", {}) or {}
    # map metric to native eval_metric (if provided)
    native = native_eval_name(lib, task, metric)
    if native and lib in ("lightgbm", "xgboost"):
        fit_args = dict(fit_args)
        fit_args["eval_metric"] = native

    start_time = time.time()
    timeout_sec = (timeout_min * 60) if (timeout_min and timeout_min > 0) else None

    for k, (tr_idx, va_idx) in enumerate(folds):
        fold_model_path = run_dir / f"fold_{k}.model"
        fold_ready = fold_model_path.exists() and resume
        t0 = time.time()

        if timeout_sec is not None and (time.time() - start_time) >= timeout_sec:
            LOG.warn(f"[{cand_name}] Таймаут кандидата — прекращаем после {k} фолдов.")
            truncated = True
            break

        if fold_ready:
            LOG.info(f"[{cand_name}] skip fold {k} (resume)")
            # подгружаем предсказания, если сохранили отдельно? — пересчитаем ниже из сохранённого .oof.npy если есть
            # здесь для простоты просто пропустим — но OOF нужно иметь целиком → если резюмируем, предполагается, что есть oof.npy
            continue

        if use_dense:
            X_tr, X_va = Xd_tr.iloc[tr_idx], Xd_tr.iloc[va_idx]
            X_te = Xd_te
        else:
            X_tr, X_va = Xs_tr[tr_idx], Xs_tr[va_idx]
            X_te = Xs_te

        y_tr, y_va = y[tr_idx], y[va_idx]

        # train one fold
        if lib == "lightgbm":
            model, p_va, p_te, best_iter = train_one_fold_lightgbm(params, fit_args, X_tr, y_tr, X_va, y_va, X_te, task, device)
        elif lib == "xgboost":
            model, p_va, p_te, best_iter = train_one_fold_xgboost(params, fit_args, X_tr, y_tr, X_va, y_va, X_te, task, device)
        elif lib == "catboost":
            model, p_va, p_te, best_iter = train_one_fold_catboost(params, fit_args, X_tr, y_tr, X_va, y_va, X_te, task, device)
        elif lib == "linear":
            model, p_va, p_te, best_iter = train_one_fold_linear(algo, params, X_tr, y_tr, X_va, y_va, X_te, task)
        else:
            raise ValueError(f"Неизвестная библиотека: {lib}")

        # save model
        try:
            if joblib is not None:
                joblib.dump(model, fold_model_path)
            else:
                with open(fold_model_path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            LOG.warn(f"[warn] Не удалось сохранить модель фолда {k}: {e}")

        # shape align
        if task == "binary":
            oof[va_idx, 0] = p_va.reshape(-1)
            test_pred_folds[k, :, 0] = p_te.reshape(-1)
        elif task == "multiclass":
            oof[va_idx, :] = p_va
            test_pred_folds[k, :, :] = p_te
        else:  # regression
            oof[va_idx, 0] = p_va.reshape(-1)
            test_pred_folds[k, :, 0] = p_te.reshape(-1)

        # metric
        mf = metric_fn(task, metric)
        score = mf(y_va, p_va if task != "binary" else p_va.reshape(-1))
        per_fold.append(float(score))
        best_iter_per_fold.append(None if best_iter is None else int(best_iter))
        times_per_fold.append(float(time.time() - t0))
        LOG.info(f"[{cand_name}] fold {k}: {metric}={score:.6f} | best_iter={best_iter} | {times_per_fold[-1]:.1f}s")

        if timeout_sec is not None and (time.time() - start_time) >= timeout_sec:
            LOG.warn(f"[{cand_name}] Таймаут после завершения фолда {k}.")
            truncated = True
            break

    # если резюмировали полностью — попробуем прочитать готовый OOF/TEST
    if resume and (run_dir/"oof.npy").exists():
        try:
            oof = np.load(run_dir/"oof.npy")
            test_pred = np.load(run_dir/"test_pred.npy") if (run_dir/"test_pred.npy").exists() else None
        except Exception:
            test_pred = None
    else:
        # aggregate test
        if len(per_fold) == 0:
            raise RuntimeError("Ни один фолд не обучен (возможно, все пропущены с resume?).")
        test_pred = np.mean(test_pred_folds[:len(per_fold), :, :], axis=0)

    # cv summary
    mf = metric_fn(task, metric)
    cv = mf(y, oof[:, 0] if (task != "multiclass") else oof)
    cv_mean = float(np.mean(per_fold)) if len(per_fold) else float(cv)
    cv_std = float(np.std(per_fold)) if len(per_fold) else 0.0

    # save arrays
    np.save(run_dir/"oof.npy", oof)
    if test_pred is not None:
        np.save(run_dir/"test_pred.npy", test_pred)

    # feature importance (если возможно)
    imp_df = None
    try:
        # загрузим любую модель последнего фолда (или 0-го, если резюм)
        any_fold_model = None
        for k in reversed(range(len(folds))):
            pth = run_dir / f"fold_{k}.model"
            if pth.exists():
                any_fold_model = load_obj(pth)
                break
        if any_fold_model is not None:
            imp_df = feature_importance_table(any_fold_model, feat_names, lib)
            if imp_df is not None:
                imp_df.to_csv(run_dir/"feature_importance.csv", index=False)
    except Exception as e:
        LOG.warn(f"[warn] feature importance: {e}")

    # metrics.json
    metrics_obj = {
        "task": task, "metric": metric,
        "per_fold": per_fold,
        "cv_mean": cv_mean, "cv_std": cv_std,
        "best_iterations": best_iter_per_fold,
        "times_sec": times_per_fold,
        "truncated": truncated
    }
    (run_dir/"metrics.json").write_text(json.dumps(metrics_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics_obj


# -------------------- CLI and runner --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train model candidates on prepared feature set.")
    p.add_argument("--tag", required=True, help="run_tag из artifacts/sets/<tag>")
    p.add_argument("--cands", required=True, help="Список кандидатов (имена YAML в configs/models), через запятую")
    p.add_argument("--profile", default=None, help="Профиль тренировки (profiles/train/<name>.yaml)")
    p.add_argument("--base", default=".", help="Корень репозитория")
    p.add_argument("--sets-dir", default=None, help="Путь к наборам (по умолчанию artifacts/sets/<tag>)")
    p.add_argument("--models-dir", default="artifacts/models", help="Папка для артефактов моделей")
    p.add_argument("--metric", default="roc_auc", help="Целевая метрика (roc_auc, pr_auc, logloss, rmse, mae, ...)")
    p.add_argument("--task", default=None, help="binary|multiclass|regression|multilabel (auto, если не задано)")
    p.add_argument("--use-dense", action="store_true", help="Принудительно использовать dense-матрицу")
    p.add_argument("--use-sparse", action="store_true", help="Принудительно использовать sparse-матрицу")
    p.add_argument("--device", default="auto", help="auto|cpu|gpu")
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Продолжить/досчитать, не перетирая готовые фолды")
    p.add_argument("--timeout-min", type=int, default=0, help="Лимит времени на кандидата (мин). 0 = без лимита")
    p.add_argument("--save-test", action="store_true", help="Сохранить test_pred.npy")
    p.add_argument("--calibrate", default="off", help="off|platt|isotonic (только binary)")
    p.add_argument("--log-file", default=None, help="Файл лога (по умолчанию train.log в run_dir)")
    p.add_argument("--dry-run", action="store_true", help="Только показать план и выйти")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base).resolve()
    models_root = ensure_dir(args.models_dir)

    # load train profile (optional)
    train_profile = {}
    if args.profile:
        prof_path = Path(base)/"profiles"/"train"/f"{args.profile}.yaml"
        if prof_path.exists():
            train_profile = read_yaml(prof_path)
        else:
            # позволяем путь напрямую
            if Path(args.profile).exists():
                train_profile = read_yaml(args.profile)
            else:
                LOG.warn(f"[warn] Профиль тренировки не найден: {prof_path}")

    sets_dir = Path(args.sets_dir) if args.sets_dir else (Path("artifacts/sets")/args.tag)
    set_dir = Path(sets_dir)

    # load set (shape checks)
    require_dense = args.use_dense
    require_sparse = args.use_sparse
    data = load_set(set_dir, require_dense=require_dense, require_sparse=require_sparse)

    # resolve task
    task = args.task or resolve_task_from_y(data["y"])
    metric = args.metric

    # read candidates
    cand_names = [s.strip() for s in args.cands.split(",") if s.strip()]
    if not cand_names:
        raise RuntimeError("Не указаны кандидаты (--cands)")

    # device
    device = args.device.lower()
    if device == "auto":
        # на простоту: попробуем детект по переменной CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        device = "gpu" if (cuda_visible not in (None, "", "-1")) else "cpu"

    # threads
    threads = args.threads if args.threads is not None else -1

    # seeds (фиксируем глобально)
    np.random.seed(args.seed)

    git_hash = get_git_commit(base)

    # pick matrix preference to pass into load_set resolution (already done above)
    # dry-run preflight
    if args.dry_run:
        LOG.info("=== DRY-RUN ===")
        LOG.info(f"tag={args.tag} | task={task} | metric={metric} | device={device} | threads={threads}")
        LOG.info(f"cands: {cand_names}")
        LOG.info(f"set_dir: {set_dir}")
        LOG.info(f"available: dense={'yes' if data['Xd_tr'] is not None else 'no'}, sparse={'yes' if data['Xs_tr'] is not None else 'no'}")
        sys.exit(0)

    # train per candidate
    for cand in cand_names:
        # load cand yaml and merge with profile defaults
        cand_cfg = load_cand_yaml(base, cand)
        cand_cfg = merge_dicts(train_profile, cand_cfg)

        lib = (cand_cfg.get("lib") or "").lower()
        algo = cand_cfg.get("algo")
        matrix_pref = (cand_cfg.get("matrix") or "auto").lower()

        # build run_id
        run_id = f"{task}-{lib}{('-'+str(algo)) if (lib=='linear' and algo) else ''}-{cand}-{args.tag}-{short_hash(str(time.time())+cand)}"
        run_dir = ensure_dir(models_root / run_id)

        log_file = args.log_file or (run_dir / "train.log")
        try:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {run_id} ====\n")
        except Exception:
            pass

        LOG.info(f"\n=== TRAIN {cand} → {run_id} ===")
        LOG.info(f"task={task} metric={metric} device={device} threads={threads} matrix={matrix_pref}")

        # save config.json
        cfg_dump = {
            "tag": args.tag,
            "cand": cand,
            "lib": lib,
            "algo": algo,
            "matrix": matrix_pref,
            "task": task,
            "metric": metric,
            "device": device,
            "threads": threads,
            "seed": args.seed,
            "timeout_min": args.timeout_min,
            "resume": bool(args.resume),
            "save_test": bool(args.save_test),
            "calibrate": args.calibrate,
            "sets_dir": str(set_dir),
            "env": env_info(),
            "git": git_hash,
            "params": cand_cfg.get("params", {}),
            "fit": cand_cfg.get("fit", {}),
        }
        (run_dir/"config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2), encoding="utf-8")

        # train
        metrics_obj = train_candidate(
            run_dir=run_dir, set_dir=set_dir, cand_name=cand, cand_cfg=cand_cfg,
            task=task, metric=metric, data=data, device=device, threads=threads,
            resume=args.resume, timeout_min=args.timeout_min
        )

        LOG.info(f"[{cand}] CV {metric}: {metrics_obj['cv_mean']:.6f} ± {metrics_obj['cv_std']:.6f}")

        # calibration (binary only)
        if args.calibrate.lower() not in ("off", "none", "") and task == "binary":
            LOG.info(f"[{cand}] calibration: {args.calibrate}")
            try:
                oof = np.load(run_dir/"oof.npy").reshape(-1)
                y = data["y"].astype(int)
                cal_obj, apply_fn = calibrate_binary(args.calibrate, y, oof)
                if cal_obj is not None:
                    # save calibrator
                    method, obj = cal_obj
                    try:
                        save_obj(run_dir/"calibrator.joblib", obj)
                    except Exception:
                        pass
                    # apply to test
                    if (run_dir/"test_pred.npy").exists():
                        test_raw = np.load(run_dir/"test_pred.npy").reshape(-1)
                        test_cal = apply_fn(test_raw)
                        np.save(run_dir/"test_pred_cal.npy", test_cal)
            except Exception as e:
                LOG.warn(f"[warn] calibration failed: {e}")
        elif args.calibrate.lower() not in ("off", "none", "") and task != "binary":
            LOG.warn("Калибровка включена, но поддерживается только для binary — пропускаю.")

        # optionally save test (уже сохранён), просто контроль
        if args.save_test and not (run_dir/"test_pred.npy").exists():
            LOG.warn("[warn] test_pred отсутствует — возможно, все фолды были пропущены из-за resume.")

        # append to models index
        try:
            idx_path = models_root / "index.json"
            index = {}
            if idx_path.exists():
                index = json.loads(idx_path.read_text(encoding="utf-8"))
            index[str(run_id)] = {
                "tag": args.tag,
                "cand": cand,
                "lib": lib,
                "algo": algo,
                "task": task,
                "metric": metric,
                "cv_mean": float(metrics_obj.get("cv_mean", 0.0)),
                "cv_std": float(metrics_obj.get("cv_std", 0.0)),
                "n_folds": len(metrics_obj.get("per_fold", [])),
                "time_sec": float(sum(metrics_obj.get("times_sec", []) or [0.0])),
                "path": str(run_dir)
            }
            idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            LOG.warn(f"[warn] не удалось обновить models index: {e}")

    LOG.info("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
