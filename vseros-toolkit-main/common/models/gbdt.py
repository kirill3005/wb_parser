from typing import Dict, Optional, List, Any
import warnings
import numpy as np

from .types import ModelRun
from . import artifacts as A, eval as ME


_DEF_METRICS = {
    "regression": "rmse",
    "binary": "roc_auc",
    "multiclass": "f1_macro",
}


_DEF_OBJECTIVES = {
    "regression": "regression",
    "binary": "binary",
    "multiclass": "multiclass",
}


def _infer_num_classes(y) -> int:
    if y is None:
        return 0
    if len(y.shape) > 1:
        return y.shape[1]
    return int(len(np.unique(y)))


def _predict_wrapper(model, X, lib: str, task: str, num_class: int):
    if lib == "lightgbm":
        pred = model.predict(X)
    elif lib == "catboost":
        pred = model.predict_proba(X) if task in {"binary", "multiclass"} else model.predict(X)
    elif lib == "xgboost":
        pred = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
    else:
        pred = model.predict(X)

    if task == "binary" and pred.ndim > 1 and pred.shape[1] == 2:
        return pred[:, 1]
    if task == "multiclass" and pred.ndim == 1 and num_class > 2:
        # xgboost might return class labels; convert to onehot probabilities
        out = np.zeros((pred.shape[0], num_class))
        out[np.arange(pred.shape[0]), pred.astype(int)] = 1.0
        return out
    return pred


def _init_pred_arrays(task: str, n_samples: int, n_test: int, num_class: int):
    if task == "regression" or task == "binary":
        oof_pred = np.zeros(n_samples)
        test_pred = np.zeros(n_test)
    else:
        oof_pred = np.zeros((n_samples, num_class))
        test_pred = np.zeros((n_test, num_class))
    return oof_pred, test_pred


def _setup_lightgbm(task: str, params: Dict[str, Any], seed: int, n_jobs: int, num_class: int):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is not installed. Please `pip install lightgbm`. ")

    params = dict(params or {})
    params.setdefault("objective", _DEF_OBJECTIVES.get(task, "binary"))
    if task == "multiclass":
        params.setdefault("num_class", num_class)
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", n_jobs)
    return lgb, params


def _setup_catboost(task: str, params: Dict[str, Any], seed: int, n_jobs: int):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost is not installed. Please `pip install catboost`. ")

    params = dict(params or {})
    params.setdefault("random_seed", seed)
    params.setdefault("thread_count", n_jobs)
    model_cls = CatBoostRegressor if task == "regression" else CatBoostClassifier
    return model_cls, params


def _setup_xgboost(task: str, params: Dict[str, Any], seed: int, n_jobs: int, num_class: int):
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is not installed. Please `pip install xgboost`. ")

    params = dict(params or {})
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", n_jobs)
    if task == "multiclass":
        params.setdefault("num_class", num_class)
        params.setdefault("objective", "multi:softprob")
    elif task == "binary":
        params.setdefault("objective", "binary:logistic")
    else:
        params.setdefault("objective", "reg:squarederror")
    model_cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    return model_cls, params


def train_cv(
    X_train,
    y,
    X_test,
    folds: List[tuple[np.ndarray, np.ndarray]],
    *,
    params: Dict[str, Any],
    run_id: Optional[str] = None,
    lib: str = "lightgbm",
    task: str = "binary",
    class_weight: Optional[dict] = None,
    sample_weight=None,
    seed: int = 42,
    n_jobs: int = 4,
    save: bool = True,
    resume: bool = True,
    show_progress: bool = False,
    verbose: bool = False,
) -> ModelRun:
    """
    Обучает по фолдам, после КАЖДОГО фолда сохраняет модель в artifacts/models/<run_id>/.
    Возвращает ModelRun с oof/test, fold_scores и manifest.

    show_progress управляет выводом прогресс-бара по фолдам, verbose — подробными логами
    и verbose-режимом библиотек.
    """

    task = task.lower()
    metric_name = params.get("metric", _DEF_METRICS.get(task, "rmse")) if params else _DEF_METRICS.get(task, "rmse")
    scorer = ME.get_scorer(task, metric_name)

    num_class = _infer_num_classes(y) if task != "binary" else 2
    n_samples = y.shape[0]
    n_test = X_test.shape[0] if X_test is not None else 0
    oof_pred, test_pred = _init_pred_arrays(task, n_samples, n_test, num_class)
    oof_true = y

    run_id = run_id or A.make_run_id(task=task, model=lib, feat_hash="manual", seed=seed)
    run_dir = A.path(run_id)
    existing = A.existing_folds(run_dir) if resume else set()

    manifest = {
        "task": task,
        "lib": lib,
        "params": params,
        "metric": metric_name,
        "seed": seed,
        "folds": len(folds),
    }

    total_folds = len(folds)
    if verbose:
        print(
            f"[gbdt.train_cv] lib={lib}, task={task}, run_id={run_id}, "
            f"folds={total_folds}, resume={resume}"
        )
        train_shape = getattr(X_train, "shape", None)
        test_shape = getattr(X_test, "shape", None)
        print(f"[gbdt.train_cv] train_shape={train_shape}, test_shape={test_shape}")
        print(f"[gbdt.train_cv] params={params}")

    fold_iterable = folds
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            fold_iterable = tqdm(folds, total=total_folds, desc=f"{lib} CV")
        except ImportError:
            if verbose:
                print("[gbdt.train_cv] tqdm is not installed; progress bar disabled")

    for k, (tr_idx, val_idx) in enumerate(fold_iterable):
        model = None
        fold_ext = ".joblib"
        try:
            if verbose:
                print(
                    f"[gbdt.train_cv] Fold {k + 1}/{total_folds}: "
                    f"train={len(tr_idx)}, val={len(val_idx)}"
                )
            if lib == "lightgbm":
                lgb, lgb_params = _setup_lightgbm(task, params, seed, n_jobs, num_class)
                fold_ext = ".lgb"
                model_path = A.fold_path(run_dir, k, fold_ext)
                if resume and k in existing and model_path.exists():
                    if verbose:
                        print(f"[gbdt.train_cv] loading LightGBM fold from {model_path}")
                    model = lgb.Booster(model_file=str(model_path))
                else:
                    train_set = lgb.Dataset(X_train[tr_idx], label=y[tr_idx], weight=None if sample_weight is None else sample_weight[tr_idx])
                    val_set = lgb.Dataset(X_train[val_idx], label=y[val_idx], weight=None if sample_weight is None else sample_weight[val_idx])
                    model = lgb.train(
                        lgb_params,
                        train_set,
                        valid_sets=[val_set],
                        num_boost_round=lgb_params.get("num_boost_round", lgb_params.get("n_estimators", 200)),
                        verbose_eval=verbose,
                    )
                    if save:
                        model.save_model(str(model_path))

            elif lib == "catboost":
                model_cls, cb_params = _setup_catboost(task, params, seed, n_jobs)
                fold_ext = ".cbm"
                model_path = A.fold_path(run_dir, k, fold_ext)
                if resume and k in existing and model_path.exists():
                    if verbose:
                        print(f"[gbdt.train_cv] loading CatBoost fold from {model_path}")
                    model = model_cls()
                    model.load_model(str(model_path))
                else:
                    model = model_cls(**cb_params)
                    fit_params = {
                        "X": X_train[tr_idx],
                        "y": y[tr_idx],
                        "eval_set": (X_train[val_idx], y[val_idx]),
                        "verbose": verbose,
                    }
                    if class_weight is not None:
                        fit_params["class_weights"] = class_weight
                    model.fit(**fit_params)
                    if save:
                        model.save_model(str(model_path))

            elif lib == "xgboost":
                model_cls, xgb_params = _setup_xgboost(task, params, seed, n_jobs, num_class)
                fold_ext = ".xgb"
                model_path = A.fold_path(run_dir, k, fold_ext)
                if resume and k in existing and model_path.exists():
                    if verbose:
                        print(f"[gbdt.train_cv] loading XGBoost fold from {model_path}")
                    model = model_cls()
                    model.load_model(str(model_path))
                else:
                    model = model_cls(**xgb_params)
                    model.fit(
                        X_train[tr_idx],
                        y[tr_idx],
                        eval_set=[(X_train[val_idx], y[val_idx])],
                        verbose=verbose,
                        sample_weight=None if sample_weight is None else sample_weight[tr_idx],
                    )
                    if save:
                        model.save_model(str(model_path))
            else:
                raise ValueError(f"Unsupported lib: {lib}")
        except ImportError as e:
            warnings.warn(str(e))
            raise

        if verbose:
            print(f"[gbdt.train_cv] Fold {k + 1}/{total_folds} finished")

        # predictions
        val_pred = _predict_wrapper(model, X_train[val_idx], lib, task, num_class)
        if task in {"multiclass"} and val_pred.ndim == 1:
            # catboost for multiclass returns class labels
            onehot = np.zeros((val_pred.shape[0], num_class))
            onehot[np.arange(val_pred.shape[0]), val_pred.astype(int)] = 1.0
            val_pred = onehot
        oof_pred[val_idx] = val_pred

        if X_test is not None:
            test_fold_pred = _predict_wrapper(model, X_test, lib, task, num_class)
            if task == "multiclass" and test_fold_pred.ndim == 1:
                onehot = np.zeros((test_fold_pred.shape[0], num_class))
                onehot[np.arange(test_fold_pred.shape[0]), test_fold_pred.astype(int)] = 1.0
                test_fold_pred = onehot
            test_pred += test_fold_pred / len(folds)

    fold_scores = ME.cv_scores_by_folds(oof_true, oof_pred, folds, scorer)
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))

    manifest["fold_scores"] = fold_scores
    manifest["cv_mean"] = cv_mean
    manifest["cv_std"] = cv_std

    if save:
        A.save_manifest(run_dir, manifest)
        A.save_array(run_dir, "oof_true.npy", oof_true)
        A.save_array(run_dir, "oof_pred.npy", oof_pred)
        if X_test is not None:
            A.save_array(run_dir, "test_pred.npy", test_pred)

    if verbose:
        print(f"[gbdt.train_cv] fold_scores={fold_scores}")
        print(f"[gbdt.train_cv] CV mean={cv_mean:.4f}, std={cv_std:.4f}")
        print(f"[gbdt.train_cv] artifacts saved to {run_dir}")

    return ModelRun(
        run_id=run_id,
        task=task,
        oof_true=oof_true,
        oof_pred=oof_pred,
        test_pred=test_pred,
        fold_scores=fold_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        artifacts_path=run_dir,
        manifest=manifest,
    )
