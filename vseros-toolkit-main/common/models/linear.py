from typing import Dict, Optional, List, Any
import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Ridge,
    Lasso,
    SGDRegressor,
)
from sklearn.multiclass import OneVsRestClassifier
import joblib

from .types import ModelRun
from . import artifacts as A, eval as ME


_DEF_METRICS = {
    "regression": "rmse",
    "binary": "roc_auc",
    "multiclass": "f1_macro",
    "multilabel": "f1_micro",
}


def _get_estimator(task: str, algo: str, params: Dict[str, Any], seed: int):
    params = dict(params or {})
    if task == "regression":
        if algo == "ridge":
            params.setdefault("random_state", seed)
            return Ridge(**params)
        if algo == "lasso":
            params.setdefault("random_state", seed)
            return Lasso(**params)
        if algo == "sgd":
            params.setdefault("random_state", seed)
            return SGDRegressor(**params)
        raise ValueError(f"Unsupported regression algo: {algo}")

    if task in {"binary", "multiclass", "multilabel"}:
        if algo == "lr":
            params.setdefault("max_iter", 200)
            params.setdefault("n_jobs", None)
            params.setdefault("random_state", seed)
            return LogisticRegression(**params)
        if algo == "sgd":
            params.setdefault("loss", "log_loss")
            params.setdefault("random_state", seed)
            params.setdefault("max_iter", 1000)
            return SGDClassifier(**params)
        raise ValueError(f"Unsupported classification algo: {algo}")

    raise ValueError(f"Unknown task: {task}")


def _predict(model, X, task: str):
    if task == "regression":
        return model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim > 1 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores
    return model.predict(X)


def train_cv(
    X_train,
    y,
    X_test,
    folds,
    *,
    algo: str = "lr",  # "lr"|"sgd"|"ridge"|"lasso"
    task: str = "binary",  # regression|binary|multiclass|multilabel
    params: Dict[str, Any] | None = None,
    run_id: Optional[str] = None,
    seed: int = 42,
    n_jobs: int = 4,
    save: bool = True,
    resume: bool = True,
    show_progress: bool = False,
    verbose: bool = False,
) -> ModelRun:
    """OVR/BR для multi*, калибровка НЕ входит (отдельный модуль).

    show_progress управляет выводом прогресс-бара по фолдам, verbose — подробными логами
    и уведомлениями о сохранении.
    """

    task = task.lower()
    params = dict(params or {})
    metric_name = params.get("metric", _DEF_METRICS.get(task, "rmse"))
    scorer = ME.get_scorer(task, metric_name)

    num_class = y.shape[1] if task == "multilabel" else (len(np.unique(y)) if task == "multiclass" else 2)
    n_samples = y.shape[0]
    n_test = X_test.shape[0] if X_test is not None else 0

    if task in {"multiclass", "multilabel"}:
        oof_pred = np.zeros((n_samples, num_class))
        test_pred = np.zeros((n_test, num_class))
    else:
        oof_pred = np.zeros(n_samples)
        test_pred = np.zeros(n_test)

    run_id = run_id or A.make_run_id(task=task, model=f"linear-{algo}", feat_hash="manual", seed=seed)
    run_dir = A.path(run_id)
    existing = A.existing_folds(run_dir) if resume else set()

    manifest = {
        "task": task,
        "algo": algo,
        "params": params,
        "metric": metric_name,
        "seed": seed,
        "folds": len(folds),
    }

    total_folds = len(folds)
    if verbose:
        print(
            f"[linear.train_cv] algo={algo}, task={task}, run_id={run_id}, "
            f"folds={total_folds}, resume={resume}"
        )
        print(f"[linear.train_cv] params={params}")

    fold_iterable = folds
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            fold_iterable = tqdm(folds, total=total_folds, desc="linear CV")
        except ImportError:
            if verbose:
                print("[linear.train_cv] tqdm is not installed; progress bar disabled")

    for k, (tr_idx, val_idx) in enumerate(fold_iterable):
        fold_path = A.fold_path(run_dir, k, ".joblib")
        if resume and k in existing and fold_path.exists():
            if verbose:
                print(f"[linear.train_cv] loading fold {k} from {fold_path}")
            model = joblib.load(fold_path)
        else:
            base_estimator = _get_estimator(task, algo, params, seed)
            if task in {"multiclass", "multilabel"}:
                model = OneVsRestClassifier(base_estimator, n_jobs=n_jobs)
            else:
                model = base_estimator
            if hasattr(model, "set_params"):
                model.set_params(**{"n_jobs": n_jobs}) if task != "regression" else None
            if verbose:
                print(
                    f"[linear.train_cv] Fold {k + 1}/{total_folds}: "
                    f"train={len(tr_idx)}, val={len(val_idx)}"
                )
            model.fit(X_train[tr_idx], y[tr_idx])
            if save:
                joblib.dump(model, fold_path)

        val_pred = _predict(model, X_train[val_idx], task)
        oof_pred[val_idx] = val_pred

        if verbose:
            print(f"[linear.train_cv] Fold {k + 1}/{total_folds} finished")

        if X_test is not None:
            test_pred += _predict(model, X_test, task) / len(folds)

    fold_scores = ME.cv_scores_by_folds(y, oof_pred, folds, scorer)
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))

    manifest["fold_scores"] = fold_scores
    manifest["cv_mean"] = cv_mean
    manifest["cv_std"] = cv_std

    if save:
        A.save_manifest(run_dir, manifest)
        A.save_array(run_dir, "oof_true.npy", y)
        A.save_array(run_dir, "oof_pred.npy", oof_pred)
        if X_test is not None:
            A.save_array(run_dir, "test_pred.npy", test_pred)

    if verbose:
        print(f"[linear.train_cv] fold_scores={fold_scores}")
        print(f"[linear.train_cv] CV mean={cv_mean:.4f}, std={cv_std:.4f}")
        print(f"[linear.train_cv] artifacts saved to {run_dir}")

    return ModelRun(
        run_id=run_id,
        task=task,
        oof_true=y,
        oof_pred=oof_pred,
        test_pred=test_pred,
        fold_scores=fold_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        artifacts_path=run_dir,
        manifest=manifest,
    )
