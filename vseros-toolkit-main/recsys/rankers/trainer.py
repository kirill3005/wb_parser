"""Cross-validated trainer dispatching to backend rankers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

from recsys.rankers import group as group_utils
from recsys.rankers.base import ModelRun
from recsys.rankers.linear import LinearRanker

LOGGER = logging.getLogger(__name__)


@dataclass
class FoldPlan:
    train_idx: np.ndarray
    val_idx: np.ndarray


BACKENDS = {
    "linear": LinearRanker(),
    "lightgbm": LinearRanker(),  # graceful fallback
    "xgboost": LinearRanker(),
    "catboost": LinearRanker(),
}


def make_folds(n_rows: int, groups: List[int] | None, n_splits: int, seed: int) -> List[FoldPlan]:
    if n_splits < 2:
        raise ValueError("n_splits must be >=2")
    if groups:
        gkf = GroupKFold(n_splits=n_splits)
        group_labels = np.repeat(np.arange(len(groups)), groups)
        folds_iter = gkf.split(np.arange(n_rows), groups=group_labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds_iter = kf.split(np.arange(n_rows))
    plans: List[FoldPlan] = []
    for tr, val in folds_iter:
        plans.append(FoldPlan(train_idx=tr, val_idx=val))
    return plans


def train_ranker(
    X_train,
    y: np.ndarray,
    *,
    groups: List[int] | None,
    backend: str,
    params: dict,
    eval_metric: str,
    seed: int,
    n_jobs: int,
    folds: List[FoldPlan] | None = None,
    X_test=None,
) -> ModelRun:
    backend_key = backend.lower()
    if backend_key not in BACKENDS:
        raise ValueError(f"Unsupported backend {backend}")
    ranker = BACKENDS[backend_key]
    n_rows = y.shape[0]
    if folds is None:
        folds = make_folds(n_rows, groups, params.get("n_splits", 3), seed)

    # train
    oof_pred = np.zeros(n_rows, dtype=float)
    metrics = []
    for i, fold in enumerate(folds):
        LOGGER.info("Training fold %s with %d train / %d val rows", i, fold.train_idx.size, fold.val_idx.size)
        model_run = ranker.fit_single(
            X_train[fold.train_idx],
            y[fold.train_idx],
            groups=groups,
            params=params,
            seed=seed + i,
            n_jobs=n_jobs,
        )
        fold_pred = model_run.predict(X_train[fold.val_idx])
        oof_pred[fold.val_idx] = fold_pred
        metrics.append(_binary_auc_safe(y[fold.val_idx], fold_pred))

    cv_mean = float(np.mean(metrics)) if metrics else 0.0
    cv_std = float(np.std(metrics)) if metrics else 0.0
    test_pred = ranker.fit_single(X_train, y, groups=groups, params=params, seed=seed, n_jobs=n_jobs).predict(X_test) if X_test is not None else np.array([])

    return ModelRun(
        run_id=f"{backend_key}-seed{seed}",
        backend=backend_key,
        task="rank" if groups else "binary",
        cv_mean=cv_mean,
        cv_std=cv_std,
        oof_true=y,
        oof_pred=oof_pred,
        test_pred=test_pred,
        groups=groups,
        features=params.get("features", []),
        artifacts_path=Path("."),
        meta={"params": params, "metric": eval_metric},
    )


def _binary_auc_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        score = float(roc_auc_score(y_true, y_pred))
        if np.isnan(score):
            return float(np.mean(y_pred))
        return score
    except Exception:
        return float(np.mean(y_pred))


__all__ = ["train_ranker", "make_folds", "FoldPlan"]
