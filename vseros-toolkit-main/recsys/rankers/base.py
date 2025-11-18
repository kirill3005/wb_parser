"""Base interfaces for rankers and model runs."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class ModelRun:
    """Container for cross-validated ranker outputs.

    Attributes
    ----------
    run_id: str
        Stable identifier of the run (backend+profile+seed).
    backend: str
        Backend name (lightgbm/xgboost/catboost/linear).
    task: str
        "rank" or "binary".
    cv_mean: float
        Mean validation metric across folds.
    cv_std: float
        Stddev validation metric across folds.
    oof_true: np.ndarray
        Ground-truth labels for train set in order of X_train rows.
    oof_pred: np.ndarray
        Out-of-fold predictions aligned with ``oof_true``.
    test_pred: np.ndarray
        Predictions for test rows (same order as X_test passed to fit_cv).
    groups: list[int] | None
        Group sizes for learning-to-rank tasks.
    features: list[str]
        Names/order of features used for training.
    artifacts_path: Path
        Where artifacts were stored.
    meta: dict
        Extra metadata (params, timings, profile, fingerprint).
    """

    run_id: str
    backend: str
    task: str
    cv_mean: float
    cv_std: float
    oof_true: np.ndarray
    oof_pred: np.ndarray
    test_pred: np.ndarray
    groups: list[int] | None
    features: list[str]
    artifacts_path: Path
    meta: dict[str, Any]


class Ranker(Protocol):
    """Protocol implemented by backend-specific rankers."""

    name: str

    def fit_cv(
        self,
        X_tr,
        y: np.ndarray,
        groups: list[int] | None,
        folds: list[tuple[np.ndarray, np.ndarray]],
        *,
        params: dict,
        eval_metric: str,
        seed: int,
        n_jobs: int,
        time_limit_min: int | None = None,
        verbose: bool = True,
    ) -> ModelRun:
        ...


__all__ = ["ModelRun", "Ranker"]
