"""Lightweight linear baseline for ranking tasks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from recsys.rankers.base import ModelRun

LOGGER = logging.getLogger(__name__)


@dataclass
class _FittedModel:
    estimator: Any

    def predict(self, X):
        if hasattr(self.estimator, "predict_proba"):
            prob = self.estimator.predict_proba(X)[:, 1]
            return prob
        return self.estimator.predict(X)


class LinearRanker:
    name = "linear"

    def fit_single(self, X, y: np.ndarray, *, groups=None, params: dict | None = None, seed: int = 0, n_jobs: int = -1) -> _FittedModel:
        params = params or {}
        task = params.get("task", "binary")
        unique = np.unique(y)
        if len(unique) < 2:
            # fall back to regression to avoid solver errors when a fold has a single class
            est = Ridge(random_state=seed)
        elif task == "rank" or set(unique) <= {0, 1}:
            est = LogisticRegression(max_iter=1000, n_jobs=n_jobs, random_state=seed)
        else:
            est = Ridge(random_state=seed)
        est.fit(X, y)
        return _FittedModel(est)

    def fit_cv(self, X_tr, y: np.ndarray, groups, folds, *, params: dict, eval_metric: str, seed: int, n_jobs: int, time_limit_min=None, verbose: bool = True) -> ModelRun:
        # alias to trainer.train_ranker, not used directly in tests
        raise NotImplementedError


__all__ = ["LinearRanker"]
