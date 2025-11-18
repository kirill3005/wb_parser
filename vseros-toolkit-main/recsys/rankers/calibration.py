"""Simple Platt and isotonic calibration wrappers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from recsys.rankers.linear import LinearRanker


@dataclass
class CalibratedModel:
    calibrator: CalibratedClassifierCV

    def predict(self, X):
        proba = self.calibrator.predict_proba(X)
        return proba[:, 1]


def platt_scale(X, y, *, seed: int = 0):
    base = LinearRanker().fit_single(X, y, params={"task": "binary"}, seed=seed)
    calibrator = CalibratedClassifierCV(base.estimator, method="sigmoid", cv=3)
    calibrator.fit(X, y)
    return CalibratedModel(calibrator)


def isotonic_scale(X, y, *, seed: int = 0):
    base = LinearRanker().fit_single(X, y, params={"task": "binary"}, seed=seed)
    calibrator = CalibratedClassifierCV(base.estimator, method="isotonic", cv=3)
    calibrator.fit(X, y)
    return CalibratedModel(calibrator)


__all__ = ["platt_scale", "isotonic_scale", "CalibratedModel"]
