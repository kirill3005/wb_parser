import numpy as np
from typing import Literal, Any, Dict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit(oof_true, oof_prob, method: Literal["platt", "isotonic"] = "platt") -> Dict[str, Any]:
    """Возвращает сериализуемый объект калибратора (dict) и может использовать sklearn внутри."""
    method = method.lower()
    calibrator = None
    if method == "platt":
        calibrator = LogisticRegression(max_iter=200)
        calibrator.fit(oof_prob.reshape(-1, 1) if oof_prob.ndim == 1 else oof_prob, oof_true)
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_prob.ravel(), oof_true.ravel())
    else:
        raise ValueError("Unknown calibration method")
    return {"method": method, "model": calibrator}


def apply(calib, prob) -> np.ndarray:
    """Применение к test_prob. Для multiclass допускается per-class калибровка (список калибраторов)."""
    method = calib.get("method")
    model = calib.get("model")
    if isinstance(model, list):
        # per-class calibration
        calibrated = []
        for c, mdl in enumerate(model):
            calibrated.append(mdl.predict_proba(prob[:, c].reshape(-1, 1))[:, 1] if method == "platt" else mdl.predict(prob[:, c]))
        return np.vstack(calibrated).T
    if method == "platt" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(prob.reshape(-1, 1) if prob.ndim == 1 else prob)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    if method == "isotonic":
        return model.predict(prob.ravel()).reshape(prob.shape)
    raise ValueError("Invalid calibrator")
