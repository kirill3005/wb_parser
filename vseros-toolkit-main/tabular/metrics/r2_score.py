# src/metrics/r2_score.py
import numpy as np
from sklearn.metrics import r2_score
from .base import MetricInterface

class R2ScoreMetric(MetricInterface):
    """Вычисляет коэффициент детерминации (R² Score).

    Показывает, какую долю дисперсии зависимой переменной объясняет модель.
    Значение 1.0 - идеальная модель.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)