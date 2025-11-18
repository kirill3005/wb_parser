# src/metrics/qwk.py
import numpy as np
from sklearn.metrics import cohen_kappa_score
from .base import MetricInterface

class QWKMetric(MetricInterface):
    """Вычисляет квадратичную взвешенную каппу (Quadratic Weighted Kappa).
    
    Часто используется в соревнованиях Kaggle для задач упорядоченной
    классификации (например, оценка эссе по шкале от 1 до 5).
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # QWK требует целочисленных классов, поэтому округляем вероятности/предсказания
        y_pred_class = np.round(y_pred).astype(int)
        return cohen_kappa_score(y_true, y_pred_class, weights='quadratic')