# src/metrics/mape.py
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from .base import MetricInterface

class MapeMetric(MetricInterface):
    """Вычисляет среднюю абсолютную ошибку в процентах (MAPE).
    
    MAPE = (1/n) * Σ(|(y_true - y_pred) / y_true|) * 100
    Показывает, на сколько процентов в среднем ошибается модель.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_percentage_error(y_true, y_pred)