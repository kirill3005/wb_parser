# src/metrics/mae.py
import numpy as np
from sklearn.metrics import mean_absolute_error
from .base import MetricInterface

class MaeMetric(MetricInterface):
    """Вычисляет среднюю абсолютную ошибку (MAE)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)