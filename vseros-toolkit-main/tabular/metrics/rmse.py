# src/metrics/rmse.py
import numpy as np
from sklearn.metrics import mean_squared_error
from .base import MetricInterface

class RmseMetric(MetricInterface):
    """Вычисляет корень из среднеквадратичной ошибки (RMSE)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))