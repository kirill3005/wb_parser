# src/metrics/rmsle.py
import numpy as np
from sklearn.metrics import mean_squared_log_error
from .base import MetricInterface

class RmsleMetric(MetricInterface):
    """Вычисляет корень из среднеквадратичной логарифмической ошибки (RMSLE).
    
    Полезна, когда таргет имеет экспоненциальный рост или когда мы больше
    заботимся об относительном, а не абсолютном размере ошибки.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Убедимся, что предсказания не отрицательные, чтобы избежать ошибок с логарифмом
        return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))