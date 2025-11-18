# src/metrics/log_loss.py
import numpy as np
from sklearn.metrics import log_loss
from .base import MetricInterface

class LogLossMetric(MetricInterface):
    """Вычисляет логистическую функцию потерь (LogLoss).
    
    Работает с предсказанными вероятностями.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return log_loss(y_true, y_pred)