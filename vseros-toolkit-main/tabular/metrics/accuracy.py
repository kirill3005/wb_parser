# src/metrics/accuracy.py
import numpy as np
from sklearn.metrics import accuracy_score
from .base import MetricInterface

class AccuracyMetric(MetricInterface):
    """Вычисляет точность (Accuracy).
    
    Преобразует вероятности в классы по порогу 0.5.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_class = (y_pred > self.threshold).astype(int)
        return accuracy_score(y_true, y_pred_class)