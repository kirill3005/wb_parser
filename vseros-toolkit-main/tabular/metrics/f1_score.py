# src/metrics/f1_score.py
import numpy as np
from sklearn.metrics import f1_score
from .base import MetricInterface

class F1ScoreMetric(MetricInterface):
    """Вычисляет F1-меру.
    
    Преобразует вероятности в классы по порогу 0.5.
    Поддерживает различные стратегии усреднения для мультиклассовой задачи.
    """
    def __init__(self, threshold: float = 0.5, average: str = 'binary'):
        self.threshold = threshold
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_class = (y_pred > self.threshold).astype(int)
        return f1_score(y_true, y_pred_class, average=self.average)