# src/metrics/precision.py
import numpy as np
from sklearn.metrics import precision_score
from .base import MetricInterface

class PrecisionMetric(MetricInterface):
    """Вычисляет точность (Precision).

    Precision = TP / (TP + FP). Доля объектов, которые модель корректно
    назвала положительными, среди всех, которых она назвала положительными.
    
    Параметры:
        threshold (float): Порог для преобразования вероятностей в классы.
        average (str): Стратегия усреднения для мультиклассовой задачи
                       ('binary', 'micro', 'macro', 'weighted').
    """
    def __init__(self, threshold: float = 0.5, average: str = 'binary'):
        self.threshold = threshold
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_class = (y_pred > self.threshold).astype(int)
        return precision_score(y_true, y_pred_class, average=self.average, zero_division=0)