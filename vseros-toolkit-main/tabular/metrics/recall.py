# src/metrics/recall.py
import numpy as np
from sklearn.metrics import recall_score
from .base import MetricInterface

class RecallMetric(MetricInterface):
    """Вычисляет полноту (Recall).

    Recall = TP / (TP + FN). Доля объектов положительного класса, которые
    модель смогла корректно обнаружить.
    
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
        return recall_score(y_true, y_pred_class, average=self.average, zero_division=0)