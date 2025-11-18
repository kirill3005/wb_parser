# src/metrics/average_precision.py
import numpy as np
from sklearn.metrics import average_precision_score
from .base import MetricInterface

class AveragePrecisionMetric(MetricInterface):
    """Вычисляет среднюю точность (Average Precision, AP).

    Эта метрика суммирует кривую precision-recall и особенно полезна для
    несбалансированных датасетов, где важно качество ранжирования.
    Работает с предсказанными вероятностями.

    Параметры:
        average (str): Стратегия усреднения для мультиклассовой задачи
                       ('macro', 'weighted', 'micro').
    """
    def __init__(self, average: str = 'macro'):
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return average_precision_score(y_true, y_pred, average=self.average)