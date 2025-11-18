# src/metrics/roc_auc.py
import numpy as np
from sklearn.metrics import roc_auc_score
from .base import MetricInterface

class RocAucMetric(MetricInterface):
    """Вычисляет площадь под ROC-кривой (ROC AUC).
    
    Работает с предсказанными вероятностями.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return roc_auc_score(y_true, y_pred)