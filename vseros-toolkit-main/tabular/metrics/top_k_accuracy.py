# src/metrics/top_k_accuracy.py
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from .base import MetricInterface

class TopKAccuracyMetric(MetricInterface):
    """Вычисляет Top-K Accuracy.

    Доля правильных ответов, при условии, что правильный класс попал
    в K самых вероятных предсказаний модели.
    
    Примечание: `y_pred` для этой метрики должен быть матрицей вероятностей
    формы (n_samples, n_classes). Наш текущий `train.py` возвращает
    только вероятности для класса 1. Потребуется небольшая адаптация
    `train.py` для многоклассовой задачи.
    
    Параметры:
        k (int): Количество "топовых" предсказаний для проверки.
    """
    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Адаптация для бинарной классификации:
        # y_pred у нас - это p(y=1). Нам нужно [p(y=0), p(y=1)].
        if y_pred.ndim == 1:
            y_pred_multiclass = np.vstack([1 - y_pred, y_pred]).T
        else:
            y_pred_multiclass = y_pred
            
        return top_k_accuracy_score(y_true, y_pred_multiclass, k=self.k, labels=np.arange(y_pred_multiclass.shape[1]))