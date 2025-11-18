# src/metrics/ranking.py

from typing import Any
import numpy as np
import pandas as pd

from .base import MetricInterface

class MapAtKMetric(MetricInterface):
    """
    Вычисляет Mean Average Precision @ k (mAP@k).

    Эта метрика является стандартом для оценки качества систем ранжирования.
    Она учитывает как наличие релевантных товаров в топ-K, так и их порядок.

    Логика расчета:
    1. Для каждого пользователя (group) предсказания сортируются по убыванию.
    2. Вычисляется Average Precision (AP@k) для топ-K предсказаний.
    3. Значения AP@k усредняются по всем пользователям.

    Параметры:
        k (int): Количество "топовых" предсказаний, которые учитываются.
    """
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k должно быть положительным целым числом.")
        self.k = k
        self.name = f"mAP@{self.k}" # Имя для логов

    def _calculate_ap_at_k(self, y_true_sorted: np.ndarray, total_relevant: int) -> float:
        """Вспомогательная функция для расчета AP@k для одного пользователя."""
        if total_relevant == 0:
            return 0.0

        hits = np.cumsum(y_true_sorted)
        positions = np.arange(1, len(y_true_sorted) + 1)
        
        # Вычисляем precision на каждой позиции
        precisions = hits / positions
        
        # Суммируем precision только на позициях, где был релевантный товар
        ap = np.sum(precisions * y_true_sorted) / total_relevant
        return ap

    def _apply_metric_by_group(self, group_df: pd.DataFrame) -> float:
        """Применяется к каждой группе (пользователю) в DataFrame."""
        
        # Находим общее количество релевантных товаров для этого пользователя
        total_relevant_in_group = group_df['y_true'].sum()
        
        # Сортируем товары по предсказанному скору
        sorted_group = group_df.sort_values('y_pred', ascending=False)
        
        # Берем топ-K и их истинные метки
        top_k_true_labels = sorted_group['y_true'].head(self.k).values
        
        return self._calculate_ap_at_k(top_k_true_labels, total_relevant_in_group)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        """
        Основной метод вычисления mAP@k.

        Требует, чтобы в `kwargs` был передан ключ 'groups' с сериями ID
        пользователей, чтобы правильно сгруппировать предсказания.
        """
        if 'groups' not in kwargs:
            raise ValueError("mAP@k требует передачи 'groups' (ID пользователей) в kwargs.")
        
        groups = kwargs['groups']

        if not isinstance(groups, pd.Series):
            raise TypeError("Параметр 'groups' должен быть объектом pd.Series.")

        # Собираем все в один DataFrame для удобства
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'groups': groups
        })
        
        # Считаем AP@k для каждого пользователя и усредняем
        mean_ap = df.groupby('groups').apply(self._apply_metric_by_group).mean()
        
        return mean_ap