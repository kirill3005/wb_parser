# src/features/categorical/target_based.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from ..base import FeatureGenerator

# ==================================================================================
# TargetEncoderGenerator
# ==================================================================================
class TargetEncoderGenerator(FeatureGenerator):
    """
    Применяет сглаженное кодирование на основе среднего значения таргета (Target Encoding).

    Заменяет каждую категорию на сглаженное среднее значение целевой переменной
    для этой категории. Сглаживание необходимо для уменьшения переобучения на
    редких категориях.

    Формула сглаживания:
    `smoothed = (group_mean * group_count + global_mean * strength) / (group_count + strength)`

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для кодирования.
        target_col (str): Имя целевой переменной.
        smoothing_strength (float): Сила сглаживания. Чем выше значение, тем
            сильнее средние по группам "смещаются" к глобальному среднему.
            Является важным гиперпараметром.
    """
    def __init__(self, name: str, cols: List[str], target_col: str, smoothing_strength: float = 20.0):
        super().__init__(name)
        self.cols = cols
        self.target_col = target_col
        self.smoothing_strength = smoothing_strength
        self.mappings_: Dict[str, pd.Series] = {}
        self.global_mean_: float = 0.0

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет сглаженные средние для каждой категории
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение TargetEncoder на колонках: {self.cols}")
        
        # 1. Вычисляем глобальное среднее
        self.global_mean_ = data[self.target_col].mean()
        print(f"  - Глобальное среднее таргета: {self.global_mean_:.4f}")
        
        for col in self.cols:
            # 2. Вычисляем среднее и количество для каждой категории
            agg = data.groupby(col)[self.target_col].agg(['mean', 'count'])
            
            # 3. Применяем формулу сглаживания
            counts = agg['count']
            means = agg['mean']
            smoothed_means = (means * counts + self.global_mean_ * self.smoothing_strength) / (counts + self.smoothing_strength)
            
            # 4. Сохраняем итоговый словарь
            self.mappings_[col] = smoothed_means

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет сохраненное отображение. Категории, не встреченные при
        обучении, получают значение глобального среднего.
        """
        df = data.copy()
        print(f"[{self.name}] Применение TargetEncoder к {len(df)} строкам.")
        for col in self.cols:
            mapped_values = df[col].map(self.mappings_[col])
            # Заполняем глобальным средним те категории, которых не было в трейне
            df[f"{col}_te"] = mapped_values.fillna(self.global_mean_)

        # Исходные колонки оставляем, чтобы последующие шаги могли использовать
        # исходные значения (например, для взаимодействий или других преобразований).
        return df

# ==================================================================================
# WoEEncoderGenerator
# ==================================================================================
class WoEEncoderGenerator(FeatureGenerator):
    """
    Применяет кодирование "Вес доказательства" (Weight of Evidence).

    WoE используется ИСКЛЮЧИТЕЛЬНО для задач бинарной классификации.
    Он измеряет "силу" признака в разделении классов 1 и 0.
    WoE = ln(% positive / % negative).
    
    Положительный WoE означает, что категория чаще встречается у класса 1.
    Отрицательный WoE - что чаще у класса 0.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для кодирования.
        target_col (str): Имя целевой переменной (должна быть 0/1).
        epsilon (float): Малая константа для избежания деления на ноль.
    """
    def __init__(self, name: str, cols: List[str], target_col: str, epsilon: float = 1e-6):
        super().__init__(name)
        self.cols = cols
        self.target_col = target_col
        self.epsilon = epsilon
        self.mappings_: Dict[str, pd.Series] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет WoE для каждой категории
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение WoE Encoder на колонках: {self.cols}")
        
        if not all(data[self.target_col].isin([0, 1])):
            raise ValueError("WoE кодирование применимо только к задачам бинарной классификации (таргет 0/1).")
        
        # 1. Считаем общее количество 1 и 0
        total_positives = data[self.target_col].sum()
        total_negatives = len(data) - total_positives
        
        for col in self.cols:
            # 2. Группируем и считаем количество и сумму (кол-во 1) для каждой категории
            agg = data.groupby(col)[self.target_col].agg(['count', 'sum'])
            agg.rename(columns={'sum': 'positives'}, inplace=True)
            
            # 3. Считаем количество 0
            agg['negatives'] = agg['count'] - agg['positives']
            
            # 4. Считаем долю 1 и 0, добавляя эпсилон для стабильности
            pct_pos = (agg['positives'] + self.epsilon) / total_positives
            pct_neg = (agg['negatives'] + self.epsilon) / total_negatives
            
            # 5. Считаем WoE и сохраняем
            self.mappings_[col] = np.log(pct_pos / pct_neg)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет сохраненное WoE-отображение. Категории, не встреченные
        при обучении, получают нейтральное значение 0.
        """
        df = data.copy()
        print(f"[{self.name}] Применение WoE Encoder к {len(df)} строкам.")
        for col in self.cols:
            mapped_values = df[col].map(self.mappings_[col])
            # WoE=0 означает отсутствие "веса доказательства"
            df[f"{col}_woe"] = mapped_values.fillna(0)

        df.drop(columns=self.cols, inplace=True)
        return df
