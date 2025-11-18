# src/features/categorical/combination.py

import pandas as pd
from typing import List, Dict, Any

from ..base import FeatureGenerator

# ==================================================================================
# RareCategoryCombiner
# ==================================================================================
class RareCategoryCombiner(FeatureGenerator):
    """
    Объединяет редкие категории в одну общую категорию "Other".

    Эта техника помогает уменьшить шум, сократить количество уникальных
    категорий (кардинальность) и сделать модели более устойчивыми.
    Порог "редкости" можно задать двумя способами (приоритет у `min_count`):

    1.  `min_count`: Все категории, которые встречаются в трейне меньше,
        чем `min_count` раз, будут объединены.
    2.  `min_freq`: Все категории, чья доля в трейне меньше, чем `min_freq`,
        будут объединены.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для обработки.
        min_count (int, optional): Минимальное количество появлений категории,
            чтобы она не считалась редкой.
        min_freq (float, optional): Минимальная частота (доля) категории,
            чтобы она не считалась редкой.
        other_value (str): Имя для новой объединенной категории.
    """
    def __init__(self, 
                 name: str, 
                 cols: List[str], 
                 min_count: int = None, 
                 min_freq: float = None, 
                 other_value: str = "Other"):
        super().__init__(name)
        if min_count is None and min_freq is None:
            raise ValueError("Необходимо указать хотя бы один из параметров: `min_count` или `min_freq`.")
        
        self.cols = cols
        self.min_count = min_count
        self.min_freq = min_freq
        self.other_value = other_value
        self.frequent_categories_: Dict[str, List[str]] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Определяет и сохраняет список "частых" категорий для каждой колонки
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение RareCategoryCombiner на колонках: {self.cols}")
        for col in self.cols:
            counts = data[col].value_counts()
            
            if self.min_count:
                # Определяем частые категории по абсолютному количеству
                frequent_cats = counts[counts >= self.min_count].index.tolist()
            else: # self.min_freq
                # Определяем частые категории по доле
                freqs = data[col].value_counts(normalize=True)
                frequent_cats = freqs[freqs >= self.min_freq].index.tolist()
            
            self.frequent_categories_[col] = frequent_cats
            print(f"  - Для '{col}': {len(frequent_cats)} из {len(counts)} категорий оставлены, остальные будут '{self.other_value}'.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Заменяет редкие категории на значение `other_value`.
        Исходные колонки изменяются на месте (in-place).
        """
        df = data.copy()
        print(f"[{self.name}] Применение RareCategoryCombiner к {len(df)} строкам.")
        for col in self.cols:
            frequent_cats = self.frequent_categories_[col]
            # Заменяем все категории, которых нет в списке частых, на 'Other'
            df[col] = df[col].where(df[col].isin(frequent_cats), self.other_value)
            
        return df