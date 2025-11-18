# src/features/categorical/ordinal.py

import pandas as pd
from typing import List, Dict, Any

from ..base import FeatureGenerator

# ==================================================================================
# OrdinalEncoderGenerator
# ==================================================================================
class OrdinalEncoderGenerator(FeatureGenerator):
    """
    Применяет порядковое кодирование к заданным категориальным колонкам.

    Этот генератор работает в двух режимах:
    1.  **Пользовательский (Custom Mapping):** Если предоставлен словарь `mapping`,
        категории кодируются в соответствии с ним. Это истинное порядковое
        кодирование, которое вносит в модель доменные знания.
        Пример: {"low": 0, "medium": 1, "high": 2}
    
    2.  **Автоматический (Label Encoding):** Если `mapping` не предоставлен,
        каждой уникальной категории присваивается целое число (0, 1, 2, ...).
        Этот режим подходит ТОЛЬКО для древовидных моделей, так как он
        создает искусственный порядок.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для кодирования.
        mapping (Dict[str, Dict[str, int]], optional): Словарь, где ключи - это
            названия колонок, а значения - словари для кодирования.
        unknown_value (int): Значение для категорий, не встреченных при обучении.
    """
    def __init__(self, 
                 name: str, 
                 cols: List[str], 
                 mapping: Dict[str, Dict[str, int]] = None, 
                 unknown_value: int = -1):
        super().__init__(name)
        self.cols = cols
        self.custom_mapping = mapping
        self.unknown_value = unknown_value
        self.mappings_: Dict[str, Dict[str, int]] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Создает или проверяет словари для кодирования ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение OrdinalEncoder на колонках: {self.cols}")
        for col in self.cols:
            if self.custom_mapping and col in self.custom_mapping:
                # Используем предоставленный пользователем словарь
                self.mappings_[col] = self.custom_mapping[col]
                print(f"  - Для '{col}' используется пользовательский словарь.")
            else:
                # Создаем словарь автоматически (Label Encoding)
                unique_cats = data[col].unique()
                self.mappings_[col] = {cat: i for i, cat in enumerate(unique_cats)}
                print(f"  - Для '{col}' создан автоматический словарь ({len(unique_cats)} категорий).")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет кодирование, используя сохраненные словари.
        """
        df = data.copy()
        print(f"[{self.name}] Применение OrdinalEncoder к {len(df)} строкам.")
        for col in self.cols:
            mapping_dict = self.mappings_[col]
            # Применяем словарь
            mapped_values = df[col].map(mapping_dict)
            
            # Заполняем пропуски и необученные категории значением unknown_value
            df[f"{col}_ordinal"] = mapped_values.fillna(self.unknown_value).astype(int)
        
        # Удаляем исходные колонки
        df.drop(columns=self.cols, inplace=True)
        return df