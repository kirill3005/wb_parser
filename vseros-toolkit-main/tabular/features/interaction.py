# src/features/interaction.py

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from itertools import combinations
from dataclasses import dataclass, field  # noqa: F401 (some generators below use dataclass)

from .base import FeatureGenerator, FitStrategy

# ==================================================================================
# NumericalInteractionGenerator
# ==================================================================================
class NumericalInteractionGenerator(FeatureGenerator):
    """Creates interactions between pairs of numerical features.

    For each pair of columns from the `cols` list, applies specified
    mathematical operations to create new interaction features.

    Args:
        name (str): Unique name for this step.
        cols (List[str]): List of numerical columns to create interactions for.
        operations (List[str]): List of operations to apply.
            Available operations: 'add', 'subtract', 'multiply', 'divide'.

    Attributes:
        cols (List[str]): Columns to create interactions between.
        operations (List[str]): Mathematical operations to perform.
        epsilon (float): Small value added to denominators for safe division.
    """
    def __init__(
        self,
        name: str,
        cols: List[str],
        operations: Optional[List[str]] = None,
        epsilon: float = 1e-6,
    ):
        super().__init__(name)
        self.cols = cols
        self.operations = operations or ['add', 'subtract', 'multiply', 'divide']
        self.epsilon = epsilon
        self.fit_strategy = "train_only"
        self._validate_columns()

    def _validate_columns(self) -> None:
        if len(self.cols) < 2:
            raise ValueError("At least 2 columns are required to create interactions.")

    def fit(self, data: pd.DataFrame) -> None:
        """This is a stateless transformation, no training required.

        Args:
            data (pd.DataFrame): Input data to validate column existence.

        Raises:
            ValueError: If any specified column is not found in the data.
        """
        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        logging.info(f"[{self.name}] NumericalInteractionGenerator requires no training.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create new interaction features between numerical columns.

        Args:
            data (pd.DataFrame): Input data containing the numerical columns.

        Returns:
            pd.DataFrame: Data with additional interaction features.

        Raises:
            ValueError: If any specified column is not found in the data.
        """
        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        df = data.copy()
        logging.info(f"[{self.name}] Creating numerical interactions for columns: {self.cols}")

        # Generate all unique column pairs
        for c1, c2 in combinations(self.cols, 2):
            if 'add' in self.operations:
                df[f"{c1}_add_{c2}"] = df[c1] + df[c2]
            if 'subtract' in self.operations:
                df[f"{c1}_sub_{c2}"] = df[c1] - df[c2]
                df[f"{c2}_sub_{c1}"] = df[c2] - df[c1]  # Subtraction is not symmetric
            if 'multiply' in self.operations:
                df[f"{c1}_mul_{c2}"] = df[c1] * df[c2]
            if 'divide' in self.operations:
                try:
                    df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + self.epsilon)
                    df[f"{c2}_div_{c1}"] = df[c2] / (df[c1] + self.epsilon)  # Division is not symmetric
                except Exception as e:
                    logging.error(f"Error in division operations for {c1} and {c2}: {e}")
                    raise

        return df

# ==================================================================================
# CategoricalInteractionGenerator
# ==================================================================================
class CategoricalInteractionGenerator(FeatureGenerator):
    """
    Создает взаимодействия между категориальными признаками путем их конкатенации.

    Например, `city='Moscow'` и `device='Desktop'` превратятся в `city_device='Moscow_Desktop'`.
    Это позволяет модели улавливать зависимости, специфичные для комбинации категорий.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[List[str]]): Список списков колонок. Взаимодействия будут
            созданы для каждой группы колонок во внутреннем списке.
            Пример: [['city', 'device'], ['product_brand', 'country']]
    """
    def __init__(self, name: str, cols_groups: List[List[str]]):
        super().__init__(name)
        if not cols_groups:
            raise ValueError("cols_groups must contain at least one group definition.")
        self.cols_groups = cols_groups
        self.fit_strategy = "train_only"

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        for group in self.cols_groups:
            for col in group:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
        logging.info(f"[{self.name}] CategoricalInteractionGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые конкатенированные категориальные признаки."""
        for group in self.cols_groups:
            for col in group:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
        df = data.copy()
        logging.info(f"[{self.name}] Создание категориальных взаимодействий.")

        for group in self.cols_groups:
            new_col_name = "_".join(group)
            # Убедимся, что все колонки строкового типа перед конкатенацией
            try:
                df[new_col_name] = df[group].astype(str).agg('_'.join, axis=1)
                logging.info(f"  - Создана колонка: {new_col_name}")
            except Exception as e:
                logging.error(f"Error creating column {new_col_name}: {e}")
                raise

        return df
        
# ==================================================================================
# NumCatInteractionGenerator
# ==================================================================================
class NumCatInteractionGenerator(FeatureGenerator):
    """
    Создает взаимодействия между числовыми и категориальными признаками.

    Вычисляет отклонение числового признака от среднего значения этого признака
    внутри его категории. Например, `income_deviation_from_city_mean`.
    Это очень мощный признак, который показывает, насколько значение является
    "типичным" для своей группе.

    Параметры:
        name (str): Уникальное имя для шага.
        interactions (Dict[str, List[str]]): Словарь, где ключ - это
            категориальный признак (группа), а значение - список числовых
            признаков, для которых нужно посчитать отклонение.
    """
    def __init__(
        self,
        name: str,
        interactions: Dict[str, List[str]],
        epsilon: float = 1e-6,
    ):
        super().__init__(name)
        if not interactions:
            raise ValueError("interactions must contain at least one mapping.")
        self.interactions = interactions
        self.epsilon = epsilon
        self.group_means_: Dict[str, pd.Series] = {}
        self.overall_means_: Dict[str, float] = {}
        self.fit_strategy = "train_only"

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет средние значения для каждой категории
        ТОЛЬКО на обучающих данных.
        """
        for cat_col, num_cols in self.interactions.items():
            if cat_col not in data.columns:
                raise ValueError(f"Column '{cat_col}' not found in data")
            for num_col in num_cols:
                if num_col not in data.columns:
                    raise ValueError(f"Column '{num_col}' not found in data")
        logging.info(f"[{self.name}] Обучение NumCatInteractionGenerator.")
        for cat_col, num_cols in self.interactions.items():
            for num_col in num_cols:
                try:
                    group_means = data.groupby(cat_col)[num_col].mean()
                    self.group_means_[f"{num_col}_in_{cat_col}"] = group_means
                    logging.info(f"  - Вычислены средние для '{num_col}' по группам '{cat_col}'.")
                except Exception as e:
                    logging.error(f"Error computing group means for {num_col} in {cat_col}: {e}")
                    raise
        # Precompute overall means to avoid data leakage
        for num_cols in self.interactions.values():
            for num_col in num_cols:
                if num_col not in self.overall_means_:
                    self.overall_means_[num_col] = data[num_col].mean()
                
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет и добавляет признаки отклонения от среднего по группе.
        """
        for cat_col, num_cols in self.interactions.items():
            if cat_col not in data.columns:
                raise ValueError(f"Column '{cat_col}' not found in data")
            for num_col in num_cols:
                if num_col not in data.columns:
                    raise ValueError(f"Column '{num_col}' not found in data")
        df = data
        logging.info(f"[{self.name}] Применение NumCatInteractionGenerator к {len(df)} строкам.")
        for cat_col, num_cols in self.interactions.items():
            for num_col in num_cols:
                # 1. Получаем словарь средних значений по группе для эффективного поиска
                group_means = self.group_means_[f"{num_col}_in_{cat_col}"]
                try:
                    # Используем map для эффективного поиска вместо merge, избегая создания большого DataFrame
                    group_mean_series = df[cat_col].map(group_means).fillna(self.overall_means_[num_col])
                except Exception as e:
                    logging.error(f"Error mapping group means for {num_col} in {cat_col}: {e}")
                    raise

                # 2. Вычисляем и создаем новые признаки
                try:
                    df[f"{num_col}_div_by_{cat_col}_mean"] = df[num_col] / (group_mean_series + self.epsilon)
                    df[f"{num_col}_sub_by_{cat_col}_mean"] = df[num_col] - group_mean_series
                except Exception as e:
                    logging.error(f"Error computing interaction features for {num_col} in {cat_col}: {e}")
                    raise

        return df
