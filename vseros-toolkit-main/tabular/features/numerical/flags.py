# src/features/numerical/flags.py

import pandas as pd
from typing import List, Dict, Any

from ..base import FeatureGenerator

# ==================================================================================
# ValueIndicator
# ==================================================================================
class ValueIndicator(FeatureGenerator):
    """
    Создает бинарный флаг (0/1), указывающий, равно ли значение в колонке
    заданной константе.

    Чаще всего используется для индикации нулей, но может применяться для
    любых "магических чисел" (например, -1, 999), которые могут нести
    особый смысл.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для проверки.
        value (Any): Значение, с которым производится сравнение.
    """
    def __init__(self, name: str, cols: List[str], value: Any = 0):
        super().__init__(name)
        self.cols = cols
        self.value = value
        self.output_col_names = [f"{col}_is_{str(value).replace('-', 'neg')}" for col in self.cols]

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        print(f"[{self.name}] ValueIndicator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые колонки с бинарными флагами."""
        df = data.copy()
        print(f"[{self.name}] Создание флагов для значения '{self.value}' в колонках: {self.cols}")
        for i, col in enumerate(self.cols):
            df[self.output_col_names[i]] = (df[col] == self.value).astype(int)
        return df

# ==================================================================================
# IsNullIndicator
# ==================================================================================
class IsNullIndicator(FeatureGenerator):
    """
    Создает бинарный флаг (0/1), указывающий, является ли значение в колонке
    пропущенным (NaN/Null).

    Это полезно, так как сам факт отсутствия данных может быть сильным
    предсказательным сигналом.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для проверки на пропуски.
    """
    def __init__(self, name: str, cols: List[str]):
        super().__init__(name)
        self.cols = cols
        self.output_col_names = [f"{col}_is_null" for col in self.cols]

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        print(f"[{self.name}] IsNullIndicator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые колонки с флагами пропущенных значений."""
        df = data.copy()
        print(f"[{self.name}] Создание флагов пропусков для колонок: {self.cols}")
        for i, col in enumerate(self.cols):
            df[self.output_col_names[i]] = df[col].isnull().astype(int)
        return df

# ==================================================================================
# OutlierIndicator
# ==================================================================================
class OutlierIndicator(FeatureGenerator):
    """
    Создает бинарный флаг (0/1), указывающий, является ли значение выбросом.

    Выбросы определяются с помощью робастного метода межквартильного размаха (IQR).
    Значение считается выбросом, если оно находится за пределами диапазона:
    [Q1 - multiplier * IQR, Q3 + multiplier * IQR].

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для поиска выбросов.
        multiplier (float): Множитель для IQR. Стандартное значение 1.5.
                          Для поиска более экстремальных выбросов можно использовать 3.0.
    """
    def __init__(self, name: str, cols: List[str], multiplier: float = 1.5):
        super().__init__(name)
        self.cols = cols
        self.multiplier = multiplier
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет верхнюю и нижнюю границы для определения
        выбросов ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение OutlierIndicator на колонках: {self.cols}")
        for col in self.cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.multiplier * iqr
            self.upper_bounds_[col] = q3 + self.multiplier * iqr
            print(f"  - Границы для '{col}': [{self.lower_bounds_[col]:.2f}, {self.upper_bounds_[col]:.2f}]")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые колонки с флагами выбросов."""
        df = data.copy()
        print(f"[{self.name}] Создание флагов выбросов для колонок: {self.cols}")
        for col in self.cols:
            lower = self.lower_bounds_[col]
            upper = self.upper_bounds_[col]
            is_outlier = (df[col] < lower) | (df[col] > upper)
            df[f"{col}_is_outlier"] = is_outlier.astype(int)
        return df