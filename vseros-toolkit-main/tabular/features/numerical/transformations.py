# src/features/transformations.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.stats import boxcox, yeojohnson

from ..base import FeatureGenerator
from ...utils import validate_type, validate_non_empty
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

# ==================================================================================
# LogTransformer
# ==================================================================================
class LogTransformer(FeatureGenerator):
    """
    Применяет натуральный логарифм `log(1 + x)` к заданным колонкам.

    Это одно из самых эффективных преобразований для данных с правым хвостом
    (right-skewed), таких как доходы, цены, количество транзакций.
    `log1p` используется для корректной обработки нулевых значений.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для преобразования.
    @validate_type(str, list)
    @validate_non_empty
    """
    def __init__(self, name: str, cols: List[str]):
        super().__init__(name)
        self.cols = cols
        self.output_col_names = [f"{col}_log1p" for col in self.cols]

    def fit(self, data: pd.DataFrame) -> None:
        """
        Это преобразование является stateless (не требует обучения),
        поэтому метод fit ничего не делает.
        """
        print(f"[{self.name}] Преобразование LogTransformer не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование log1p и создает новые колонки.
        """
        df = data.copy()
        print(f"[{self.name}] Применение log1p к колонкам: {self.cols}")
        for col in self.cols:
            df[f"{col}_log1p"] = np.log1p(df[col])
        return df

# ==================================================================================
# SqrtTransformer
# ==================================================================================
class SqrtTransformer(FeatureGenerator):
    """
    Применяет преобразование квадратного корня к заданным колонкам.

    Похоже на логарифмическое, но эффект менее выражен. Также полезно для
    данных с правым хвостом. Для избежания ошибок с отрицательными числами,
    значения сначала обрезаются по нулю.

    Параметры:
    @validate_type(str, list)
    @validate_non_empty
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для преобразования.
    """
    def __init__(self, name: str, cols: List[str]):
        super().__init__(name)
        self.cols = cols

    def fit(self, data: pd.DataFrame) -> None:
        """
        Это преобразование является stateless, метод fit ничего не делает.
        """
        print(f"[{self.name}] Преобразование SqrtTransformer не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование sqrt и создает новые колонки.
        """
        df = data.copy()
        print(f"[{self.name}] Применение sqrt к колонкам: {self.cols}")
        for col in self.cols:
            # Обрезаем по нулю, чтобы избежать NaN для отрицательных значений
            df[f"{col}_sqrt"] = np.sqrt(df[col].clip(0))
        return df

# ==================================================================================
# BoxCoxTransformer
# ==================================================================================
class BoxCoxTransformer(FeatureGenerator):
    """
    Применяет преобразование Бокса-Кокса.

    Это мощное степенное преобразование, которое автоматически подбирает
    оптимальный параметр `lambda`, чтобы сделать распределение данных
    максимально похожим на нормальное.

    ВАЖНО: Требует, чтобы все значения в колонке были строго положительными (> 0).
    @validate_type(str, list)
    @validate_non_empty

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для преобразования.
    """
    def __init__(self, name: str, cols: List[str]):
        super().__init__(name)
        self.cols = cols
        self.lambdas_: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет оптимальный параметр `lambda` для каждой колонки
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение BoxCoxTransformer на колонках: {self.cols}")
        for col in self.cols:
            if (data[col] <= 0).any():
                raise ValueError(
                    f"Колонка '{col}' содержит неположительные значения. "
                    "Преобразование Бокса-Кокса требует строго положительных данных."
                )
            # boxcox возвращает преобразованные данные и лямбду, нам нужна только лямбда
            _, lmbda = boxcox(data[col])
            self.lambdas_[col] = lmbda
            print(f"  - Для '{col}' найдена lambda = {lmbda:.4f}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование Бокса-Кокса с использованием сохраненных `lambda`.
        """
        df = data.copy()
        print(f"[{self.name}] Применение Box-Cox к колонкам: {self.cols}")
        for col in self.cols:
            lmbda = self.lambdas_[col]
            df[f"{col}_boxcox"] = boxcox(df[col], lmbda=lmbda)
        return df

# ==================================================================================
# YeoJohnsonTransformer
# ==================================================================================
class YeoJohnsonTransformer(FeatureGenerator):
    """
    Применяет преобразование Йео-Джонсона.

    Это более общая версия преобразования Бокса-Кокса, которая работает
    @validate_type(str, list)
    @validate_non_empty
    как с положительными, так и с отрицательными значениями. Является
    отличной и более безопасной альтернативой.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для преобразования.
    """
    def __init__(self, name: str, cols: List[str]):
        super().__init__(name)
        self.cols = cols
        self.lambdas_: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет оптимальный параметр `lambda` для каждой колонки
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение YeoJohnsonTransformer на колонках: {self.cols}")
        for col in self.cols:
            _, lmbda = yeojohnson(data[col])
            self.lambdas_[col] = lmbda
            print(f"  - Для '{col}' найдена lambda = {lmbda:.4f}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование Йео-Джонсона с использованием сохраненных `lambda`.
        """
        df = data.copy()
        print(f"[{self.name}] Применение Yeo-Johnson к колонкам: {self.cols}")
        for col in self.cols:
            lmbda = self.lambdas_[col]
            df[f"{col}_yeojohnson"] = yeojohnson(df[col], lmbda=lmbda)
        return df