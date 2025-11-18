# src/features/scaling.py

import pandas as pd
from typing import List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..base import FeatureGenerator
from typing import TYPE_CHECKING
from ...utils import validate_type, validate_non_empty

if TYPE_CHECKING:
    from omegaconf import DictConfig

# ==================================================================================
# StandardScalerGenerator
# ==================================================================================
class StandardScalerGenerator(FeatureGenerator):
    """
    Применяет стандартизацию (StandardScaler) к заданным числовым колонкам.

    Этот скейлер вычитает среднее значение и делит на стандартное отклонение.
    В результате данные имеют среднее 0 и дисперсию 1.
    Хорошо подходит для моделей, которые предполагают нормальное распределение
    признаков, таких как линейные модели и SVM.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для масштабирования.
        **kwargs: Дополнительные аргументы, передаваемые в sklearn.preprocessing.StandardScaler.
    @validate_type(str, list)
    @validate_non_empty
    """
    def __init__(self, name: str, cols: List[str], **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        # Инстанциируем скейлер из scikit-learn с любыми дополнительными параметрами
        self.scaler = StandardScaler(**kwargs)
        self.output_col_names = [f"{col}_scaled" for col in self.cols]

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает скейлер, вычисляя среднее и стандартное отклонение
        по указанным колонкам ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение StandardScaler на колонках: {self.cols}")
        self.scaler.fit(data[self.cols])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обученное преобразование к данным.
        Создает новые колонки с отмасштабированными значениями.
        """
        df = data.copy()
        print(f"[{self.name}] Применение StandardScaler к {len(df)} строкам.")
        
        # .transform возвращает numpy array, поэтому мы создаем новый DataFrame
        # с правильными индексами и именами колонок.
        scaled_data = self.scaler.transform(df[self.cols])
        df[self.output_col_names] = scaled_data
        
        return df

# ==================================================================================
# MinMaxScalerGenerator
# ==================================================================================
class MinMaxScalerGenerator(FeatureGenerator):
    """
    Применяет нормализацию (MinMaxScaler) к заданным числовым колонкам.

    Этот скейлер масштабирует данные в заданный диапазон, по умолчанию [0, 1].
    Полезен для нейронных сетей и алгоритмов, чувствительных к абсолютным
    значениям признаков.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для масштабирования.
    @validate_type(str, list)
    @validate_non_empty
        **kwargs: Дополнительные аргументы, передаваемые в sklearn.preprocessing.MinMaxScaler
                  (например, feature_range=(0, 1)).
    """
    def __init__(self, name: str, cols: List[str], **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.scaler = MinMaxScaler(**kwargs)
        self.output_col_names = [f"{col}_minmax_scaled" for col in self.cols]

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает скейлер, вычисляя минимум и максимум по указанным колонкам
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение MinMaxScaler на колонках: {self.cols}")
        self.scaler.fit(data[self.cols])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обученное преобразование к данным.
        Создает новые колонки с отмасштабированными значениями.
        """
        df = data.copy()
        print(f"[{self.name}] Применение MinMaxScaler к {len(df)} строкам.")
        
        scaled_data = self.scaler.transform(df[self.cols])
        df[self.output_col_names] = scaled_data
        
        return df

# ==================================================================================
# RobustScalerGenerator
# ==================================================================================
class RobustScalerGenerator(FeatureGenerator):
    """
    Применяет устойчивое масштабирование (RobustScaler) к заданным числовым колонкам.

    Этот скейлер использует статистику, устойчивую к выбросам (медиану и
    межквартильный размах). Он является хорошей альтернативой StandardScaler,
    если в ваших данных присутствуют значительные выбросы.

    Параметры:
    @validate_type(str, list)
    @validate_non_empty
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для масштабирования.
        **kwargs: Дополнительные аргументы, передаваемые в sklearn.preprocessing.RobustScaler
                  (например, quantile_range=(25.0, 75.0)).
    """
    def __init__(self, name: str, cols: List[str], **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.scaler = RobustScaler(**kwargs)
        self.output_col_names = [f"{col}_robust_scaled" for col in self.cols]

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает скейлер, вычисляя медиану и квантили по указанным колонкам
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение RobustScaler на колонках: {self.cols}")
        self.scaler.fit(data[self.cols])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обученное преобразование к данным.
        Создает новые колонки с отмасштабированными значениями.
        """
        df = data.copy()
        print(f"[{self.name}] Применение RobustScaler к {len(df)} строкам.")
        
        scaled_data = self.scaler.transform(df[self.cols])
        df[self.output_col_names] = scaled_data
        
        return df
    def __enter__(self):
        """Enter the context manager.

        Returns:
            StandardScalerGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup scaler resources
        if hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler = None
    def __enter__(self):
        """Enter the context manager.

        Returns:
            MinMaxScalerGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup scaler resources
        if hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler = None
    def __enter__(self):
        """Enter the context manager.

        Returns:
            RobustScalerGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup scaler resources
        if hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler = None
