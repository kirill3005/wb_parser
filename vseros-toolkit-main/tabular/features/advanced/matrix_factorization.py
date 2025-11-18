# src/features/advanced/matrix_factorization.py

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

# Импортируем наш базовый класс
from ..base import FeatureGenerator, FitStrategy
from ...utils import validate_type, validate_non_empty


class MatrixFactorizationFeatureGenerator(FeatureGenerator):
    """
    Создает признаки-эмбеддинги для пользователей и товаров с помощью
    матричной факторизации (алгоритм ALS) из библиотеки `implicit`.

    Этот генератор обучается на всех доступных взаимодействиях (train+test)
    для построения наиболее полных представлений. Затем для каждой пары
    (user, item) он добавляет:
    1. Вектор-эмбеддинг пользователя (user_embed_0, ...).
    2. Вектор-эмбеддинг товара (item_embed_0, ...).
    3. Признаки их взаимодействия (скалярное произведение, косинусное сходство).
    """
    # Обучаемся на всех данных для получения наиболее полных эмбеддингов.
    # Это не является утечкой, так как целевая переменная не используется.
    fit_strategy: FitStrategy = "combined"
    @validate_type(str, str, int)
    @validate_non_empty()

    def __init__(self, name: str, user_col: str, item_col: str, factors: int = 32, **model_kwargs: Any):
        """
        Конструктор генератора.

        Параметры:
            name (str): Уникальное имя для шага.
            user_col (str): Название колонки с ID пользователя.
            item_col (str): Название колонки с ID товара.
            factors (int): Размерность эмбеддингов (ключевой гиперпараметр).
            **model_kwargs: Дополнительные параметры, передаваемые напрямую в
                            `implicit.als.AlternatingLeastSquares`
                            (например, regularization, iterations, use_gpu).
        """
        super().__init__(name)
        self.user_col = user_col
        self.item_col = item_col
        self.factors = factors
        self.model_kwargs = model_kwargs
        self.epsilon = 1e-6  # для безопасного деления

        # --- Атрибуты, которые будут вычислены в .fit() ---
        self.model_: implicit.als.AlternatingLeastSquares = None
        # Словари для преобразования raw ID -> internal integer ID
        self.user_map_: Dict[Any, int] = {}
        self.item_map_: Dict[Any, int] = {}
        # Матрицы эмбеддингов
        self.user_factors_: np.ndarray = None
        self.item_factors_: np.ndarray = None
        # Вектор для "холодных" сущностей
        self.zero_vector_: np.ndarray = np.zeros(self.factors)

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает ALS модель на всех предоставленных данных.
        """
        print(f"[{self.name}] Обучение MatrixFactorizationFeatureGenerator...")

        # --- 1. Создание внутренних маппингов ID -> integer ---
        # pd.Categorical - эффективный способ создать маппинги
        user_cat = data[self.user_col].astype("category")
        item_cat = data[self.item_col].astype("category")
        
        self.user_map_ = {cat: code for code, cat in enumerate(user_cat.cat.categories)}
        self.item_map_ = {cat: code for code, cat in enumerate(item_cat.cat.categories)}

        print(f"  - Найдено {len(self.user_map_)} уникальных пользователей и {len(self.item_map_)} товаров.")

        # --- 2. Создание разреженной user-item матрицы ---
        # Получаем целочисленные коды для каждой строки
        user_codes = user_cat.cat.codes
        item_codes = item_cat.cat.codes
        
        # Данные о взаимодействиях (просто 1 для каждого клика)
        interaction_data = np.ones(len(data), dtype=np.float32)

        user_item_matrix = csr_matrix(
            (interaction_data, (user_codes, item_codes)),
            shape=(len(self.user_map_), len(self.item_map_))
        )
        
        # --- 3. Обучение ALS модели ---
        self.model_ = implicit.als.AlternatingLeastSquares(factors=self.factors, **self.model_kwargs)
        
        print(f"  - Обучение ALS модели ({self.model_kwargs.get('iterations', 15)} итераций)...")
        self.model_.fit(user_item_matrix, show_progress=True)
        
        # --- 4. Сохранение матриц эмбеддингов ---
        self.user_factors_ = self.model_.user_factors
        self.item_factors_ = self.model_.item_factors
        print(f"  - Обучение завершено. Размер эмбеддингов: user_factors {self.user_factors_.shape}, item_factors {self.item_factors_.shape}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обогащает DataFrame эмбеддингами и признаками их взаимодействия.
        """
        df = data.copy()
        print(f"[{self.name}] Применение эмбеддингов к {len(df)} строкам.")

        # --- 1. Получение целочисленных индексов ---
        # .map вернет NaN для ID, которых не было в fit. Заполняем -1 как маркер "холодного старта".
        user_indices = df[self.user_col].map(self.user_map_).fillna(-1).astype(int)
        item_indices = df[self.item_col].map(self.item_map_).fillna(-1).astype(int)

        # --- 2. Извлечение векторов-эмбеддингов ---
        user_vectors = self.user_factors_[user_indices]
        item_vectors = self.item_factors_[item_indices]
        
        # Заменяем векторы для "холодных" сущностей на нулевые
        user_vectors[user_indices == -1] = self.zero_vector_
        item_vectors[item_indices == -1] = self.zero_vector_

        # --- 3. Создание колонок с эмбеддингами ---
        user_embed_cols = [f"{self.name}_user_{i}" for i in range(self.factors)]
        item_embed_cols = [f"{self.name}_item_{i}" for i in range(self.factors)]

        df[user_embed_cols] = user_vectors
        df[item_embed_cols] = item_vectors

        # --- 4. Создание признаков взаимодействия ---
        # Скалярное произведение - основной сигнал "совместимости"
        dot_products = np.sum(user_vectors * item_vectors, axis=1)
        df[f"{self.name}_dot_product"] = dot_products
        
        # Косинусное сходство
        user_norms = np.linalg.norm(user_vectors, axis=1)
        item_norms = np.linalg.norm(item_vectors, axis=1)
        
        cosine_similarity = dot_products / (user_norms * item_norms + self.epsilon)
        df[f"{self.name}_cosine_similarity"] = cosine_similarity

        print(f"  - Добавлено {len(user_embed_cols) + len(item_embed_cols) + 2} новых признаков.")
        return df
    def __enter__(self):
        """Enter the context manager.

        Returns:
            MatrixFactorizationFeatureGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup matrix factorization resources
        if hasattr(self, 'model_') and self.model_ is not None:
            # Clear the model to free memory
            self.model_ = None
        if hasattr(self, 'user_factors_') and self.user_factors_ is not None:
            self.user_factors_ = None
        if hasattr(self, 'item_factors_') and self.item_factors_ is not None:
            self.item_factors_ = None
        if hasattr(self, 'user_map_'):
            self.user_map_.clear()
        if hasattr(self, 'item_map_'):
            self.item_map_.clear()