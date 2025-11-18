# src/features/advanced/neighbors.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.neighbors import NearestNeighbors

from ..base import FeatureGenerator

# ==================================================================================
# NearestNeighborsFeatureGenerator
# ==================================================================================
class NearestNeighborsFeatureGenerator(FeatureGenerator):
    """
    Создает признаки на основе `k` ближайших соседей.

    Для каждого объекта находит `k` ближайших соседей в обучающей выборке
    и вычисляет статистики по их признакам или таргету.

    Параметры:
        name (str): Уникальное имя для шага.
        feature_cols (List[str]): Список числовых признаков для вычисления расстояний.
        target_cols (List[str]): Список колонок (включая таргет), по которым
            будут вычисляться статистики соседей.
        n_neighbors (int): Количество соседей для поиска (k).
        agg_funcs (List[str]): Список функций агрегации ('mean', 'std', 'median').
        **kwargs: Дополнительные параметры для sklearn.neighbors.NearestNeighbors
                  (например, metric='minkowski', p=2 для евклидова расстояния).
    """
    def __init__(self, name: str, feature_cols: List[str], target_cols: List[str],
                 n_neighbors: int = 5, agg_funcs: List[str] = ['mean', 'std'], **kwargs: Any):
        super().__init__(name)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.n_neighbors = n_neighbors
        self.agg_funcs = agg_funcs
        self.model_kwargs = kwargs
        
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, **self.model_kwargs)
        self.fit_features: pd.DataFrame = None
        self.fit_targets: pd.DataFrame = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает модель NearestNeighbors и сохраняет данные, на которых она
        была обучена, для последующего извлечения статистик.
        """
        print(f"[{self.name}] Обучение NearestNeighbors на {len(data)} объектах.")
        
        # Заполняем пропуски, так как NearestNeighbors не работает с ними
        self.fit_features = data[self.feature_cols].fillna(0)
        self.fit_targets = data[self.target_cols].fillna(0)
        
        self.model.fit(self.fit_features)

    def _calculate_neighbor_stats(self, distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
        """Вспомогательная функция для вычисления статистик по найденным соседям."""
        
        num_samples = indices.shape[0]
        results = {}
        
        # 1. Считаем статистики по расстояниям
        for func in self.agg_funcs:
            if func == 'mean': results[f'{self.name}_dist_mean'] = np.mean(distances, axis=1)
            if func == 'std': results[f'{self.name}_dist_std'] = np.std(distances, axis=1)
            if func == 'median': results[f'{self.name}_dist_median'] = np.median(distances, axis=1)

        # 2. Считаем статистики по таргет-колонкам соседей
        for t_col in self.target_cols:
            # `indices` имеет форму (n_samples, k). Нам нужно извлечь значения
            # таргет-колонки для каждого соседа.
            neighbor_targets = self.fit_targets[t_col].values[indices]
            
            for func in self.agg_funcs:
                col_name = f'{self.name}_{t_col}_neighbor_{func}'
                if func == 'mean': results[col_name] = np.mean(neighbor_targets, axis=1)
                if func == 'std': results[col_name] = np.std(neighbor_targets, axis=1)
                if func == 'median': results[col_name] = np.median(neighbor_targets, axis=1)
                
        return pd.DataFrame(results)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Находит соседей и вычисляет статистики для переданного датасета.
        """
        df = data.copy()
        print(f"[{self.name}] Поиск соседей и вычисление статистик для {len(df)} строк.")
        
        query_features = df[self.feature_cols].fillna(0)
        
        # Находим k+1 соседей, так как если мы ищем соседей для объекта из трейна,
        # он сам будет своим ближайшим соседом (с расстоянием 0).
        distances, indices = self.model.kneighbors(query_features, n_neighbors=self.n_neighbors + 1)
        
        # Исключаем сам объект из списка своих соседей
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Вычисляем статистики
        stats_df = self._calculate_neighbor_stats(distances, indices)
        stats_df.index = df.index
        
        return pd.concat([df, stats_df], axis=1)