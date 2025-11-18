# src/features/advanced/model_based.py

import pandas as pd
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from ..base import FeatureGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

# ==================================================================================
# KMeansFeatureGenerator
# ==================================================================================
class KMeansFeatureGenerator(FeatureGenerator):
    """
    Создает новый категориальный признак `cluster_id` с помощью кластеризации K-Means.

    Группирует похожие объекты в кластеры на основе заданных признаков.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список числовых колонок для кластеризации.
        n_clusters (int): Количество кластеров (гиперпараметр K).
        **kwargs: Дополнительные параметры для sklearn.cluster.KMeans.
    """
    def __init__(self, name: str, cols: List[str], n_clusters: int = 8, **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, **kwargs)

    def fit(self, data: pd.DataFrame) -> None:
        """Обучает модель K-Means, находя центроиды кластеров."""
        print(f"[{self.name}] Обучение KMeans на {len(self.cols)} признаках.")
        # K-Means чувствителен к пропускам, используем простую стратегию заполнения
        self.model.fit(data[self.cols].fillna(0))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Присваивает каждому объекту номер ближайшего кластера."""
        df = data.copy()
        print(f"[{self.name}] Применение KMeans для {len(df)} строк.")
        cluster_ids = self.model.predict(df[self.cols].fillna(0))
        df[f"{self.name}_cluster_id"] = cluster_ids
        return df

# ==================================================================================
# PCAGenerator / TruncatedSVDGenerator
# ==================================================================================
class PCAGenerator(FeatureGenerator):
    """
    Понижает размерность данных с помощью метода главных компонент (PCA).

    Создает новые, некоррелированные признаки (компоненты), которые
    объясняют максимальное количество дисперсии в исходных данных.
    Используется для плотных (dense) данных.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список числовых колонок для преобразования.
        n_components (int): Количество компонент на выходе.
        **kwargs: Дополнительные параметры для sklearn.decomposition.PCA.
    """
    def __init__(self, name: str, cols: List[str], n_components: int = 2, **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.n_components = n_components
        self.model = PCA(n_components=self.n_components, random_state=42, **kwargs)

    def fit(self, data: pd.DataFrame) -> None:
        """Обучает PCA, находя главные компоненты."""
        print(f"[{self.name}] Обучение PCA на {len(self.cols)} признаках.")
        self.model.fit(data[self.cols].fillna(0))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Проецирует данные на найденные главные компоненты."""
        df = data.copy()
        print(f"[{self.name}] Применение PCA для {len(df)} строк.")
        
        components = self.model.transform(df[self.cols].fillna(0))
        
        col_names = [f"{self.name}_pca_{i}" for i in range(self.n_components)]
        comp_df = pd.DataFrame(components, columns=col_names, index=df.index)
        
        return pd.concat([df, comp_df], axis=1)

# ==================================================================================
# TreeLeafFeatureGenerator
# ==================================================================================
class TreeLeafFeatureGenerator(FeatureGenerator):
    """
    Использует обученную древовидную модель (RandomForest, LightGBM и т.д.)
    для создания категориальных признаков на основе индексов листьев.

    Параметры:
        name (str): Уникальное имя для шага.
        feature_cols (List[str]): Список признаков для обучения модели.
        target_col (str): Имя целевой переменной.
        model_config (DictConfig): Конфигурация Hydra для инстанцирования
                                    древовидной модели.
    """
    # Этот генератор - supervised, поэтому обучаем его только на трейне
    fit_strategy = "train_only"
    
    def __init__(self, name: str, feature_cols: List[str], target_col: str, model_config: "DictConfig"):
        super().__init__(name)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model_config = model_config
        
        # Инстанциируем модель из переданного конфига
        print(f"[{self.name}] Инициализация модели для генерации листьев...")
        import hydra
        self.model = hydra.utils.instantiate(self.model_config)

    def fit(self, data: pd.DataFrame) -> None:
        """Обучает древовидную модель на признаках и таргете."""
        model_name = self.model.__class__.__name__
        print(f"[{self.name}] Обучение {model_name} для извлечения листьев.")
        self.model.fit(data[self.feature_cols].fillna(0), data[self.target_col])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлекает индексы листьев для каждого объекта.
        """
        df = data.copy()
        model_name = self.model.__class__.__name__
        print(f"[{self.name}] Извлечение индексов листьев с помощью {model_name} для {len(df)} строк.")
        
        # RandomForest и XGBoost имеют метод .apply()
        if hasattr(self.model, 'apply'):
            leaf_indices = self.model.apply(df[self.feature_cols].fillna(0))
        # LightGBM имеет метод .predict(..., pred_leaf=True)
        elif hasattr(self.model, 'predict') and 'pred_leaf' in self.model.predict.__code__.co_varnames:
            leaf_indices = self.model.predict(df[self.feature_cols].fillna(0), pred_leaf=True)
        else:
            raise TypeError(f"Модель {model_name} не поддерживает извлечение индексов листьев "
                            "(.apply или .predict(pred_leaf=True)).")
        
        # n_estimators может называться по-разному
        n_estimators = getattr(self.model, 'n_estimators', 
                               getattr(self.model, 'n_estimators_', # для некоторых sklearn моделей
                                       getattr(self.model, 'best_iteration_', -1))) # для LGBM с early stopping

        if n_estimators == -1 or leaf_indices.ndim != 2:
             # Для одиночных деревьев или неизвестных случаев
             n_estimators = 1
             leaf_indices = leaf_indices.reshape(-1, 1)

        col_names = [f"{self.name}_tree_{i}_leaf" for i in range(n_estimators)]
        leaf_df = pd.DataFrame(leaf_indices, columns=col_names, index=df.index)
        
        return pd.concat([df, leaf_df], axis=1)
    def __enter__(self):
        """Enter the context manager.

        Returns:
            KMeansFeatureGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup clustering model resources
        if hasattr(self, 'model') and self.model is not None:
            self.model = None
    def __enter__(self):
        """Enter the context manager.

        Returns:
            PCAGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup PCA model resources
        if hasattr(self, 'model') and self.model is not None:
            self.model = None
    def __enter__(self):
        """Enter the context manager.

        Returns:
            TreeLeafFeatureGenerator: The generator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup tree model resources
        if hasattr(self, 'model') and self.model is not None:
            self.model = None