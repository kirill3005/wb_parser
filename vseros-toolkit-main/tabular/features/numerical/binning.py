# src/features/numerical/binning.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.tree import DecisionTreeClassifier

from ..base import FeatureGenerator

# ==================================================================================
# EqualWidthBinner
# ==================================================================================
class EqualWidthBinner(FeatureGenerator):
    """
    Разбивает непрерывный признак на N корзин одинаковой ширины.

    Например, если признак варьируется от 0 до 100 и n_bins=10, то
    бины будут [0-10], [10-20], ..., [90-100].
    Этот метод прост, но очень чувствителен к выбросам.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для биннинга.
        n_bins (int): Количество создаваемых корзин (бинов).
    """
    def __init__(self, name: str, cols: List[str], n_bins: int = 10):
        super().__init__(name)
        self.cols = cols
        self.n_bins = n_bins
        self.bin_edges_: Dict[str, np.ndarray] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет границы бинов на основе минимального и
        максимального значений ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение EqualWidthBinner на колонках: {self.cols}")
        for col in self.cols:
            min_val, max_val = data[col].min(), data[col].max()
            # Создаем n_bins+1 границ
            self.bin_edges_[col] = np.linspace(min_val, max_val, self.n_bins + 1)
            print(f"  - Границы для '{col}': {np.round(self.bin_edges_[col], 2)}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет биннинг, используя сохраненные границы.
        Возвращает целочисленные метки для каждого бина.
        """
        df = data.copy()
        print(f"[{self.name}] Применение EqualWidthBinner к {len(df)} строкам.")
        for col in self.cols:
            bin_edges = self.bin_edges_[col]
            # labels=False возвращает целые числа, а не интервалы
            # include_lowest=True, чтобы включить минимальное значение
            df[f"{col}_ew_binned"] = pd.cut(
                df[col], bins=bin_edges, labels=False, include_lowest=True
            )
            # Заполняем NaN, которые могут возникнуть, если значения в тесте
            # выходят за пределы обученных границ
            df[f"{col}_ew_binned"].fillna(-1, inplace=True)
            df[f"{col}_ew_binned"] = df[f"{col}_ew_binned"].astype(int)
        return df

# ==================================================================================
# QuantileBinner
# ==================================================================================
class QuantileBinner(FeatureGenerator):
    """
    Разбивает непрерывный признак на N корзин с примерно одинаковым
    количеством наблюдений в каждой.

    Этот метод гораздо более устойчив к выбросам и скошенным распределениям.
    Является отличным выбором по умолчанию для биннинга.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для биннинга.
        n_bins (int): Количество создаваемых корзин (квантилей).
    """
    def __init__(self, name: str, cols: List[str], n_bins: int = 10):
        super().__init__(name)
        self.cols = cols
        self.n_bins = n_bins
        self.bin_edges_: Dict[str, np.ndarray] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет границы бинов на основе квантилей
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение QuantileBinner на колонках: {self.cols}")
        for col in self.cols:
            # retbins=True возвращает границы бинов
            # duplicates='drop' обрабатывает случай, когда квантили не уникальны
            _, self.bin_edges_[col] = pd.qcut(
                data[col], q=self.n_bins, retbins=True, duplicates='drop'
            )
            print(f"  - Границы для '{col}': {np.round(self.bin_edges_[col], 2)}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет биннинг, используя сохраненные границы квантилей.
        """
        df = data.copy()
        print(f"[{self.name}] Применение QuantileBinner к {len(df)} строкам.")
        for col in self.cols:
            bin_edges = self.bin_edges_[col]
            df[f"{col}_q_binned"] = pd.cut(
                df[col], bins=bin_edges, labels=False, include_lowest=True
            )
            df[f"{col}_q_binned"].fillna(-1, inplace=True)
            df[f"{col}_q_binned"] = df[f"{col}_q_binned"].astype(int)
        return df

# ==================================================================================
# DecisionTreeBinner
# ==================================================================================
class DecisionTreeBinner(FeatureGenerator):
    """
    Использует обученное дерево решений для нахождения "оптимальных"
    границ бинов. Это supervised-метод, который ищет точки разделения,
    наиболее эффективно разделяющие целевую переменную.

    Очень мощная техника для выявления нелинейных зависимостей.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для биннинга.
        target_col (str): Имя целевой переменной.
        max_depth (int): Максимальная глубина дерева. Контролирует количество бинов.
        **kwargs: Дополнительные параметры для sklearn.tree.DecisionTreeClassifier.
    """
    def __init__(self, name: str, cols: List[str], target_col: str, max_depth: int = 3, **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.target_col = target_col
        self.max_depth = max_depth
        self.tree_kwargs = kwargs
        self.bin_edges_: Dict[str, np.ndarray] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает дерево решений для каждой колонки и извлекает из него
        границы для биннинга.
        """
        print(f"[{self.name}] Обучение DecisionTreeBinner на колонках: {self.cols}")
        y = data[self.target_col]
        
        for col in self.cols:
            X = data[[col]]
            clf = DecisionTreeClassifier(max_depth=self.max_depth, **self.tree_kwargs)
            clf.fit(X, y)
            
            # Извлекаем пороги (точки разделения) из обученного дерева
            thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
            
            # Добавляем минимальное и максимальное значения, чтобы покрыть весь диапазон
            min_val, max_val = data[col].min(), data[col].max()
            bin_edges = np.unique(np.concatenate(([min_val, max_val], thresholds)))
            
            self.bin_edges_[col] = bin_edges
            print(f"  - Найденные границы для '{col}': {np.round(bin_edges, 2)}")
            
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет биннинг, используя границы, найденные деревом.
        """
        df = data.copy()
        print(f"[{self.name}] Применение DecisionTreeBinner к {len(df)} строкам.")
        for col in self.cols:
            bin_edges = self.bin_edges_[col]
            df[f"{col}_tree_binned"] = pd.cut(
                df[col], bins=bin_edges, labels=False, include_lowest=True
            )
            df[f"{col}_tree_binned"].fillna(-1, inplace=True)
            df[f"{col}_tree_binned"] = df[f"{col}_tree_binned"].astype(int)
        return df