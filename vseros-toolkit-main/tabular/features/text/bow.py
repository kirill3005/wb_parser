# src/features/text/bow.py

import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse

from ..base import FeatureGenerator

# ==================================================================================
# CountVectorizerGenerator
# ==================================================================================
class CountVectorizerGenerator(FeatureGenerator):
    """
    Преобразует текстовые данные в матрицу с количеством токенов (слов).

    Использует `sklearn.feature_extraction.text.CountVectorizer`.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список текстовых колонок для векторизации.
        prefix (str): Префикс для новых колонок.
        **kwargs: Дополнительные аргументы, передаваемые в CountVectorizer.
            Ключевые параметры:
            - ngram_range (tuple): (min_n, max_n), например (1, 2) для уни- и биграмм.
            - max_features (int): Ограничивает словарь N самыми частыми словами.
            - min_df (int/float): Игнорировать слова с частотой ниже порога.
            - max_df (int/float): Игнорировать слова с частотой выше порога (стоп-слова).
            - stop_words (str/list): Список стоп-слов.
    """
    def __init__(self, name: str, cols: List[str], prefix: str = 'countvec', **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.prefix = prefix
        # Мы будем создавать отдельный векторизатор для каждой колонки
        self.vectorizers: Dict[str, CountVectorizer] = {
            col: CountVectorizer(**kwargs) for col in self.cols
        }

    def fit(self, data: pd.DataFrame) -> None:
        """Обучает векторизаторы, создавая словарь для каждой колонки."""
        print(f"[{self.name}] Обучение CountVectorizer на колонках: {self.cols}")
        for col in self.cols:
            # Заполняем NaN пустыми строками, чтобы избежать ошибок
            self.vectorizers[col].fit(data[col].fillna(''))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет векторизацию и возвращает DataFrame с разреженной матрицей."""
        df = data.copy()
        print(f"[{self.name}] Применение CountVectorizer к {len(df)} строкам.")
        
        for col in self.cols:
            vectorizer = self.vectorizers[col]
            # Transform возвращает разреженную матрицу (sparse matrix)
            sparse_matrix = vectorizer.transform(df[col].fillna(''))
            
            # Создаем DataFrame из разреженной матрицы
            feature_names = [f"{self.prefix}_{col}_{name}" for name in vectorizer.get_feature_names_out()]
            sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=feature_names, index=df.index)
            dense_df = sparse_df.sparse.to_dense()
            
            # Присоединяем новые признаки
            df = pd.concat([df, dense_df], axis=1)

        # Удаляем исходные текстовые колонки
        df.drop(columns=self.cols, inplace=True)
        return df

# ==================================================================================
# TfidfVectorizerGenerator
# ==================================================================================
class TfidfVectorizerGenerator(FeatureGenerator):
    """
    Преобразует текстовые данные в матрицу TF-IDF признаков.

    TF-IDF (Term Frequency-Inverse Document Frequency) учитывает не только
    частоту слова в документе, но и его "важность" во всем корпусе.
    Часто работает лучше, чем простой CountVectorizer.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список текстовых колонок для векторизации.
        prefix (str): Префикс для новых колонок.
        **kwargs: Дополнительные аргументы, передаваемые в TfidfVectorizer.
            (ngram_range, max_features, min_df, max_df, stop_words и др.)
    """
    def __init__(self, name: str, cols: List[str], prefix: str = 'tfidf', **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.prefix = prefix
        self.vectorizers: Dict[str, TfidfVectorizer] = {
            col: TfidfVectorizer(**kwargs) for col in self.cols
        }

    def fit(self, data: pd.DataFrame) -> None:
        """Обучает векторизаторы, создавая словарь и вычисляя IDF веса."""
        print(f"[{self.name}] Обучение TfidfVectorizer на колонках: {self.cols}")
        for col in self.cols:
            self.vectorizers[col].fit(data[col].fillna(''))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет TF-IDF векторизацию."""
        df = data.copy()
        print(f"[{self.name}] Применение TfidfVectorizer к {len(df)} строкам.")
        
        for col in self.cols:
            vectorizer = self.vectorizers[col]
            sparse_matrix = vectorizer.transform(df[col].fillna(''))
            
            feature_names = [f"{self.prefix}_{col}_{name}" for name in vectorizer.get_feature_names_out()]
            sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=feature_names, index=df.index)
            
            df = pd.concat([df, sparse_df], axis=1)

        df.drop(columns=self.cols, inplace=True)
        return df
