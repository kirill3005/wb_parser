# src/features/text/embeddings.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any

try:
    import gensim
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from ..base import FeatureGenerator

# ==================================================================================
# PretrainedEmbeddingGenerator
# ==================================================================================
class PretrainedEmbeddingGenerator(FeatureGenerator):
    """
    Создает векторные представления документов (эмбеддинги) с использованием
    предобученных моделей (Word2Vec, GloVe, FastText).

    Процесс:
    1. Текст токенизируется (разбивается на слова).
    2. Для каждого слова ищется его предобученный вектор.
    3. Векторы всех слов в тексте агрегируются в один вектор документа.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список текстовых колонок для обработки.
        embedding_path (str): Путь к файлу с предобученными эмбеддингами.
        agg_methods (List[str]): Список методов агрегации.
            Поддерживаются: 'mean', 'max'.
        binary (bool): Укажите True, если файл эмбеддингов в бинарном формате
            (например, Google News Word2Vec). False для текстовых (GloVe).
    """
    SUPPORTED_AGGS: set = {'mean', 'max'}

    def __init__(self, name: str, cols: List[str], embedding_path: str, 
                 agg_methods: List[str], binary: bool = False):
        super().__init__(name)
        if not set(agg_methods).issubset(self.SUPPORTED_AGGS):
            raise ValueError(f"Обнаружены неподдерживаемые методы агрегации. Доступные: {self.SUPPORTED_AGGS}")
        
        self.cols = cols
        self.embedding_path = embedding_path
        self.agg_methods = agg_methods
        self.binary = binary
        
        print(f"[{self.name}] Загрузка предобученной модели из {self.embedding_path}... (может занять время)")
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_path, binary=self.binary)
        self.embedding_dim = self.model.vector_size
        self.zero_vector = np.zeros(self.embedding_dim)
        print(f"[{self.name}] Модель загружена. Размерность вектора: {self.embedding_dim}")

    def fit(self, data: pd.DataFrame) -> None:
        """Предобученные эмбеддинги не требуют обучения на наших данных."""
        print(f"[{self.name}] PretrainedEmbeddingGenerator не требует обучения.")
        pass

    def _text_to_aggregated_vectors(self, text: str) -> Dict[str, np.ndarray]:
        """Вспомогательная функция для преобразования одного текста."""
        if pd.isna(text):
            return {method: self.zero_vector for method in self.agg_methods}
            
        # Простая токенизация и приведение к нижнему регистру
        tokens = text.lower().split()
        
        # Получаем векторы для слов, которые есть в словаре модели
        vectors = [self.model[word] for word in tokens if word in self.model]
        
        if not vectors:
            return {method: self.zero_vector for method in self.agg_methods}
            
        # Вычисляем агрегации
        aggregations = {}
        if 'mean' in self.agg_methods:
            aggregations['mean'] = np.mean(vectors, axis=0)
        if 'max' in self.agg_methods:
            aggregations['max'] = np.max(vectors, axis=0)
            
        return aggregations

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые признаки-эмбеддинги."""
        df = data.copy()
        print(f"[{self.name}] Применение PretrainedEmbeddingGenerator к {len(df)} строкам.")

        for col in self.cols:
            # Используем генератор для обработки текстов по одному, избегая создания большого списка результатов
            def _generate_embeddings():
                for text in df[col]:
                    yield self._text_to_aggregated_vectors(text)

            # Создаем генератор результатов
            results_gen = _generate_embeddings()

            # "Распаковываем" результаты в новые колонки
            for method in self.agg_methods:
                # Извлекаем векторы для данного метода агрегации с помощью генератора
                vectors = [result[method] for result in results_gen]

                # Создаем имена для новых колонок
                new_col_names = [f"{col}_{method}_dim_{i}" for i in range(self.embedding_dim)]

                # Создаем новый DataFrame из векторов
                emb_df = pd.DataFrame(vectors, columns=new_col_names, index=df.index)

                # Присоединяем к основному DataFrame
                df = pd.concat([df, emb_df], axis=1)

        # Удаляем исходные текстовые колонки
        df.drop(columns=self.cols, inplace=True)
        return df