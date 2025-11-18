# src/features/text/statistics.py

import pandas as pd
from typing import List, Set
import string
import re

from ..base import FeatureGenerator

# ==================================================================================
# TextStatisticsGenerator
# ==================================================================================
class TextStatisticsGenerator(FeatureGenerator):
    """
    Вычисляет набор простых статистических признаков для текстовых колонок.

    Эти признаки описывают не семантику, а структуру и стиль текста.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список текстовых колонок для обработки.
        stats (List[str]): Список статистик для вычисления.
            Доступные статистики:
            - 'text_length': Общее количество символов.
            - 'word_count': Общее количество слов.
            - 'unique_word_count': Количество уникальных слов.
            - 'avg_word_length': Средняя длина слова.
            - 'sentence_count': Приблизительное количество предложений.
            - 'punctuation_count': Количество знаков препинания.
            - 'uppercase_count': Количество заглавных букв.
    """
    SUPPORTED_STATS: Set[str] = {
        'text_length', 'word_count', 'unique_word_count', 'avg_word_length',
        'sentence_count', 'punctuation_count', 'uppercase_count'
    }

    def __init__(self, name: str, cols: List[str], stats: List[str]):
        super().__init__(name)
        if not set(stats).issubset(self.SUPPORTED_STATS):
            raise ValueError(f"Обнаружены неподдерживаемые статистики. Доступные: {self.SUPPORTED_STATS}")
        self.cols = cols
        self.stats = stats
        self.epsilon = 1e-6 # для безопасного деления

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        print(f"[{self.name}] TextStatisticsGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет и добавляет статистические признаки."""
        df = data.copy()
        print(f"[{self.name}] Вычисление статистик для колонок: {self.cols}")

        for col in self.cols:
            # Предварительно заполняем NaN, чтобы избежать ошибок
            text_series = df[col].fillna('')
            
            if 'text_length' in self.stats:
                df[f'{col}_text_length'] = text_series.str.len()

            if 'word_count' in self.stats:
                df[f'{col}_word_count'] = text_series.str.split().str.len()

            if 'unique_word_count' in self.stats:
                df[f'{col}_unique_word_count'] = text_series.str.split().apply(lambda x: len(set(x)))

            if 'avg_word_length' in self.stats:
                # Считаем количество символов, исключая пробелы
                char_count = text_series.str.replace(r'\s+', '', regex=True).str.len()
                word_count = df.get(f'{col}_word_count', text_series.str.split().str.len())
                df[f'{col}_avg_word_length'] = char_count / (word_count + self.epsilon)

            if 'sentence_count' in self.stats:
                # Приблизительный подсчет по знакам конца предложения
                df[f'{col}_sentence_count'] = text_series.str.count(r'[.!?]') + 1

            if 'punctuation_count' in self.stats:
                punct_regex = f'[{re.escape(string.punctuation)}]'
                df[f'{col}_punctuation_count'] = text_series.str.count(punct_regex)

            if 'uppercase_count' in self.stats:
                df[f'{col}_uppercase_count'] = text_series.str.findall(r'[A-Z]').str.len()

        # Исходные текстовые колонки обычно не удаляются, так как они нужны
        # для других генераторов (например, TfidfVectorizer)
        return df