# src/features/categorical/nominal.py

import pandas as pd
from typing import List, Dict, Any

try:
    from category_encoders import HashingEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

from ..base import FeatureGenerator

# ==================================================================================
# OneHotEncoderGenerator
# ==================================================================================
class OneHotEncoderGenerator(FeatureGenerator):
    """
    Применяет One-Hot Encoding (OHE) к заданным категориальным колонкам.

    OHE создает новые бинарные (0/1) колонки для каждой уникальной категории.
    Идеально подходит для линейных моделей и признаков с низкой кардинальностью
    (небольшим количеством уникальных значений).

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для кодирования.
        max_categories (int, optional): Если задано, кодирует только N самых
            частых категорий, а остальные объединяет в категорию 'other'.
            Крайне полезно для борьбы с высокой кардинальностью.
    """
    def __init__(self, name: str, cols: List[str], max_categories: int = None):
        super().__init__(name)
        self.cols = cols
        self.max_categories = max_categories
        self.top_categories_: Dict[str, List[str]] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Определяет и сохраняет уникальные категории (или топ-N категорий)
        для каждой колонки ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение OneHotEncoder на колонках: {self.cols}")
        for col in self.cols:
            if self.max_categories:
                # Находим N самых частых категорий
                top_cats = data[col].value_counts().nlargest(self.max_categories).index.tolist()
                self.top_categories_[col] = top_cats
                print(f"  - Для '{col}' выбраны топ-{len(top_cats)} категорий.")
            else:
                # Используем все уникальные категории
                all_cats = data[col].unique().tolist()
                self.top_categories_[col] = all_cats
                print(f"  - Для '{col}' выбраны все {len(all_cats)} категорий.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет OHE, используя сохраненные категории.
        Категории, не встреченные при обучении, игнорируются или попадают в 'other'.
        """
        df = data.copy()
        print(f"[{self.name}] Применение OneHotEncoder к {len(df)} строкам.")
        for col in self.cols:
            for cat in self.top_categories_[col]:
                if pd.isna(cat):
                    col_suffix = "nan"
                    mask = df[col].isna()
                else:
                    col_suffix = self._sanitize_category(str(cat))
                    mask = df[col] == cat
                df[f"{col}_{col_suffix}"] = mask.astype(int)

            if self.max_categories:
                is_other = ~df[col].isin(self.top_categories_[col]) & df[col].notna()
                df[f"{col}_other"] = is_other.astype(int)
        
        # Удаляем исходные колонки после кодирования
        df.drop(columns=self.cols, inplace=True)
        return df

    @staticmethod
    def _sanitize_category(value: str) -> str:
        return (
            value.strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

# ==================================================================================
# CountFrequencyEncoderGenerator
# ==================================================================================
class CountFrequencyEncoderGenerator(FeatureGenerator):
    """
    Заменяет каждую категорию на количество ее появлений (Count) или
    долю (Frequency) в наборе данных.

    Очень простой и очень эффективный метод для древовидных моделей.
    Он захватывает "популярность" категории, что может быть сильным сигналом.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для кодирования.
        normalize (bool): Если True, кодирует в долю (частоту), иначе в количество.
    """
    def __init__(self, name: str, cols: List[str], normalize: bool = False):
        super().__init__(name)
        self.cols = cols
        self.normalize = normalize
        self.mappings_: Dict[str, pd.Series] = {}
        self.suffix = "_freq" if normalize else "_count"

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет количество/частоту для каждой категории
        ТОЛЬКО на обучающих данных.
        """
        print(f"[{self.name}] Обучение Count/Frequency Encoder на колонках: {self.cols}")
        for col in self.cols:
            self.mappings_[col] = data[col].value_counts(normalize=self.normalize)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет сохраненное отображение. Категории, не встреченные при
        обучении, получают значение 0.
        """
        df = data.copy()
        print(f"[{self.name}] Применение Count/Frequency Encoder к {len(df)} строкам.")
        for col in self.cols:
            mapped_values = df[col].map(self.mappings_[col])
            # Заполняем нулем те категории, которых не было в трейне
            df[f"{col}{self.suffix}"] = mapped_values.fillna(0)
        
        # Этот кодировщик обычно не заменяет исходные признаки, а дополняет их,
        # но для единообразия можно добавить опцию удаления.
        # df.drop(columns=self.cols, inplace=True)
        return df

# ==================================================================================
# HashingEncoderGenerator
# ==================================================================================
class HashingEncoderGenerator(FeatureGenerator):
    """
    Использует хеш-функцию для преобразования категорий в N новых признаков.

    Основное преимущество - управляемая размерность на выходе, что делает его
    идеальным для признаков с экстремально высокой кардинальностью (user_id,
    URL, zip-код), где OHE и Target Encoding неприменимы.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для кодирования.
        n_components (int): Количество признаков на выходе (размер хеша).
    """
    def __init__(self, name: str, cols: List[str], n_components: int = 8, **kwargs: Any):
        super().__init__(name)
        self.cols = cols
        self.n_components = n_components
        self.encoder = HashingEncoder(
            n_components=self.n_components,
            cols=self.cols,
            **kwargs
        )
        self.output_col_names = [f"{self.name}_hash_{i}" for i in range(self.n_components)]

    def fit(self, data: pd.DataFrame) -> None:
        """
        Hashing Encoder является stateless, но мы вызываем fit для
        совместимости с API.
        """
        print(f"[{self.name}] Обучение HashingEncoder на колонках: {self.cols}")
        self.encoder.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет хеш-преобразование и добавляет новые колонки.
        """
        df = data.copy()
        print(f"[{self.name}] Применение HashingEncoder к {len(df)} строкам.")
        
        # Преобразуем данные
        hashed_features = self.encoder.transform(df)
        hashed_features.columns = self.output_col_names
        
        # Соединяем с исходным датафреймом
        df = pd.concat([df.reset_index(drop=True), hashed_features.reset_index(drop=True)], axis=1)
        
        # Удаляем исходные колонки
        df.drop(columns=self.cols, inplace=True)
        return df
