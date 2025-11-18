# src/features/text/transformer.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tqdm import tqdm

from ..base import FeatureGenerator

# ==================================================================================
# TransformerEmbeddingGenerator
# ==================================================================================
class TransformerEmbeddingGenerator(FeatureGenerator):
    """
    Создает векторные представления документов (эмбеддинги) с использованием
    трансформерных моделей из библиотеки Hugging Face (BERT, RoBERTa и т.д.).

    Процесс:
    1. Текст токенизируется специальным токенизатором модели.
    2. Токены пропускаются через модель для получения контекстуализированных
       эмбеддингов для каждого токена.
    3. Эмбеддинги токенов агрегируются в один вектор документа (чаще всего
       используется эмбеддинг специального [CLS] токена или mean-pooling).

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список текстовых колонок для обработки.
        model_name (str): Имя модели из Hugging Face Hub (например,
            'distilbert-base-uncased', 'bert-base-multilingual-cased').
        batch_size (int): Размер батча для обработки. Увеличьте для ускорения на GPU.
        pooling_strategy (str): Метод агрегации токенов.
            - 'cls': использовать эмбеддинг [CLS] токена (стандарт для BERT).
            - 'mean': усреднить эмбеддинги всех токенов (часто работает лучше).
    """
    SUPPORTED_POOLING: set = {'cls', 'mean'}

    def __init__(self, name: str, cols: List[str], model_name: str, 
                 batch_size: int = 8, pooling_strategy: str = 'mean'):
        super().__init__(name)
        if pooling_strategy not in self.SUPPORTED_POOLING:
            raise ValueError(f"Неподдерживаемая стратегия пулинга. Доступные: {self.SUPPORTED_POOLING}")
            
        self.cols = cols
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling_strategy = pooling_strategy
        
        # Определяем устройство (GPU, если доступно, иначе CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.name}] Используется устройство: {self.device}")
        
        # Загружаем токенизатор и модель
        print(f"[{self.name}] Загрузка модели '{self.model_name}'... (может занять время)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval() # Переводим модель в режим инференса
        print(f"[{self.name}] Модель загружена.")

    def fit(self, data: pd.DataFrame) -> None:
        """Трансформерные модели не требуют обучения на наших данных."""
        print(f"[{self.name}] TransformerEmbeddingGenerator не требует обучения.")
        pass
    
    @torch.no_grad() # Отключаем вычисление градиентов для экономии памяти
    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Вспомогательная функция для обработки одного батча текстов."""
        # Токенизируем батч
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем выход модели
        outputs = self.model(**inputs)
        
        # Применяем стратегию пулинга
        if self.pooling_strategy == 'cls':
            # Используем эмбеддинг [CLS] токена
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif self.pooling_strategy == 'mean':
            # Усредняем эмбеддинги всех токенов, учитывая маску
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
        return embeddings

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые признаки-эмбеддинги."""
        df = data.copy()
        print(f"[{self.name}] Применение TransformerEmbeddingGenerator к {len(df)} строкам.")

        for col in self.cols:
            # Используем генератор для текстов вместо создания полного списка
            def _text_generator():
                for text in df[col].fillna(''):
                    yield text

            text_gen = _text_generator()
            all_embeddings = []

            # Обрабатываем данные батчами с прогресс-баром, используя генератор
            for i in tqdm(range(0, len(df), self.batch_size), desc=f"Обработка '{col}'"):
                # Берем следующий батч из генератора
                batch_texts = []
                for _ in range(min(self.batch_size, len(df) - i)):
                    try:
                        batch_texts.append(next(text_gen))
                    except StopIteration:
                        break
                if not batch_texts:
                    break
                batch_embeddings = self._get_embeddings_batch(batch_texts)
                all_embeddings.append(batch_embeddings)

            # Соединяем результаты всех батчей
            embeddings_matrix = np.vstack(all_embeddings)
            embedding_dim = embeddings_matrix.shape[1]

            # Создаем имена для новых колонок
            new_col_names = [f"{col}_transformer_{self.pooling_strategy}_dim_{i}" for i in range(embedding_dim)]

            # Создаем DataFrame и присоединяем
            emb_df = pd.DataFrame(embeddings_matrix, columns=new_col_names, index=df.index)
            df = pd.concat([df, emb_df], axis=1)

        df.drop(columns=self.cols, inplace=True)
        return df