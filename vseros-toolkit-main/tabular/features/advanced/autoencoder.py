# src/features/advanced/autoencoder.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..base import FeatureGenerator

# ==================================================================================
# 1. Архитектура Автоэнкодера (nn.Module)
# ==================================================================================
class TabularAutoencoder(nn.Module):
    def __init__(self, n_features: int, bottleneck_dim: int, layers: List[int], dropout: float):
        super().__init__()
        
        # --- Encoder ---
        encoder_layers = []
        input_dim = n_features
        for layer_dim in layers:
            encoder_layers.append(nn.Linear(input_dim, layer_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(layer_dim))
            encoder_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        # Последний слой кодировщика
        encoder_layers.append(nn.Linear(input_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- Decoder ---
        decoder_layers = []
        # Разворачиваем слои в обратном порядке
        layers.reverse()
        input_dim = bottleneck_dim
        for layer_dim in layers:
            decoder_layers.append(nn.Linear(input_dim, layer_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(layer_dim))
            decoder_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        # Последний слой декодера восстанавливает исходную размерность
        decoder_layers.append(nn.Linear(input_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Метод для использования только кодировщика на этапе transform."""
        return self.encoder(x)

# ==================================================================================
# 2. Генератор Признаков
# ==================================================================================
class AutoencoderFeatureGenerator(FeatureGenerator):
    """
    Создает новые признаки с помощью Denoising Autoencoder, обученного
    в режиме self-supervised.
    """
    # Обучаемся на всех данных, чтобы выучить общую структуру
    fit_strategy = "combined"

    def __init__(self, name: str, feature_cols: List[str], bottleneck_dim: int = 16,
                 layers: List[int] = [128, 64], dropout: float = 0.1,
                 noise_level: float = 0.1, epochs: int = 10, batch_size: int = 256,
                 optimizer_params: Dict[str, Any] = {'lr': 1e-3}):
        super().__init__(name)
        self.feature_cols = feature_cols
        self.bottleneck_dim = bottleneck_dim
        self.layers = layers
        self.dropout = dropout
        self.noise_level = noise_level
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_params = optimizer_params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model: TabularAutoencoder = None

    def fit(self, data: pd.DataFrame) -> None:
        """Обучает Denoising Autoencoder."""
        print(f"[{self.name}] Обучение Denoising Autoencoder на {len(data)} объектах.")
        
        # 1. Подготовка данных
        X = data[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Инициализация модели, функции потерь и оптимизатора
        n_features = X.shape[1]
        self.model = TabularAutoencoder(n_features, self.bottleneck_dim, self.layers, self.dropout)
        self.model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_params)
        
        # 3. Цикл обучения
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X_tuple in loader:
                batch_X = batch_X_tuple[0].to(self.device)
                
                # Создаем "испорченную" версию (добавляем шум)
                noise = torch.randn_like(batch_X) * self.noise_level
                noisy_batch_X = batch_X + noise
                
                # Forward pass
                outputs = self.model(noisy_batch_X)
                # Считаем ошибку между выходом и ЧИСТЫМ входом
                loss = criterion(outputs, batch_X)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"  - Эпоха {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
        # Сохраняем только encoder для этапа transform
        self.encoder = self.model.encoder.eval()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Прогоняет данные через обученный encoder для получения признаков."""
        df = data.copy()
        if self.encoder is None:
            raise RuntimeError("Модель не обучена. Вызовите .fit() перед .transform().")
            
        print(f"[{self.name}] Генерация признаков с помощью автоэнкодера для {len(df)} строк.")
        
        # Подготовка данных с использованием уже обученного scaler'а
        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Получаем эмбеддинги
        with torch.no_grad():
            embeddings = self.encoder(X_tensor).cpu().numpy()
            
        # Создаем DataFrame с новыми признаками
        col_names = [f"{self.name}_ae_{i}" for i in range(self.bottleneck_dim)]
        emb_df = pd.DataFrame(embeddings, columns=col_names, index=df.index)
        
        return pd.concat([df, emb_df], axis=1)