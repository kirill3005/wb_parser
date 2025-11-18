# src/models/catboost.py

from typing import Any, Dict, List
from dataclasses import dataclass, field

import joblib
import pandas as pd
import catboost as cb
from catboost.utils import get_gpu_device_count

from .base import ModelInterface # Импортируем наш базовый "контракт"

# ==================================================================================
# CatBoostModel
# ==================================================================================
@dataclass
class CatBoostModel(ModelInterface):
    """Wrapper class for CatBoost Classifier and Regressor models.

    This class provides a unified interface for CatBoost models, automatically
    detecting categorical features and configuring GPU usage when available.
    It supports both classification and regression tasks based on the loss function.

    Attributes:
        params (Dict[str, Any]): Model parameters passed to CatBoost.
        is_regressor (bool): Whether the model is configured for regression.
        model: The underlying CatBoost model instance.

    Example:
        >>> params = {'iterations': 100, 'learning_rate': 0.1}
        >>> model = CatBoostModel(params)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    params: Dict[str, Any]
    is_regressor: bool = field(init=False, default=False)
    model: Any = field(init=False)

    def __post_init__(self):
        """
        Инициализирует модель CatBoost.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                      Например, {'iterations': 1000, ...}
        """
        self.params = self.params.copy() # Копируем, чтобы безопасно изменять

        # --- Автоматическая настройка параметров ---

        # 1. Настройка verbose по умолчанию для чистоты логов
        if 'verbose' not in self.params:
            self.params['verbose'] = False

        # 2. Автоматическое включение GPU, если он доступен и не указан task_type
        if 'task_type' not in self.params and get_gpu_device_count() > 0:
            print("Обнаружен GPU. Установка 'task_type': 'GPU' для CatBoost.")
            self.params['task_type'] = 'GPU'

        # 3. Выбираем класс в зависимости от задачи
        loss_function = self.params.get('loss_function', '').lower()
        regression_losses = {'rmse', 'mae', 'rmsle', 'quantile', 'mape'}

        if any(reg_loss in loss_function for reg_loss in regression_losses):
            self.is_regressor = True
            self.model = cb.CatBoostRegressor(**self.params)
        else:
            self.is_regressor = False
            self.model = cb.CatBoostClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Train the CatBoost model.

        Automatically detects categorical features in the data and converts them
        to the appropriate format. Accepts eval_set and other fit parameters directly.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            **kwargs: Additional arguments passed to the model's fit method,
                such as eval_set, early_stopping_rounds, etc.

        Note:
            Categorical features are automatically detected and converted to 'category' dtype.
            GPU training is automatically enabled if available and not explicitly disabled.
        """
        print("Обучение модели CatBoost...")
        
        # Находим категориальные признаки в данных
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_features:
            print(f"Найдены категориальные признаки для CatBoost: {cat_features}")
        
        # CatBoost предпочитает, чтобы категориальные признаки имели тип 'category'
        X_train_copy = X_train.copy()
        for col in cat_features:
            X_train_copy[col] = X_train_copy[col].astype('category')
        
        # Адаптируем eval_set, если он передан
        if 'eval_set' in kwargs:
            eval_set = kwargs['eval_set']
            X_valid, y_valid = eval_set[0]
            X_valid_copy = X_valid.copy()
            for col in cat_features:
                X_valid_copy[col] = X_valid_copy[col].astype('category')
            kwargs['eval_set'] = [(X_valid_copy, y_valid)]

        self.model.fit(X_train_copy, y_train, cat_features=cat_features, **kwargs)

    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions for the input data.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: Predicted class labels (classification) or continuous values (regression).
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Generate probability predictions for classification or numeric predictions for regression.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: For classification, returns probabilities for the positive class.
                For regression, returns numeric predictions (same as predict()).

        Raises:
            AttributeError: If the underlying model doesn't support probability predictions
                (should not occur for properly configured models).
        """
        if self.is_regressor:
            # Для регрессора predict_proba не существует
            return self.model.predict(X)
        else:
            # Для классификации возвращаем только вероятности для класса "1"
            return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        """Save the trained model to disk using joblib.

        Args:
            filepath (str): Path where the model should be saved.
                Should include the file extension (e.g., '.pkl' or '.joblib').

        Raises:
            IOError: If the file cannot be written to the specified path.
        """
        print(f"Saving model to {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CatBoostModel':
        """Load a saved model from disk using joblib.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            CatBoostModel: Loaded model instance ready for prediction.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or corrupted.
        """
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)
    def __enter__(self):
        """Enter the context manager.

        Returns:
            CatBoostModel: The model instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup CatBoost model resources
        if hasattr(self, 'model') and self.model is not None:
            self.model = None