# src/models/xgb.py

from typing import Any, Dict
from dataclasses import dataclass, field

import joblib
import pandas as pd
import xgboost as xgb

from .base import ModelInterface # Импортируем наш базовый "контракт"

# ==================================================================================
# XGBModel
# ==================================================================================
@dataclass
class XGBModel(ModelInterface):
    """Wrapper class for XGBoost Classifier and Regressor models.

    This class provides a unified interface for XGBoost models, automatically
    determining whether to use classification or regression based on the objective
    parameter. It supports both binary/multiclass classification and regression tasks.

    Attributes:
        params (Dict[str, Any]): Model parameters passed to XGBoost.
        model: The underlying XGBoost model instance (XGBClassifier or XGBRegressor).

    Example:
        >>> params = {'objective': 'binary:logistic', 'max_depth': 6}
        >>> model = XGBModel(params)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    params: Dict[str, Any]
    model: Any = field(init=False)

    def __post_init__(self):
        """
        Инициализирует модель XGBoost.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                      Например, {'objective': 'binary:logistic', ...}
        """
        # Выбираем класс в зависимости от задачи (классификация или регрессия)
        if 'regressor' in str(self.params.get('objective', '')).lower():
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Train the XGBoost model.

        Accepts eval_set and other fit parameters directly, enabling features
        like early stopping and validation monitoring.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            **kwargs: Additional arguments passed to the model's fit method,
                such as eval_set, early_stopping_rounds, eval_metric, etc.

        Note:
            Use eval_set parameter to monitor validation performance during training.
            Early stopping can be enabled with early_stopping_rounds parameter.
        """
        print("Обучение модели XGBoost...")
        self.model.fit(X_train, y_train, **kwargs)

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
        if hasattr(self.model, 'predict_proba'):
            # Для классификации возвращаем только вероятности для класса "1"
            return self.model.predict_proba(X)[:, 1]
        else:
            # Для регрессии predict_proba не существует, возвращаем обычные предсказания
            return self.model.predict(X)

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
    def load(cls, filepath: str) -> 'XGBModel':
        """Load a saved model from disk using joblib.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            XGBModel: Loaded model instance ready for prediction.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or corrupted.
        """
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)