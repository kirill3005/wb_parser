# src/models/sklearn_model.py

import os
import re
from typing import Any, Dict
from dataclasses import dataclass, field

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn import ensemble, linear_model, svm

from .base import ModelInterface
from ..utils import validate_type, validate_non_empty

# Whitelist of allowed sklearn model classes for security
ALLOWED_MODELS = {
    'sklearn.ensemble.ExtraTreesClassifier': ensemble.ExtraTreesClassifier,
    'sklearn.ensemble.RandomForestClassifier': ensemble.RandomForestClassifier,
    'sklearn.ensemble.RandomForestRegressor': ensemble.RandomForestRegressor,
    'sklearn.linear_model.LogisticRegression': linear_model.LogisticRegression,
    'sklearn.linear_model.Ridge': linear_model.Ridge,
    'sklearn.svm.SVC': svm.SVC,
}

# ==================================================================================
# SklearnModel
# ==================================================================================
@dataclass
class SklearnModel(ModelInterface):
    """Universal wrapper class for scikit-learn models.

    This class provides a secure and unified interface for scikit-learn models,
    with built-in validation and a whitelist of allowed model classes for security.
    It supports both classification and regression tasks depending on the chosen model.

    Attributes:
        model_class_path (str): Full path to the sklearn model class.
        params (Dict[str, Any]): Parameters passed to the model constructor.
        model: The underlying sklearn model instance.

    Example:
        >>> model = SklearnModel('sklearn.ensemble.RandomForestClassifier', {'n_estimators': 100})
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    model_class: str
    params: Dict[str, Any]
    model_class_path: str = field(init=False)
    model: Any = field(init=False)
    @validate_type(str, dict)
    @validate_non_empty()

    def __post_init__(self):
        """
        Инициализирует sklearn-совместимую модель.

        Args:
            model_class (str): Полный путь к классу модели в scikit-learn.
                                Например, 'sklearn.linear_model.LogisticRegression'.
            params (Dict[str, Any]): Словарь с параметрами для конструктора модели.
        """
        # Validate model_class string for security
        if not isinstance(self.model_class, str):
            raise ValueError("model_class must be a string")
        if not re.match(r'^[a-zA-Z_.]+$', self.model_class):
            raise ValueError("model_class contains invalid characters. Only letters, dots, and underscores are allowed.")
        if '..' in self.model_class or self.model_class.startswith('.') or self.model_class.endswith('.'):
            raise ValueError("model_class has invalid format")

        # Validate params for security (basic check)
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        for key, value in self.params.items():
            if not isinstance(key, str):
                raise ValueError("Parameter keys must be strings")
            # Prevent potentially dangerous values (basic check)
            if isinstance(value, str) and ('..' in value or value.startswith('/') or value.startswith('\\')):
                raise ValueError(f"Parameter '{key}' contains potentially dangerous path-like value")

        self.model_class_path = self.model_class

        try:
            # Validate and get model class from whitelist
            if self.model_class not in ALLOWED_MODELS:
                raise ValueError(f"Model class '{self.model_class}' is not in the allowed list for security reasons")
            model_constructor = ALLOWED_MODELS[self.model_class]
            self.model = model_constructor(**self.params)
        except (KeyError, TypeError) as e:
            raise ImportError(f"Не удалось импортировать или создать класс модели: {self.model_class}") from e

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Train the sklearn model. Additional kwargs are ignored for compatibility.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            **kwargs: Additional arguments (ignored for sklearn compatibility).

        Note:
            This method ignores additional kwargs to maintain compatibility
            with the ModelInterface while sklearn models may not accept them.
        """
        print(f"Training model {self.model.__class__.__name__}...")
        self.model.fit(X_train, y_train)

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
            Any: For classifiers with predict_proba, returns probabilities for the positive class.
                For regressors or classifiers without predict_proba, returns numeric predictions.

        Note:
            For binary classification, returns probabilities for the positive class (index 1).
            For multiclass classification, this may not work as expected.
        """
        if hasattr(self.model, 'predict_proba'):
            # For classifiers return probabilities for class "1"
            return self.model.predict_proba(X)[:, 1]
        else:
            # For regressors return just the prediction
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
    def load(cls, filepath: str) -> 'SklearnModel':
        """Load a saved model from disk using joblib.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            SklearnModel: Loaded model instance ready for prediction.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or corrupted.
        """
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)