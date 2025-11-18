# src/models/lgbm.py

from typing import Any, Dict
from dataclasses import dataclass, field
import warnings

import joblib
import pandas as pd

try:
    import lightgbm as lgb
except (ImportError, OSError) as exc:  # pragma: no cover - depends on system libs
    lgb = None  # type: ignore[assignment]
    _LGB_IMPORT_ERROR = exc  # pragma: no cover - stored for diagnostics
else:  # pragma: no cover - lightgbm is optional in tests
    _LGB_IMPORT_ERROR = None

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .base import ModelInterface  # Импортируем наш базовый "контракт"

# ==================================================================================
# LGBMModel
# ==================================================================================
@dataclass
class LGBMModel(ModelInterface):
    """Wrapper class for LightGBM Classifier and Regressor models.

    This class provides a unified interface for LightGBM models, automatically
    determining whether to use classification or regression based on the objective
    parameter. It supports both binary/multiclass classification and regression tasks.

    Attributes:
        params (Dict[str, Any]): Model parameters passed to LightGBM.
        is_regressor (bool): Whether the model is configured for regression.
        model: The underlying LightGBM model instance.

    Example:
        >>> params = {'objective': 'binary', 'num_leaves': 31}
        >>> model = LGBMModel(params)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    params: Dict[str, Any]
    is_regressor: bool = field(init=False, default=False)
    model: Any = field(init=False)
    _using_fallback: bool = field(init=False, default=False)

    def __post_init__(self):
        """
        Инициализирует модель LightGBM.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                      Например, {'objective': 'binary', ...}
        """
        # Выбираем класс в зависимости от задачи (классификация или регрессия)
        objective = self.params.get('objective', '').lower()

        if 'regression' in objective or 'mae' in objective or 'mse' in objective:
            self.is_regressor = True
        else:
            self.is_regressor = False

        if lgb is not None:
            if self.is_regressor:
                self.model = lgb.LGBMRegressor(**self.params)
            else:
                self.model = lgb.LGBMClassifier(**self.params)
            return

        self._using_fallback = True
        warnings.warn(
            "LightGBM backend is unavailable. Falling back to HistGradientBoosting. "
            "Install libomp (e.g. `brew install libomp`) to enable native LightGBM.",
            RuntimeWarning,
        )
        fallback_params = self._prepare_fallback_params()
        if self.is_regressor:
            self.model = HistGradientBoostingRegressor(**fallback_params)
        else:
            self.model = HistGradientBoostingClassifier(**fallback_params)

    def _prepare_fallback_params(self) -> Dict[str, Any]:
        """Map LightGBM parameters to HistGradientBoosting equivalents."""
        params: Dict[str, Any] = {}
        if 'learning_rate' in self.params:
            params['learning_rate'] = self.params['learning_rate']

        n_estimators = self.params.get('n_estimators')
        if isinstance(n_estimators, int) and n_estimators > 0:
            params['max_iter'] = n_estimators

        max_depth = self.params.get('max_depth')
        if isinstance(max_depth, int) and max_depth > 0:
            params['max_depth'] = max_depth

        num_leaves = self.params.get('num_leaves')
        if isinstance(num_leaves, int) and num_leaves > 1:
            params['max_leaf_nodes'] = num_leaves

        min_child_samples = self.params.get('min_child_samples')
        if isinstance(min_child_samples, int) and min_child_samples > 0:
            params['min_samples_leaf'] = min_child_samples

        reg_lambda = self.params.get('reg_lambda')
        if isinstance(reg_lambda, (int, float)):
            params['l2_regularization'] = max(reg_lambda, 0.0)

        random_state = self.params.get('random_state')
        if isinstance(random_state, int):
            params['random_state'] = random_state

        return params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Train the LightGBM model.

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
        backend_name = "HistGradientBoosting" if self._using_fallback else "LightGBM"
        print(f"Обучение модели {backend_name}...")
        kwargs = dict(kwargs)
        if self._using_fallback:
            # HistGradientBoosting не поддерживает eval_set и ранний стоп — убираем их
            kwargs.pop('eval_set', None)
            kwargs.pop('early_stopping_rounds', None)
            kwargs.pop('eval_metric', None)
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
        if self.is_regressor:
            # Для регрессора predict_proba не существует, возвращаем обычные предсказания
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
    def load(cls, filepath: str) -> 'LGBMModel':
        """Load a saved model from disk using joblib.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            LGBMModel: Loaded model instance ready for prediction.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or corrupted.
        """
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)
    def __enter__(self):
        """Enter the context manager.

        Returns:
            LGBMModel: The model instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup LightGBM model resources
        if hasattr(self, 'model') and self.model is not None:
            self.model = None
