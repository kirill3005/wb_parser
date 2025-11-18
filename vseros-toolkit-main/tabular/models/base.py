# src/models/base.py

from abc import ABC, abstractmethod
from typing import Any, Type
import pandas as pd

from ..utils import performance_monitor

class ModelInterface(ABC):
    """Abstract base class (interface) for all models.

    This class defines the standard interface that all machine learning models
    in the project must implement. It ensures consistency across different
    model implementations and provides a unified API for training, prediction,
    and model persistence.
    """

    @abstractmethod
    @performance_monitor
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Train the model on the provided data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            **kwargs: Additional keyword arguments for specific model implementations.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
        """
        pass

    @abstractmethod
    @performance_monitor
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions for the input data.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: Predicted values. For classification, returns class labels.
                For regression, returns continuous values.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
        """
        pass

    @abstractmethod
    @performance_monitor
    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Generate probability predictions for classification tasks.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: Predicted probabilities. For binary classification, returns
                probabilities for the positive class. For multiclass, returns
                probabilities for each class.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
            NotSupportedError: If the model doesn't support probability predictions.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath (str): Path where the model should be saved.
                Should include the file extension appropriate for the model type.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
            IOError: If the file cannot be written.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls: Type['ModelInterface'], filepath: str) -> 'ModelInterface':
        """Load a saved model from disk.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            ModelInterface: Loaded model instance ready for prediction.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or corrupted.
        """
        pass
    def __enter__(self):
        """Enter the context manager.

        Returns:
            ModelInterface: The model instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Default cleanup - subclasses can override for specific cleanup logic
        pass