# src/features/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from typing import Literal

from ..utils import performance_monitor

# Define possible fitting strategies
FitStrategy = Literal["train_only", "combined"]

@dataclass
class FeatureGenerator(ABC):
    """Abstract base class for all feature generators.

    This class defines the interface that all feature generators must implement.
    Feature generators are responsible for creating new features from existing data
    through various transformations, encodings, and aggregations.

    Attributes:
        name (str): Unique name identifier for the feature generator
        fit_strategy (FitStrategy): Strategy for fitting the generator.
            - "train_only": Fit only on training data (default, safest option)
            - "combined": Fit on combined train/validation data
    """
    name: str
    # Default to the safest strategy
    fit_strategy: FitStrategy = "train_only"

    @abstractmethod
    @performance_monitor
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the generator on training data.

        This method learns parameters from the training data (e.g., computes means,
        medians, creates encoding dictionaries). This method is called ONLY on
        the training dataset to prevent data leakage.

        Args:
            data (pd.DataFrame): Training data to fit the generator on.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
        """
        pass

    @abstractmethod
    @performance_monitor
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform input data by creating new features.

        This method applies the learned transformations to create new features
        from the input data. Can be applied to training, validation, or test data.

        Args:
            data (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: DataFrame with additional generated features.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
        """
        pass

    @performance_monitor
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the generator and transform the data in one step.

        This is a convenience method that calls fit() followed by transform().
        Use this when you want to fit and transform the same dataset.

        Args:
            data (pd.DataFrame): Input data to fit on and transform.

        Returns:
            pd.DataFrame: Transformed data with new features.
        """
        self.fit(data)
        return self.transform(data)
    def __enter__(self):
        """Enter the context manager.

        Returns:
            FeatureGenerator: The generator instance.
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