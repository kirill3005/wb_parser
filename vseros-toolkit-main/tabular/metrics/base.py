# src/metrics/base.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class MetricInterface(ABC):
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        # kwargs может содержать 'groups', 'sample_weight' и т.д.
        pass