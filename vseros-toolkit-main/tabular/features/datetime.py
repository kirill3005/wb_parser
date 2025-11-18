# src/features/datetime.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass, field

from .base import FeatureGenerator, FitStrategy

# ==================================================================================
# DatetimeFeatureGenerator
# ==================================================================================
@dataclass
class DatetimeFeatureGenerator(FeatureGenerator):
    """Extracts various components from date and time columns.

    Transforms date/time columns into a set of numeric and cyclical features
    that can be used by machine learning models.

    Args:
        name (str): Unique name for this step.
        cols (List[str]): List of columns to process.
        components (List[str]): List of components to extract.
            Available components:
            - 'year', 'month', 'day', 'hour', 'minute', 'second'
            - 'weekday' (day of week, 0=Mon), 'dayofyear', 'weekofyear'
            - 'quarter'
            - 'is_weekend', 'is_month_start', 'is_month_end'
            - 'time_of_day' ('Morning', 'Afternoon', 'Evening', 'Night')
            - 'cyclical' (creates sin/cos transformations for month, weekday, hour)

    Attributes:
        cols (List[str]): Columns to process.
        components (set): Set of components to extract.
    """
    name: str
    cols: List[str] = field()
    components: List[str] = field()
    fit_strategy: FitStrategy = "train_only"

    def __post_init__(self):
        self.components = set(self.components)

    def fit(self, data: pd.DataFrame) -> None:
        """This is a stateless transformation, no training required."""
        print(f"[{self.name}] DatetimeFeatureGenerator requires no training.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create new features based on date and time components.

        Args:
            data (pd.DataFrame): Input data containing datetime columns.

        Returns:
            pd.DataFrame: Data with additional datetime-derived features.
        """
        df = data.copy()
        print(f"[{self.name}] Extracting datetime components from: {self.cols}")

        for col in self.cols:
            # Convert column to datetime if not already in that format
            dt_series = pd.to_datetime(df[col], errors='coerce')

            if 'year' in self.components: df[f'{col}_year'] = dt_series.dt.year
            if 'month' in self.components: df[f'{col}_month'] = dt_series.dt.month
            if 'day' in self.components: df[f'{col}_day'] = dt_series.dt.day
            if 'hour' in self.components: df[f'{col}_hour'] = dt_series.dt.hour
            if 'minute' in self.components: df[f'{col}_minute'] = dt_series.dt.minute
            if 'second' in self.components: df[f'{col}_second'] = dt_series.dt.second
            if 'weekday' in self.components: df[f'{col}_weekday'] = dt_series.dt.weekday
            if 'dayofyear' in self.components: df[f'{col}_dayofyear'] = dt_series.dt.dayofyear
            if 'weekofyear' in self.components: df[f'{col}_weekofyear'] = dt_series.dt.isocalendar().week.astype(int)
            if 'quarter' in self.components: df[f'{col}_quarter'] = dt_series.dt.quarter

            if 'is_weekend' in self.components: df[f'{col}_is_weekend'] = (dt_series.dt.weekday >= 5).astype(int)
            if 'is_month_start' in self.components: df[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
            if 'is_month_end' in self.components: df[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
            
            if 'time_of_day' in self.components:
                hour = dt_series.dt.hour
                bins = [0, 6, 12, 18, 24]
                labels = ['Night', 'Morning', 'Afternoon', 'Evening']
                df[f'{col}_time_of_day'] = pd.cut(hour, bins=bins, labels=labels, right=False, ordered=False)

            if 'cyclical' in self.components:
                # Sine/cosine transformations to capture cyclical nature
                if 'month' in self.components:
                    df[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                    df[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                if 'weekday' in self.components:
                    df[f'{col}_weekday_sin'] = np.sin(2 * np.pi * dt_series.dt.weekday / 7)
                    df[f'{col}_weekday_cos'] = np.cos(2 * np.pi * dt_series.dt.weekday / 7)
                if 'hour' in self.components:
                    df[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                    df[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)

        return df

# ==================================================================================
# DateDifferenceGenerator
# ==================================================================================
@dataclass
class DateDifferenceGenerator(FeatureGenerator):
    """Computes the difference between two date/time columns.

    Args:
        name (str): Unique name for this step.
        col1 (str): First (later) date column.
        col2 (str): Second (earlier) date column.
        unit (str): Unit for the difference ('D' for days, 'h' for hours, etc.).

    Attributes:
        col1 (str): First date column name.
        col2 (str): Second date column name.
        unit (str): Time unit for difference calculation.
    """
    name: str
    col1: str = field()
    col2: str = field()
    unit: str = 'D'

    def fit(self, data: pd.DataFrame) -> None:
        """This is a stateless transformation, no training required."""
        print(f"[{self.name}] DateDifferenceGenerator requires no training.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a new feature with the time difference.

        Args:
            data (pd.DataFrame): Input data containing the date columns.

        Returns:
            pd.DataFrame: Data with additional time difference feature.

        Raises:
            ValueError: If an unsupported time unit is specified.
        """
        df = data.copy()
        print(f"[{self.name}] Computing difference between {self.col1} and {self.col2}")
        
        dt1 = pd.to_datetime(df[self.col1], errors='coerce')
        dt2 = pd.to_datetime(df[self.col2], errors='coerce')
        
        time_delta = (dt1 - dt2).dt.total_seconds()
        
        # Convert seconds to the desired unit
        if self.unit == 'D':
            divisor = 86400  # 24 * 60 * 60
        elif self.unit == 'h':
            divisor = 3600  # 60 * 60
        elif self.unit == 'm':
            divisor = 60
        elif self.unit == 's':
            divisor = 1
        else:
            raise ValueError(f"Unsupported unit: {self.unit}. Use 'D', 'h', 'm', or 's'.")
            
        df[f"{self.col1}_minus_{self.col2}_in_{self.unit}"] = time_delta / divisor
        return df