# src/features/aggregation.py

import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass, field

from .base import FeatureGenerator, FitStrategy

# ==================================================================================
# AggregationGenerator (Standard group statistics)
# ==================================================================================
@dataclass
class AggregationGenerator(FeatureGenerator):
    """Creates aggregated features using groupby().agg() operations.

    Summarizes entity behavior across their entire history. Supports an extended
    list of statistical functions for comprehensive feature engineering.

    Args:
        name (str): Unique name for this step.
        group_keys (List[str]): List of columns to group by.
        group_values (List[str]): List of columns to aggregate.
        agg_funcs (List[str]): List of aggregation functions.
            Supported: 'mean', 'std', 'sum', 'median', 'min', 'max', 'count',
            'nunique', 'skew', 'kurt' (for kurtosis).

    Attributes:
        agg_df_ (pd.DataFrame): Fitted aggregation DataFrame (set during fit).
    """
    name: str
    group_keys: List[str] = field()
    group_values: List[str] = field()
    agg_funcs: List[str] = field()
    agg_df_: pd.DataFrame = field(default=None, init=False)
    fit_strategy: FitStrategy = "combined"

    def fit(self, data: pd.DataFrame) -> None:
        """Compute and store aggregated statistics.

        For feature stability, this method is often called on combined data
        (train + test). The computed aggregations are stored for later use
        in transform().

        Args:
            data (pd.DataFrame): Input data to compute aggregations from.
        """
        print(f"[{self.name}] Fitting AggregationGenerator: grouping by {self.group_keys}")
        
        agg_df = data.groupby(self.group_keys)[self.group_values].agg(self.agg_funcs)
        
        new_cols = [f"{'_'.join(self.group_keys)}_{col[1]}_{col[0]}" for col in agg_df.columns.values]
        agg_df.columns = new_cols
        agg_df.reset_index(inplace=True)
        
        self.agg_df_ = agg_df
        print(f"  - Created {len(self.agg_df_)} aggregated records with {self.agg_df_.shape[1]} features.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Join computed aggregations to the original dataframe.

        Args:
            data (pd.DataFrame): Input data to merge aggregations with.

        Returns:
            pd.DataFrame: Data with additional aggregated features.
        """
        print(f"[{self.name}] Applying AggregationGenerator to {len(data)} rows.")
        df = pd.merge(data, self.agg_df_, on=self.group_keys, how='left')
        return df

# ==================================================================================
# RollingAggregationGenerator (Rolling window statistics)
# ==================================================================================
@dataclass
class RollingAggregationGenerator(FeatureGenerator):
    """Creates aggregated features in rolling time windows for each group.

    Computes statistics (e.g., mean, sum) for specified features over a previous
    time period (e.g., last 7 days) for each group.

    IMPORTANT: Results are shifted by 1 step (lagged) to prevent data leakage
    from the current event. The feature describes the state *before* the current moment.

    Args:
        name (str): Unique name for this step.
        group_keys (List[str]): List of columns to group by (e.g., ['user_id']).
        date_col (str): Date/time column to build the window on.
        value_cols (List[str]): List of numeric columns to aggregate in the window.
        window_sizes (List[str]): List of window sizes in pandas format
            (e.g., '3D', '7D', '30D' - 3 days, 7 days, 30 days).
        agg_funcs (List[str]): List of aggregation functions ('mean', 'sum', 'count').
    """
    name: str
    group_keys: List[str] = field()
    date_col: str = field()
    value_cols: List[str] = field()
    window_sizes: List[str] = field()
    agg_funcs: List[str] = field()
    fit_strategy: FitStrategy = "combined"

    def fit(self, data: pd.DataFrame) -> None:
        """This is a stateless transformation, no training required."""
        print(f"[{self.name}] RollingAggregationGenerator requires no training.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute and join rolling window features.

        Args:
            data (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Data with additional rolling window features.
        """
        df = data.copy()
        print(f"[{self.name}] Applying RollingAggregationGenerator to {len(df)} rows.")

        # Ensure the date column has the correct type
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Sorting is mandatory for correct rolling operation
        df.sort_values(by=self.group_keys + [self.date_col], inplace=True)
        
        grouped = df.groupby(self.group_keys)
        
        for value_col in self.value_cols:
            for window in self.window_sizes:
                for func in self.agg_funcs:
                    new_col_name = f"{'_'.join(self.group_keys)}_{value_col}_rolling_{window}_{func}"
                    print(f"  - Creating feature: {new_col_name}")

                    # Compute rolling aggregation
                    rolling_agg = grouped[value_col].rolling(window, on=self.date_col).agg(func)

                    # Shift result by 1 to prevent leakage.
                    # reset_index needed to return group_keys for proper merge
                    lagged_agg = rolling_agg.reset_index(0, drop=True).shift(1)
                    
                    df[new_col_name] = lagged_agg

        return df