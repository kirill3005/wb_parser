from typing import Iterable

import numpy as np
import pandas as pd


def assert_no_nan_inf(df: pd.DataFrame) -> None:
    if df.isnull().values.any():
        raise ValueError("NaN values detected")
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty and ~np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite values detected")


def assert_no_duplicate_columns(df: pd.DataFrame) -> None:
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate columns: {dupes}")


def assert_same_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test columns mismatch")


def assert_unique_names(packages: Iterable[str]) -> None:
    names = list(packages)
    if len(names) != len(set(names)):
        raise ValueError("Feature package names must be unique")
