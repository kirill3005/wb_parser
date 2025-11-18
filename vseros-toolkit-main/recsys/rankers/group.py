"""Utilities for forming learning-to-rank groups."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def build_groups(pairs: pd.DataFrame, query_col: str = "query_id") -> List[int]:
    """Return group sizes preserving order of rows.

    Parameters
    ----------
    pairs: pd.DataFrame
        Table with at least ``query_col`` column sorted as training matrix.
    query_col: str
        Column denoting query/session/user identifier.

    Returns
    -------
    list[int]
        Group sizes in the same order as pairs are provided.
    """

    if query_col not in pairs.columns:
        raise KeyError(f"{query_col} missing in pairs")
    sizes: List[int] = []
    last_q = None
    count = 0
    for q in pairs[query_col].tolist():
        if last_q is None:
            last_q = q
            count = 1
            continue
        if q == last_q:
            count += 1
        else:
            sizes.append(count)
            last_q = q
            count = 1
    if last_q is not None:
        sizes.append(count)
    return sizes


def assert_group_sums(groups: Iterable[int], total: int) -> None:
    groups_list = list(groups)
    if sum(groups_list) != total:
        raise ValueError("Sum of group sizes does not match number of rows")


__all__ = ["build_groups", "assert_group_sums"]
