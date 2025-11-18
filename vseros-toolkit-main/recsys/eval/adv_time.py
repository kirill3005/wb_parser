from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_time_shift(train_inter: pd.DataFrame, val_inter: pd.DataFrame) -> Dict:
    """Lightweight detector using item popularity change."""

    train_counts = train_inter["item_id"].value_counts(normalize=True)
    val_counts = val_inter["item_id"].value_counts(normalize=True)
    joined = pd.concat(
        [train_counts.rename("train"), val_counts.rename("val")], axis=1
    ).fillna(0)
    # simple proxy: mean absolute difference
    mad = float((joined["train"] - joined["val"]).abs().mean())
    auc_proxy = 0.5 + mad
    flag = auc_proxy > 0.6
    top_shift = joined.sort_values(("train", "val"), ascending=False).head(5).index.tolist()
    return {
        "auc_proxy": auc_proxy,
        "suspected_shift": flag,
        "top_items": top_shift,
    }
