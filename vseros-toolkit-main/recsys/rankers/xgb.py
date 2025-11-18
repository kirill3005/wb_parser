"""XGBoost ranker placeholder with linear fallback."""
from __future__ import annotations

import logging

from recsys.rankers.linear import LinearRanker

LOGGER = logging.getLogger(__name__)


class XGBoostRanker(LinearRanker):
    name = "xgboost"

    def __init__(self):
        try:
            import xgboost  # type: ignore

            self.available = True
        except Exception:
            self.available = False
            LOGGER.warning("XGBoost not installed; using linear model")
        super().__init__()


__all__ = ["XGBoostRanker"]
