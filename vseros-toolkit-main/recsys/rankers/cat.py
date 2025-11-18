"""CatBoost ranker placeholder with fallback."""
from __future__ import annotations

import logging

from recsys.rankers.linear import LinearRanker

LOGGER = logging.getLogger(__name__)


class CatBoostRanker(LinearRanker):
    name = "catboost"

    def __init__(self):
        try:
            import catboost  # type: ignore

            self.available = True
        except Exception:
            self.available = False
            LOGGER.warning("CatBoost not installed; using linear model")
        super().__init__()


__all__ = ["CatBoostRanker"]
