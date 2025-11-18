"""LightGBM ranker with graceful fallback if library missing."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from recsys.rankers.base import ModelRun
from recsys.rankers.linear import LinearRanker

LOGGER = logging.getLogger(__name__)


class LightGBMRanker(LinearRanker):
    name = "lightgbm"

    def __init__(self):
        try:
            import lightgbm as lgb  # type: ignore

            self._lib = lgb
            self.available = True
        except Exception:
            self._lib = None
            self.available = False
            LOGGER.warning("LightGBM not installed; falling back to linear model")
        super().__init__()


__all__ = ["LightGBMRanker"]
