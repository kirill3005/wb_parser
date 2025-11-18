from __future__ import annotations

import logging
import pandas as pd

from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema

logger = logging.getLogger(__name__)


class ContentImageGenerator(CandidateGenerator):
    name = "content_image"

    def __init__(self, enabled: bool = False, **kwargs) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.enabled = enabled

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame | None, *, cutoff_ts, schema: Schema, rng):
        if not self.enabled:
            logger.info("content_image disabled")
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        return pd.DataFrame(columns=["query_id", "item_id", "source", "score_raw", "rank_src"])
