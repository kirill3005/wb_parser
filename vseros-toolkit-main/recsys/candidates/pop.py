from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema
from recsys.dataio.utils_time import filter_by_cutoff

logger = logging.getLogger(__name__)


class PopularityGenerator(CandidateGenerator):
    name = "pop"

    def __init__(self, decay_days: int = 30) -> None:
        super().__init__(decay_days=decay_days)
        self.ranking = pd.DataFrame()

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        cutoff_ts,
        schema: Schema,
        rng: np.random.RandomState,
    ) -> "PopularityGenerator":
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")
        interactions = interactions.copy()
        max_ts = interactions["ts"].max()
        decay_const = self.params["decay_days"]
        delta_days = (max_ts - interactions["ts"]).dt.days.clip(lower=0)
        interactions["w"] = np.exp(-delta_days / max(decay_const, 1))

        pop = interactions.groupby("item_id")["w"].sum().sort_values(ascending=False)
        self.ranking = pop.reset_index().rename(columns={"w": "score_raw"})
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        rows = []
        top = self.ranking.head(k)
        for q in queries.itertuples():
            for rank, row in enumerate(top.itertuples(index=False)):
                rows.append(
                    {
                        "query_id": getattr(q, "query_id"),
                        "item_id": row.item_id,
                        "source": self.name,
                        "score_raw": float(row.score_raw),
                        "rank_src": rank,
                    }
                )
        return pd.DataFrame(rows)
