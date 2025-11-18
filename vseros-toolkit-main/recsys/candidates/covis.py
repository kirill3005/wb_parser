from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema
from recsys.dataio.utils_time import filter_by_cutoff

logger = logging.getLogger(__name__)


class CoVisGenerator(CandidateGenerator):
    name = "covis"

    def __init__(
        self,
        window_days: int = 30,
        alpha: float = 0.05,
        min_count: int = 1,
        max_neighbors_per_item: int = 200,
    ) -> None:
        super().__init__(
            window_days=window_days,
            alpha=alpha,
            min_count=min_count,
            max_neighbors_per_item=max_neighbors_per_item,
        )
        self.co_counts: Dict[str, Dict[str, float]] = {}
        self.history: Dict[str, List[str]] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        cutoff_ts,
        schema: Schema,
        rng: np.random.RandomState,
    ) -> "CoVisGenerator":
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")
        cutoff_lower = pd.to_datetime(cutoff_ts) - pd.Timedelta(days=self.params["window_days"])
        interactions = interactions[interactions["ts"] >= cutoff_lower]
        interactions = interactions.sort_values("ts")

        query_key = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        co_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        history: Dict[str, List[str]] = defaultdict(list)

        for qid, group in interactions.groupby(query_key):
            items_seq = list(group["item_id"])
            history[qid] = items_seq
            timestamps = list(group["ts"])
            for i, item_i in enumerate(items_seq):
                ts_i = timestamps[i]
                for j in range(i + 1, len(items_seq)):
                    item_j = items_seq[j]
                    delta_days = max((timestamps[j] - ts_i).days, 0)
                    w = np.exp(-self.params["alpha"] * delta_days)
                    co_map[item_i][item_j] += w
                    co_map[item_j][item_i] += w

        # prune
        pruned = {}
        for item, neigh in co_map.items():
            filtered = {k: v for k, v in neigh.items() if v >= self.params["min_count"]}
            sorted_nb = sorted(filtered.items(), key=lambda x: -x[1])[: self.params["max_neighbors_per_item"]]
            pruned[item] = dict(sorted_nb)
        self.co_counts = pruned
        self.history = history
        logger.info("CoVis built for %d items", len(pruned))
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        rows = []
        pos_decay = 0.9
        for q in queries.itertuples():
            qid = getattr(q, "query_id")
            hist_items = self.history.get(qid, [])[-10:][::-1]
            cand_scores: Dict[str, float] = defaultdict(float)
            for pos, it in enumerate(hist_items):
                w_pos = pos_decay ** pos
                for nb, w in self.co_counts.get(it, {}).items():
                    cand_scores[nb] += w_pos * w
            top = sorted(cand_scores.items(), key=lambda x: -x[1])[:k]
            for rank, (item_id, score_raw) in enumerate(top):
                rows.append(
                    {
                        "query_id": qid,
                        "item_id": item_id,
                        "source": self.name,
                        "score_raw": float(score_raw),
                        "rank_src": rank,
                    }
                )
        return pd.DataFrame(rows)
