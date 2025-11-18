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


class SessionNgramGenerator(CandidateGenerator):
    name = "session_ngram"

    def __init__(self, max_n: int = 3, last_k: int = 10, decay: float = 0.9) -> None:
        super().__init__(max_n=max_n, last_k=last_k, decay=decay)
        self.transitions: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.history: Dict[str, List[str]] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        cutoff_ts,
        schema: Schema,
        rng: np.random.RandomState,
    ) -> "SessionNgramGenerator":
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")
        query_key = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        for qid, group in interactions.groupby(query_key):
            seq = list(group.sort_values("ts")["item_id"])
            self.history[qid] = seq
            for n in range(1, self.params["max_n"] + 1):
                for i in range(len(seq) - n):
                    prefix = tuple(seq[i : i + n])
                    next_item = seq[i + n]
                    self.transitions[prefix][next_item] += 1.0
        # normalize
        for prefix, nxt in self.transitions.items():
            total = sum(nxt.values())
            for k in list(nxt.keys()):
                nxt[k] = nxt[k] / max(total, 1e-9)
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        rows = []
        for q in queries.itertuples():
            qid = getattr(q, "query_id")
            seq = self.history.get(qid, [])[-self.params["last_k"] :]
            cand_scores: Dict[str, float] = defaultdict(float)
            for n in range(1, min(self.params["max_n"], len(seq)) + 1):
                prefix = tuple(seq[-n:])
                decay_weight = self.params["decay"] ** (n - 1)
                for item_id, prob in self.transitions.get(prefix, {}).items():
                    cand_scores[item_id] += prob * decay_weight
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
