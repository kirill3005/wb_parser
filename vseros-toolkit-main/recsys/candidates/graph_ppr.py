from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema
from recsys.dataio.utils_time import filter_by_cutoff

logger = logging.getLogger(__name__)


class GraphPPRGenerator(CandidateGenerator):
    name = "graph_ppr"

    def __init__(self, restart_prob: float = 0.15, iters: int = 10):
        super().__init__(restart_prob=restart_prob, iters=iters)
        self.neighbors: Dict[str, Dict[str, float]] = {}
        self.history: Dict[str, list[str]] = {}

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame | None, *, cutoff_ts, schema: Schema, rng) -> "GraphPPRGenerator":
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")
        query_key = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        adj: Dict[str, set[str]] = defaultdict(set)
        for _, group in interactions.groupby(query_key):
            seq = list(group.sort_values("ts")["item_id"])
            self.history[_] = seq
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    a, b = seq[i], seq[j]
                    adj[a].add(b)
                    adj[b].add(a)
        neigh = {}
        for item, nbrs in adj.items():
            scores = {}
            for nb in nbrs:
                inter = len(adj[item].intersection(adj[nb]))
                union = len(adj[item].union(adj[nb]))
                scores[nb] = inter / union if union > 0 else 0
            neigh[item] = scores
        self.neighbors = neigh
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        rows = []
        for q in queries.itertuples():
            qid = getattr(q, "query_id")
            hist = self.history.get(qid, [])
            cand_scores: Dict[str, float] = defaultdict(float)
            for it in hist[-5:]:
                for nb, sc in self.neighbors.get(it, {}).items():
                    cand_scores[nb] += sc
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
