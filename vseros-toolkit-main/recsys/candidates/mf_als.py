from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema
from recsys.dataio.utils_time import filter_by_cutoff

logger = logging.getLogger(__name__)


class MFALSCandidate(CandidateGenerator):
    name = "mf_als"

    def __init__(self, factors: int = 64, reg: float = 0.01, iterations: int = 10):
        super().__init__(factors=factors, reg=reg, iterations=iterations)
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_map: dict[str, int] = {}
        self.item_map: dict[str, int] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        cutoff_ts,
        schema: Schema,
        rng: np.random.RandomState,
    ) -> "MFALSCandidate":
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")
        self.user_map = {u: i for i, u in enumerate(interactions["user_id"].unique())}
        self.item_map = {i: j for j, i in enumerate(interactions["item_id"].unique())}
        rows = [self.user_map[u] for u in interactions["user_id"]]
        cols = [self.item_map[i] for i in interactions["item_id"]]
        data = np.ones(len(interactions))
        mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.user_map), len(self.item_map)))
        svd = TruncatedSVD(n_components=min(self.params["factors"], min(mat.shape) - 1))
        user_f = svd.fit_transform(mat)
        item_f = svd.components_.T
        self.user_factors = user_f
        self.item_factors = item_f
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        if self.user_factors is None or self.item_factors is None:
            return pd.DataFrame()
        rows = []
        item_list = list(self.item_map.keys())
        for q in queries.itertuples():
            qid = getattr(q, "query_id")
            if qid not in self.user_map:
                continue
            uidx = self.user_map[qid]
            uvec = self.user_factors[uidx]
            scores = self.item_factors @ uvec
            top_idx = np.argsort(-scores)[:k]
            for rank, idx in enumerate(top_idx):
                rows.append(
                    {
                        "query_id": qid,
                        "item_id": item_list[idx],
                        "source": self.name,
                        "score_raw": float(scores[idx]),
                        "rank_src": rank,
                    }
                )
        return pd.DataFrame(rows)
