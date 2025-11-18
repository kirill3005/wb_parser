from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recsys.candidates.ann import ANNIndex
from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema
from recsys.dataio.utils_time import filter_by_cutoff

logger = logging.getLogger(__name__)


class Item2VecGenerator(CandidateGenerator):
    name = "item2vec"

    def __init__(self, dim: int = 64, window: int = 5, epochs: int = 3, min_count: int = 1):
        super().__init__(dim=dim, window=window, epochs=epochs, min_count=min_count)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.index: ANNIndex | None = None
        self.item_list: List[str] = []

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        cutoff_ts,
        schema: Schema,
        rng: np.random.RandomState,
    ) -> "Item2VecGenerator":
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")
        query_key = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        co_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for _, group in interactions.groupby(query_key):
            seq = list(group.sort_values("ts")["item_id"])
            for i, item in enumerate(seq):
                left = max(0, i - self.params["window"])
                right = min(len(seq), i + self.params["window"] + 1)
                context = seq[left:i] + seq[i + 1 : right]
                for ctx in context:
                    co_matrix[item][ctx] += 1.0

        items_unique = list(co_matrix.keys())
        self.item_list = items_unique
        mat = np.zeros((len(items_unique), len(items_unique)), dtype=float)
        for i, it in enumerate(items_unique):
            for ctx, val in co_matrix[it].items():
                if ctx in co_matrix:
                    j = items_unique.index(ctx)
                    mat[i, j] = val
        svd = TruncatedSVD(n_components=min(self.params["dim"], max(1, len(items_unique) - 1)))
        emb = svd.fit_transform(mat)
        self.embeddings = {it: emb[i] for i, it in enumerate(items_unique)}
        if len(items_unique) > 0:
            self.index = ANNIndex(embeddings=np.vstack(list(self.embeddings.values())), n_neighbors=50)
        logger.info("Item2Vec fitted for %d items", len(items_unique))
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        if not self.embeddings:
            return pd.DataFrame(columns=["query_id", "item_id", "source", "score_raw", "rank_src"])
        rows = []
        item_to_idx = {it: i for i, it in enumerate(self.item_list)}
        for q in queries.itertuples():
            qid = getattr(q, "query_id")
            if hasattr(q, "hist_items"):
                hist_items = getattr(q, "hist_items")
            else:
                hist_items = list(self.embeddings.keys())[:1]
            vectors = []
            for it in hist_items[-5:]:
                if it in self.embeddings:
                    vectors.append(self.embeddings[it])
            if not vectors:
                continue
            mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
            idx, scores = self.index.query(mean_vec.reshape(1, -1), topk=k)
            for rank, (i_idx, score) in enumerate(zip(idx[0], scores[0])):
                item_id = self.item_list[i_idx]
                rows.append(
                    {
                        "query_id": qid,
                        "item_id": item_id,
                        "source": self.name,
                        "score_raw": float(score),
                        "rank_src": rank,
                    }
                )
        return pd.DataFrame(rows)
