from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recsys.candidates.base import CandidateGenerator
from recsys.dataio.schema import Schema

logger = logging.getLogger(__name__)


class ContentTextGenerator(CandidateGenerator):
    name = "content_text"

    def __init__(self, backend: str = "tfidf") -> None:
        super().__init__(backend=backend)
        self.vectorizer: TfidfVectorizer | None = None
        self.item_vectors: np.ndarray | None = None
        self.item_ids: list[str] = []

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame | None, *, cutoff_ts, schema: Schema, rng):
        if items is None or items.empty:
            logger.warning("No items table provided; content_text disabled")
            return self
        text_col = None
        for col in ["text", "title", "description"]:
            if col in items.columns:
                text_col = col
                break
        if text_col is None:
            logger.warning("No text columns found; content_text skipped")
            return self
        texts = items[text_col].fillna("").astype(str)
        self.vectorizer = TfidfVectorizer(min_df=1)
        self.item_vectors = self.vectorizer.fit_transform(texts)
        self.item_ids = list(items["item_id"])
        return self

    def score(self, queries: pd.DataFrame, *, k: int, schema: Schema) -> pd.DataFrame:
        if self.item_vectors is None or self.vectorizer is None:
            return pd.DataFrame(columns=["query_id", "item_id", "source", "score_raw", "rank_src"])
        rows = []
        # naive strategy: recommend globally top by norm
        norms = np.asarray(self.item_vectors.sum(axis=1)).ravel()
        top_idx = np.argsort(-norms)[:k]
        for q in queries.itertuples():
            qid = getattr(q, "query_id")
            for rank, idx in enumerate(top_idx):
                rows.append(
                    {
                        "query_id": qid,
                        "item_id": self.item_ids[int(idx)],
                        "source": self.name,
                        "score_raw": float(norms[int(idx)]),
                        "rank_src": rank,
                    }
                )
        return pd.DataFrame(rows)
