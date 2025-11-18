from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class ANNIndex:
    """Lightweight wrapper over sklearn NearestNeighbors with cosine metric."""

    def __init__(self, embeddings: np.ndarray, n_neighbors: int = 20) -> None:
        self.embeddings = embeddings
        self.index = NearestNeighbors(metric="cosine")
        self.index.fit(embeddings)
        self.n_neighbors = n_neighbors

    def query(self, vectors: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = self.index.kneighbors(vectors, n_neighbors=min(topk, self.n_neighbors))
        scores = 1.0 - dist
        return idx, scores
