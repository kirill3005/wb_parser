from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .metrics import recall_at_k, ndcg_at_k, ap_at_k, mrr_at_k


METRIC_MAP = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": ap_at_k,
    "mrr": mrr_at_k,
}


def bootstrap_metrics(joined: List[Tuple[Sequence, Sequence]], parsed_metrics: Iterable[Tuple[str, int]], *, iters: int, rng: np.random.RandomState) -> Dict:
    results: Dict = {}
    joined = list(joined)
    if not joined:
        return results
    n = len(joined)
    for name, k in parsed_metrics:
        fn = METRIC_MAP.get(name)
        if fn is None:
            continue
        samples = []
        for _ in range(iters):
            idx = rng.randint(0, n, size=n)
            subset = [joined[i] for i in idx]
            samples.append(np.mean([fn(t, p, k) for t, p in subset]))
        samples_arr = np.array(samples)
        results[f"{name}@{k}"] = {
            "mean": float(samples_arr.mean()),
            "p5": float(np.percentile(samples_arr, 5)),
            "p95": float(np.percentile(samples_arr, 95)),
        }
    return results
