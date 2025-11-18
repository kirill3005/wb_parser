import numpy as np
from typing import Callable, Dict, Any


def find_global_tau(y_true, y_prob, scorer: Callable) -> float:
    """Брутфорс/поиск τ на OOF по выбранной метрике (binary)."""
    best_tau = 0.5
    best_score = -np.inf
    for tau in np.linspace(0, 1, 101):
        preds = (y_prob >= tau).astype(int)
        score = scorer(y_true, preds)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


def find_per_class_tau(y_true_onehot, y_prob, scorer: Callable) -> np.ndarray:
    """Пер-класс пороги (multiclass one-vs-rest или multilabel)."""
    n_classes = y_true_onehot.shape[1]
    taus = np.zeros(n_classes)
    for c in range(n_classes):
        taus[c] = find_global_tau(y_true_onehot[:, c], y_prob[:, c], scorer)
    return taus


def apply_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Возвращает индексы/onehot top-K."""
    if scores.ndim == 1:
        idx = np.argsort(-scores)[:k]
        return idx
    topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    onehot = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        cols = topk_idx[i][np.argsort(-scores[i, topk_idx[i]])]
        onehot[i, cols] = 1
    return onehot
