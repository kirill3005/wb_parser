from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np


def _topk(y_pred_items: Sequence, k: int) -> List:
    return list(y_pred_items[:k])


def recall_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int) -> float:
    true_set = set(y_true_items)
    if not true_set:
        return 0.0
    hit = len(true_set.intersection(_topk(y_pred_items, k)))
    return hit / len(true_set)


def hitrate_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int) -> float:
    return 1.0 if set(y_true_items).intersection(_topk(y_pred_items, k)) else 0.0


def precision_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int) -> float:
    top = _topk(y_pred_items, k)
    if not top:
        return 0.0
    return len(set(y_true_items).intersection(top)) / len(top)


def dcg_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int, gains: str = "exp2") -> float:
    top = _topk(y_pred_items, k)
    dcg = 0.0
    for idx, item in enumerate(top):
        if item in y_true_items:
            gain = 1.0 if gains == "linear" else (2.0 ** 1 - 1)
            dcg += gain / math.log2(idx + 2)
    return dcg


def ndcg_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int, gains: str = "exp2") -> float:
    ideal = dcg_at_k(y_true_items, y_true_items, min(k, len(y_true_items)), gains=gains)
    if ideal == 0:
        return 0.0
    return dcg_at_k(y_true_items, y_pred_items, k, gains=gains) / ideal


def ap_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int) -> float:
    top = _topk(y_pred_items, k)
    if not top:
        return 0.0
    score = 0.0
    hits = 0
    for idx, item in enumerate(top, start=1):
        if item in y_true_items:
            hits += 1
            score += hits / idx
    return score / min(len(y_true_items), k) if y_true_items else 0.0


def mrr_at_k(y_true_items: Sequence, y_pred_items: Sequence, k: int) -> float:
    top = _topk(y_pred_items, k)
    for idx, item in enumerate(top, start=1):
        if item in y_true_items:
            return 1.0 / idx
    return 0.0


def coverage_at_k(all_pred_items: Iterable[Sequence], item_universe_size: int, k: int) -> float:
    uniq = set()
    for pred in all_pred_items:
        uniq.update(_topk(pred, k))
    if item_universe_size == 0:
        return 0.0
    return len(uniq) / float(item_universe_size)


def novelty_at_k(all_pred_items: Iterable[Sequence], item_pop: dict, k: int) -> float:
    values = []
    for pred in all_pred_items:
        for item in _topk(pred, k):
            pop = max(item_pop.get(item, 1), 1)
            values.append(-math.log(pop / sum(item_pop.values())))
    return float(np.mean(values)) if values else 0.0


def diversity_at_k(all_pred_items: Iterable[Sequence], item_to_category: dict, k: int) -> float:
    values = []
    for pred in all_pred_items:
        top = _topk(pred, k)
        if len(top) <= 1:
            values.append(0.0)
            continue
        distinct = len({item_to_category.get(i, i) for i in top})
        values.append(distinct / len(top))
    return float(np.mean(values)) if values else 0.0
