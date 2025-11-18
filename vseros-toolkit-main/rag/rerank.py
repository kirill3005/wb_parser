from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .indices import DenseIndex
from .candidates import (
    Candidate,
    from_idx_score,
    to_idx_score,
    sort_and_rank,
)
from .utils import ensure_numpy


# =====================================
# Cross-encoder rerank (Candidate API)
# =====================================

def rerank_cross_encoder(
    query: str,
    candidates: Sequence[Candidate],
    docs: Sequence[str],
    cross_encoder: Any,
    top_k: Optional[int] = None,
    score_name: str = "ce",
    source: str = "ce_rerank",
) -> List[Candidate]:
    """
    Переранк кандидатов с помощью cross-encoder'а.

    Логика:
      - для каждого кандидата берём текст чанка docs[c.chunk_idx],
      - считаем скор cross-encoder'ом для пары (query, text),
      - записываем скор в cand.scores[score_name], обновляем cand.score и cand.source,
      - сортируем по новому score и проставляем rank,
      - опционально обрезаем до top_k.

    Параметры
    ---------
    query:
        Текст запроса.
    candidates:
        Список кандидатов (Candidate), обычно после hybrid_candidates / bm25_candidates / dense_candidates.
    docs:
        Список текстов чанков. docs[i] должен соответствовать chunk_idx=i.
    cross_encoder:
        Модель cross-encoder с методом .predict(pairs) или вызываемая как функция (pairs),
        где pairs — список [(query, text), ...].
    top_k:
        Сколько кандидатов вернуть после переранка. Если None — возвращаем всех валидных.
    score_name:
        Имя канала для записи скорa в Candidate.scores (по умолчанию "ce").
    source:
        Значение для Candidate.source (по умолчанию "ce_rerank").

    Возвращает
    ----------
    List[Candidate]
        Список кандидатов, отсортированный по скору cross-encoder'а.
    """
    if not candidates:
        return []

    n_docs = len(docs)

    # Фильтруем кандидатов с некорректным chunk_idx
    valid_cands: List[Candidate] = [
        c for c in candidates
        if 0 <= c.chunk_idx < n_docs
    ]

    if not valid_cands:
        return []

    # Собираем пары (query, text) для cross-encoder'а
    pairs: List[Tuple[str, str]] = []
    for c in valid_cands:
        text = docs[c.chunk_idx] if docs[c.chunk_idx] is not None else ""
        pairs.append((query, text))

    # Считаем скор cross-encoder'ом
    if hasattr(cross_encoder, "predict"):
        raw_scores = cross_encoder.predict(pairs)
    elif callable(cross_encoder):
        raw_scores = cross_encoder(pairs)
    else:
        raise TypeError(
            "cross_encoder должен иметь метод .predict(pairs) или быть вызываемым объектом."
        )

    scores = ensure_numpy(raw_scores)
    scores = scores.reshape(-1).astype("float32")

    if scores.shape[0] != len(valid_cands):
        raise ValueError(
            f"rerank_cross_encoder: получено {scores.shape[0]} скор(ов) "
            f"для {len(valid_cands)} кандидатов."
        )

    # Обновляем кандидатов
    for cand, s in zip(valid_cands, scores):
        cand.set_score(float(s), name=score_name, source=source)

    # Сортируем по новому score и проставляем rank
    ranked = sort_and_rank(valid_cands, descending=True)

    if top_k is not None and top_k > 0:
        ranked = ranked[: min(top_k, len(ranked))]

    return ranked


# ==================================================
# MMR (Maximal Marginal Relevance) поверх кандидатов
# ==================================================

def apply_mmr(
    candidates: Sequence[Candidate],
    dense_index: DenseIndex,
    lambda_: float = 0.5,
    top_k: int = 10,
    base_score_key: Optional[str] = None,
) -> List[Candidate]:
    """
    Применяет MMR (Maximal Marginal Relevance) к списку кандидатов.

    Цель:
      - уравновесить "релевантность" (score) и "диверсификацию" (похожесть на уже выбранные),
      - на выходе получить top_k кандидатов в порядке выбора MMR.

    Формула:
      MMR(j) = λ * relevance(j) - (1 - λ) * max_{k ∈ S} sim(j, k),

    где:
      - relevance(j) — базовый скор кандидата j (либо из cand.score, либо из cand.scores[base_score_key]),
      - sim(j, k) — косинусное сходство эмбеддингов (dot product, если они нормированы),
      - S — уже выбранные кандидаты.

    Параметры
    ---------
    candidates:
        Список кандидатов (обычно уже отсортированных по какому-то скору).
    dense_index:
        DenseIndex с эмбеддингами для всех чанков.
    lambda_:
        Вес релевантности (0..1). Чем ближе к 1.0 — тем меньше диверсификация.
    top_k:
        Сколько кандидатов выбрать MMR'ом.
    base_score_key:
        Если задано — брать релевантность из cand.scores[base_score_key],
        иначе используется cand.score.

    Возвращает
    ----------
    List[Candidate]
        Новый список кандидатов в порядке выбора MMR. В каждом:
          - score = mmr_score,
          - scores["mmr"] = mmr_score,
          - source = "mmr",
          - rank = порядок выбора (0-based).
    """
    if not candidates or top_k <= 0:
        return []

    n_docs = dense_index.n_docs
    # Фильтруем кандидатов, для которых есть эмбеддинги
    valid_cands: List[Candidate] = [
        c for c in candidates
        if 0 <= c.chunk_idx < n_docs
    ]
    if not valid_cands:
        return []

    top_k = min(top_k, len(valid_cands))

    # Вектор релевантности
    if base_score_key is not None:
        relevance = np.array(
            [
                float(c.scores.get(base_score_key, c.score))
                for c in valid_cands
            ],
            dtype="float32",
        )
    else:
        relevance = np.array(
            [float(c.score) for c in valid_cands],
            dtype="float32",
        )

    # Эмбеддинги кандидатов
    emb = dense_index.embeddings
    cand_indices = np.array([c.chunk_idx for c in valid_cands], dtype="int64")
    cand_vecs = emb[cand_indices]  # shape = (M, D)
    M = cand_vecs.shape[0]

    # Матрица попарных сходств (dot product). Если эмбеддинги нормированы — это cosine.
    sim_matrix = cand_vecs @ cand_vecs.T  # shape = (M, M)

    # Нормализуем lambda_
    lam = float(lambda_)
    if lam < 0.0:
        lam = 0.0
    elif lam > 1.0:
        lam = 1.0

    selected_indices: List[int] = []
    selected_mmr_scores: List[float] = []

    # 1) Начинаем с кандидата с максимальной релевантностью
    first_idx = int(np.argmax(relevance))
    selected_indices.append(first_idx)
    # для первого diversity=0, поэтому mmr_score = relevance
    selected_mmr_scores.append(float(relevance[first_idx]))

    remaining = [i for i in range(M) if i != first_idx]

    # 2) Жадно добираем top_k
    while remaining and len(selected_indices) < top_k:
        best_j: Optional[int] = None
        best_score: float = -1e30

        for j in remaining:
            if selected_indices:
                sims = sim_matrix[j, selected_indices]  # shape = (len(S),)
                max_sim = float(sims.max())
            else:
                max_sim = 0.0

            mmr_score = lam * float(relevance[j]) - (1.0 - lam) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_j = j

        if best_j is None:
            break

        selected_indices.append(best_j)
        selected_mmr_scores.append(float(best_score))
        remaining.remove(best_j)

    # 3) Обновляем кандидатов и собираем результат
    result: List[Candidate] = []
    for rank, (idx, mmr_score) in enumerate(zip(selected_indices, selected_mmr_scores)):
        cand = valid_cands[idx]
        cand.scores["mmr"] = float(mmr_score)
        cand.score = float(mmr_score)
        cand.source = "mmr"
        cand.rank = rank
        result.append(cand)

    return result


# ==========================================
# Обёртки для обратной совместимости (pairs)
# ==========================================

def rerank_cross_encoder_pairs(
    query: str,
    candidates: Sequence[Tuple[int, float]],
    docs: Sequence[str],
    cross_encoder: Any,
    top_k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    Обратная совместимость: версия rerank_cross_encoder, работающая с (chunk_idx, score).

    Параметры
    ---------
    query:
        Текст запроса.
    candidates:
        Список (chunk_idx, score), например выход hybrid_retrieve.
    docs:
        Список текстов чанков.
    cross_encoder:
        См. rerank_cross_encoder.
    top_k:
        Сколько кандидатов вернуть.

    Возвращает
    ----------
    List[Tuple[int, float]]
        Список (chunk_idx, score), где score — скор cross-encoder'а.
    """
    if not candidates:
        return []

    cands = from_idx_score(
        pairs=candidates,
        score_name="pre_ce",
        source="pre_ce",
        base_meta=None,
        assign_rank=True,
    )
    reranked = rerank_cross_encoder(
        query=query,
        candidates=cands,
        docs=docs,
        cross_encoder=cross_encoder,
        top_k=top_k,
        score_name="ce",
        source="ce_rerank",
    )
    return to_idx_score(reranked)


def apply_mmr_pairs(
    candidates: Sequence[Tuple[int, float]],
    dense_index: DenseIndex,
    lambda_: float = 0.5,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Обратная совместимость: версия apply_mmr, работающая с (chunk_idx, score).

    Параметры
    ---------
    candidates:
        Список (chunk_idx, score) — обычно выход hybrid_retrieve / rerank_cross_encoder_pairs.
    dense_index:
        DenseIndex с эмбеддингами.
    lambda_:
        Параметр MMR (см. apply_mmr).
    top_k:
        Сколько кандидатов выбрать.

    Возвращает
    ----------
    List[Tuple[int, float]]
        Список (chunk_idx, score), где score — mmr_score.
    """
    if not candidates or top_k <= 0:
        return []

    cands = from_idx_score(
        pairs=candidates,
        score_name="base",
        source="pre_mmr",
        base_meta=None,
        assign_rank=True,
    )

    mmr_cands = apply_mmr(
        candidates=cands,
        dense_index=dense_index,
        lambda_=lambda_,
        top_k=top_k,
        base_score_key="base",
    )

    return to_idx_score(mmr_cands)
