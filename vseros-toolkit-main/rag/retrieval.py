from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .indices import BM25Index, DenseIndex, _default_tokenizer
from .embeddings import EmbeddingCache, encode_query_with_cache
from .candidates import Candidate, from_idx_score, to_idx_score, sort_and_rank


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def _normalize_scores_0_1(scores: np.ndarray) -> np.ndarray:
    """
    Нормализация массива скороов в [0, 1].

    Если все значения одинаковые или массив пустой — возвращает нули.
    """
    if scores.size == 0:
        return scores

    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores, dtype="float32")

    return ((scores - s_min) / (s_max - s_min)).astype("float32")


def _top_k_from_scores(scores: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """
    Возвращает top-k индексов и их скоры из массива scores.
    """
    n = scores.shape[0]
    if n == 0 or k <= 0:
        return []

    k_eff = min(k, n)
    # argpartition для быстрого top-k, затем сортировка внутри
    idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
    idx_sorted = idx[np.argsort(-scores[idx])]

    return [(int(i), float(scores[i])) for i in idx_sorted]


def _bm25_scores_for_query(
    query: str,
    index: BM25Index,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> np.ndarray:
    """
    Считает BM25-скоры для ВСЕХ документов индекса под один query.

    Возвращает массив scores shape = (n_docs,).
    """
    n_docs = index.n_docs
    if n_docs == 0:
        return np.zeros((0,), dtype="float32")

    tok = tokenizer or index.tokenizer or _default_tokenizer
    tokens = tok(query or "")
    if not tokens:
        return np.zeros((n_docs,), dtype="float32")

    k1 = index.k1
    b = index.b
    doc_len = index.doc_len
    avg_dl = index.avg_doc_len if index.avg_doc_len > 0 else 1.0

    # нормирующий множитель на документ
    # k1 * (1 - b + b * |D|/avgDL)
    doc_norm = k1 * (1.0 - b + b * doc_len / avg_dl)  # shape = (N,)

    scores = np.zeros((n_docs,), dtype="float32")
    idf = index.idf
    tf_mat = index.term_freqs  # csr_matrix (N, V)

    for tok in tokens:
        term_id = index.vocab.get(tok)
        if term_id is None:
            continue

        # столбец tf по всем документам
        col = tf_mat[:, term_id]  # shape (N, 1)
        tf = col.toarray().ravel().astype("float32")  # [N]

        if not np.any(tf):
            continue

        denom = tf + doc_norm  # [N]
        contrib = idf[term_id] * (tf * (k1 + 1.0) / denom)  # [N]
        scores += contrib

    return scores


def _dense_scores_for_query(
    query: str,
    index: DenseIndex,
    emb_model: Any,
    cache: Optional[EmbeddingCache],
    model_id: Optional[str],
) -> np.ndarray:
    """
    Считает dense-скоры (dot product / cosine) для ВСЕХ документов индекса под один query.

    Возвращает массив scores shape = (n_docs,).
    """
    n_docs = index.n_docs
    if n_docs == 0:
        return np.zeros((0,), dtype="float32")

    # Если model_id не задан, пытаемся взять из индекса, иначе "default"
    eff_model_id = model_id or index.model_id or "default"

    q_emb = encode_query_with_cache(
        query=query,
        emb_model=emb_model,
        model_id=eff_model_id,
        cache=cache,
        normalize=index.normalize,
    )  # shape = (1, D)

    q_vec = q_emb[0]  # shape = (D,)
    # (N, D) @ (D,) -> (N,)
    scores = index.embeddings @ q_vec
    return scores.astype("float32")


# =========================
# CANDIDATE-API
# =========================

def bm25_candidates(
    query: str,
    index: BM25Index,
    k: int = 10,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[Candidate]:
    """
    BM25-поиск: возвращает top-k кандидатов в формате Candidate.

    Параметры
    ---------
    query:
        Запрос.
    index:
        BM25Index.
    k:
        Сколько кандидатов вернуть.
    tokenizer:
        Кастомный токенайзер. Если None — берётся index.tokenizer или _default_tokenizer.

    Возвращает
    ----------
    List[Candidate]
        Отсортированный по score список кандидатов.
    """
    scores = _bm25_scores_for_query(query, index, tokenizer=tokenizer)
    if scores.size == 0:
        return []

    top_pairs = _top_k_from_scores(scores, k)

    base_meta: Optional[Dict[int, Dict[str, Any]]] = None
    if index.meta is not None:
        base_meta = {i: index.meta[i] for i, _ in top_pairs if 0 <= i < len(index.meta)}

    cands = from_idx_score(
        pairs=top_pairs,
        score_name="bm25",
        source="bm25",
        base_meta=base_meta,
        assign_rank=True,
    )
    return cands


def dense_candidates(
    query: str,
    index: DenseIndex,
    emb_model: Any,
    k: int = 10,
    cache: Optional[EmbeddingCache] = None,
    model_id: Optional[str] = None,
) -> List[Candidate]:
    """
    Dense-поиск: возвращает top-k кандидатов в формате Candidate.

    Параметры
    ---------
    query:
        Запрос.
    index:
        DenseIndex.
    emb_model:
        Модель эмбеддингов с методом .encode.
    k:
        Сколько кандидатов вернуть.
    cache:
        EmbeddingCache для query-эмбеддингов (опционально).
    model_id:
        Идентификатор модели для кэша (должен совпадать с тем, что использовался
        при build_dense_index, но не обязательно).

    Возвращает
    ----------
    List[Candidate]
        Отсортированный по score список кандидатов.
    """
    scores = _dense_scores_for_query(
        query=query,
        index=index,
        emb_model=emb_model,
        cache=cache,
        model_id=model_id,
    )
    if scores.size == 0:
        return []

    top_pairs = _top_k_from_scores(scores, k)

    base_meta: Optional[Dict[int, Dict[str, Any]]] = None
    if index.meta is not None:
        base_meta = {i: index.meta[i] for i, _ in top_pairs if 0 <= i < len(index.meta)}

    cands = from_idx_score(
        pairs=top_pairs,
        score_name="dense",
        source="dense",
        base_meta=base_meta,
        assign_rank=True,
    )
    return cands


def hybrid_candidates(
    query: str,
    bm25_index: Optional[BM25Index],
    dense_index: Optional[DenseIndex],
    emb_model: Any,
    k: int = 10,
    alpha: float = 0.5,
    cache: Optional[EmbeddingCache] = None,
    model_id: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[Candidate]:
    """
    Гибридный поиск: BM25 + dense.

    Общая формула:
        hybrid_score = alpha * bm25_norm + (1 - alpha) * dense_norm,
    где bm25_norm и dense_norm — скоры, нормированные в [0, 1] отдельно.

    Параметры
    ---------
    query:
        Запрос.
    bm25_index:
        BM25Index или None (если хотите только dense).
    dense_index:
        DenseIndex или None (если хотите только BM25).
    emb_model:
        Модель эмбеддингов для dense.
    k:
        Сколько кандидатов вернуть.
    alpha:
        Вес BM25 в итоговом скоре (0..1).
        1.0 -> только BM25, 0.0 -> только dense (если оба есть).
    cache:
        EmbeddingCache для query-эмбеддингов.
    model_id:
        Идентификатор модели эмбеддингов.
    tokenizer:
        Токенайзер для BM25 (если None — берётся из bm25_index или дефолтный).

    Возвращает
    ----------
    List[Candidate]
        Отсортированный по hybrid_score список кандидатов.
    """
    if bm25_index is None and dense_index is None:
        raise ValueError("hybrid_candidates: требуется хотя бы один индекс (BM25 или Dense).")

    # Деградация в один канал, если второй не задан
    if bm25_index is not None and dense_index is None:
        return bm25_candidates(query, bm25_index, k=k, tokenizer=tokenizer)

    if dense_index is not None and bm25_index is None:
        return dense_candidates(
            query=query,
            index=dense_index,
            emb_model=emb_model,
            k=k,
            cache=cache,
            model_id=model_id,
        )

    # Здесь оба индекса есть
    assert bm25_index is not None and dense_index is not None

    # Проверим, что размеры коллекции совпадают
    n_bm25 = bm25_index.n_docs
    n_dense = dense_index.n_docs
    if n_bm25 != n_dense:
        raise ValueError(
            f"hybrid_candidates: n_docs BM25 ({n_bm25}) != n_docs Dense ({n_dense}). "
            "Убедись, что индексы строились по одному и тому же списку чанков."
        )

    n_docs = n_bm25
    if n_docs == 0 or k <= 0:
        return []

    # Считаем полные скоры
    bm25_scores = _bm25_scores_for_query(
        query=query,
        index=bm25_index,
        tokenizer=tokenizer,
    )  # shape = (N,)

    dense_scores = _dense_scores_for_query(
        query=query,
        index=dense_index,
        emb_model=emb_model,
        cache=cache,
        model_id=model_id,
    )  # shape = (N,)

    # Нормализация
    bm25_norm = _normalize_scores_0_1(bm25_scores)
    dense_norm = _normalize_scores_0_1(dense_scores)

    alpha = float(alpha)
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    hybrid_scores = alpha * bm25_norm + (1.0 - alpha) * dense_norm  # [N]

    top_pairs = _top_k_from_scores(hybrid_scores, k)

    cands: List[Candidate] = []
    for i, s in top_pairs:
        meta: Dict[str, Any] = {}

        # meta из BM25
        if bm25_index.meta is not None and 0 <= i < len(bm25_index.meta):
            meta.update(bm25_index.meta[i])

        # meta из Dense (может дополнять/перекрывать)
        if dense_index.meta is not None and 0 <= i < len(dense_index.meta):
            meta.update(dense_index.meta[i])

        scores_dict: Dict[str, float] = {
            "hybrid": float(s),
            "bm25": float(bm25_scores[i]),
            "dense": float(dense_scores[i]),
        }

        cand = Candidate(
            chunk_idx=i,
            score=float(s),
            scores=scores_dict,
            source="hybrid",
            rank=None,
            meta=meta,
        )
        cands.append(cand)

    # Сортируем по hybrid_score и проставляем rank
    cands = sort_and_rank(cands, descending=True)
    return cands


# =========================
# ОБРАТНАЯ СОВМЕСТИМОСТЬ:
# ФУНКЦИИ, ВОЗВРАЩАЮЩИЕ (idx, score)
# =========================

def bm25_search(
    query: str,
    index: BM25Index,
    k: int = 10,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[Tuple[int, float]]:
    """
    Совместимая обёртка над bm25_candidates: возвращает список (chunk_idx, score).
    """
    cands = bm25_candidates(query=query, index=index, k=k, tokenizer=tokenizer)
    return to_idx_score(cands)


def dense_search(
    query: str,
    index: DenseIndex,
    emb_model: Any,
    k: int = 10,
    cache: Optional[EmbeddingCache] = None,
    model_id: Optional[str] = None,
) -> List[Tuple[int, float]]:
    """
    Совместимая обёртка над dense_candidates: возвращает список (chunk_idx, score).
    """
    cands = dense_candidates(
        query=query,
        index=index,
        emb_model=emb_model,
        k=k,
        cache=cache,
        model_id=model_id,
    )
    return to_idx_score(cands)


def hybrid_retrieve(
    query: str,
    bm25_index: Optional[BM25Index],
    dense_index: Optional[DenseIndex],
    emb_model: Any,
    k: int = 10,
    alpha: float = 0.5,
    cache: Optional[EmbeddingCache] = None,
    model_id: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[Tuple[int, float]]:
    """
    Совместимая обёртка над hybrid_candidates: возвращает список (chunk_idx, score).
    """
    cands = hybrid_candidates(
        query=query,
        bm25_index=bm25_index,
        dense_index=dense_index,
        emb_model=emb_model,
        k=k,
        alpha=alpha,
        cache=cache,
        model_id=model_id,
        tokenizer=tokenizer,
    )
    return to_idx_score(cands)
