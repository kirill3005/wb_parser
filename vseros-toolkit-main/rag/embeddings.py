from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hashlib
import numpy as np

from .utils import ensure_numpy

# tqdm — аккуратно, чтобы не падать, если его нет в окружении
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class EmbeddingCache:
    """
    Простой in-memory кэш эмбеддингов.

    doc_cache:
        Кэш для документов/чанков.
        Ключ: (model_id, normalize, text_hash).
        Значение: np.ndarray shape = (dim,).

    query_cache:
        Кэш для запросов (query-текстов).
        Ключ: (model_id, normalize, text_hash).
        Значение: np.ndarray shape = (dim,).

    max_size:
        Общий лимит записей (doc_cache + query_cache). Если None — без лимита.
        Если лимит исчерпан, новые в кэш не кладём, но эмбеддинги всё равно считаем.
    """

    doc_cache: Dict[Tuple[str, bool, str], np.ndarray] = field(default_factory=dict)
    query_cache: Dict[Tuple[str, bool, str], np.ndarray] = field(default_factory=dict)
    max_size: Optional[int] = None

    def _can_store_more(self) -> bool:
        if self.max_size is None:
            return True
        return (len(self.doc_cache) + len(self.query_cache)) < self.max_size


def _make_key(model_id: str, normalize: bool, text: str) -> Tuple[str, bool, str]:
    """
    Строит ключ для кэша по (model_id, normalize, sha1(text)).
    """
    # text может быть довольно длинным, поэтому берём именно хэш
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (model_id, bool(normalize), h)


def _encode_batch(
    emb_model: Any,
    texts: Sequence[str],
    normalize: bool,
) -> np.ndarray:
    """
    Внутренняя обёртка над emb_model.encode(...) для одного батча.
    Без логов и tqdm — логика логирования живёт на уровне encode_docs_with_cache.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    # Пытаемся вызвать encode с параметрами SentenceTransformers,
    # если не получится — fallback на простой вариант.
    try:
        vecs = emb_model.encode(
            list(texts),
            batch_size=len(texts),
            show_progress_bar=False,
        )
    except TypeError:
        vecs = emb_model.encode(list(texts))

    arr = ensure_numpy(vecs)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms

    return arr


def encode_docs_with_cache(
    texts: Sequence[str],
    emb_model: Any,
    model_id: str,
    cache: Optional[EmbeddingCache] = None,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Считает эмбеддинги для списка текстов, по возможности используя кэш.

    Параметры
    ---------
    texts:
        Список строк (документов/чанков).
    emb_model:
        Модель с методом .encode([...]).
    model_id:
        Идентификатор модели (любая строка, главное быть консистентным).
        Используется как часть ключа кэша.
    cache:
        EmbeddingCache. Если None — кэш не используется.
    batch_size:
        Размер батча для encode (используется только для новых текстов).
    normalize:
        Нормализовать ли эмбеддинги по L2.
    show_progress:
        Если True и установлен tqdm — показывает прогресс-бар по батчам
        (особенно полезно на больших корпусах).
    verbose:
        Если True — печатает полезные логи: размеры, долю кэша, форму матрицы и т.п.

    Возвращает
    ----------
    np.ndarray
        Матрица эмбеддингов shape = (len(texts), dim).
    """
    texts = [t if t is not None else "" for t in texts]
    n_texts = len(texts)

    if n_texts == 0:
        if verbose:
            print("[encode_docs_with_cache] пустой список текстов, возвращаю (0, 0).")
        return np.zeros((0, 0), dtype="float32")

    use_cache = cache is not None and model_id is not None

    if verbose:
        print(
            f"[encode_docs_with_cache] n_texts={n_texts}, "
            f"batch_size={batch_size}, normalize={normalize}, "
            f"use_cache={use_cache}, model_id={model_id!r}"
        )

    # =========================
    # Вариант без кэша
    # =========================
    if not use_cache:
        if verbose:
            print("[encode_docs_with_cache] cache=None или model_id=None — считаем всё без кэша.")

        all_vecs: List[np.ndarray] = []

        iter_range = range(0, n_texts, batch_size)
        if show_progress and tqdm is not None and n_texts > batch_size:
            iter_range = tqdm(iter_range, desc="Encoding docs (no cache)", unit="batch")
        elif show_progress and tqdm is None and verbose:
            print("[encode_docs_with_cache] tqdm не установлен, прогресс-бар отключён.")

        for start in iter_range:
            batch = texts[start : start + batch_size]
            if not batch:
                continue
            arr = _encode_batch(emb_model, batch, normalize=normalize)
            all_vecs.append(arr)

        if not all_vecs:
            if verbose:
                print("[encode_docs_with_cache] не удалось получить ни одного батча, возвращаю (0, 0).")
            return np.zeros((0, 0), dtype="float32")

        emb = np.vstack(all_vecs)
        if verbose:
            print(f"[encode_docs_with_cache] готово (no cache). shape={emb.shape}")
        return emb

    # =========================
    # Вариант с кэшем
    # =========================

    # Сначала определим, какие тексты уже есть в кэше
    keys: List[Tuple[str, bool, str]] = []
    cached_vecs: Dict[int, np.ndarray] = {}
    missing_indices: List[int] = []

    for i, text in enumerate(texts):
        key = _make_key(model_id, normalize, text)
        keys.append(key)
        if key in cache.doc_cache:
            cached_vecs[i] = cache.doc_cache[key]
        else:
            missing_indices.append(i)

    if verbose:
        print(
            f"[encode_docs_with_cache] cache stats: "
            f"total={n_texts}, cached={len(cached_vecs)}, missing={len(missing_indices)}"
        )

    new_vecs: Dict[int, np.ndarray] = {}

    # Для "пропавших" считаем эмбеддинги батчами
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]

        if verbose:
            print(f"[encode_docs_with_cache] считаю эмбеддинги для {len(missing_texts)} новых текстов...")

        all_vecs: List[np.ndarray] = []

        iter_range = range(0, len(missing_texts), batch_size)
        if show_progress and tqdm is not None and len(missing_texts) > batch_size:
            iter_range = tqdm(iter_range, desc="Encoding missing docs", unit="batch")
        elif show_progress and tqdm is None and verbose:
            print("[encode_docs_with_cache] tqdm не установлен, прогресс-бар отключён.")

        for start in iter_range:
            batch = missing_texts[start : start + batch_size]
            if not batch:
                continue
            arr = _encode_batch(emb_model, batch, normalize=normalize)
            all_vecs.append(arr)

        if all_vecs:
            enc_missing = np.vstack(all_vecs)
            if enc_missing.shape[0] != len(missing_indices):
                raise ValueError(
                    f"encode_docs_with_cache: получено {enc_missing.shape[0]} "
                    f"эмбеддингов для {len(missing_indices)} текстов"
                )

            for local_idx, global_idx in enumerate(missing_indices):
                v = enc_missing[local_idx]
                new_vecs[global_idx] = v

                # Сохраняем в кэш, если есть место
                key = keys[global_idx]
                if cache._can_store_more():
                    cache.doc_cache[key] = v

        if verbose and cache.max_size is not None:
            print(
                "[encode_docs_with_cache] doc_cache size после обновления: "
                f"{len(cache.doc_cache)} (max_size={cache.max_size})"
            )

    # Если вообще нет векторов (маловероятно), вернём пустую матрицу
    if not (cached_vecs or new_vecs):
        if verbose:
            print("[encode_docs_with_cache] ни cached_vecs, ни new_vecs — возвращаю (n_texts, 0).")
        return np.zeros((n_texts, 0), dtype="float32")

    # Определяем размерность по любому найденному вектору
    any_vec: Optional[np.ndarray] = None
    for i in range(n_texts):
        if i in new_vecs:
            any_vec = new_vecs[i]
            break
        if i in cached_vecs:
            any_vec = cached_vecs[i]
            break

    if any_vec is None:
        if verbose:
            print("[encode_docs_with_cache] не удалось найти ни одного вектора, возвращаю (n_texts, 0).")
        return np.zeros((n_texts, 0), dtype="float32")

    dim = any_vec.shape[-1]
    embeddings = np.zeros((n_texts, dim), dtype="float32")

    # Собираем итоговую матрицу в исходном порядке
    for i in range(n_texts):
        if i in new_vecs:
            embeddings[i] = new_vecs[i]
        elif i in cached_vecs:
            embeddings[i] = cached_vecs[i]
        else:
            # На всякий случай fallback — не должен срабатывать при нормальном сценарии
            if verbose:
                print(
                    f"[encode_docs_with_cache] WARN: fallback encode для текста {i}, "
                    "он отсутствует и в new_vecs, и в cached_vecs."
                )
            arr = _encode_batch(emb_model, [texts[i]], normalize=normalize)
            if arr.shape[1] != dim:
                raise ValueError(
                    f"encode_docs_with_cache: несовместимая размерность эмбеддинга "
                    f"для текста {i}: ожидали {dim}, получили {arr.shape[1]}"
                )
            embeddings[i] = arr[0]
            key = keys[i]
            if cache._can_store_more():
                cache.doc_cache[key] = arr[0]

    if verbose:
        print(f"[encode_docs_with_cache] готово (cache). shape={embeddings.shape}")

    return embeddings


def encode_query_with_cache(
    query: str,
    emb_model: Any,
    model_id: str,
    cache: Optional[EmbeddingCache] = None,
    normalize: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Считает эмбеддинг для одного запроса (query), по возможности используя кэш.

    Параметры
    ---------
    query:
        Текст запроса.
    emb_model:
        Модель с методом .encode([...]).
    model_id:
        Идентификатор модели (такой же, как при encode_docs_with_cache).
    cache:
        EmbeddingCache. Если None — кэш не используется.
    normalize:
        Нормализовать ли эмбеддинг по L2.
    verbose:
        Если True — печатает, был ли cache hit/miss и размер кэша.

    Возвращает
    ----------
    np.ndarray
        Вектор эмбеддинга shape = (1, dim).
    """
    query = query if query is not None else ""
    use_cache = cache is not None and model_id is not None

    if verbose:
        print(
            f"[encode_query_with_cache] normalize={normalize}, "
            f"use_cache={use_cache}, model_id={model_id!r}"
        )

    # Вариант без кэша
    if not use_cache:
        if verbose:
            print("[encode_query_with_cache] cache=None или model_id=None — считаем без кэша.")
        arr = _encode_batch(emb_model, [query], normalize=normalize)
        if verbose:
            print(f"[encode_query_with_cache] готово (no cache). shape={arr.shape}")
        return arr

    key = _make_key(model_id, normalize, query)

    # cache hit
    if key in cache.query_cache:
        v = cache.query_cache[key]
        if verbose:
            print("[encode_query_with_cache] cache hit для запроса.")
        if v.ndim == 1:
            return v.reshape(1, -1)
        return v

    # cache miss
    if verbose:
        print("[encode_query_with_cache] cache miss для запроса — считаем и кладём в кэш.")

    arr = _encode_batch(emb_model, [query], normalize=normalize)
    vec = arr[0]

    if cache._can_store_more():
        cache.query_cache[key] = vec
        if verbose and cache.max_size is not None:
            print(
                "[encode_query_with_cache] query-эмбеддинг сохранён в кэш. "
                f"query_cache_size={len(cache.query_cache)} (max_size={cache.max_size})"
            )
    elif verbose and cache.max_size is not None:
        print(
            "[encode_query_with_cache] кэш переполнен, query-эмбеддинг не сохранён. "
            f"query_cache_size={len(cache.query_cache)} (max_size={cache.max_size})"
        )

    return arr
