from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import re

import numpy as np
from scipy.sparse import csr_matrix

from .embeddings import EmbeddingCache, encode_docs_with_cache

# tqdm — аккуратно, чтобы не падать, если его нет
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# =========================
# BM25 INDEX
# =========================

def _default_tokenizer(text: str) -> List[str]:
    """
    Простейший токенайзер по словам: [a-zA-Z0-9_]+, в lower-case.

    Можно спокойно заменить на свой и передать в build_bm25_index.
    """
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Index:
    """
    Индекс для BM25-поиска по коллекции чанков/документов.

    Поля
    ----
    vocab:
        Словарь token -> term_id.
    idf:
        Массив IDF-значений shape = (V,), где V = |vocab|.
    doc_len:
        Длины документов в токенах shape = (N,).
    avg_doc_len:
        Средняя длина документа.
    term_freqs:
        CSR-матрица частот по (doc, term_id) shape = (N, V).
    raw_docs:
        Оригинальные строки документов/чанков.
    meta:
        Доп. мета по документу/чанку, по индексу.
    k1, b:
        Параметры BM25.
    tokenizer:
        Токенайзер, который использовался при построении индекса
        (можно использовать по умолчанию в bm25_search).
    """

    vocab: Dict[str, int]
    idf: np.ndarray
    doc_len: np.ndarray
    avg_doc_len: float
    term_freqs: csr_matrix
    raw_docs: List[str]
    meta: Optional[List[Dict[str, Any]]] = None
    k1: float = 1.5
    b: float = 0.75
    tokenizer: Optional[Callable[[str], List[str]]] = None

    @property
    def n_docs(self) -> int:
        return len(self.raw_docs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_bm25_index(
    docs: Sequence[str],
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    meta: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    verbose: bool = False,
    show_progress: bool = False,
) -> BM25Index:
    """
    Строит BM25-индекс по коллекции текстов.

    Параметры
    ---------
    docs:
        Список строк (документы или чанки).
    tokenizer:
        Функция: str -> List[str]. Если None — используется _default_tokenizer.
    k1, b:
        Параметры BM25.
    meta:
        Необязательный список метадаты длины len(docs). Каждый элемент либо dict, либо None.
    verbose:
        Если True — печатает краткие логи (размеры, статистики).
    show_progress:
        Если True и установлен tqdm — показывает прогресс по токенизации и сборке tf/df.

    Возвращает
    ----------
    BM25Index
    """
    docs_list = [d if d is not None else "" for d in docs]
    n_docs = len(docs_list)

    if tokenizer is None:
        tokenizer = _default_tokenizer

    if verbose:
        print(
            f"[build_bm25_index] n_docs={n_docs}, k1={k1}, b={b}, "
            f"meta_present={meta is not None}"
        )

    if n_docs == 0:
        # Пустой индекс — нормальный кейс
        if verbose:
            print("[build_bm25_index] пустой список документов, возвращаю пустой индекс.")
        empty_arr = np.zeros((0,), dtype="float32")
        empty_mat = csr_matrix((0, 0), dtype="float32")
        return BM25Index(
            vocab={},
            idf=empty_arr,
            doc_len=empty_arr,
            avg_doc_len=0.0,
            term_freqs=empty_mat,
            raw_docs=[],
            meta=[] if meta is not None else None,
            k1=float(k1),
            b=float(b),
            tokenizer=tokenizer,
        )

    # 1. Токенизация
    tokenized: List[List[str]] = []

    iter_range = range(n_docs)
    if show_progress and tqdm is not None:
        iter_range = tqdm(iter_range, desc="BM25: tokenizing docs", unit="doc")
    elif show_progress and tqdm is None and verbose:
        print("[build_bm25_index] tqdm не установлен, прогресс-бар отключён.")

    for i in iter_range:
        text = docs_list[i]
        tokens = tokenizer(text)
        tokenized.append(tokens)

    if verbose:
        total_tokens = sum(len(toks) for toks in tokenized)
        avg_tokens = total_tokens / float(n_docs) if n_docs > 0 else 0.0
        print(
            f"[build_bm25_index] токенизация завершена: "
            f"total_tokens={total_tokens}, avg_tokens_per_doc={avg_tokens:.2f}"
        )

    # 2. Словарь token -> term_id
    vocab: Dict[str, int] = {}
    for tokens in tokenized:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    vocab_size = len(vocab)

    if verbose:
        print(f"[build_bm25_index] vocab_size={vocab_size}")

    # 3. Длины документов
    doc_len = np.array([len(tokens) for tokens in tokenized], dtype="float32")
    avg_doc_len = float(doc_len.mean()) if n_docs > 0 else 0.0

    if verbose:
        print(
            f"[build_bm25_index] avg_doc_len={avg_doc_len:.2f}, "
            f"min_len={float(doc_len.min()) if n_docs > 0 else 0.0}, "
            f"max_len={float(doc_len.max()) if n_docs > 0 else 0.0}"
        )

    # 4. Частоты и document frequency
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    df = np.zeros(vocab_size, dtype="int32")  # document frequency per term

    iter_docs = range(n_docs)
    if show_progress and tqdm is not None:
        iter_docs = tqdm(iter_docs, desc="BM25: building tf/df", unit="doc")

    for doc_id in iter_docs:
        tokens = tokenized[doc_id]
        if not tokens:
            continue

        # считаем tf в документе
        counts: Dict[int, int] = {}
        for tok in tokens:
            term_id = vocab[tok]
            counts[term_id] = counts.get(term_id, 0) + 1

        # записываем в COO-списки
        for term_id, freq in counts.items():
            rows.append(doc_id)
            cols.append(term_id)
            data.append(freq)

        # обновляем df
        for term_id in counts.keys():
            df[term_id] += 1

    term_freqs = csr_matrix(
        (np.array(data, dtype="float32"), (rows, cols)),
        shape=(n_docs, vocab_size),
        dtype="float32",
    )

    if verbose:
        nnz = term_freqs.nnz
        print(
            f"[build_bm25_index] term_freqs shape={term_freqs.shape}, nnz={nnz}, "
            f"avg_terms_per_doc={nnz / float(n_docs) if n_docs > 0 else 0.0:.2f}"
        )

    # 5. IDF (классическая формула BM25)
    df_float = df.astype("float32")
    # добавляем 0.5 для устойчивости при df=0/df=N
    idf = np.log(1.0 + (n_docs - df_float + 0.5) / (df_float + 0.5)).astype("float32")

    # 6. meta-выравнивание
    meta_list: Optional[List[Dict[str, Any]]]
    if meta is not None:
        meta_list = []
        meta_seq = list(meta)
        if len(meta_seq) != n_docs:
            raise ValueError(
                f"build_bm25_index: len(meta)={len(meta_seq)} "
                f"не совпадает с количеством документов={n_docs}"
            )
        for m in meta_seq:
            meta_list.append(dict(m) if m is not None else {})
        if verbose:
            print(f"[build_bm25_index] meta прикреплена к {len(meta_list)} документам.")
    else:
        meta_list = None
        if verbose:
            print("[build_bm25_index] meta не передана (None).")

    if verbose:
        print("[build_bm25_index] BM25Index готов.")

    return BM25Index(
        vocab=vocab,
        idf=idf,
        doc_len=doc_len,
        avg_doc_len=avg_doc_len,
        term_freqs=term_freqs,
        raw_docs=docs_list,
        meta=meta_list,
        k1=float(k1),
        b=float(b),
        tokenizer=tokenizer,
    )


# =========================
# DENSE INDEX
# =========================

@dataclass
class DenseIndex:
    """
    Простой dense-индекс: эмбеддинги + сырые тексты.

    Поля
    ----
    embeddings:
        Матрица эмбеддингов shape = (N, D), float32.
    raw_docs:
        Список текстов (документов/чанков), по индексу совпадает с embeddings.
    meta:
        Необязательный список метадаты длины N.
    metric:
        Какой метрикой предполагается пользоваться при поиске:
        "ip" (inner product) или "cosine" — по сути одно и то же при normalize=True.
    model_id:
        Идентификатор модели эмбеддингов (для кэша/логов).
    normalize:
        Были ли эмбеддинги нормализованы по L2.
    """

    embeddings: np.ndarray
    raw_docs: List[str]
    meta: Optional[List[Dict[str, Any]]] = None
    metric: str = "ip"
    model_id: Optional[str] = None
    normalize: bool = True

    @property
    def n_docs(self) -> int:
        return self.embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1] if self.embeddings.ndim == 2 else 0


def build_dense_index(
    docs: Sequence[str],
    emb_model: Any,
    batch_size: int = 32,
    normalize: bool = True,
    meta: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    metric: str = "ip",
    cache: Optional[EmbeddingCache] = None,
    model_id: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = False,
) -> DenseIndex:
    """
    Строит dense-индекс поверх эмбеддингов документов/чанков.

    Параметры
    ---------
    docs:
        Список строк.
    emb_model:
        Модель с методом .encode([...]).
    batch_size:
        Размер батча для encode (используется внутри encode_docs_with_cache).
    normalize:
        Нормализовать ли эмбеддинги по L2 (рекомендуется=True, тогда dot = cosine).
    meta:
        Необязательный список метадаты длины len(docs).
    metric:
        Строка, описывающая предполагаемую метрику ("ip" или "cosine").
    cache:
        EmbeddingCache, если хочешь переиспользовать эмбеддинги между индексами/запросами.
    model_id:
        Идентификатор модели эмбеддингов (любая строка, но должна быть консистентной).
        Если None — encode_docs_with_cache просто не будет использовать кэш (использует "default").
    verbose:
        Если True — печатает логи о размерностях, использовании кэша и т.п.
    show_progress:
        Если True — прокидывает флаг в encode_docs_with_cache, чтобы видеть tqdm по батчам.

    Возвращает
    ----------
    DenseIndex
    """
    docs_list = [d if d is not None else "" for d in docs]
    n_docs = len(docs_list)

    if verbose:
        print(
            f"[build_dense_index] n_docs={n_docs}, batch_size={batch_size}, "
            f"normalize={normalize}, metric={metric}, model_id={model_id!r}, "
            f"use_cache={cache is not None}"
        )

    if n_docs == 0:
        if verbose:
            print("[build_dense_index] пустой список документов, возвращаю пустой DenseIndex.")
        empty_emb = np.zeros((0, 0), dtype="float32")
        return DenseIndex(
            embeddings=empty_emb,
            raw_docs=[],
            meta=[] if meta is not None else None,
            metric=metric,
            model_id=model_id,
            normalize=normalize,
        )

    eff_model_id = model_id if model_id is not None else "default"

    # Эмбеддинги с кэшом (или без него, если cache=None)
    emb = encode_docs_with_cache(
        texts=docs_list,
        emb_model=emb_model,
        model_id=eff_model_id,
        cache=cache,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
        verbose=verbose,
    )
    # emb уже float32 и 2D

    if emb.shape[0] != n_docs:
        raise ValueError(
            f"build_dense_index: количество эмбеддингов ({emb.shape[0]}) "
            f"не совпадает с количеством документов ({n_docs})"
        )

    if verbose:
        print(f"[build_dense_index] embeddings shape={emb.shape}")

    # meta-выравнивание
    meta_list: Optional[List[Dict[str, Any]]]
    if meta is not None:
        meta_seq = list(meta)
        if len(meta_seq) != n_docs:
            raise ValueError(
                f"build_dense_index: len(meta)={len(meta_seq)} "
                f"не совпадает с количеством документов={n_docs}"
            )
        meta_list = []
        for m in meta_seq:
            meta_list.append(dict(m) if m is not None else {})
        if verbose:
            print(f"[build_dense_index] meta прикреплена к {len(meta_list)} документам.")
    else:
        meta_list = None
        if verbose:
            print("[build_dense_index] meta не передана (None).")

    if verbose:
        print("[build_dense_index] DenseIndex готов.")

    return DenseIndex(
        embeddings=emb,
        raw_docs=docs_list,
        meta=meta_list,
        metric=metric,
        model_id=model_id,
        normalize=normalize,
    )
