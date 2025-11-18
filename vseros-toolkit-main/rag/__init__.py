# rag/__init__.py

from .chunkers import BaseChunker, CharChunker, Chunker
from .indices import (
    BM25Index,
    DenseIndex,
    build_bm25_index,
    build_dense_index,
)

from .retrieval import (
    bm25_search,
    dense_search,
    hybrid_retrieve,
)

from .rerank import (
    rerank_cross_encoder,
    apply_mmr,
)

from .context import (
    build_context_window,
)

from .candidates import (
    Candidate,
    assign_ranks_inplace,
    from_idx_score,
    sort_and_rank,
    to_idx_score,
)

from .corpus import (
    Document,
    Chunk,
    Corpus,
    build_corpus_from_texts,
    add_chunks_to_corpus,
    get_chunk_texts,
    get_chunk,
    get_document,
)

# Опционально: токены/дебаг, если решишь их делать
# from .tokens import count_tokens_simple, truncate_text_to_tokens
# from .debug import inspect_retrieval, log_retrieval_stats

__all__ = [
    "BaseChunker",
    "CharChunker",
    "Chunker",
    "BM25Index",
    "DenseIndex",
    "build_bm25_index",
    "build_dense_index",
    "bm25_search",
    "dense_search",
    "hybrid_retrieve",
    "rerank_cross_encoder",
    "apply_mmr",
    "build_context_window",
    "Candidate",
    "from_idx_score",
    "to_idx_score",
    "sort_and_rank",
    "assign_ranks_inplace",
    "Document",
    "Chunk",
    "Corpus",
    "build_corpus_from_texts",
    "add_chunks_to_corpus",
    "get_chunk_texts",
    "get_chunk",
    "get_document",
    # "count_tokens_simple",
    # "truncate_text_to_tokens",
    # "inspect_retrieval",
    # "log_retrieval_stats",
]
