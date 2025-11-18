from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Callable


@dataclass
class Document:
    """
    Один исходный документ в корпусе.

    doc_id:
        Целочисленный идентификатор документа внутри корпуса.
        По умолчанию совпадает с позицией в Corpus.documents.
    text:
        Полный текст документа.
    title:
        Необязательный заголовок документа.
    meta:
        Произвольная метаинформация (источник, путь к файлу, дата и т.п.).
    """
    doc_id: int
    text: str
    title: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    """
    Текстовый чанк, полученный из документа.

    chunk_idx:
        Глобальный индекс чанка в корпусе. Важно: должен совпадать
        с индексом в Corpus.chunks.
    doc_id:
        Идентификатор документа, из которого взят чанк.
    text:
        Текст чанка.
    start_char, end_char:
        Диапазон [start_char, end_char) в исходном Document.text.
    meta:
        Доп. метаинформация (страница, тип секции, и т.п.).
    """
    chunk_idx: int
    doc_id: int
    text: str
    start_char: int
    end_char: int
    meta: Optional[Dict[str, Any]] = None

@dataclass
class LocalChunk:
    """
    Чанк внутри одного документа, без глобального chunk_idx.
    """
    text: str
    start_char: int
    end_char: int
    meta: Optional[Dict[str, Any]] = None


@dataclass
class Corpus:
    """
    Корпус документов и чанков.

    documents:
        Список исходных документов.
    chunks:
        Список чанков, обычно результат add_chunks_to_corpus.
        Инвариант: chunks[i].chunk_idx == i.
    """
    documents: List[Document]
    chunks: List[Chunk]


def build_corpus_from_texts(
    texts: Sequence[str],
    titles: Optional[Sequence[Optional[str]]] = None,
    doc_meta: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> Corpus:
    """
    Строит Corpus только с документами (без чанков) из списка текстов.

    Параметры
    ---------
    texts:
        Список строк, каждая — отдельный документ.
    titles:
        Необязательный список заголовков той же длины, что texts.
    doc_meta:
        Необязательный список словарей метаинформации той же длины.

    Возвращает
    ----------
    Corpus
        Corpus с заполненным .documents и пустым .chunks.
    """
    texts = list(texts)

    if titles is not None:
        titles = list(titles)
        if len(titles) != len(texts):
            raise ValueError("build_corpus_from_texts: len(titles) != len(texts)")
    if doc_meta is not None:
        doc_meta = list(doc_meta)
        if len(doc_meta) != len(texts):
            raise ValueError("build_corpus_from_texts: len(doc_meta) != len(texts)")

    documents: List[Document] = []

    for i, text in enumerate(texts):
        title_i: Optional[str] = None
        meta_i: Optional[Dict[str, Any]] = None

        if titles is not None:
            title_i = titles[i]
        if doc_meta is not None:
            meta_i = doc_meta[i]

        documents.append(
            Document(
                doc_id=i,
                text=text if text is not None else "",
                title=title_i,
                meta=meta_i,
            )
        )

    return Corpus(documents=documents, chunks=[])


def add_chunks_to_corpus(
    corpus: Corpus,
    chunker: Callable[[Document], Sequence[LocalChunk]],
) -> Corpus:
    """
    Применяет переданный chunker ко всем документам и заполняет corpus.chunks.

    chunk_idx выставляется глобально: 0..len(chunks)-1.
    """
    chunks: List[Chunk] = []
    next_chunk_idx = 0

    for doc in corpus.documents:
        local_chunks = chunker(doc)  # <- тут твоя кастомная логика

        for lc in local_chunks:
            chunks.append(
                Chunk(
                    chunk_idx=next_chunk_idx,
                    doc_id=doc.doc_id,
                    text=lc.text,
                    start_char=lc.start_char,
                    end_char=lc.end_char,
                    meta=lc.meta,
                )
            )
            next_chunk_idx += 1

    corpus.chunks = chunks
    return corpus



def get_chunk_texts(corpus: Corpus) -> List[str]:
    """
    Возвращает список текстов всех чанков в корпусе.

    Удобно использовать как вход для build_bm25_index / build_dense_index.
    """
    return [chunk.text for chunk in corpus.chunks]


def get_chunk(corpus: Corpus, chunk_idx: int) -> Chunk:
    """
    Безопасно возвращает чанк по его chunk_idx.

    Предполагается, что chunk_idx совпадает с индексом в corpus.chunks,
    как гарантирует add_chunks_to_corpus.
    """
    if chunk_idx < 0 or chunk_idx >= len(corpus.chunks):
        raise IndexError(f"chunk_idx {chunk_idx} вне диапазона [0, {len(corpus.chunks)})")
    return corpus.chunks[chunk_idx]


def get_document(corpus: Corpus, doc_id: int) -> Document:
    """
    Возвращает Document по doc_id.

    По умолчанию doc_id == индексу в corpus.documents (так их создаёт
    build_corpus_from_texts).
    """
    if doc_id < 0 or doc_id >= len(corpus.documents):
        raise IndexError(f"doc_id {doc_id} вне диапазона [0, {len(corpus.documents)})")
    return corpus.documents[doc_id]
