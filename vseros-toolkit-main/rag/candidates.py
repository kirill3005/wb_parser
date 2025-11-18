from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class Candidate:
    """
    Унифицированный формат кандидата для RAG-пайплайна.

    chunk_idx:
        Индекс чанка в corpus.chunks и в raw_docs индексов (BM25/Dense и т.п.).
    score:
        Текущий "главный" скор, по которому сортируем на данном этапе.
    scores:
        Все известные скоры по каналам, например:
            {
                "bm25":  ...,
                "dense": ...,
                "hybrid": ...,
                "ce":    ...,
                "mmr":   ...,
            }
    source:
        Последний источник/этап, который обновил поле score
        (например: "bm25", "dense", "hybrid", "ce_rerank", "mmr").
    rank:
        Позиция кандидата после сортировки на текущем этапе (0-based).
    meta:
        Дополнительная метаинформация:
        doc_id, флаги, тип секции и всё, что удобно хранить рядом.
    """

    chunk_idx: int
    score: float
    scores: Dict[str, float] = field(default_factory=dict)
    source: Optional[str] = None
    rank: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def set_score(
        self,
        value: float,
        name: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Обновить основной score и, при желании, записать его под именем name
        в scores, а также обновить source.
        """
        v = float(value)
        self.score = v
        if name is not None:
            self.scores[name] = v
        if source is not None:
            self.source = source


def from_idx_score(
    pairs: Sequence[Tuple[int, float]],
    score_name: str,
    source: Optional[str] = None,
    base_meta: Optional[Mapping[int, Dict[str, Any]]] = None,
    assign_rank: bool = True,
) -> List[Candidate]:
    """
    Превращает список (chunk_idx, score) в список Candidate.

    Параметры
    ---------
    pairs:
        Список (chunk_idx, score), например результат bm25_search / dense_search / hybrid_retrieve.
    score_name:
        Имя канала, под которым скор попадёт в Candidate.scores (например "bm25", "dense", "hybrid").
    source:
        Значение для поля Candidate.source. Если None — берётся score_name.
    base_meta:
        Необязательная мапа chunk_idx -> meta-словарь, который будет положен в Candidate.meta.
        Удобно, если заранее есть doc_id или другая инфа по чанкам.
    assign_rank:
        Проставлять ли rank по порядку (0, 1, 2, ...).

    Возвращает
    ----------
    List[Candidate]
        Список кандидатов, упорядоченный так же, как pairs.
    """
    result: List[Candidate] = []
    src = source or score_name

    for i, (chunk_idx, score) in enumerate(pairs):
        meta: Dict[str, Any] = {}
        if base_meta is not None:
            m = base_meta.get(chunk_idx)
            if m is not None:
                meta = dict(m)  # копия, чтобы не трогать исходный словарь

        cand = Candidate(
            chunk_idx=int(chunk_idx),
            score=float(score),
            scores={score_name: float(score)},
            source=src,
            rank=i if assign_rank else None,
            meta=meta,
        )
        result.append(cand)

    return result


def to_idx_score(
    candidates: Sequence[Candidate],
) -> List[Tuple[int, float]]:
    """
    Превращает список Candidate обратно в список (chunk_idx, score).

    Удобно, если нужно использовать низкоуровневые функции,
    которые ещё работают с парами, а не с Candidate.
    """
    return [(int(c.chunk_idx), float(c.score)) for c in candidates]


def sort_and_rank(
    candidates: Sequence[Candidate],
    descending: bool = True,
) -> List[Candidate]:
    """
    Сортирует кандидатов по текущему полю score и проставляет rank по порядку.

    ВАЖНО:
        - Возвращает НОВЫЙ список Candidate, но сами объекты внутри те же самые.
        - Поле rank обновляется в соответствии с новой позицией.

    Параметры
    ---------
    candidates:
        Последовательность Candidate.
    descending:
        Если True — сортировка по убыванию score (больший score = выше).

    Возвращает
    ----------
    List[Candidate]
        Новый список, отсортированный по score.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=descending)
    for i, cand in enumerate(sorted_cands):
        cand.rank = i
    return sorted_cands


def assign_ranks_inplace(
    candidates: Sequence[Candidate],
) -> None:
    """
    Проставляет rank в соответствии с текущим порядком в последовательности.

    Ничего не сортирует — просто ставит:
        candidates[0].rank = 0,
        candidates[1].rank = 1,
        ...
    """
    for i, cand in enumerate(candidates):
        cand.rank = i
