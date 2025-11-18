from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .candidates import Candidate


# Тип функции подсчёта токенов
CountTokensFn = Callable[[str], int]


def simple_token_counter(text: str) -> int:
    """
    Простой подсчёт "токенов" по количеству слов.

    Это грубая аппроксимация. Для реальной модели можно передать
    свой count_tokens, который считает токены через HF-токенайзер
    или tiktoken.
    """
    if not text:
        return 0
    return len(text.split())


@dataclass
class ContextItem:
    """
    Один фрагмент контекста, который попал в итоговое окно.

    chunk_idx:
        Индекс чанка (совпадает с индексом в списке docs и corpus.chunks).
    text:
        Текст чанка (возможно, уже усечённый под токен-бюджет).
    tokens:
        Число токенов в этом тексте (по выбранному count_tokens).
    score:
        Итоговый скор кандидата, по которому он был отсортирован
        при отборе в окно (обычно после всех стадий: hybrid/CE/MMR).
    scores:
        Все известные канальные скоры кандидата (bm25/dense/hybrid/ce/mmr и т.п.).
    meta:
        Метаинформация, пришедшая из кандидата (doc_id, тип секции и т.п.).
    """

    chunk_idx: int
    text: str
    tokens: int
    score: float
    scores: Dict[str, float]
    meta: Dict[str, Any]


def _truncate_by_words(text: str, max_tokens: int) -> tuple[str, int]:
    """
    Простое усечение текста по количеству "токенов"=слов.

    Используется только в том случае, когда не передан кастомный
    count_tokens (т.е. мы знаем, что simple_token_counter — это split()).
    """
    if max_tokens <= 0:
        return "", 0

    words = text.split()
    if len(words) <= max_tokens:
        return text, len(words)

    truncated = " ".join(words[:max_tokens])
    return truncated, max_tokens


def build_context_window(
    query: str,
    docs: Sequence[str],
    candidates: Sequence[Candidate],
    max_tokens: int,
    k_final: Optional[int] = None,
    count_tokens: Optional[CountTokensFn] = None,
    join_with: str = "\n\n",
) -> Dict[str, Any]:
    """
    Собирает итоговое контекстное окно для LLM из уже отранжированных кандидатов.

    ВАЖНО:
      - Эта функция НИЧЕГО не знает про BM25/dense/CE/MMR.
      - Она не запускает retriever / rerank / MMR — только отбирает
        уже готовых кандидатов в рамках токен-бюджета.

    Типичный пайплайн:
        cands = hybrid_candidates(...)
        cands = rerank_cross_encoder(...)
        cands = apply_mmr(...)
        window = build_context_window(
            query=query,
            docs=get_chunk_texts(corpus),
            candidates=cands,
            max_tokens=1024,
            k_final=8,
        )
        context_str = window["context"]

    Параметры
    ---------
    query:
        Текст запроса (здесь используется только для контекста, на отбор не влияет).
        Оставлен в сигнатуре на будущее (например, для логов или более умного окна).
    docs:
        Список текстов чанков. docs[i] должен соответствовать chunk_idx=i
        в Candidate.
    candidates:
        Список кандидатов (обычно уже после всех стадий ретрива/ренка),
        отсортированный по убыванию score.
    max_tokens:
        Максимальное количество токенов на контекст (приблизительно).
        Считается по функции count_tokens (или simple_token_counter).
    k_final:
        Максимальное число чанков в контексте. Если None — ограничиваем
        только токен-бюджетом.
    count_tokens:
        Функция, считающая токены в строке. Если None — используем
        simple_token_counter (по словам).
    join_with:
        Разделитель между чанками в финальной строке контекста.

    Возвращает
    ----------
    Dict[str, Any]
        {
          "context": str,              # собранный текст контекста
          "items": List[ContextItem],  # подробная инфа по каждому включённому чанку
          "total_tokens": int,         # сколько токенов ушло на контекст
        }
    """
    if max_tokens <= 0 or not candidates or not docs:
        return {
            "context": "",
            "items": [],
            "total_tokens": 0,
        }

    token_fn: CountTokensFn = count_tokens or simple_token_counter

    seen_chunk_ids: set[int] = set()
    items: List[ContextItem] = []
    total_tokens = 0

    # Гарантируем, что работаем по текущему порядку кандидатов
    for cand in candidates:
        # Ограничение по количеству чанков
        if k_final is not None and len(items) >= k_final:
            break

        idx = cand.chunk_idx
        if idx < 0 or idx >= len(docs):
            continue
        if idx in seen_chunk_ids:
            continue

        raw_text = docs[idx] if docs[idx] is not None else ""
        if not raw_text.strip():
            continue

        # Считаем токены текущего чанка
        tokens = token_fn(raw_text)

        # Если прибавление этого чанка сильно превышает бюджет:
        if total_tokens + tokens > max_tokens:
            # Если это НЕ первый чанк — просто останавливаемся
            if items:
                break

            # Если это первый чанк — попробуем усечь его по словам,
            # но только если используется simple_token_counter.
            if count_tokens is None:
                truncated_text, truncated_tokens = _truncate_by_words(raw_text, max_tokens)
                if truncated_tokens <= 0:
                    # вообще не смогли ничего положить — выходим
                    return {
                        "context": "",
                        "items": [],
                        "total_tokens": 0,
                    }

                item = ContextItem(
                    chunk_idx=idx,
                    text=truncated_text,
                    tokens=truncated_tokens,
                    score=float(cand.score),
                    scores=dict(cand.scores),
                    meta=dict(cand.meta),
                )
                items.append(item)
                total_tokens += truncated_tokens
                seen_chunk_ids.add(idx)
                break
            else:
                # Пользователь передал кастомный count_tokens — не лезем в усечение,
                # просто кладём целиком первый чанк, даже если он чуть превышает бюджет.
                item = ContextItem(
                    chunk_idx=idx,
                    text=raw_text,
                    tokens=tokens,
                    score=float(cand.score),
                    scores=dict(cand.scores),
                    meta=dict(cand.meta),
                )
                items.append(item)
                total_tokens += tokens
                seen_chunk_ids.add(idx)
                break

        # Обычный кейс: чанк влезает в бюджет
        item = ContextItem(
            chunk_idx=idx,
            text=raw_text,
            tokens=tokens,
            score=float(cand.score),
            scores=dict(cand.scores),
            meta=dict(cand.meta),
        )
        items.append(item)
        total_tokens += tokens
        seen_chunk_ids.add(idx)

    if not items:
        return {
            "context": "",
            "items": [],
            "total_tokens": 0,
        }

    context_str = join_with.join(item.text for item in items)

    return {
        "context": context_str,
        "items": items,
        "total_tokens": total_tokens,
    }
