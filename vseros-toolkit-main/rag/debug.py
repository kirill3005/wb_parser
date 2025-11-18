# rag/debug.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from collections import Counter
import statistics as stats

from .corpus import Corpus
from .candidates import Candidate


# =============================
# 1. Быстрый просмотр ретрива
# =============================

def _order_candidates_for_view(
    candidates: Sequence[Candidate],
) -> List[Candidate]:
    """
    Вспомогательная функция:
    - если у кандидатов проставлен rank — сортируем по нему;
    - иначе сохраняем исходный порядок.
    """
    cand_list = list(candidates)
    if any(c.rank is not None for c in cand_list):
        return sorted(
            cand_list,
            key=lambda c: (c.rank if c.rank is not None else float("inf"))
        )
    return cand_list


def inspect_retrieval(
    query: str,
    candidates: Sequence[Candidate],
    corpus: Corpus,
    top_n: int = 5,
    text_chars: int = 300,
    score_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Возвращает "человеческий" обзор результатов ретрива для одного запроса.

    Для top_n кандидатов:
      - rank, chunk_idx, doc_id, doc_title
      - текущий score и scores по каналам
      - source
      - укороченный текст чанка (text_preview)

    Параметры
    ---------
    query:
        Текст запроса (для контекста).
    candidates:
        Кандидаты (обычно уже после какого-то шага ретрива/ренка).
    corpus:
        Corpus с документами и чанками.
    top_n:
        Сколько кандидатов показать.
    text_chars:
        Сколько символов текста чанка включить в preview.
    score_keys:
        Какие ключи из Candidate.scores явно вытаскивать.
        Если None — берётся объединение всех ключей из scores.

    Возвращает
    ----------
    List[Dict[str, Any]]
        Список словарей с диагностической информацией по кандидатам.
    """
    if not candidates:
        return []

    ordered = _order_candidates_for_view(candidates)
    top_n = min(top_n, len(ordered))
    ordered = ordered[:top_n]

    # если score_keys не заданы — возьмём все ключи, которые вообще встречаются
    if score_keys is None:
        keys_set = set()
        for c in ordered:
            keys_set.update(c.scores.keys())
        score_keys = sorted(keys_set)

    rows: List[Dict[str, Any]] = []

    for i, cand in enumerate(ordered):
        chunk_idx = cand.chunk_idx
        if chunk_idx < 0 or chunk_idx >= len(corpus.chunks):
            # что-то странное, пропустим
            continue

        chunk = corpus.chunks[chunk_idx]
        doc_id = chunk.doc_id
        doc_title = None
        if 0 <= doc_id < len(corpus.documents):
            doc = corpus.documents[doc_id]
            doc_title = doc.title

        text = chunk.text or ""
        preview = text[:text_chars].replace("\n", " ")

        row_scores = {k: cand.scores.get(k) for k in score_keys}

        rows.append(
            {
                "query": query,
                "rank": cand.rank if cand.rank is not None else i,
                "chunk_idx": chunk_idx,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source": cand.source,
                "score": cand.score,
                "scores": row_scores,
                "text_preview": preview,
                "text_len": len(text),
            }
        )

    return rows


def summarize_candidates(
    candidates: Sequence[Candidate],
    k: int = 20,
) -> Dict[str, Any]:
    """
    Краткая сводка по списку кандидатов.

    Считает:
      - количество кандидатов
      - min/median/max по score
      - распределение по source
      - сколько уникальных doc_id в топ-k
      - распределение doc_id в топ-k (если doc_id есть в meta)

    Параметры
    ---------
    candidates:
        Список кандидатов (в любом порядке).
    k:
        Для анализа doc_id смотрим только top-k кандидатов (в текущем порядке,
        либо по rank, если он задан).

    Возвращает
    ----------
    Dict[str, Any]
        Словарь с агрегированной информацией.
    """
    cands = list(candidates)
    n = len(cands)
    if n == 0:
        return {
            "num_candidates": 0,
            "score_min": None,
            "score_max": None,
            "score_median": None,
            "sources_counts": {},
            "unique_doc_ids_in_top_k": 0,
            "doc_id_counts_top_k": {},
        }

    # упорядочим для top-k
    ordered = _order_candidates_for_view(cands)

    scores = [c.score for c in ordered]
    score_min = min(scores)
    score_max = max(scores)
    score_median = stats.median(scores) if len(scores) > 1 else scores[0]

    # распределение по source
    src_counter = Counter(c.source or "None" for c in ordered)

    # анализ doc_id в топ-k
    k = min(k, len(ordered))
    top_k = ordered[:k]
    doc_ids = []
    for c in top_k:
        # предполагаем, что doc_id мог быть положен в meta или доступен только через meta
        if "doc_id" in c.meta and c.meta["doc_id"] is not None:
            doc_ids.append(c.meta["doc_id"])

    doc_id_counts_top_k: Dict[Any, int] = dict(Counter(doc_ids))
    unique_doc_ids_in_top_k = len(doc_id_counts_top_k)

    return {
        "num_candidates": n,
        "score_min": float(score_min),
        "score_max": float(score_max),
        "score_median": float(score_median),
        "sources_counts": dict(src_counter),
        "unique_doc_ids_in_top_k": unique_doc_ids_in_top_k,
        "doc_id_counts_top_k": doc_id_counts_top_k,
    }


# =====================================
# 2. Анализ перехода между этапами
# =====================================

def analyze_stage_transition(
    before: Sequence[Candidate],
    after: Sequence[Candidate],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Анализирует, как меняются позиции кандидатов между двумя стадиями пайплайна.

    Пример:
      - before: кандидаты после hybrid_retrieve
      - after:  кандидаты после rerank_cross_encoder

    В расчёт берутся кандидаты, которые попали в top_n хотя бы на одной стадии.

    Для каждого chunk_idx возвращает:
      - rank_before, rank_after, delta_rank
      - score_before, score_after
      - source_before, source_after

    Параметры
    ---------
    before:
        Кандидаты "до" стадии.
    after:
        Кандидаты "после" стадии.
    top_n:
        Смотрим только top_n по каждой стадии (в текущем порядке / по rank).

    Возвращает
    ----------
    List[Dict[str, Any]]
        Список словарей по кандидату, отсортированный в основном по rank_after.
    """
    before_ord = _order_candidates_for_view(before)
    after_ord = _order_candidates_for_view(after)

    top_n_before = before_ord[: min(top_n, len(before_ord))]
    top_n_after = after_ord[: min(top_n, len(after_ord))]

    # строим мапы chunk_idx -> (rank, score, source)
    before_map: Dict[int, Tuple[int, float, Optional[str]]] = {}
    for i, c in enumerate(top_n_before):
        r = c.rank if c.rank is not None else i
        before_map[c.chunk_idx] = (r, c.score, c.source)

    after_map: Dict[int, Tuple[int, float, Optional[str]]] = {}
    for i, c in enumerate(top_n_after):
        r = c.rank if c.rank is not None else i
        after_map[c.chunk_idx] = (r, c.score, c.source)

    ids = set(before_map.keys()) | set(after_map.keys())
    rows: List[Dict[str, Any]] = []

    for chunk_idx in ids:
        rb: Optional[int]
        sb: Optional[float]
        sb_src: Optional[str]

        ra: Optional[int]
        sa: Optional[float]
        sa_src: Optional[str]

        if chunk_idx in before_map:
            rb, sb, sb_src = before_map[chunk_idx]
        else:
            rb, sb, sb_src = None, None, None

        if chunk_idx in after_map:
            ra, sa, sa_src = after_map[chunk_idx]
        else:
            ra, sa, sa_src = None, None, None

        # delta_rank: положительно, если документ поднялся вверх (улучшение позиции)
        if rb is not None and ra is not None:
            delta_rank = rb - ra
        else:
            delta_rank = None

        rows.append(
            {
                "chunk_idx": chunk_idx,
                "rank_before": rb,
                "rank_after": ra,
                "delta_rank": delta_rank,
                "score_before": float(sb) if sb is not None else None,
                "score_after": float(sa) if sa is not None else None,
                "source_before": sb_src,
                "source_after": sa_src,
            }
        )

    # сортируем по rank_after, затем по rank_before
    def sort_key(row: Dict[str, Any]) -> Tuple[float, float]:
        ra = row["rank_after"]
        rb = row["rank_before"]
        ra_val = float(ra) if ra is not None else float("inf")
        rb_val = float(rb) if rb is not None else float("inf")
        return (ra_val, rb_val)

    rows.sort(key=sort_key)
    return rows


# ==========================
# 3. Метрики ретрива
# ==========================

def recall_at_k(
    candidates: Sequence[Candidate],
    relevant_chunk_ids: Sequence[int],
    k: int,
) -> float:
    """
    Recall@k: доля релевантных чанков, попавших в top-k.

    Если relevant_chunk_ids пуст, возвращает 0.0.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set or k <= 0:
        return 0.0

    top_k = list(candidates)[: min(k, len(candidates))]
    hit_count = sum(1 for c in top_k if c.chunk_idx in rel_set)

    return hit_count / float(len(rel_set))


def hit_at_k(
    candidates: Sequence[Candidate],
    relevant_chunk_ids: Sequence[int],
    k: int,
) -> float:
    """
    Hit@k: есть ли хотя бы один релевантный чанк в top-k.

    Возвращает 1.0, если есть, иначе 0.0.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set or k <= 0:
        return 0.0

    top_k = list(candidates)[: min(k, len(candidates))]
    for c in top_k:
        if c.chunk_idx in rel_set:
            return 1.0
    return 0.0


def mrr(
    candidates: Sequence[Candidate],
    relevant_chunk_ids: Sequence[int],
) -> float:
    """
    Mean Reciprocal Rank для одного запроса.

    Берёт первый релевантный чанк по порядку кандидатов:
      MRR = 1 / rank (rank 1-based).
    Если релевантных нет — 0.0.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return 0.0

    for i, c in enumerate(candidates):
        if c.chunk_idx in rel_set:
            return 1.0 / float(i + 1)

    return 0.0


def eval_retrieval_batch(
    all_candidates: Mapping[str, Sequence[Candidate]],
    all_relevants: Mapping[str, Sequence[int]],
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    """
    Оценивает качество ретрива на батче запросов.

    Параметры
    ---------
    all_candidates:
        Мапа query_id -> список кандидатов.
    all_relevants:
        Мапа query_id -> список релевантных chunk_idx для этого запроса.
    ks:
        Список значений k, для которых считаем recall@k и hit@k.

    Возвращает
    ----------
    Dict[str, Any]
        {
          "num_queries": ...,
          "per_query": {query_id: {...}},
          "macro_avg": {...},
        }
    """
    ks = list(ks)
    common_ids = [qid for qid in all_candidates.keys() if qid in all_relevants]

    per_query: Dict[str, Dict[str, float]] = {}
    agg: Dict[str, float] = {}

    if not common_ids:
        return {
            "num_queries": 0,
            "per_query": per_query,
            "macro_avg": {},
        }

    for qid in common_ids:
        cands = all_candidates[qid]
        rels = all_relevants[qid]

        metrics: Dict[str, float] = {}

        # recall@k и hit@k
        for k in ks:
            r = recall_at_k(cands, rels, k)
            h = hit_at_k(cands, rels, k)
            metrics[f"recall@{k}"] = r
            metrics[f"hit@{k}"] = h

            agg[f"recall@{k}"] = agg.get(f"recall@{k}", 0.0) + r
            agg[f"hit@{k}"] = agg.get(f"hit@{k}", 0.0) + h

        # MRR
        m = mrr(cands, rels)
        metrics["mrr"] = m
        agg["mrr"] = agg.get("mrr", 0.0) + m

        per_query[qid] = metrics

    num_q = float(len(common_ids))
    macro_avg = {name: val / num_q for name, val in agg.items()}

    return {
        "num_queries": len(common_ids),
        "per_query": per_query,
        "macro_avg": macro_avg,
    }


# ==========================
# 4. Профайлинг пайплайна
# ==========================

@dataclass
class RetrievalStats:
    """
    Простая структура для логирования времени и размеров на этапах ретрива.

    Время измеряется в секундах (float), но ты можешь логировать как захочешь.
    """

    time_bm25: Optional[float] = None
    time_dense: Optional[float] = None
    time_hybrid: Optional[float] = None
    time_ce: Optional[float] = None
    time_mmr: Optional[float] = None

    num_candidates_initial: Optional[int] = None
    num_candidates_after_ce: Optional[int] = None
    num_candidates_after_mmr: Optional[int] = None


def log_retrieval_stats(
    query: str,
    stats: RetrievalStats,
) -> Dict[str, Any]:
    """
    Превращает RetrievalStats в словарь, который удобно логировать / складывать в DataFrame.

    Параметры
    ---------
    query:
        Текст или id запроса.
    stats:
        Объект RetrievalStats, заполненный в ноуте/коде.

    Возвращает
    ----------
    Dict[str, Any]
        Плоский словарь с полями времени и размеров.
    """
    times = [
        t
        for t in [
            stats.time_bm25,
            stats.time_dense,
            stats.time_hybrid,
            stats.time_ce,
            stats.time_mmr,
        ]
        if t is not None
    ]
    total_time = sum(times) if times else None

    return {
        "query": query,
        "time_bm25": stats.time_bm25,
        "time_dense": stats.time_dense,
        "time_hybrid": stats.time_hybrid,
        "time_ce": stats.time_ce,
        "time_mmr": stats.time_mmr,
        "time_total": total_time,
        "num_candidates_initial": stats.num_candidates_initial,
        "num_candidates_after_ce": stats.num_candidates_after_ce,
        "num_candidates_after_mmr": stats.num_candidates_after_mmr,
    }

def inspect_and_print(
    query: str,
    candidates: Sequence[Candidate],
    corpus: Corpus,
    top_n: int = 5,
    text_chars: int = 300,
    score_keys: Optional[Sequence[str]] = None,
) -> None:
    """
    Утилита-обёртка: вызывает inspect_retrieval и печатает результат
    в человеко-читаемом виде (для ноутбуков / быстрых проверок).
    """
    rows = inspect_retrieval(
        query=query,
        candidates=candidates,
        corpus=corpus,
        top_n=top_n,
        text_chars=text_chars,
        score_keys=score_keys,
    )

    if not rows:
        print("inspect_and_print: кандидатов нет.")
        return

    print(f"=== RETRIEVAL INSPECT ===")
    print(f"Query: {query}")
    print(f"Top-{len(rows)} candidates\n")

    for row in rows:
        rank = row["rank"]
        chunk_idx = row["chunk_idx"]
        doc_id = row["doc_id"]
        doc_title = row["doc_title"]
        source = row["source"]
        score = row["score"]
        scores = row["scores"]
        text_preview = row["text_preview"]

        print(f"[{rank}] chunk_idx={chunk_idx} | doc_id={doc_id} | title={doc_title!r}")
        print(f"    source={source} | score={score:.4f}")
        if scores:
            scores_str = ", ".join(
                f"{k}={v:.4f}" for k, v in scores.items() if v is not None
            )
            print(f"    scores: {scores_str}")
        print(f"    text: {text_preview!r}")
        print()


def print_candidates_summary(
    candidates: Sequence[Candidate],
    k: int = 20,
) -> None:
    """
    Печатает краткую сводку по кандидатам (см. summarize_candidates).
    """
    summary = summarize_candidates(candidates, k=k)

    print("=== CANDIDATES SUMMARY ===")
    print(f"num_candidates:          {summary['num_candidates']}")
    print(f"score_min / max / median: "
          f"{summary['score_min']} / {summary['score_max']} / {summary['score_median']}")
    print("sources_counts:")
    for src, cnt in summary["sources_counts"].items():
        print(f"  {src}: {cnt}")

    print(f"unique_doc_ids_in_top_{k}: {summary['unique_doc_ids_in_top_k']}")
    print("doc_id_counts_top_k:")
    for doc_id, cnt in summary["doc_id_counts_top_k"].items():
        print(f"  doc_id={doc_id}: {cnt}")


def print_stage_transition(
    before: Sequence[Candidate],
    after: Sequence[Candidate],
    top_n: int = 20,
) -> None:
    """
    Утилита: вызывает analyze_stage_transition и печатает,
    как меняются ранги/скоры между стадиями.
    """
    rows = analyze_stage_transition(before, after, top_n=top_n)

    if not rows:
        print("print_stage_transition: нет общих кандидатов в top_n.")
        return

    print("=== STAGE TRANSITION ===")
    print(f"top_n (по каждой стадии): {top_n}")
    print("chunk_idx | rank_before -> rank_after (Δrank) | score_before -> score_after")

    for row in rows:
        cid = row["chunk_idx"]
        rb = row["rank_before"]
        ra = row["rank_after"]
        dr = row["delta_rank"]
        sb = row["score_before"]
        sa = row["score_after"]

        print(
            f"{cid:8d} | "
            f"{str(rb):>3} -> {str(ra):>3} "
            f"({str(dr):>3}) | "
            f"{str(round(sb, 4) if sb is not None else None):>8} -> "
            f"{str(round(sa, 4) if sa is not None else None):>8}"
        )


def print_eval_retrieval_batch(
    all_candidates: Mapping[str, Sequence[Candidate]],
    all_relevants: Mapping[str, Sequence[int]],
    ks: Sequence[int] = (5, 10, 20),
    show_per_query: bool = False,
) -> None:
    """
    Утилита: вызывает eval_retrieval_batch и печатает macro-метрики,
    а при желании — метрики по каждому запросу.
    """
    res = eval_retrieval_batch(
        all_candidates=all_candidates,
        all_relevants=all_relevants,
        ks=ks,
    )

    num_q = res["num_queries"]
    macro = res["macro_avg"]
    per_q = res["per_query"]

    print("=== RETRIEVAL EVAL BATCH ===")
    print(f"num_queries: {num_q}")
    print("macro_avg:")
    for name, val in macro.items():
        print(f"  {name}: {val:.4f}")

    if show_per_query:
        print("\nper_query:")
        for qid, metrics in per_q.items():
            print(f"- query_id={qid}:")
            for name, val in metrics.items():
                print(f"    {name}: {val:.4f}")
