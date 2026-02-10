"""
Hybrid retrieval with domain boosting: after BM25+BGE top-K, boost chunks whose
domain_hint matches query-detected domains, re-sort, normalize confidence to 0-1.
Does not modify BGE/BM25 scoring algorithms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.schema import HybridRetrievalConfig, RetrievalHit, RetrievalResult
from pipeline.retrieval.retrieval_engine import retrieve as engine_retrieve

from ragops.config import (
    METADATA_BOOST_ENABLED,
    METADATA_BOOST_FACTOR,
    RETRIEVAL_CANDIDATE_TOP_K,
    RETRIEVAL_FINAL_TOP_K,
)
from ragops.retrieval.domain_detector import detect_domains

if TYPE_CHECKING:
    from pipeline.embedding.embedding_engine import EmbeddingBackend
    from pipeline.retrieval.retrieval_engine import IndexSnapshot


def _normalize_scores_to_01(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Normalize similarity_score to 0.0-1.0 over the given hits (min-max)."""
    if not hits:
        return hits
    scores = [h.similarity_score for h in hits]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        # Single hit or all ties: treat as max confidence (1.0)
        norm = 1.0
        use_max = True
    else:
        norm = hi - lo
        use_max = False
    new_hits: list[RetrievalHit] = []
    for h in hits:
        if use_max:
            val = 1.0
        else:
            val = (h.similarity_score - lo) / norm if norm else 1.0
        new_hits.append(
            RetrievalHit(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                raw_text=h.raw_text,
                embedding_vector_id=h.embedding_vector_id,
                index_version=h.index_version,
                similarity_score=round(val, 4),
                page_number=h.page_number,
                chapter=h.chapter,
                section=h.section,
                domain_hint=h.domain_hint,
                boosted=h.boosted,
                original_score=h.original_score,
            )
        )
    return new_hits


def hybrid_retrieve(
    query: str,
    index: "IndexSnapshot",
    backend: "EmbeddingBackend",
    final_top_k: int | None = None,
    candidate_top_k: int | None = None,
    hybrid_config: HybridRetrievalConfig | None = None,
    boost_enabled: bool | None = None,
    boost_factor: float | None = None,
) -> tuple[RetrievalResult, list[str]]:
    """
    Run retrieval with optional domain boosting and normalized confidence.

    1. Retrieve top candidate_top_k (default 20) using existing BGE (+ BM25 if hybrid_config).
    2. If boost_enabled: detect_domains(query); for each hit, if domain_hint in query_domains
       then score *= boost_factor and set boosted=True, original_score=previous score.
    3. Re-sort by boosted score descending.
    4. Take top final_top_k (default 5).
    5. Normalize similarity_score to 0.0-1.0 over returned hits.

    Returns:
        (RetrievalResult with final_top_k hits and normalized scores, detected_domains list).
    """
    k_cand = candidate_top_k if candidate_top_k is not None else RETRIEVAL_CANDIDATE_TOP_K
    k_final = final_top_k if final_top_k is not None else RETRIEVAL_FINAL_TOP_K
    do_boost = boost_enabled if boost_enabled is not None else METADATA_BOOST_ENABLED
    factor = boost_factor if boost_factor is not None else METADATA_BOOST_FACTOR

    raw = engine_retrieve(query, index, backend, top_k=k_cand, hybrid_config=hybrid_config)
    query_domains = detect_domains(query)

    if not do_boost or not query_domains:
        # No boosting: just take top final_top_k and normalize
        hits = raw.hits[:k_final]
        hits = _normalize_scores_to_01(hits)
        return (
            RetrievalResult(
                hits=hits,
                index_version=raw.index_version,
                top_k_requested=k_final,
                truncated=len(hits) < k_final,
                corpus_size=raw.corpus_size,
            ),
            query_domains,
        )

    # Apply domain boost and re-sort
    boosted_list: list[tuple[float, RetrievalHit]] = []
    for h in raw.hits:
        score = h.similarity_score
        orig = score
        boosted = False
        if h.domain_hint and h.domain_hint in query_domains:
            score = score * factor
            boosted = True
        boosted_list.append((score, h if not boosted else RetrievalHit(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            raw_text=h.raw_text,
            embedding_vector_id=h.embedding_vector_id,
            index_version=h.index_version,
            similarity_score=score,
            page_number=h.page_number,
            chapter=h.chapter,
            section=h.section,
            domain_hint=h.domain_hint,
            boosted=True,
            original_score=orig,
        )))
    boosted_list.sort(key=lambda x: (-x[0], x[1].chunk_id))
    top_pairs = boosted_list[:k_final]
    hits = [p[1] for p in top_pairs]
    # Normalize to 0-1
    hits = _normalize_scores_to_01(hits)

    return (
        RetrievalResult(
            hits=hits,
            index_version=raw.index_version,
            top_k_requested=k_final,
            truncated=len(hits) < k_final,
            corpus_size=raw.corpus_size,
        ),
        query_domains,
    )
