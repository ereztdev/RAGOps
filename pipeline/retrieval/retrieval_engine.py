"""
Retrieval engine: query embedding, top-k similarity search, trace payload and logging.
Deterministic for fixed query, index_version, and top_k (brute-force search; no ANN).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from core.schema import (
    Chunk,
    Document,
    Embedding,
    HybridRetrievalConfig,
    RetrievalHit,
    RetrievalResult,
)
from pipeline.embedding.embedding_engine import EmbeddingBackend


logger = logging.getLogger("ragops.retrieval")

# Deterministic float format for serialization (no scientific notation drift).
_FLOAT_FORMAT = ".17g"


# ---------------------------------------------------------------------------
# In-memory index (no vector storage layer yet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexEntry:
    """One index row: chunk identity, text, vector, and optional metadata for domain boosting."""
    chunk_id: str
    document_id: str
    raw_text: str
    embedding_vector_id: str
    vector: tuple[float, ...]
    page_number: int | None = None
    chapter: str | None = None
    section: str | None = None
    domain_hint: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON. Vector as deterministic float strings for byte-identical output."""
        out: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "raw_text": self.raw_text,
            "embedding_vector_id": self.embedding_vector_id,
            "vector": [format(v, _FLOAT_FORMAT) for v in self.vector],
        }
        if self.page_number is not None:
            out["page_number"] = self.page_number
        if self.chapter is not None:
            out["chapter"] = self.chapter
        if self.section is not None:
            out["section"] = self.section
        if self.domain_hint is not None:
            out["domain_hint"] = self.domain_hint
        return out

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> IndexEntry:
        """Deserialize from dict. Fails loudly on missing or invalid fields. Optional metadata for backward compat."""
        required = ("chunk_id", "document_id", "raw_text", "embedding_vector_id", "vector")
        for key in required:
            if key not in data:
                raise ValueError(f"IndexEntry missing required field: {key!r}")
        chunk_id = data["chunk_id"]
        document_id = data["document_id"]
        raw_text = data["raw_text"]
        embedding_vector_id = data["embedding_vector_id"]
        raw_vector = data["vector"]
        if not isinstance(chunk_id, str):
            raise ValueError(f"IndexEntry chunk_id must be str, got {type(chunk_id).__name__}")
        if not isinstance(document_id, str):
            raise ValueError(f"IndexEntry document_id must be str, got {type(document_id).__name__}")
        if not isinstance(raw_text, str):
            raise ValueError(f"IndexEntry raw_text must be str, got {type(raw_text).__name__}")
        if not isinstance(embedding_vector_id, str):
            raise ValueError(f"IndexEntry embedding_vector_id must be str, got {type(embedding_vector_id).__name__}")
        if not isinstance(raw_vector, list):
            raise ValueError(f"IndexEntry vector must be list, got {type(raw_vector).__name__}")
        try:
            vector = tuple(float(x) for x in raw_vector)
        except (TypeError, ValueError) as e:
            raise ValueError(f"IndexEntry vector must be list of numbers: {e}") from e
        page_number = data.get("page_number")
        if page_number is not None:
            page_number = int(page_number)
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            raw_text=raw_text,
            embedding_vector_id=embedding_vector_id,
            vector=vector,
            page_number=page_number,
            chapter=data.get("chapter"),
            section=data.get("section"),
            domain_hint=data.get("domain_hint"),
        )


@dataclass
class IndexSnapshot:
    """Immutable snapshot of an index for retrieval. No storage IO."""
    index_version_id: str
    entries: list[IndexEntry]

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON. Entries sorted by chunk_id for deterministic ordering."""
        sorted_entries = sorted(self.entries, key=lambda e: e.chunk_id)
        return {
            "index_version_id": self.index_version_id,
            "entries": [e.to_serializable() for e in sorted_entries],
        }

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> IndexSnapshot:
        """Deserialize from dict. Fails loudly on missing or invalid fields."""
        if "index_version_id" not in data:
            raise ValueError("IndexSnapshot missing required field: index_version_id")
        if "entries" not in data:
            raise ValueError("IndexSnapshot missing required field: entries")
        index_version_id = data["index_version_id"]
        if not isinstance(index_version_id, str):
            raise ValueError(
                f"IndexSnapshot index_version_id must be str, got {type(index_version_id).__name__}"
            )
        if not index_version_id.strip():
            raise ValueError("IndexSnapshot index_version_id must be non-empty")
        raw_entries = data["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError(f"IndexSnapshot entries must be list, got {type(raw_entries).__name__}")
        entries = [IndexEntry.from_serializable(item) for item in raw_entries]
        return cls(index_version_id=index_version_id, entries=entries)


def build_index_snapshot(
    document: Document,
    embeddings: list[Embedding],
    index_version_id: str,
) -> IndexSnapshot:
    """
    Build an in-memory index from a Document and its Embeddings.
    Order must match: document.chunks[i] <-> embeddings[i].
    """
    if len(document.chunks) != len(embeddings):
        raise ValueError(
            f"chunk/embedding count mismatch: {len(document.chunks)} vs {len(embeddings)}"
        )
    entries: list[IndexEntry] = []
    for c, e in zip(document.chunks, embeddings):
        if c.chunk_id != e.chunk_id:
            raise ValueError(
                f"chunk_id mismatch: chunk {c.chunk_id} vs embedding {e.chunk_id}"
            )
        entries.append(
            IndexEntry(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                raw_text=c.text,
                embedding_vector_id=e.embedding_vector_id,
                vector=e.vector,
                page_number=c.page_number if c.page_number else None,
                chapter=getattr(c, "chapter", None),
                section=getattr(c, "section", None),
                domain_hint=getattr(c, "domain_hint", None),
            )
        )
    return IndexSnapshot(index_version_id=index_version_id, entries=entries)


def merge_index_snapshots(index_version_id: str, snapshots: list[IndexSnapshot]) -> IndexSnapshot:
    """Merge multiple snapshots (e.g. multiple docs) into one index. All must share index_version_id."""
    all_entries: list[IndexEntry] = []
    for s in snapshots:
        if s.index_version_id != index_version_id:
            raise ValueError(
                f"index_version_id mismatch: {s.index_version_id} vs {index_version_id}"
            )
        all_entries.extend(s.entries)
    return IndexSnapshot(index_version_id=index_version_id, entries=all_entries)


# ---------------------------------------------------------------------------
# Similarity (deterministic)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity in [-1, 1]. Deterministic. Raises if dimension mismatch."""
    if len(a) != len(b):
        raise ValueError(f"vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _scores_to_ranks(scores: list[float], descending: bool = True) -> list[int]:
    """
    Convert scores to 1-based ranks. Ties get the same rank (min rank of the group).
    Order: same as input (rank[i] = rank of entry i).
    """
    n = len(scores)
    if n == 0:
        return []
    # Sort (index, score) by score; then assign rank by position
    order = sorted(range(n), key=lambda i: scores[i], reverse=descending)
    ranks = [0] * n
    r = 1
    i = 0
    while i < n:
        idx = order[i]
        s = scores[idx]
        j = i
        while j < n and scores[order[j]] == s:
            ranks[order[j]] = r
            j += 1
        r += j - i
        i = j
    return ranks


def _rrf_merge(
    bge_scores: list[float],
    bm25_scores: list[float],
    k: int = 60,
) -> list[float]:
    """
    Reciprocal Rank Fusion: merge BGE and BM25 rankings into a single score.
    score[i] = 1/(k + rank_bge[i]) + 1/(k + rank_bm25[i])
    Higher BGE/BM25 score => lower rank (1 = best). Deterministic for same inputs.
    """
    n = len(bge_scores)
    if n != len(bm25_scores):
        raise ValueError("bge_scores and bm25_scores must have same length")
    if n == 0:
        return []
    rank_bge = _scores_to_ranks(bge_scores, descending=True)
    rank_bm25 = _scores_to_ranks(bm25_scores, descending=True)
    return [
        1.0 / (k + rank_bge[i]) + 1.0 / (k + rank_bm25[i])
        for i in range(n)
    ]


def _extract_query_phrases(query: str, min_words: int = 2, max_words: int = 4) -> list[str]:
    """
    Extract candidate multi-word phrases from query.
    Returns lowercased phrases of length min_words to max_words. Deterministic.
    """
    words = query.lower().split()
    phrases: list[str] = []
    for length in range(min_words, min(max_words + 1, len(words) + 1)):
        for i in range(len(words) - length + 1):
            phrases.append(" ".join(words[i : i + length]))
    return phrases


def _phrase_match_bonus(
    query: str, chunk_text: str, bonus_per_phrase: float = 0.15, cap: float = 0.45
) -> float:
    """
    Score bonus for multi-word phrase matches between query and chunk.
    Each matching phrase adds bonus_per_phrase to the score. Capped at cap.
    Deterministic for same inputs.
    """
    phrases = _extract_query_phrases(query)
    chunk_lower = chunk_text.lower()
    bonus = 0.0
    for phrase in phrases:
        if phrase in chunk_lower:
            bonus += bonus_per_phrase
    return min(bonus, cap)


# ---------------------------------------------------------------------------
# Query chunk for embedding (no protocol change)
# ---------------------------------------------------------------------------


def _query_chunk(query: str) -> Chunk:
    """Synthetic chunk for query embedding. Same query -> same chunk_id and vector."""
    # Deterministic query chunk_id: inputs explicit (not stored in index; for embedding only)
    qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    chunk_id = f"query:{qhash}"
    return Chunk(
        chunk_id=chunk_id,
        chunk_key="query:0",
        document_id="",
        source_type="query",
        page_number=0,
        chunk_index=0,
        text=query,
    )


# ---------------------------------------------------------------------------
# Retrieval: top-k enforced, trace payload, JSON logging
# ---------------------------------------------------------------------------
# Determinism: Same query, index_version, top_k -> same results (brute-force search;
# no ANN or randomness). If ANN is introduced later, document non-determinism in code comments.


def retrieve(
    query: str,
    index: IndexSnapshot,
    backend: EmbeddingBackend,
    top_k: int,
    hybrid_config: HybridRetrievalConfig | None = None,
) -> RetrievalResult:
    """
    Run retrieval: embed query, score all index entries, return top_k hits with full trace.

    Scoring modes:
    - When hybrid_config is None: BGE cosine similarity only (Phase 1 behavior)
    - When hybrid_config is provided: Merge BGE + BM25 via RRF (rrf_k)

    Args:
        query: Query string
        index: IndexSnapshot with entries
        backend: EmbeddingBackend for query embedding
        top_k: Number of results to return
        hybrid_config: Optional hybrid retrieval configuration

    Returns:
        RetrievalResult with hits sorted by final score descending
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # Embed query (always needed for BGE component)
    query_chunk = _query_chunk(query)
    query_embeddings = backend.embed_chunks([query_chunk])
    query_vector = query_embeddings[0].vector

    # Compute BGE cosine scores
    bge_scores: list[float] = []
    for entry in index.entries:
        score = _cosine_similarity(entry.vector, query_vector)
        bge_scores.append(score)

    # Compute final scores
    if hybrid_config is None:
        # Phase 1 behavior: BGE-only
        final_scores = list(bge_scores)
    else:
        # Hybrid: BGE + BM25 merged via RRF
        from pipeline.retrieval.bm25_backend import Bm25Backend

        corpus = [entry.raw_text for entry in index.entries]
        bm25 = Bm25Backend(corpus)
        bm25_scores = bm25.score(query)
        final_scores = _rrf_merge(
            bge_scores,
            bm25_scores,
            k=hybrid_config.rrf_k,
        )

    # Phrase-match bonus: reward chunks containing exact query phrases (e.g. "engine switch", "clip-lead jumper")
    phrase_bonuses: list[float] = []
    for entry in index.entries:
        bonus = _phrase_match_bonus(query, entry.raw_text)
        phrase_bonuses.append(bonus)
    scores_with_phrase = [s + b for s, b in zip(final_scores, phrase_bonuses)]

    # Sort by final score descending, then chunk_id for determinism
    scored: list[tuple[float, IndexEntry]] = list(
        zip(scores_with_phrase, index.entries)
    )
    scored.sort(key=lambda x: (-x[0], x[1].chunk_id))

    # Take top_k
    k = min(top_k, len(scored))
    top_pairs = scored[:k]

    hits: list[RetrievalHit] = []
    for score, entry in top_pairs:
        hit = RetrievalHit(
            chunk_id=entry.chunk_id,
            document_id=entry.document_id,
            raw_text=entry.raw_text,
            embedding_vector_id=entry.embedding_vector_id,
            index_version=index.index_version_id,
            similarity_score=score,  # final_score (hybrid or BGE-only)
            page_number=entry.page_number,
            chapter=entry.chapter,
            section=entry.section,
            domain_hint=entry.domain_hint,
            boosted=False,
            original_score=None,
        )
        hits.append(hit)

    truncated = len(hits) < top_k
    corpus_size = len(index.entries)
    result = RetrievalResult(
        hits=hits,
        index_version=index.index_version_id,
        top_k_requested=top_k,
        truncated=truncated,
        corpus_size=corpus_size,
    )

    # Structured JSON log (include phrase_bonus for tuning)
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    log_payload: dict[str, Any] = {
        "query_hash": query_hash,
        "index_version": index.index_version_id,
        "top_k": top_k,
        "hybrid": hybrid_config is not None,
        "returned_chunk_ids": [h.chunk_id for h in hits],
    }
    if phrase_bonuses:
        log_payload["phrase_bonus_max"] = max(phrase_bonuses)
        log_payload["phrase_bonus_top_k"] = [
            phrase_bonuses[index.entries.index(entry)]
            for _sc, entry in scored[:k]
        ]
    logger.info("%s", json.dumps(log_payload, sort_keys=True))

    return result
