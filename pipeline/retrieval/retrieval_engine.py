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
    """One index row: chunk identity, text, and vector for similarity."""
    chunk_id: str
    document_id: str
    raw_text: str
    embedding_vector_id: str
    vector: tuple[float, ...]

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON. Vector as deterministic float strings for byte-identical output."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "raw_text": self.raw_text,
            "embedding_vector_id": self.embedding_vector_id,
            "vector": [format(v, _FLOAT_FORMAT) for v in self.vector],
        }

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> IndexEntry:
        """Deserialize from dict. Fails loudly on missing or invalid fields."""
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
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            raw_text=raw_text,
            embedding_vector_id=embedding_vector_id,
            vector=vector,
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


def _normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores to [0, 1] using min-max normalization.

    Edge cases:
    - Empty list: return []
    - Single score or all equal: return [0.5, 0.5, ...]
    - Normal case: (score - min) / (max - min)
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if abs(max_score - min_score) < 1e-9:
        # All scores equal or single score
        return [0.5] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


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
    - When hybrid_config is provided: Merge BGE + BM25 scores with alpha/beta weights

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
        final_scores = bge_scores
    else:
        # Phase 2 behavior: BGE + BM25 hybrid
        from pipeline.retrieval.bm25_backend import Bm25Backend

        corpus = [entry.raw_text for entry in index.entries]
        bm25 = Bm25Backend(corpus)
        bm25_scores = bm25.score(query)

        # Normalize both to [0, 1]
        bge_normalized = _normalize_scores(bge_scores)
        bm25_normalized = _normalize_scores(bm25_scores)

        # Merge: final = alpha * BGE + beta * BM25
        final_scores = [
            hybrid_config.alpha * bg + hybrid_config.beta * bm
            for bg, bm in zip(bge_normalized, bm25_normalized)
        ]

    # Sort by final score descending, then chunk_id for determinism
    scored: list[tuple[float, IndexEntry]] = list(
        zip(final_scores, index.entries)
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

    # Structured JSON log
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    log_payload = {
        "query_hash": query_hash,
        "index_version": index.index_version_id,
        "top_k": top_k,
        "hybrid": hybrid_config is not None,
        "returned_chunk_ids": [h.chunk_id for h in hits],
    }
    logger.info("%s", json.dumps(log_payload, sort_keys=True))

    return result
