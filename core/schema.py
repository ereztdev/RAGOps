"""
Canonical internal schema for RAGOps pipeline artifacts.
Pure data models: no IO, no side effects. All fields explicit and typed.
"""
from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Stable identifiers
# ---------------------------------------------------------------------------


def document_id_from_bytes(raw_pdf_bytes: bytes) -> str:
    """Deterministic document ID: sha256 of raw PDF bytes."""
    return hashlib.sha256(raw_pdf_bytes).hexdigest()


def chunk_key(document_id: str, page_number: int) -> str:
    """Deterministic chunk key: document_id:page_number."""
    return f"{document_id}:{page_number}"


def chunk_id_from_components(
    document_id: str,
    source_type: str,
    page_number: int,
    chunk_index: int,
) -> str:
    """
    Deterministic chunk_id from explicit inputs. Hash-based for stability.
    Inputs (logged at creation): document_id, source_type, page_number, chunk_index.
    """
    canonical = f"{document_id}:{source_type}:{page_number}:{chunk_index}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One addressable unit of document content (e.g. one page)."""
    chunk_id: str
    chunk_key: str
    document_id: str
    source_type: str
    page_number: int
    chunk_index: int
    text: str

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON: field order consistent."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_key": self.chunk_key,
            "document_id": self.document_id,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


@dataclass
class Document:
    """Ingested document with ordered chunks."""
    document_id: str
    source_path: str
    chunks: list[Chunk] = field(default_factory=list)

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON."""
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "chunks": [c.to_serializable() for c in self.chunks],
        }


def embedding_vector_id_from_chunk_and_vector(chunk_id: str, vector: tuple[float, ...]) -> str:
    """Deterministic embedding_vector_id from chunk_id and vector bytes."""
    buf = struct.pack(f"{len(vector)}d", *vector)
    canonical = chunk_id.encode("utf-8") + base64.b64encode(buf)
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class Embedding:
    """Structure only: vector + chunk reference. No generation logic."""
    chunk_key: str
    chunk_id: str
    embedding_vector_id: str
    vector: tuple[float, ...]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "chunk_key": self.chunk_key,
            "chunk_id": self.chunk_id,
            "embedding_vector_id": self.embedding_vector_id,
            "vector": list(self.vector),
        }


@dataclass
class IndexVersion:
    """Structure only: index version metadata. No storage logic."""
    index_version_id: str
    created_at: str
    embedding_model: str
    chunking_strategy: str
    document_ids: list[str] = field(default_factory=list)

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for index_manifest.json."""
        return {
            "index_version_id": self.index_version_id,
            "created_at": self.created_at,
            "embedding_model": self.embedding_model,
            "chunking_strategy": self.chunking_strategy,
            "document_ids": list(self.document_ids),
        }


@dataclass(frozen=True)
class RetrievalHit:
    """
    Trace payload for one retrieved chunk. All fields required; no silent defaults.
    Enables traceability: chunk -> document -> source, and embedding state.
    """
    chunk_id: str
    document_id: str
    raw_text: str
    embedding_vector_id: str
    index_version: str
    similarity_score: float

    def to_serializable(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "raw_text": self.raw_text,
            "embedding_vector_id": self.embedding_vector_id,
            "index_version": self.index_version,
            "similarity_score": self.similarity_score,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval call. Explicit top_k enforcement and truncation flag."""
    hits: list[RetrievalHit]
    index_version: str
    top_k_requested: int
    truncated: bool  # True when fewer than top_k_requested returned
    corpus_size: int  # Number of chunks in the index at retrieval time (for effective_k semantics)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "hits": [h.to_serializable() for h in self.hits],
            "index_version": self.index_version,
            "top_k_requested": self.top_k_requested,
            "truncated": self.truncated,
            "corpus_size": self.corpus_size,
        }


@dataclass
class AnswerWithCitations:
    """
    Output of grounded inference: answer plus citation references, or refusal.
    found=True → answer_text + citation_chunk_ids; found=False → refusal_reason required.
    """
    answer_text: str | None
    citation_chunk_ids: list[str] = field(default_factory=list)
    found: bool = True
    refusal_reason: str | None = None  # REQUIRED when found is False

    def to_serializable(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "citation_chunk_ids": list(self.citation_chunk_ids),
            "found": self.found,
            "refusal_reason": self.refusal_reason,
        }


# ---------------------------------------------------------------------------
# Evaluation (Segment 7): retrieval quality gating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration-driven thresholds for evaluation. No hardcoded magic numbers."""
    min_confidence_score: float  # top similarity_score must be >= this (low_confidence signal)
    min_alphabetic_ratio: float  # gibberish: ratio of alphabetic chars in text (gibberish_detected)
    max_entropy: float  # gibberish: max allowed character entropy (gibberish_detected)


@dataclass(frozen=True)
class KnownAnswerFixture:
    """Expected document_id or chunk_id set for a query (known-answer evaluation)."""
    expected_chunk_ids: frozenset[str] = field(default_factory=frozenset)
    expected_document_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SignalResult:
    """One evaluation signal: passed (bool) and supporting evidence."""
    passed: bool
    details: str | dict[str, Any]


@dataclass
class EvaluationReport:
    """
    Stored evaluation report for retrieval quality gating.
    overall_pass is derived: True only if all mandatory signals pass.
    """
    evaluation_id: str
    index_version: str
    query_hash: str
    top_k_requested: int
    hit_count: int
    signals: dict[str, SignalResult]
    overall_pass: bool  # derived: all mandatory signals pass
    created_at: str  # UTC ISO

    def to_serializable(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "index_version": self.index_version,
            "query_hash": self.query_hash,
            "top_k_requested": self.top_k_requested,
            "hit_count": self.hit_count,
            "signals": {
                name: {"passed": sr.passed, "details": sr.details}
                for name, sr in self.signals.items()
            },
            "overall_pass": self.overall_pass,
            "created_at": self.created_at,
        }
