"""
Canonical internal schema for RAGOps pipeline artifacts.
Pure data models: no IO, no side effects. All fields explicit and typed.
"""
from __future__ import annotations

import hashlib
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


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One addressable unit of document content (e.g. one page)."""
    chunk_key: str
    document_id: str
    page_number: int
    text: str

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON: field order consistent."""
        return {
            "chunk_key": self.chunk_key,
            "document_id": self.document_id,
            "page_number": self.page_number,
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


@dataclass(frozen=True)
class Embedding:
    """Structure only: vector + chunk reference. No generation logic."""
    chunk_key: str
    vector: tuple[float, ...]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "chunk_key": self.chunk_key,
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
    """Structure only: one retrieved chunk with score and metadata."""
    chunk_key: str
    score: float
    document_id: str
    page_number: int
    text: str

    def to_serializable(self) -> dict[str, Any]:
        return {
            "chunk_key": self.chunk_key,
            "score": self.score,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "text": self.text,
        }


@dataclass
class AnswerWithCitations:
    """Structure only: answer plus citation references."""
    answer: str
    citation_chunk_keys: list[str] = field(default_factory=list)
    found: bool = True

    def to_serializable(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citation_chunk_keys": list(self.citation_chunk_keys),
            "found": self.found,
        }
