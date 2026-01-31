"""
Embedding engine: pluggable backends that transform Chunk -> Embedding.
Segment 3: abstraction and deterministic fake backend only; no vector storage.
"""
from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from core.schema import Chunk, Document, Embedding


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingBackend(Protocol):
    """
    Protocol for embedding generation: accepts typed Chunks, returns typed Embeddings.
    Deterministic for a fixed backend and input.
    """
    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        """Produce one Embedding per Chunk in the same order."""
        ...


class AbstractEmbeddingBackend(ABC):
    """Abstract base for embedding backends. Implements the same contract as EmbeddingBackend."""

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        """Produce one Embedding per Chunk in the same order."""
        ...


# ---------------------------------------------------------------------------
# Deterministic fake backend (no external services, no ML libs)
# ---------------------------------------------------------------------------

FAKE_EMBEDDING_VECTOR_SIZE = 8
FAKE_EMBEDDING_MODEL_ID = "fake_v1"


def _deterministic_floats(seed_bytes: bytes, count: int) -> tuple[float, ...]:
    """
    Generate `count` floats in [-1, 1] from a seed.
    Uses SHA256(seed_bytes) then deterministic expansion so output is stable.
    """
    h = hashlib.sha256(seed_bytes).digest()
    # Use first 4 bytes as seed for simple deterministic sequence
    seed = struct.unpack("<I", h[:4])[0]
    # LCG-like sequence: deterministic, no external RNG
    out: list[float] = []
    for i in range(count):
        seed = (seed * 1103515245 + 12345) & 0x7FFF_FFFF
        # Map to [-1, 1]
        out.append((seed / 0x7FFF_FFFF) * 2.0 - 1.0)
    return tuple(out)


class FakeEmbeddingBackend(AbstractEmbeddingBackend):
    """
    Deterministic fake backend for testing and pipeline validation.
    No external services, no ML libraries. Vector size fixed at 8.
    Values derived from chunk_key and text only.
    """
    vector_size: int = FAKE_EMBEDDING_VECTOR_SIZE
    embedding_model: str = FAKE_EMBEDDING_MODEL_ID

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        result: list[Embedding] = []
        for c in chunks:
            seed_bytes = f"{c.chunk_key}:{c.text}".encode("utf-8")
            vector = _deterministic_floats(seed_bytes, self.vector_size)
            result.append(Embedding(chunk_key=c.chunk_key, vector=vector))
        return result


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def run_embedding_pipeline(
    document: Document,
    backend: EmbeddingBackend,
) -> list[Embedding]:
    """
    Transform a Document's chunks into Embeddings using the given backend.
    Returns one Embedding per Chunk in document order.
    """
    return backend.embed_chunks(document.chunks)


# ---------------------------------------------------------------------------
# Minimal entrypoint for running embeddings (no extra CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from repo root: python -m pipeline.embedding.embedding_engine [pdf_path]
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_pdfs/ragops_semantic_test_pdf.pdf"
    path = Path(pdf_path)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline

    doc = run_ingestion_pipeline(str(path))
    backend = FakeEmbeddingBackend()
    embeddings = run_embedding_pipeline(doc, backend)
    print("document_id:", doc.document_id)
    print("chunks:", len(doc.chunks))
    print("embeddings:", len(embeddings))
    if embeddings:
        e = embeddings[0]
        print("first embedding chunk_key:", e.chunk_key)
        print("first embedding vector (len=%d):" % len(e.vector), e.vector)
    print("backend model:", backend.embedding_model)
