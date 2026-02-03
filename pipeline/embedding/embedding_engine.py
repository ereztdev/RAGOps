"""
Embedding engine: pluggable backends that transform Chunk -> Embedding.
Segment 3: abstraction, deterministic fake backend (test-only), and real local BGE backend.
"""
from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from core.schema import (
    Chunk,
    Document,
    Embedding,
    embedding_vector_id_from_chunk_and_vector,
)

# Index versions starting with this prefix are test-only (fake backend); they must not pass evaluation.
TEST_ONLY_INDEX_VERSION_PREFIX = "fake"


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


def is_test_only_index_version(index_version_id: str) -> bool:
    """
    True if this index version denotes a test-only (fake) backend.
    Such indexes must not pass evaluation or be used in production.
    """
    return index_version_id.strip().startswith(TEST_ONLY_INDEX_VERSION_PREFIX)


class FakeEmbeddingBackend(AbstractEmbeddingBackend):
    """
    Deterministic fake backend for testing and pipeline validation only.
    Not for production. No external services, no ML libraries. Vector size fixed at 8.
    Values derived from chunk_key and text only.
    Indexes built with this backend must use an index_version_id starting with
    'fake'; such indexes will not pass evaluation.
    """
    vector_size: int = FAKE_EMBEDDING_VECTOR_SIZE
    embedding_model: str = FAKE_EMBEDDING_MODEL_ID

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        result: list[Embedding] = []
        for c in chunks:
            seed_bytes = f"{c.chunk_key}:{c.text}".encode("utf-8")
            vector = _deterministic_floats(seed_bytes, self.vector_size)
            eid = embedding_vector_id_from_chunk_and_vector(c.chunk_id, vector)
            result.append(
                Embedding(
                    chunk_key=c.chunk_key,
                    chunk_id=c.chunk_id,
                    embedding_vector_id=eid,
                    vector=vector,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Real local embedding backend (bge-base via sentence-transformers)
# ---------------------------------------------------------------------------

BGE_BASE_MODEL_ID = "BAAI/bge-base-en-v1.5"
BGE_EMBEDDING_MODEL_ID = "bge-base-en-v1.5"
BGE_VECTOR_SIZE = 768


class BgeEmbeddingBackend(AbstractEmbeddingBackend):
    """
    Real local embedding backend using bge-base (sentence-transformers).
    Deterministic: same chunk text -> same embedding vector (~768 dimensions).
    For production and semantically meaningful retrieval.
    """
    vector_size: int = BGE_VECTOR_SIZE
    embedding_model: str = BGE_EMBEDDING_MODEL_ID

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(BGE_BASE_MODEL_ID)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        # normalize_embeddings=True for cosine similarity; deterministic for same input
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result: list[Embedding] = []
        for c, vec in zip(chunks, vectors):
            vector = tuple(float(x) for x in vec)
            eid = embedding_vector_id_from_chunk_and_vector(c.chunk_id, vector)
            result.append(
                Embedding(
                    chunk_key=c.chunk_key,
                    chunk_id=c.chunk_id,
                    embedding_vector_id=eid,
                    vector=vector,
                )
            )
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
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run embedding pipeline (BGE or fake)")
    parser.add_argument("pdf", nargs="?", default="data/test_pdfs/ragops_semantic_test_pdf.pdf", help="Path to PDF")
    parser.add_argument("--out-index", dest="out_index", default=None, help="Path to write index JSON (uses BGE backend)")
    parser.add_argument("--index-version", dest="index_version", default=None, help="index_version_id (required if --out-index)")
    args = parser.parse_args()

    path = Path(args.pdf)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline

    doc = run_ingestion_pipeline(str(path))
    if args.out_index and args.index_version:
        backend = BgeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        from pipeline.retrieval.retrieval_engine import build_index_snapshot
        from storage.vector_index_store import save_index
        snapshot = build_index_snapshot(doc, embeddings, args.index_version)
        out_path = Path(args.out_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_index(snapshot, out_path)
        print("document_id:", doc.document_id)
        print("chunks:", len(doc.chunks))
        print("index_version_id:", args.index_version)
        print("saved:", str(out_path.resolve()))
    else:
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
