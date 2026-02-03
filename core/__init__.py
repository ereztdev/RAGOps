"""Core pure logic: schema and identifiers. No IO in schema module."""

from core.schema import (
    AnswerWithCitations,
    Chunk,
    ChunkingConfig,
    Document,
    Embedding,
    IndexVersion,
    RetrievalHit,
    RetrievalResult,
    chunk_id_from_components,
    chunk_key,
    document_id_from_bytes,
    embedding_vector_id_from_chunk_and_vector,
)

__all__ = [
    "AnswerWithCitations",
    "Chunk",
    "ChunkingConfig",
    "Document",
    "Embedding",
    "IndexVersion",
    "RetrievalHit",
    "RetrievalResult",
    "chunk_id_from_components",
    "chunk_key",
    "document_id_from_bytes",
    "embedding_vector_id_from_chunk_and_vector",
]
