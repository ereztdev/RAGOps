"""
Embedding pipeline: pluggable backends that transform Chunk -> Embedding.
Segment 3: abstraction and deterministic fake backend; no vector storage.
"""
from pipeline.embedding.embedding_engine import (
    AbstractEmbeddingBackend,
    EmbeddingBackend,
    FakeEmbeddingBackend,
    run_embedding_pipeline,
)

__all__ = [
    "AbstractEmbeddingBackend",
    "EmbeddingBackend",
    "FakeEmbeddingBackend",
    "run_embedding_pipeline",
]
