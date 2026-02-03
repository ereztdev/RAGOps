"""
Embedding pipeline: pluggable backends that transform Chunk -> Embedding.
Segment 3: abstraction, deterministic fake backend (test-only), real BGE backend.
"""
from pipeline.embedding.embedding_engine import (
    AbstractEmbeddingBackend,
    BgeEmbeddingBackend,
    EmbeddingBackend,
    FakeEmbeddingBackend,
    is_test_only_index_version,
    run_embedding_pipeline,
)

__all__ = [
    "AbstractEmbeddingBackend",
    "BgeEmbeddingBackend",
    "EmbeddingBackend",
    "FakeEmbeddingBackend",
    "is_test_only_index_version",
    "run_embedding_pipeline",
]
