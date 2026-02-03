"""Retrieval pipeline: top-k similarity search with trace payload and logging."""

from pipeline.retrieval.bm25_backend import Bm25Backend
from pipeline.retrieval.retrieval_engine import (
    IndexEntry,
    IndexSnapshot,
    build_index_snapshot,
    merge_index_snapshots,
    retrieve,
)

__all__ = [
    "Bm25Backend",
    "IndexEntry",
    "IndexSnapshot",
    "build_index_snapshot",
    "merge_index_snapshots",
    "retrieve",
]
