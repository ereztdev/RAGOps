"""Retrieval pipeline: top-k similarity search with trace payload and logging."""

from pipeline.retrieval.retrieval_engine import (
    IndexEntry,
    IndexSnapshot,
    build_index_snapshot,
    merge_index_snapshots,
    retrieve,
)

__all__ = [
    "IndexEntry",
    "IndexSnapshot",
    "build_index_snapshot",
    "merge_index_snapshots",
    "retrieve",
]
