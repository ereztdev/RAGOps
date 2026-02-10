"""
RAGOps configuration: domain boosting, retrieval, and run log provenance.
"""
from __future__ import annotations

# Domain boosting (keyword-based, post-retrieval)
METADATA_BOOST_ENABLED: bool = True
METADATA_BOOST_FACTOR: float = 1.3

# Retrieval: internal top-K before re-rank, then final top-K returned
RETRIEVAL_CANDIDATE_TOP_K: int = 20
RETRIEVAL_FINAL_TOP_K: int = 10

# Run log provenance
LOG_TOP_K_CHUNKS: int = 5
LOG_CHUNK_PREVIEW_LENGTH: int = 100
LOG_RETRIEVAL_DETAILS: bool = True
