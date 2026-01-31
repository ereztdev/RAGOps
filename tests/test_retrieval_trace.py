"""
Retrieval traceability: trace payload fields present, top_k enforced, order by score.
Integration test: ingest -> embed -> build index -> retrieve -> assert trace fields and order.
"""
from __future__ import annotations

import unittest

from core.schema import Document
from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.retrieval.retrieval_engine import (
    build_index_snapshot,
    retrieve,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"


# ---------------------------------------------------------------------------
# Trace payload: required fields present
# ---------------------------------------------------------------------------


class TestRetrievalTraceFields(unittest.TestCase):
    """Assert every retrieval hit has required trace fields; no silent defaults."""

    def test_retrieval_hits_have_all_trace_fields(self) -> None:
        """Each RetrievalHit must have chunk_id, document_id, raw_text, embedding_vector_id, index_version, similarity_score."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        if not doc.chunks:
            self.skipTest("no chunks in fixture")
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "v1")
        result = retrieve("test query", index, backend, top_k=5)

        required = {"chunk_id", "document_id", "raw_text", "embedding_vector_id", "index_version", "similarity_score"}
        for hit in result.hits:
            ser = hit.to_serializable()
            self.assertEqual(set(ser.keys()), required, f"hit missing keys: {ser}")
            for k in required:
                self.assertIsNotNone(ser[k], f"hit.{k} must not be None")
            self.assertIsInstance(hit.similarity_score, float)
            self.assertEqual(hit.index_version, "v1")

    def test_retrieval_results_ordered_by_similarity_score(self) -> None:
        """Returned hits must be strictly sorted by similarity_score descending."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        if not doc.chunks:
            self.skipTest("no chunks in fixture")
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "v1")
        result = retrieve("semantic content", index, backend, top_k=10)

        scores = [h.similarity_score for h in result.hits]
        self.assertEqual(scores, sorted(scores, reverse=True), "hits must be ordered by similarity_score descending")

    def test_top_k_enforced(self) -> None:
        """Retrieval must return at most top_k hits; truncated when fewer exist."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        if not doc.chunks:
            self.skipTest("no chunks in fixture")
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "v1")
        n = len(doc.chunks)

        result = retrieve("query", index, backend, top_k=n + 10)
        self.assertLessEqual(len(result.hits), n + 10)
        self.assertEqual(result.top_k_requested, n + 10)
        self.assertTrue(result.truncated, "truncated must be True when fewer than top_k returned")
        self.assertEqual(len(result.hits), n)

        result_exact = retrieve("query", index, backend, top_k=n)
        self.assertEqual(len(result_exact.hits), n)
        self.assertFalse(result_exact.truncated)

    def test_same_query_index_top_k_reproducible(self) -> None:
        """Same query, index_version, top_k -> same returned_chunk_ids order (deterministic)."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        if not doc.chunks:
            self.skipTest("no chunks in fixture")
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "v1")

        r1 = retrieve("reproducible query", index, backend, top_k=5)
        r2 = retrieve("reproducible query", index, backend, top_k=5)
        self.assertEqual([h.chunk_id for h in r1.hits], [h.chunk_id for h in r2.hits])
        self.assertEqual([h.similarity_score for h in r1.hits], [h.similarity_score for h in r2.hits])
