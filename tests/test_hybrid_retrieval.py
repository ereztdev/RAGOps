"""Tests for hybrid retrieval: domain boosting, score normalization, retrieve_for_query."""
from __future__ import annotations

import unittest

from core.schema import HybridRetrievalConfig
from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.retrieval.retrieval_engine import IndexEntry, IndexSnapshot, build_index_snapshot
from ragops.retrieval.domain_detector import detect_domains
from ragops.retrieval.hybrid_retrieval import hybrid_retrieve, retrieve_for_query


def _make_fake_index() -> IndexSnapshot:
    """Build a minimal index from semantic test PDF for retrieval tests."""
    from pathlib import Path
    pdf = Path(__file__).resolve().parent.parent / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"
    if not pdf.is_file():
        raise FileNotFoundError(f"fixture not found: {pdf}")
    doc = run_ingestion_pipeline(str(pdf))
    embeddings = run_embedding_pipeline(doc, FakeEmbeddingBackend())
    return build_index_snapshot(doc, embeddings, "fake_v1")


class TestDetectDomains(unittest.TestCase):
    def test_engine_keywords_detected(self) -> None:
        domains = detect_domains("What is the engine oil pressure?")
        self.assertIn("Engine", domains)

    def test_troubleshooting_keywords_detected(self) -> None:
        domains = detect_domains("The unit fails to build voltage")
        self.assertIn("Troubleshooting", domains)

    def test_empty_query_returns_empty(self) -> None:
        domains = detect_domains("")
        self.assertEqual(domains, [])


class TestRetrieveForQuery(unittest.TestCase):
    def test_retrieve_for_query_returns_tuple(self) -> None:
        index = _make_fake_index()
        backend = FakeEmbeddingBackend()
        result, domains = retrieve_for_query(
            "what is the main topic?",
            index,
            backend,
            top_k=3,
            hybrid_config=HybridRetrievalConfig(),
        )
        self.assertIsInstance(result.hits, list)
        self.assertIsInstance(domains, list)
        self.assertLessEqual(len(result.hits), 3)

    def test_retrieve_for_query_respects_top_k(self) -> None:
        index = _make_fake_index()
        backend = FakeEmbeddingBackend()
        result, _ = retrieve_for_query(
            "test query",
            index,
            backend,
            top_k=2,
            hybrid_config=HybridRetrievalConfig(),
        )
        self.assertLessEqual(len(result.hits), 2)


class TestHybridRetrieveNormalization(unittest.TestCase):
    def test_hybrid_retrieve_normalizes_scores_to_01(self) -> None:
        index = _make_fake_index()
        backend = FakeEmbeddingBackend()
        result, _ = hybrid_retrieve(
            "test",
            index,
            backend,
            final_top_k=5,
            hybrid_config=HybridRetrievalConfig(),
        )
        for hit in result.hits:
            self.assertGreaterEqual(hit.similarity_score, 0.0)
            self.assertLessEqual(hit.similarity_score, 1.0)
