"""
Phase 3: Pluggable LLM backends and RRF scoring.
Tests: InferenceBackend protocol compliance, Ollama graceful failure when ollama missing,
RRF scoring and determinism, mocked Ollama (no real ollama in CI).
"""
from __future__ import annotations

import unittest
from unittest import mock

from core.schema import HybridRetrievalConfig
from pipeline.inference.backends.base import FakeInferenceBackend, InferenceBackend
from pipeline.inference.backends.ollama_backend import OllamaInferenceBackend
from pathlib import Path

from core.serialization import load_ingestion_output
from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
from pipeline.retrieval.retrieval_engine import (
    build_index_snapshot,
    _rrf_merge,
    _scores_to_ranks,
    retrieve,
)


# ---------------------------------------------------------------------------
# InferenceBackend protocol compliance
# ---------------------------------------------------------------------------


class TestInferenceBackendProtocol(unittest.TestCase):
    def test_fake_backend_has_model_id_and_generate(self) -> None:
        backend = FakeInferenceBackend()
        self.assertEqual(backend.model_id, "fake")
        out = backend.generate("test", "Some context.")
        self.assertIsInstance(out, str)

    def test_ollama_backend_has_model_id_when_mocked(self) -> None:
        with mock.patch("pipeline.inference.backends.ollama_backend._OLLAMA_AVAILABLE", True):
            with mock.patch(
                "pipeline.inference.backends.ollama_backend.OllamaClient",
                create=True,
            ) as MockClient:
                MockClient.return_value = mock.MagicMock()
                backend = OllamaInferenceBackend(model="test:1b")
                self.assertEqual(backend.model_id, "test:1b")


class TestOllamaGracefulFailureWhenNotInstalled(unittest.TestCase):
    def test_raises_clear_import_error_when_ollama_unavailable(self) -> None:
        with mock.patch("pipeline.inference.backends.ollama_backend._OLLAMA_AVAILABLE", False):
            with self.assertRaises(ImportError) as ctx:
                OllamaInferenceBackend(model="x")
            self.assertIn("ollama", str(ctx.exception).lower())
            self.assertIn("pip install", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# RRF scoring
# ---------------------------------------------------------------------------


class TestRRFScoring(unittest.TestCase):
    def test_rrf_merge_produces_correct_ranks(self) -> None:
        # BGE: best first (index 0); BM25: best last (index 2). RRF should favor docs that rank well in both.
        bge = [0.9, 0.5, 0.3]   # ranks 1, 2, 3
        bm25 = [0.1, 0.5, 0.9]  # ranks 3, 2, 1
        k = 60
        scores = _rrf_merge(bge, bm25, k=k)
        self.assertEqual(len(scores), 3)
        # Index 0: rank_bge=1, rank_bm25=3 => 1/61 + 1/63
        # Index 1: rank_bge=2, rank_bm25=2 => 1/62 + 1/62
        # Index 2: rank_bge=3, rank_bm25=1 => 1/63 + 1/61
        # So index 0 and 2 get same RRF (symmetric), index 1 gets 2/62. Highest is index 0 or 2.
        self.assertAlmostEqual(scores[0], 1 / (k + 1) + 1 / (k + 3))
        self.assertAlmostEqual(scores[1], 2 / (k + 2))
        self.assertAlmostEqual(scores[2], 1 / (k + 3) + 1 / (k + 1))
        self.assertAlmostEqual(scores[0], scores[2])
        self.assertLess(scores[1], scores[0])

    def test_rrf_determinism_same_input_same_output(self) -> None:
        bge = [0.8, 0.2, 0.5, 0.5]
        bm25 = [0.1, 0.9, 0.3, 0.3]
        a = _rrf_merge(bge, bm25, k=60)
        b = _rrf_merge(bge, bm25, k=60)
        self.assertEqual(a, b)

    def test_rrf_merge_empty_returns_empty(self) -> None:
        self.assertEqual(_rrf_merge([], [], k=60), [])

    def test_rrf_merge_mismatch_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            _rrf_merge([1.0, 2.0], [1.0], k=60)


class TestScoresToRanks(unittest.TestCase):
    def test_ranks_descending_order(self) -> None:
        scores = [0.9, 0.5, 0.3]
        ranks = _scores_to_ranks(scores, descending=True)
        self.assertEqual(ranks, [1, 2, 3])

    def test_ties_get_same_rank(self) -> None:
        scores = [0.5, 0.9, 0.5]
        ranks = _scores_to_ranks(scores, descending=True)
        # Index 1 is best (rank 1); indices 0 and 2 tie (rank 2)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[0], ranks[2])
        self.assertEqual(ranks[0], 2)


# ---------------------------------------------------------------------------
# Retrieval with RRF (integration)
# ---------------------------------------------------------------------------


class TestRetrievalWithRRF(unittest.TestCase):
    def test_retrieve_with_hybrid_config_uses_rrf(self) -> None:
        # Use fake embedding and minimal index so we don't need real BGE
        fixtures_dir = Path(__file__).parent / "fixtures"
        ingestion_path = fixtures_dir / "ingestion_output.json"
        if not ingestion_path.exists():
            self.skipTest("fixtures/ingestion_output.json not found")
        doc = load_ingestion_output(ingestion_path)
        if not doc.chunks:
            self.skipTest("no chunks in ingestion fixture")
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "rrf_test_v1")
        config = HybridRetrievalConfig(rrf_k=60)
        result = retrieve("test query", index, backend, top_k=3, hybrid_config=config)
        self.assertEqual(len(result.hits), min(3, len(index.entries)))
        # Determinism: same call again gives same order
        result2 = retrieve("test query", index, backend, top_k=3, hybrid_config=config)
        self.assertEqual([h.chunk_id for h in result.hits], [h.chunk_id for h in result2.hits])


# ---------------------------------------------------------------------------
# Ollama backend with mocked client (no real ollama)
# ---------------------------------------------------------------------------


class TestOllamaBackendMocked(unittest.TestCase):
    def test_generate_returns_stripped_response(self) -> None:
        with mock.patch("pipeline.inference.backends.ollama_backend._OLLAMA_AVAILABLE", True):
            with mock.patch(
                "pipeline.inference.backends.ollama_backend.OllamaClient",
                create=True,
            ) as MockClient:
                mock_instance = mock.MagicMock()
                mock_instance.chat.return_value = mock.MagicMock(
                    message=mock.MagicMock(content="  The answer is 42.  ")
                )
                MockClient.return_value = mock_instance
                from pipeline.inference.backends.ollama_backend import OllamaInferenceBackend
                backend = OllamaInferenceBackend(model="llama3.1:8b")
                out = backend.generate("What is it?", "Context here.")
                self.assertEqual(out, "The answer is 42.")
                mock_instance.chat.assert_called_once()
                call_kw = mock_instance.chat.call_args[1]
                self.assertEqual(call_kw.get("options", {}).get("temperature"), 0)

    def test_generate_retries_once_on_timeout_then_returns_empty(self) -> None:
        with mock.patch("pipeline.inference.backends.ollama_backend._OLLAMA_AVAILABLE", True):
            with mock.patch(
                "pipeline.inference.backends.ollama_backend.OllamaClient",
                create=True,
            ) as MockClient:
                mock_instance = mock.MagicMock()
                mock_instance.chat.side_effect = [TimeoutError("timeout"), TimeoutError("timeout")]
                MockClient.return_value = mock_instance
                from pipeline.inference.backends.ollama_backend import OllamaInferenceBackend
                backend = OllamaInferenceBackend(model="x")
                out = backend.generate("q", "ctx")
                self.assertEqual(out, "")
                self.assertEqual(mock_instance.chat.call_count, 2)
