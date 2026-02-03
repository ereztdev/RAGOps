"""
Segment 7: Evaluation engine — retrieval quality gating.
Tests: determinism of evaluation_id, each signal, overall_pass gating,
known-answer pass/fail, empty retrieval -> overall_pass false.
Full-pipeline integration: PDF -> ingest -> embed -> index -> retrieve -> evaluate.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.schema import (
    EvaluationConfig,
    KnownAnswerFixture,
    RetrievalHit,
    RetrievalResult,
)
from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
from pipeline.evaluation.evaluation_engine import (
    EMPTY_RETRIEVAL,
    GIBBERISH_DETECTED,
    HIT_COUNT_INSUFFICIENT,
    KNOWN_ANSWER_MISS,
    LOW_CONFIDENCE,
    TEST_ONLY_BACKEND,
    evaluation_id_from_index_and_query,
    evaluate_retrieval,
    query_hash_from_query,
)
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"


def _hit(
    chunk_id: str = "c1",
    document_id: str = "d1",
    raw_text: str = "Normal readable content for retrieval.",
    similarity_score: float = 0.9,
    index_version: str = "v1",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        raw_text=raw_text,
        embedding_vector_id="ev1",
        index_version=index_version,
        similarity_score=similarity_score,
    )


def _config(
    min_confidence_score: float = 0.5,
    min_alphabetic_ratio: float = 0.5,
    max_entropy: float = 5.0,
) -> EvaluationConfig:
    return EvaluationConfig(
        min_confidence_score=min_confidence_score,
        min_alphabetic_ratio=min_alphabetic_ratio,
        max_entropy=max_entropy,
    )


# ---------------------------------------------------------------------------
# Determinism of evaluation_id
# ---------------------------------------------------------------------------


class TestEvaluationIdDeterminism(unittest.TestCase):
    def test_same_index_and_query_hash_produces_same_evaluation_id(self) -> None:
        qh = query_hash_from_query("test query")
        eid1 = evaluation_id_from_index_and_query("v1", qh)
        eid2 = evaluation_id_from_index_and_query("v1", qh)
        self.assertEqual(eid1, eid2)

    def test_different_query_hash_produces_different_evaluation_id(self) -> None:
        eid1 = evaluation_id_from_index_and_query("v1", query_hash_from_query("q1"))
        eid2 = evaluation_id_from_index_and_query("v1", query_hash_from_query("q2"))
        self.assertNotEqual(eid1, eid2)

    def test_different_index_version_produces_different_evaluation_id(self) -> None:
        qh = query_hash_from_query("same query")
        eid1 = evaluation_id_from_index_and_query("v1", qh)
        eid2 = evaluation_id_from_index_and_query("v2", qh)
        self.assertNotEqual(eid1, eid2)

    def test_report_evaluation_id_matches_deterministic_hash(self) -> None:
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        report = evaluate_retrieval(result, "my query", config, evaluations_dir=None)
        expected_id = evaluation_id_from_index_and_query(
            "v1", query_hash_from_query("my query")
        )
        self.assertEqual(report.evaluation_id, expected_id)


# ---------------------------------------------------------------------------
# Each signal triggering independently
# ---------------------------------------------------------------------------


class TestSignalsTriggerIndependently(unittest.TestCase):
    def test_empty_retrieval_signal_fails_when_no_hits(self) -> None:
        result = RetrievalResult(
            hits=[],
            index_version="v1",
            top_k_requested=5,
            truncated=True,
            corpus_size=0,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertIn(EMPTY_RETRIEVAL, report.signals)
        self.assertFalse(report.signals[EMPTY_RETRIEVAL].passed)
        self.assertIn("hit_count", report.signals[EMPTY_RETRIEVAL].details)

    def test_empty_retrieval_signal_passes_when_hits_exist(self) -> None:
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertTrue(report.signals[EMPTY_RETRIEVAL].passed)

    def test_low_confidence_signal_fails_when_top_score_below_threshold(self) -> None:
        result = RetrievalResult(
            hits=[_hit(similarity_score=0.3)],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config(min_confidence_score=0.5)
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertFalse(report.signals[LOW_CONFIDENCE].passed)
        self.assertEqual(report.signals[LOW_CONFIDENCE].details["top_score"], 0.3)
        self.assertEqual(report.signals[LOW_CONFIDENCE].details["threshold"], 0.5)

    def test_low_confidence_signal_passes_when_top_score_at_or_above_threshold(
        self,
    ) -> None:
        result = RetrievalResult(
            hits=[_hit(similarity_score=0.6)],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config(min_confidence_score=0.5)
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertTrue(report.signals[LOW_CONFIDENCE].passed)

    def test_gibberish_detected_signal_fails_when_chunk_has_low_alphabetic_ratio(
        self,
    ) -> None:
        # Mostly digits/symbols -> low alphabetic ratio
        result = RetrievalResult(
            hits=[_hit(raw_text="12345 67890 !@#$% 111")],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config(min_alphabetic_ratio=0.5)
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertFalse(report.signals[GIBBERISH_DETECTED].passed)
        self.assertIn("gibberish_chunk_ids", report.signals[GIBBERISH_DETECTED].details)

    def test_gibberish_detected_signal_passes_when_content_is_readable(self) -> None:
        result = RetrievalResult(
            hits=[_hit(raw_text="Clear alphabetic content for retrieval.")],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config(min_alphabetic_ratio=0.5, max_entropy=5.0)
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertTrue(report.signals[GIBBERISH_DETECTED].passed)

    def test_known_answer_miss_signal_fails_when_expected_chunk_not_in_hits(
        self,
    ) -> None:
        result = RetrievalResult(
            hits=[_hit(chunk_id="actual_chunk")],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        fixtures = {
            "q": KnownAnswerFixture(expected_chunk_ids=frozenset({"expected_chunk"}))
        }
        report = evaluate_retrieval(result, "q", config, fixtures=fixtures, evaluations_dir=None)
        self.assertIn(KNOWN_ANSWER_MISS, report.signals)
        self.assertFalse(report.signals[KNOWN_ANSWER_MISS].passed)
        self.assertIn("expected_chunk", report.signals[KNOWN_ANSWER_MISS].details["missing_chunk_ids"])

    def test_known_answer_miss_signal_passes_when_expected_chunk_in_hits(self) -> None:
        result = RetrievalResult(
            hits=[_hit(chunk_id="expected_chunk")],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        fixtures = {
            "q": KnownAnswerFixture(expected_chunk_ids=frozenset({"expected_chunk"}))
        }
        report = evaluate_retrieval(result, "q", config, fixtures=fixtures, evaluations_dir=None)
        self.assertTrue(report.signals[KNOWN_ANSWER_MISS].passed)

    def test_known_answer_not_evaluated_when_no_fixture_for_query(self) -> None:
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        fixtures = {"other_query": KnownAnswerFixture(expected_chunk_ids=frozenset({"x"}))}
        report = evaluate_retrieval(result, "q", config, fixtures=fixtures, evaluations_dir=None)
        self.assertNotIn(KNOWN_ANSWER_MISS, report.signals)

    def test_hit_count_insufficient_signal_fails_when_corpus_large_but_few_hits_returned(
        self,
    ) -> None:
        """corpus_size >= requested_k but returned_hits < requested_k -> FAIL."""
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=5,
            truncated=True,
            corpus_size=5,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertFalse(report.signals[HIT_COUNT_INSUFFICIENT].passed)
        self.assertEqual(report.signals[HIT_COUNT_INSUFFICIENT].details["hit_count"], 1)
        self.assertEqual(report.signals[HIT_COUNT_INSUFFICIENT].details["top_k_requested"], 5)
        self.assertEqual(report.signals[HIT_COUNT_INSUFFICIENT].details["corpus_size"], 5)
        self.assertEqual(report.signals[HIT_COUNT_INSUFFICIENT].details["effective_k"], 5)

    def test_hit_count_insufficient_signal_passes_when_hits_ge_top_k(self) -> None:
        result = RetrievalResult(
            hits=[_hit(chunk_id=f"c{i}") for i in range(5)],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=5,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertTrue(report.signals[HIT_COUNT_INSUFFICIENT].passed)

    def test_hit_count_insufficient_signal_passes_when_corpus_smaller_than_requested_k(
        self,
    ) -> None:
        """corpus_size < requested_k: 1 chunk, top_k=5 -> pass (quality over count)."""
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=5,
            truncated=True,
            corpus_size=1,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertTrue(report.signals[HIT_COUNT_INSUFFICIENT].passed)
        self.assertEqual(report.signals[HIT_COUNT_INSUFFICIENT].details["effective_k"], 1)


# ---------------------------------------------------------------------------
# overall_pass gating behavior
# ---------------------------------------------------------------------------


class TestOverallPassGating(unittest.TestCase):
    def test_overall_pass_true_only_when_all_signals_pass(self) -> None:
        result = RetrievalResult(
            hits=[_hit(similarity_score=0.8, raw_text="Good readable content.")],
            index_version="v1",
            top_k_requested=1,
            truncated=False,
            corpus_size=1,
        )
        config = _config(min_confidence_score=0.5, min_alphabetic_ratio=0.3, max_entropy=5.0)
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertTrue(report.overall_pass)
        for name, sr in report.signals.items():
            self.assertTrue(sr.passed, f"signal {name} should pass")

    def test_overall_pass_false_when_any_mandatory_signal_fails(self) -> None:
        result = RetrievalResult(
            hits=[_hit(similarity_score=0.1)],  # low_confidence fails
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config(min_confidence_score=0.5)
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertFalse(report.overall_pass)
        self.assertFalse(report.signals[LOW_CONFIDENCE].passed)


# ---------------------------------------------------------------------------
# Known-answer pass vs fail
# ---------------------------------------------------------------------------


class TestKnownAnswerPassFail(unittest.TestCase):
    def test_known_answer_pass_when_expected_document_id_in_hits(self) -> None:
        result = RetrievalResult(
            hits=[_hit(document_id="doc_expected")],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        fixtures = {
            "q": KnownAnswerFixture(expected_document_ids=frozenset({"doc_expected"}))
        }
        report = evaluate_retrieval(result, "q", config, fixtures=fixtures, evaluations_dir=None)
        self.assertTrue(report.signals[KNOWN_ANSWER_MISS].passed)
        self.assertEqual(report.signals[KNOWN_ANSWER_MISS].details["missing_document_ids"], [])

    def test_known_answer_fail_when_expected_document_id_not_in_hits(self) -> None:
        result = RetrievalResult(
            hits=[_hit(document_id="other_doc")],
            index_version="v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        fixtures = {
            "q": KnownAnswerFixture(expected_document_ids=frozenset({"doc_expected"}))
        }
        report = evaluate_retrieval(result, "q", config, fixtures=fixtures, evaluations_dir=None)
        self.assertFalse(report.signals[KNOWN_ANSWER_MISS].passed)
        self.assertIn("doc_expected", report.signals[KNOWN_ANSWER_MISS].details["missing_document_ids"])


# ---------------------------------------------------------------------------
# Empty retrieval produces overall_pass = false
# ---------------------------------------------------------------------------


class TestEmptyRetrievalOverallFail(unittest.TestCase):
    def test_empty_retrieval_produces_overall_pass_false(self) -> None:
        result = RetrievalResult(
            hits=[],
            index_version="v1",
            top_k_requested=5,
            truncated=True,
            corpus_size=0,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertFalse(report.overall_pass)
        self.assertFalse(report.signals[EMPTY_RETRIEVAL].passed)

    def test_fake_index_version_fails_test_only_backend_signal(self) -> None:
        """Index versions starting with 'fake' must not pass evaluation (test-only backend)."""
        result = RetrievalResult(
            hits=[_hit(index_version="fake_v1")],
            index_version="fake_v1",
            top_k_requested=5,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertIn(TEST_ONLY_BACKEND, report.signals)
        self.assertFalse(report.signals[TEST_ONLY_BACKEND].passed)
        self.assertFalse(report.overall_pass)


# ---------------------------------------------------------------------------
# Persistence and report schema
# ---------------------------------------------------------------------------


class TestEvaluationPersistenceAndSchema(unittest.TestCase):
    def test_persist_report_writes_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = RetrievalResult(
                hits=[_hit()],
                index_version="v1",
                top_k_requested=1,
                truncated=False,
                corpus_size=1,
            )
            config = _config()
            report = evaluate_retrieval(
                result, "persist query", config, evaluations_dir=base
            )
            path = base / f"{report.evaluation_id}.json"
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn(report.evaluation_id, content)
            self.assertIn("overall_pass", content)
            self.assertIn("signals", content)

    def test_report_has_required_fields(self) -> None:
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=1,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        report = evaluate_retrieval(result, "q", config, evaluations_dir=None)
        self.assertIsInstance(report.evaluation_id, str)
        self.assertEqual(report.index_version, "v1")
        self.assertIsInstance(report.query_hash, str)
        self.assertEqual(report.top_k_requested, 1)
        self.assertEqual(report.hit_count, 1)
        self.assertIsInstance(report.signals, dict)
        self.assertIn(EMPTY_RETRIEVAL, report.signals)
        self.assertIn(LOW_CONFIDENCE, report.signals)
        self.assertIn(GIBBERISH_DETECTED, report.signals)
        self.assertIn(HIT_COUNT_INSUFFICIENT, report.signals)
        self.assertIsInstance(report.overall_pass, bool)
        self.assertRegex(report.created_at, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_deterministic_report_for_fixed_input(self) -> None:
        result = RetrievalResult(
            hits=[_hit()],
            index_version="v1",
            top_k_requested=1,
            truncated=False,
            corpus_size=1,
        )
        config = _config()
        r1 = evaluate_retrieval(result, "fixed query", config, evaluations_dir=None)
        r2 = evaluate_retrieval(result, "fixed query", config, evaluations_dir=None)
        self.assertEqual(r1.evaluation_id, r2.evaluation_id)
        self.assertEqual(r1.overall_pass, r2.overall_pass)
        self.assertEqual(r1.signals.keys(), r2.signals.keys())


# ---------------------------------------------------------------------------
# Full-pipeline integration: PDF -> ingest -> embed -> index -> retrieve -> evaluate
# ---------------------------------------------------------------------------


class TestFullPipelineEvaluation(unittest.TestCase):
    """
    Run evaluation on a real PDF through the full pipeline.
    Printed JSON appears after the test result when run with unittest.
    With pytest, use -s to see the print (no capture).
    """

    def test_full_pipeline_produces_valid_evaluation_report_and_prints_json(self) -> None:
        """Ingest PDF -> embed -> index -> retrieve -> evaluate; assert report valid; print JSON."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")

        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        self.assertGreater(len(doc.chunks), 0, "semantic PDF must yield chunks")

        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "v1")
        query = "what is the main topic?"
        top_k = 5
        result = retrieve(query, index, backend, top_k=top_k)

        config = EvaluationConfig(
            min_confidence_score=0.5,
            min_alphabetic_ratio=0.5,
            max_entropy=5.0,
        )
        report = evaluate_retrieval(result, query, config, evaluations_dir=None)

        # Assert report structure and that serialization is valid JSON
        self.assertIsInstance(report.evaluation_id, str)
        self.assertEqual(report.index_version, "v1")
        self.assertEqual(report.query_hash, query_hash_from_query(query))
        self.assertEqual(report.top_k_requested, top_k)
        self.assertEqual(report.hit_count, len(result.hits))
        self.assertIn(EMPTY_RETRIEVAL, report.signals)
        self.assertIn(LOW_CONFIDENCE, report.signals)
        self.assertIn(GIBBERISH_DETECTED, report.signals)
        self.assertIn(HIT_COUNT_INSUFFICIENT, report.signals)
        self.assertIsInstance(report.overall_pass, bool)

        serialized = report.to_serializable()
        json_str = json.dumps(serialized, indent=2, sort_keys=True)
        # Round-trip: response is proper JSON
        decoded = json.loads(json_str)
        self.assertEqual(decoded["evaluation_id"], report.evaluation_id)
        self.assertEqual(decoded["index_version"], report.index_version)
        self.assertEqual(decoded["hit_count"], report.hit_count)
        self.assertEqual(decoded["overall_pass"], report.overall_pass)
        self.assertIn("signals", decoded)
        self.assertIn("created_at", decoded)

        # Print so user can see the JSON (use -s when running unittest)
        print("\n--- Evaluation report JSON (full pipeline) ---")
        print(json_str)
        print("--- End evaluation report ---\n")
