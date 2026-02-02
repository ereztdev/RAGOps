"""
Segment 10: Regression tests and golden fixtures.
Load fixtures from disk, run pipeline stages, assert deep equality (structure + values).
Fail on reordered outputs, changed numeric values, missing/extra fields.
No approximate assertions. No logic changes to production code.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"
CANONICAL_SOURCE_PATH = "data/test_pdfs/ragops_semantic_test_pdf.pdf"

# Fixed inputs used to generate fixtures (must match generate_fixtures.py)
REGRESSION_QUERY = "regression test query"
REGRESSION_TOP_K = 5
REGRESSION_INDEX_VERSION_ID = "regression_v1"


def _load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Fixture not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ingestion determinism
# ---------------------------------------------------------------------------


class TestIngestionDeterminism(unittest.TestCase):
    """Run ingestion on known input; assert deep equality against golden fixture."""

    def test_ingestion_output_matches_fixture(self) -> None:
        if not SEMANTIC_PDF.is_file():
            self.skipTest(f"Test input not found: {SEMANTIC_PDF}")
        from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
        from core.serialization import ingestion_output_to_dict

        expected = _load_fixture("ingestion_output.json")
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        actual = ingestion_output_to_dict(doc)
        actual["source_path"] = CANONICAL_SOURCE_PATH

        self.assertEqual(expected, actual, "Ingestion output must match golden fixture (structure + values)")
        self.assertEqual(
            [c["chunk_id"] for c in expected["chunks"]],
            [c["chunk_id"] for c in actual["chunks"]],
            "Chunk order must match fixture",
        )

    def test_ingestion_fails_on_extra_or_missing_fields(self) -> None:
        """Structural drift: extra or missing top-level keys must fail equality."""
        expected = _load_fixture("ingestion_output.json")
        from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
        from core.serialization import ingestion_output_to_dict

        if not SEMANTIC_PDF.is_file():
            self.skipTest(f"Test input not found: {SEMANTIC_PDF}")
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        actual = ingestion_output_to_dict(doc)
        actual["source_path"] = CANONICAL_SOURCE_PATH
        self.assertEqual(set(expected.keys()), set(actual.keys()), "Top-level keys must match")


# ---------------------------------------------------------------------------
# Index build determinism
# ---------------------------------------------------------------------------


class TestIndexBuildDeterminism(unittest.TestCase):
    """Build index from fixture document + embeddings; assert deep equality against golden snapshot."""

    def test_index_snapshot_matches_fixture(self) -> None:
        from core.serialization import load_ingestion_output
        from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
        from pipeline.retrieval.retrieval_engine import build_index_snapshot

        ingestion_path = FIXTURES_DIR / "ingestion_output.json"
        if not ingestion_path.is_file():
            self.skipTest("Ingestion fixture not found")
        doc = load_ingestion_output(ingestion_path)
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, REGRESSION_INDEX_VERSION_ID)

        expected = _load_fixture("index_snapshot.json")
        actual = index.to_serializable()

        self.assertEqual(expected, actual, "Index snapshot must match golden fixture (structure + values)")
        self.assertEqual(
            [e["chunk_id"] for e in expected["entries"]],
            [e["chunk_id"] for e in actual["entries"]],
            "Entry order (by chunk_id) must match fixture",
        )


# ---------------------------------------------------------------------------
# Retrieval determinism
# ---------------------------------------------------------------------------


class TestRetrievalDeterminism(unittest.TestCase):
    """Same query + index -> same top-k, same scores, same metadata. Deep equality against fixture."""

    def test_retrieval_result_matches_fixture(self) -> None:
        from storage.vector_index_store import load_index
        from pipeline.retrieval.retrieval_engine import retrieve
        from pipeline.embedding.embedding_engine import FakeEmbeddingBackend

        index_path = FIXTURES_DIR / "index_snapshot.json"
        if not index_path.is_file():
            self.skipTest("Index snapshot fixture not found")
        index = load_index(index_path)
        backend = FakeEmbeddingBackend()
        result = retrieve(REGRESSION_QUERY, index, backend, REGRESSION_TOP_K)

        expected = _load_fixture("retrieval_result.json")
        actual = result.to_serializable()

        self.assertEqual(expected, actual, "Retrieval result must match golden fixture (structure + values)")
        self.assertEqual(
            [h["chunk_id"] for h in expected["hits"]],
            [h["chunk_id"] for h in actual["hits"]],
            "Hit order must match fixture",
        )
        self.assertEqual(
            [h["similarity_score"] for h in expected["hits"]],
            [h["similarity_score"] for h in actual["hits"]],
            "Similarity scores and order must match fixture (exact numeric equality)",
        )


# ---------------------------------------------------------------------------
# Inference determinism (same active index + query -> same retrieval path and output structure)
# ---------------------------------------------------------------------------


class TestInferenceDeterminism(unittest.TestCase):
    """Same index + query + FakeInferenceBackend -> same AnswerWithCitations structure. Deep equality against fixture."""

    def test_inference_output_structure_matches_fixture(self) -> None:
        from storage.vector_index_store import load_index
        from pipeline.retrieval.retrieval_engine import retrieve
        from pipeline.embedding.embedding_engine import FakeEmbeddingBackend
        from pipeline.evaluation.evaluation_engine import evaluate_retrieval
        from pipeline.inference.inference_runner import (
            FakeInferenceBackend,
            run_grounded_inference,
        )
        from core.schema import EvaluationConfig

        index_path = FIXTURES_DIR / "index_snapshot.json"
        if not index_path.is_file():
            self.skipTest("Index snapshot fixture not found")
        index = load_index(index_path)
        backend = FakeEmbeddingBackend()
        retrieval_result = retrieve(REGRESSION_QUERY, index, backend, REGRESSION_TOP_K)
        eval_config = EvaluationConfig(
            min_confidence_score=0.0,
            min_alphabetic_ratio=0.0,
            max_entropy=10.0,
        )
        evaluation_report = evaluate_retrieval(
            retrieval_result, REGRESSION_QUERY, eval_config
        )
        llm = FakeInferenceBackend()
        output = run_grounded_inference(
            REGRESSION_QUERY, retrieval_result, evaluation_report, llm
        )

        expected = _load_fixture("inference_output.json")
        actual = output.to_serializable()

        self.assertEqual(
            expected, actual,
            "Inference output structure must match golden fixture (found, citation_chunk_ids, refusal_reason, answer_text)",
        )
        self.assertEqual(
            expected.get("citation_chunk_ids", []),
            actual.get("citation_chunk_ids", []),
            "Citation chunk_id order must match fixture",
        )


# ---------------------------------------------------------------------------
# Failure semantics: reorder / wrong value / missing field
# ---------------------------------------------------------------------------


class TestRegressionFailureSemantics(unittest.TestCase):
    """Tests must fail on reordered outputs, changed numeric values, missing/extra fields."""

    def test_retrieval_reordered_hits_fails_equality(self) -> None:
        """Reordering hits must not equal fixture."""
        expected = _load_fixture("retrieval_result.json")
        if len(expected["hits"]) < 2:
            self.skipTest("Fixture has fewer than 2 hits; reorder test N/A")
        actual = json.loads(json.dumps(expected))
        actual["hits"] = [actual["hits"][1], actual["hits"][0]] + actual["hits"][2:]
        self.assertNotEqual(expected, actual, "Reordered hits must not compare equal")

    def test_retrieval_changed_score_fails_equality(self) -> None:
        """Changed similarity_score must not equal fixture."""
        expected = _load_fixture("retrieval_result.json")
        if not expected["hits"]:
            self.skipTest("Fixture has no hits")
        actual = json.loads(json.dumps(expected))
        actual["hits"][0]["similarity_score"] = actual["hits"][0]["similarity_score"] + 0.0001
        self.assertNotEqual(expected, actual, "Changed numeric value must not compare equal")

    def test_ingestion_missing_chunk_field_fails_equality(self) -> None:
        """Missing chunk field must not equal fixture."""
        expected = _load_fixture("ingestion_output.json")
        if not expected["chunks"]:
            self.skipTest("Fixture has no chunks")
        actual = json.loads(json.dumps(expected))
        del actual["chunks"][0]["chunk_id"]
        self.assertNotEqual(expected, actual, "Missing field must not compare equal")

    def test_ingestion_extra_top_level_key_fails_equality(self) -> None:
        """Extra top-level key must not compare equal when asserting exact keys."""
        expected = _load_fixture("ingestion_output.json")
        actual = json.loads(json.dumps(expected))
        actual["extra_key"] = "value"
        self.assertNotEqual(expected, actual, "Extra field must not compare equal")
