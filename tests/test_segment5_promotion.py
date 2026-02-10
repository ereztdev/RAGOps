"""
Segment 5: Index versioning and registry (promotion).
Tests: cannot promote without passing evaluation; only one promoted at a time;
previously promoted demoted; inference refuses when active index missing/invalid;
registry serialization deterministic; promotion does not alter index contents.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.schema import (
    EvaluationConfig,
    EvaluationReport,
    RetrievalHit,
    RetrievalResult,
    SignalResult,
)
from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
from pipeline.evaluation.evaluation_engine import (
    EMPTY_RETRIEVAL,
    HIT_COUNT_INSUFFICIENT,
    LOW_CONFIDENCE,
    evaluate_retrieval,
)
from pipeline.inference.inference_runner import (
    FakeInferenceBackend,
    REFUSAL_NO_ACTIVE_INDEX,
    run_inference_using_active_index,
)
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.promotion import (
    EvaluationFailedError,
    IndexNotInRegistryError,
    IndexVersionMismatchError,
    PromotionError,
    REGISTRY_STATUS_EVALUATED,
    REGISTRY_STATUS_PROMOTED,
    get_entry,
    get_promoted_entry,
    load_active_index,
    load_registry,
    promote_index,
    register_index,
    resolve_active_index,
    save_registry,
)
from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve
from storage.vector_index_store import load_index, save_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"


def _hit(
    chunk_id: str = "c1",
    document_id: str = "d1",
    raw_text: str = "Content for retrieval.",
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


def _evaluation_report(
    overall_pass: bool,
    evaluation_id: str = "eid1",
    index_version: str = "v1",
    query_hash: str = "qh1",
    top_k_requested: int = 5,
    hit_count: int = 1,
    created_at: str = "2026-02-02T12:00:00.000000Z",
) -> EvaluationReport:
    signals = {
        EMPTY_RETRIEVAL: SignalResult(passed=True, details={"hit_count": hit_count}),
        LOW_CONFIDENCE: SignalResult(passed=overall_pass, details={"top_score": 0.9}),
        HIT_COUNT_INSUFFICIENT: SignalResult(
            passed=overall_pass, details={"hit_count": hit_count, "top_k_requested": top_k_requested}
        ),
    }
    return EvaluationReport(
        evaluation_id=evaluation_id,
        index_version=index_version,
        query_hash=query_hash,
        top_k_requested=top_k_requested,
        hit_count=hit_count,
        signals=signals,
        overall_pass=overall_pass,
        created_at=created_at,
    )


def _eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        min_confidence_score=0.5,
        min_alphabetic_ratio=0.5,
        max_entropy=5.0,
    )


# ---------------------------------------------------------------------------
# Cannot promote without passing evaluation
# ---------------------------------------------------------------------------


class TestPromotionRequiresPassingEvaluation(unittest.TestCase):
    def test_promote_with_overall_pass_false_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from pipeline.promotion.index_registry import RegistryEntry

            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            entry = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path=str(Path(tmp) / "v1.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(registry_path, [entry])
            report = _evaluation_report(overall_pass=False, index_version="v1")
            with self.assertRaises(EvaluationFailedError):
                promote_index("v1", report, registry_path, active_path)

    def test_promote_with_overall_pass_true_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            from pipeline.promotion.index_registry import RegistryEntry

            entry = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path=str(Path(tmp) / "v1.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(registry_path, [entry])
            Path(tmp).joinpath("v1.json").write_text("{}")
            report = _evaluation_report(overall_pass=True, index_version="v1")
            promote_index("v1", report, registry_path, active_path)
            active = load_active_index(active_path)
            self.assertEqual(active.index_version_id, "v1")


# ---------------------------------------------------------------------------
# Only one promoted index at a time; previously promoted demoted
# ---------------------------------------------------------------------------


class TestOnlyOnePromotedAndDemotion(unittest.TestCase):
    def test_only_one_promoted_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            from pipeline.promotion.index_registry import RegistryEntry

            e1 = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path=str(Path(tmp) / "v1.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            e2 = RegistryEntry(
                index_version_id="v2",
                created_at="2026-02-02T12:01:00Z",
                index_path=str(Path(tmp) / "v2.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(registry_path, [e1, e2])
            Path(tmp).joinpath("v1.json").write_text("{}")
            Path(tmp).joinpath("v2.json").write_text("{}")
            report1 = _evaluation_report(overall_pass=True, index_version="v1", evaluation_id="e1")
            report2 = _evaluation_report(overall_pass=True, index_version="v2", evaluation_id="e2")
            promote_index("v1", report1, registry_path, active_path)
            entries = load_registry(registry_path)
            promoted = get_promoted_entry(entries)
            self.assertIsNotNone(promoted)
            self.assertEqual(promoted.index_version_id, "v1")
            promote_index("v2", report2, registry_path, active_path)
            entries = load_registry(registry_path)
            promoted = get_promoted_entry(entries)
            self.assertIsNotNone(promoted)
            self.assertEqual(promoted.index_version_id, "v2")

    def test_previously_promoted_is_demoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            from pipeline.promotion.index_registry import RegistryEntry

            e1 = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path=str(Path(tmp) / "v1.json"),
                evaluation_report_id="e1",
                status=REGISTRY_STATUS_PROMOTED,
                notes=None,
            )
            e2 = RegistryEntry(
                index_version_id="v2",
                created_at="2026-02-02T12:01:00Z",
                index_path=str(Path(tmp) / "v2.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(registry_path, [e1, e2])
            Path(tmp).joinpath("v1.json").write_text("{}")
            Path(tmp).joinpath("v2.json").write_text("{}")
            report2 = _evaluation_report(overall_pass=True, index_version="v2", evaluation_id="e2")
            promote_index("v2", report2, registry_path, active_path)
            entries = load_registry(registry_path)
            entry_v1 = get_entry(entries, "v1")
            entry_v2 = get_entry(entries, "v2")
            self.assertIsNotNone(entry_v1)
            self.assertIsNotNone(entry_v2)
            self.assertEqual(entry_v1.status, REGISTRY_STATUS_EVALUATED)
            self.assertEqual(entry_v2.status, REGISTRY_STATUS_PROMOTED)


# ---------------------------------------------------------------------------
# Inference refuses when active index missing or invalid
# ---------------------------------------------------------------------------


class TestInferenceRefusesWhenNoActiveIndex(unittest.TestCase):
    def test_missing_active_index_returns_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indexes_dir = Path(tmp)
            backend = FakeEmbeddingBackend()
            llm = FakeInferenceBackend()
            config = _eval_config()
            out, _, _ = run_inference_using_active_index(
                "query",
                indexes_dir,
                backend,
                top_k=5,
                llm=llm,
                evaluation_config=config,
            )
            self.assertFalse(out.found)
            self.assertEqual(out.refusal_reason, REFUSAL_NO_ACTIVE_INDEX)
            self.assertIsNone(out.answer_text)
            self.assertEqual(out.citation_chunk_ids, [])

    def test_missing_registry_returns_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indexes_dir = Path(tmp)
            active_path = indexes_dir / "active_index.json"
            active_path.write_text(
                json.dumps(
                    {"index_version_id": "v1", "promoted_at": "2026-02-02T12:00:00Z", "evaluation_report_id": "e1"},
                    sort_keys=True,
                )
            )
            backend = FakeEmbeddingBackend()
            llm = FakeInferenceBackend()
            config = _eval_config()
            out, _, _ = run_inference_using_active_index(
                "query",
                indexes_dir,
                backend,
                top_k=5,
                llm=llm,
                evaluation_config=config,
            )
            self.assertFalse(out.found)
            self.assertEqual(out.refusal_reason, REFUSAL_NO_ACTIVE_INDEX)


# ---------------------------------------------------------------------------
# Registry serialization deterministic
# ---------------------------------------------------------------------------


class TestRegistrySerializationDeterministic(unittest.TestCase):
    def test_save_twice_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from pipeline.promotion.index_registry import RegistryEntry

            path = Path(tmp) / "registry.json"
            entry = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path="/indexes/v1.json",
                evaluation_report_id="e1",
                status="promoted",
                notes="test",
            )
            save_registry(path, [entry])
            content1 = path.read_bytes()
            save_registry(path, [entry])
            content2 = path.read_bytes()
            self.assertEqual(content1, content2, "Registry serialization must be deterministic")

    def test_entries_sorted_keys_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from pipeline.promotion.index_registry import RegistryEntry

            path = Path(tmp) / "registry.json"
            entry = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path="/indexes/v1.json",
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(path, [entry])
            data = json.loads(path.read_text())
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            keys = list(data[0].keys())
            self.assertEqual(keys, sorted(keys), "Registry entry keys must be sorted")


# ---------------------------------------------------------------------------
# Promotion does not alter index contents
# ---------------------------------------------------------------------------


class TestPromotionDoesNotAlterIndexContents(unittest.TestCase):
    def test_promote_does_not_mutate_index_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from pipeline.retrieval.retrieval_engine import IndexEntry, IndexSnapshot

            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            index_path = Path(tmp) / "v1.json"
            entry = IndexEntry(
                chunk_id="c1",
                document_id="d1",
                raw_text="Normal readable content for retrieval.",
                embedding_vector_id="ev1",
                vector=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            )
            index = IndexSnapshot(index_version_id="v1", entries=[entry])
            save_index(index, index_path)
            before_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
            register_index(
                registry_path,
                index_version_id="v1",
                index_path=str(index_path),
                created_at="2026-02-02T12:00:00Z",
            )
            report = _evaluation_report(overall_pass=True, index_version="v1")
            promote_index("v1", report, registry_path, active_path)
            after_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
            self.assertEqual(before_hash, after_hash, "Promotion must not alter index file contents")
            loaded = load_index(index_path)
            self.assertEqual(loaded.index_version_id, index.index_version_id)
            self.assertEqual(len(loaded.entries), len(index.entries))


# ---------------------------------------------------------------------------
# Typed errors: index not in registry, version mismatch
# ---------------------------------------------------------------------------


class TestPromotionTypedErrors(unittest.TestCase):
    def test_index_not_in_registry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            from pipeline.promotion.index_registry import RegistryEntry

            entry = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path=str(Path(tmp) / "v1.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(registry_path, [entry])
            report = _evaluation_report(overall_pass=True, index_version="v2")
            with self.assertRaises(IndexNotInRegistryError):
                promote_index("v2", report, registry_path, active_path)

    def test_evaluation_report_version_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            active_path = Path(tmp) / "active_index.json"
            from pipeline.promotion.index_registry import RegistryEntry

            entry = RegistryEntry(
                index_version_id="v1",
                created_at="2026-02-02T12:00:00Z",
                index_path=str(Path(tmp) / "v1.json"),
                evaluation_report_id=None,
                status="created",
                notes=None,
            )
            save_registry(registry_path, [entry])
            report = _evaluation_report(overall_pass=True, index_version="v2")
            with self.assertRaises(IndexVersionMismatchError):
                promote_index("v1", report, registry_path, active_path)
