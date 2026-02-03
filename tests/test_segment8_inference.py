"""
Segment 8: Grounded inference runner — answer only when supported by evaluated retrieval.
Tests: evaluation_fail → refusal; empty hits → refusal; success → answer + citations;
determinism; refusal_reason correctness; citations match chunk_ids used.
No disk IO. No network. No randomness.
"""
from __future__ import annotations

import unittest

from core.schema import (
    EvaluationReport,
    RetrievalHit,
    RetrievalResult,
    SignalResult,
)
from pipeline.evaluation.evaluation_engine import (
    EMPTY_RETRIEVAL,
    HIT_COUNT_INSUFFICIENT,
    LOW_CONFIDENCE,
)
from pipeline.inference.inference_runner import (
    FakeInferenceBackend,
    REFUSAL_EVALUATION_FAILED_PREFIX,
    REFUSAL_NO_RETRIEVAL_HITS,
    REFUSAL_UNSUPPORTED_BY_CONTEXT,
    run_grounded_inference,
)


def _hit(
    chunk_id: str = "chunk_a",
    document_id: str = "doc1",
    raw_text: str = "The capital of France is Paris. It is a major European city.",
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


def _retrieval_result(
    hits: list[RetrievalHit] | None = None,
    index_version: str = "v1",
    top_k_requested: int = 5,
    truncated: bool = False,
    corpus_size: int = 1,
) -> RetrievalResult:
    return RetrievalResult(
        hits=hits if hits is not None else [_hit()],
        index_version=index_version,
        top_k_requested=top_k_requested,
        truncated=truncated,
        corpus_size=corpus_size,
    )


def _evaluation_report(
    overall_pass: bool,
    signals: dict[str, SignalResult] | None = None,
    evaluation_id: str = "eid1",
    index_version: str = "v1",
    query_hash: str = "qh1",
    top_k_requested: int = 5,
    hit_count: int = 1,
    created_at: str = "2026-01-31T12:00:00.000000Z",
) -> EvaluationReport:
    if signals is None:
        signals = {
            EMPTY_RETRIEVAL: SignalResult(passed=True, details={"hit_count": 1}),
            LOW_CONFIDENCE: SignalResult(passed=overall_pass, details={"top_score": 0.9}),
            HIT_COUNT_INSUFFICIENT: SignalResult(passed=overall_pass, details={"hit_count": 1, "top_k_requested": 5}),
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


# ---------------------------------------------------------------------------
# Evaluation fail → refusal, no answer
# ---------------------------------------------------------------------------


class TestEvaluationFailRefusal(unittest.TestCase):
    def test_evaluation_fail_returns_found_false(self) -> None:
        retrieval = _retrieval_result(hits=[_hit(raw_text="Paris is the capital.")])
        report = _evaluation_report(overall_pass=False)
        backend = FakeInferenceBackend()
        out = run_grounded_inference("What is the capital?", retrieval, report, backend)
        self.assertFalse(out.found)
        self.assertIsNone(out.answer_text)
        self.assertEqual(out.citation_chunk_ids, [])
        self.assertIsNotNone(out.refusal_reason)
        self.assertTrue(out.refusal_reason.startswith(REFUSAL_EVALUATION_FAILED_PREFIX))

    def test_refusal_reason_lists_failed_signals(self) -> None:
        retrieval = _retrieval_result(hits=[_hit()])
        report = _evaluation_report(
            overall_pass=False,
            signals={
                EMPTY_RETRIEVAL: SignalResult(passed=True, details={}),
                LOW_CONFIDENCE: SignalResult(passed=False, details={"top_score": 0.1}),
                HIT_COUNT_INSUFFICIENT: SignalResult(passed=False, details={"hit_count": 1, "top_k_requested": 5}),
            },
        )
        backend = FakeInferenceBackend()
        out = run_grounded_inference("query", retrieval, report, backend)
        self.assertFalse(out.found)
        self.assertIn("low_confidence", out.refusal_reason)
        self.assertIn("hit_count_insufficient", out.refusal_reason)


# ---------------------------------------------------------------------------
# Empty hits → refusal
# ---------------------------------------------------------------------------


class TestEmptyHitsRefusal(unittest.TestCase):
    def test_empty_hits_returns_found_false(self) -> None:
        retrieval = _retrieval_result(hits=[])
        report = _evaluation_report(overall_pass=True, hit_count=0)
        backend = FakeInferenceBackend()
        out = run_grounded_inference("Anything", retrieval, report, backend)
        self.assertFalse(out.found)
        self.assertIsNone(out.answer_text)
        self.assertEqual(out.citation_chunk_ids, [])
        self.assertEqual(out.refusal_reason, REFUSAL_NO_RETRIEVAL_HITS)

    def test_empty_hits_refusal_even_if_evaluation_pass(self) -> None:
        retrieval = _retrieval_result(hits=[])
        report = _evaluation_report(overall_pass=True, hit_count=0)
        out = run_grounded_inference("q", retrieval, report, FakeInferenceBackend())
        self.assertFalse(out.found)
        self.assertEqual(out.refusal_reason, REFUSAL_NO_RETRIEVAL_HITS)


# ---------------------------------------------------------------------------
# Successful inference → answer + citations
# ---------------------------------------------------------------------------


class TestSuccessfulInference(unittest.TestCase):
    def test_success_returns_answer_and_citations(self) -> None:
        hit1 = _hit(chunk_id="c1", raw_text="France capital is Paris. It is in Europe.")
        retrieval = _retrieval_result(hits=[hit1])
        report = _evaluation_report(overall_pass=True, hit_count=1)
        backend = FakeInferenceBackend()
        out = run_grounded_inference("What is the capital of France?", retrieval, report, backend)
        self.assertTrue(out.found)
        self.assertIsNotNone(out.answer_text)
        self.assertIn("Paris", out.answer_text or "")
        self.assertEqual(out.citation_chunk_ids, ["c1"])
        self.assertIsNone(out.refusal_reason)

    def test_citations_match_chunk_ids_used(self) -> None:
        hit_a = _hit(chunk_id="id_a", raw_text="Alpha content. Beta here.")
        hit_b = _hit(chunk_id="id_b", raw_text="Gamma content. Delta here.")
        retrieval = _retrieval_result(hits=[hit_a, hit_b])
        report = _evaluation_report(overall_pass=True, hit_count=2)
        out = run_grounded_inference("Alpha", retrieval, report, FakeInferenceBackend())
        self.assertTrue(out.found)
        self.assertEqual(out.citation_chunk_ids, ["id_a", "id_b"])

    def test_multiple_hits_context_concatenated_in_order(self) -> None:
        hit1 = _hit(chunk_id="first", raw_text="First chunk only.")
        hit2 = _hit(chunk_id="second", raw_text="Second chunk only.")
        retrieval = _retrieval_result(hits=[hit1, hit2])
        report = _evaluation_report(overall_pass=True, hit_count=2)
        out = run_grounded_inference("First", retrieval, report, FakeInferenceBackend())
        self.assertTrue(out.found)
        self.assertEqual(out.citation_chunk_ids, ["first", "second"])
        self.assertIn("First", out.answer_text or "")


# ---------------------------------------------------------------------------
# Deterministic output for fixed input
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_same_output(self) -> None:
        retrieval = _retrieval_result(hits=[_hit(raw_text="Deterministic answer here.")])
        report = _evaluation_report(overall_pass=True)
        backend = FakeInferenceBackend()
        out1 = run_grounded_inference("Deterministic", retrieval, report, backend)
        out2 = run_grounded_inference("Deterministic", retrieval, report, backend)
        self.assertEqual(out1.found, out2.found)
        self.assertEqual(out1.answer_text, out2.answer_text)
        self.assertEqual(out1.citation_chunk_ids, out2.citation_chunk_ids)
        self.assertEqual(out1.refusal_reason, out2.refusal_reason)

    def test_refusal_path_deterministic(self) -> None:
        retrieval = _retrieval_result(hits=[])
        report = _evaluation_report(overall_pass=True, hit_count=0)
        out1 = run_grounded_inference("q", retrieval, report, FakeInferenceBackend())
        out2 = run_grounded_inference("q", retrieval, report, FakeInferenceBackend())
        self.assertEqual(out1.refusal_reason, out2.refusal_reason)
        self.assertFalse(out1.found)
        self.assertFalse(out2.found)


# ---------------------------------------------------------------------------
# Refusal reason correctness
# ---------------------------------------------------------------------------


class TestRefusalReasonCorrectness(unittest.TestCase):
    def test_no_retrieval_hits_reason(self) -> None:
        retrieval = _retrieval_result(hits=[])
        report = _evaluation_report(overall_pass=True, hit_count=0)
        out = run_grounded_inference("q", retrieval, report, FakeInferenceBackend())
        self.assertEqual(out.refusal_reason, REFUSAL_NO_RETRIEVAL_HITS)

    def test_evaluation_failed_reason_format(self) -> None:
        retrieval = _retrieval_result(hits=[_hit()])
        report = _evaluation_report(
            overall_pass=False,
            signals={
                EMPTY_RETRIEVAL: SignalResult(passed=True, details={}),
                LOW_CONFIDENCE: SignalResult(passed=False, details={}),
            },
        )
        out = run_grounded_inference("q", retrieval, report, FakeInferenceBackend())
        self.assertTrue(out.refusal_reason.startswith(REFUSAL_EVALUATION_FAILED_PREFIX))
        self.assertIn("low_confidence", out.refusal_reason)


# ---------------------------------------------------------------------------
# Citations correspond to chunk_ids used
# ---------------------------------------------------------------------------


class TestCitationsCorrespondToChunks(unittest.TestCase):
    def test_single_hit_citation_is_that_chunk_id(self) -> None:
        hit = _hit(chunk_id="only_chunk_99", raw_text="Only content.")
        retrieval = _retrieval_result(hits=[hit])
        report = _evaluation_report(overall_pass=True)
        out = run_grounded_inference("Only", retrieval, report, FakeInferenceBackend())
        self.assertEqual(out.citation_chunk_ids, ["only_chunk_99"])

    def test_citations_in_retrieval_order(self) -> None:
        hits = [
            _hit(chunk_id="z_last", raw_text="Z last sentence."),
            _hit(chunk_id="a_first", raw_text="A first sentence."),
        ]
        retrieval = _retrieval_result(hits=hits)
        report = _evaluation_report(overall_pass=True, hit_count=2)
        out = run_grounded_inference("sentence", retrieval, report, FakeInferenceBackend())
        self.assertEqual(out.citation_chunk_ids, ["z_last", "a_first"])


# ---------------------------------------------------------------------------
# FakeInferenceBackend: no hallucination
# ---------------------------------------------------------------------------


class TestFakeInferenceBackend(unittest.TestCase):
    def test_returns_only_context_text(self) -> None:
        backend = FakeInferenceBackend()
        context = "The answer is forty-two. Nothing else."
        out = backend.generate("answer", context)
        self.assertIn("forty-two", out)
        self.assertTrue(out.strip().endswith(".") or out in context)

    def test_first_sentence_overlapping_query_terms(self) -> None:
        backend = FakeInferenceBackend()
        context = "Paris is the capital. London is not."
        out = backend.generate("Paris capital", context)
        self.assertIn("Paris", out)

    def test_empty_context_returns_empty(self) -> None:
        backend = FakeInferenceBackend()
        self.assertEqual(backend.generate("q", ""), "")
        self.assertEqual(backend.generate("q", "   "), "")


# ---------------------------------------------------------------------------
# Unsupported by context (empty answer from backend)
# ---------------------------------------------------------------------------


class TestUnsupportedByContext(unittest.TestCase):
    def test_empty_backend_answer_refuses_with_unsupported(self) -> None:
        class EmptyBackend:
            def generate(self, query: str, context: str) -> str:
                return ""

        retrieval = _retrieval_result(hits=[_hit(raw_text="Some content.")])
        report = _evaluation_report(overall_pass=True)
        out = run_grounded_inference("query", retrieval, report, EmptyBackend())
        self.assertFalse(out.found)
        self.assertEqual(out.refusal_reason, REFUSAL_UNSUPPORTED_BY_CONTEXT)
        self.assertIsNone(out.answer_text)
        self.assertEqual(out.citation_chunk_ids, [])
