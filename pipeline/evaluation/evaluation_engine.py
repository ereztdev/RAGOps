"""
Evaluation engine: scores retrieval quality and produces stored evaluation reports.
Used to gate index promotion. No inference; only judges retrieval behavior.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schema import (
    EvaluationConfig,
    EvaluationReport,
    KnownAnswerFixture,
    RetrievalResult,
    SignalResult,
)


# ---------------------------------------------------------------------------
# Deterministic evaluation_id
# ---------------------------------------------------------------------------


def evaluation_id_from_index_and_query(index_version: str, query_hash: str) -> str:
    """Deterministic evaluation_id = hash(index_version + query_hash)."""
    canonical = f"{index_version}:{query_hash}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def query_hash_from_query(query: str) -> str:
    """Deterministic query_hash from query string."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Gibberish detection (entropy + alphabetic ratio, no inference)
# ---------------------------------------------------------------------------


def _alphabetic_ratio(text: str) -> float:
    """Ratio of alphabetic (a-z A-Z) characters in text. Empty -> 0."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def _character_entropy(text: str) -> float:
    """Shannon entropy of character distribution (bits). Empty -> 0."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def _is_gibberish(text: str, config: EvaluationConfig) -> bool:
    """
    Flag text as low-signal (gibberish) if alphabetic ratio too low or entropy too high.
    Deterministic; no inference.
    """
    if not text.strip():
        return True
    if _alphabetic_ratio(text) < config.min_alphabetic_ratio:
        return True
    if _character_entropy(text) > config.max_entropy:
        return True
    return False


# ---------------------------------------------------------------------------
# Signals (boolean + evidence)
# ---------------------------------------------------------------------------

EMPTY_RETRIEVAL = "empty_retrieval"
LOW_CONFIDENCE = "low_confidence"
GIBBERISH_DETECTED = "gibberish_detected"
KNOWN_ANSWER_MISS = "known_answer_miss"
HIT_COUNT_INSUFFICIENT = "hit_count_insufficient"
TEST_ONLY_BACKEND = "test_only_backend"


def _signal_empty_retrieval(result: RetrievalResult) -> SignalResult:
    """No hits returned."""
    hit_count = len(result.hits)
    passed = hit_count > 0
    details = {"hit_count": hit_count, "message": "no hits returned" if not passed else "at least one hit"}
    return SignalResult(passed=passed, details=details)


def _signal_low_confidence(result: RetrievalResult, config: EvaluationConfig) -> SignalResult:
    """Top similarity_score below threshold."""
    if not result.hits:
        return SignalResult(passed=False, details={"message": "no hits", "top_score": None})
    top_score = result.hits[0].similarity_score
    passed = top_score >= config.min_confidence_score
    details = {"top_score": top_score, "threshold": config.min_confidence_score}
    return SignalResult(passed=passed, details=details)


def _signal_gibberish_detected(result: RetrievalResult, config: EvaluationConfig) -> SignalResult:
    """Any retrieved chunk flagged as low-signal (gibberish)."""
    if not result.hits:
        return SignalResult(passed=True, details={"message": "no hits to check"})
    gibberish_chunk_ids: list[str] = []
    for hit in result.hits:
        if _is_gibberish(hit.raw_text, config):
            gibberish_chunk_ids.append(hit.chunk_id)
    passed = len(gibberish_chunk_ids) == 0
    details = {"gibberish_chunk_ids": gibberish_chunk_ids, "checked": len(result.hits)}
    return SignalResult(passed=passed, details=details)


def _signal_known_answer_miss(
    result: RetrievalResult,
    fixture: KnownAnswerFixture,
) -> SignalResult:
    """Expected chunk_id or document_id not present in hits (only when fixture exists)."""
    hit_chunk_ids = {h.chunk_id for h in result.hits}
    hit_document_ids = {h.document_id for h in result.hits}
    missing_chunks = fixture.expected_chunk_ids - hit_chunk_ids
    missing_docs = fixture.expected_document_ids - hit_document_ids
    passed = len(missing_chunks) == 0 and len(missing_docs) == 0
    details = {
        "expected_chunk_ids": list(fixture.expected_chunk_ids),
        "expected_document_ids": list(fixture.expected_document_ids),
        "missing_chunk_ids": list(missing_chunks),
        "missing_document_ids": list(missing_docs),
    }
    return SignalResult(passed=passed, details=details)


def _signal_hit_count_insufficient(result: RetrievalResult) -> SignalResult:
    """
    Fail only when corpus has enough chunks to satisfy requested_k but returned fewer.
    effective_k = min(requested_k, corpus_size). Small corpora (corpus_size < requested_k)
    can pass if retrieval quality is good; large corpora must supply requested_k hits.
    """
    hit_count = len(result.hits)
    requested_k = result.top_k_requested
    corpus_size = result.corpus_size
    effective_k = min(requested_k, corpus_size)
    # FAIL only when: corpus_size >= requested_k AND returned_hits < requested_k
    passed = corpus_size < requested_k or hit_count >= requested_k
    details = {
        "hit_count": hit_count,
        "top_k_requested": requested_k,
        "corpus_size": corpus_size,
        "effective_k": effective_k,
    }
    return SignalResult(passed=passed, details=details)


# ---------------------------------------------------------------------------
# Report assembly and overall_pass
# ---------------------------------------------------------------------------


def _signal_test_only_backend(result: RetrievalResult) -> SignalResult:
    """Fail when index was built with test-only (fake) backend; such indexes must not pass evaluation."""
    from pipeline.embedding.embedding_engine import is_test_only_index_version
    is_test_only = is_test_only_index_version(result.index_version)
    details = {
        "message": "index built with test-only (fake) backend; cannot pass evaluation" if is_test_only else "production embedding backend",
        "index_version": result.index_version,
    }
    return SignalResult(passed=not is_test_only, details=details)


def _build_signals(
    result: RetrievalResult,
    config: EvaluationConfig,
    fixture: KnownAnswerFixture | None,
) -> dict[str, SignalResult]:
    """Build all signals. known_answer_miss only when fixture provided."""
    signals: dict[str, SignalResult] = {}
    signals[EMPTY_RETRIEVAL] = _signal_empty_retrieval(result)
    signals[LOW_CONFIDENCE] = _signal_low_confidence(result, config)
    signals[GIBBERISH_DETECTED] = _signal_gibberish_detected(result, config)
    if fixture is not None:
        signals[KNOWN_ANSWER_MISS] = _signal_known_answer_miss(result, fixture)
    signals[HIT_COUNT_INSUFFICIENT] = _signal_hit_count_insufficient(result)
    signals[TEST_ONLY_BACKEND] = _signal_test_only_backend(result)

    return signals


def _overall_pass(signals: dict[str, SignalResult]) -> bool:
    """True only if all signals pass. Deterministic for fixed input."""
    return all(sr.passed for sr in signals.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Fixtures: optional dict query -> KnownAnswerFixture (only used when key matches current query)
FixturesMap = dict[str, KnownAnswerFixture]


def evaluate_retrieval(
    retrieval_result: RetrievalResult,
    query: str,
    config: EvaluationConfig,
    fixtures: FixturesMap | None = None,
    evaluations_dir: Path | str | None = None,
) -> EvaluationReport:
    """
    Score retrieval quality and produce a stored evaluation report.
    No inference; only judges retrieval behavior.
    - overall_pass is derived: True only if all mandatory signals pass.
    - Known-answer check is mandatory only when a fixture exists for this query.
    - Deterministic for fixed input.
    If evaluations_dir is set, persist report as JSON under evaluations_dir.
    """
    query_hash = query_hash_from_query(query)
    index_version = retrieval_result.index_version
    evaluation_id = evaluation_id_from_index_and_query(index_version, query_hash)

    fixture: KnownAnswerFixture | None = None
    if fixtures and query in fixtures:
        fixture = fixtures[query]

    signals = _build_signals(retrieval_result, config, fixture)
    overall_pass = _overall_pass(signals)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    report = EvaluationReport(
        evaluation_id=evaluation_id,
        index_version=index_version,
        query_hash=query_hash,
        top_k_requested=retrieval_result.top_k_requested,
        hit_count=len(retrieval_result.hits),
        signals=signals,
        overall_pass=overall_pass,
        created_at=created_at,
    )

    if evaluations_dir is not None:
        _persist_report(report, Path(evaluations_dir))

    return report


def _persist_report(report: EvaluationReport, base_dir: Path) -> None:
    """Write one JSON file per evaluation_id under base_dir (e.g. storage/evaluations/)."""
    base_dir = base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{report.evaluation_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_serializable(), f, indent=2, sort_keys=True)


def load_evaluation_report(path: Path | str) -> EvaluationReport:
    """
    Load EvaluationReport from JSON file. Fails loudly on missing file or invalid schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation report root must be dict, got {type(data).__name__}")
    required = (
        "evaluation_id",
        "index_version",
        "query_hash",
        "top_k_requested",
        "hit_count",
        "signals",
        "overall_pass",
        "created_at",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"Evaluation report missing required field: {key!r}")
    if not isinstance(data["signals"], dict):
        raise ValueError(f"Evaluation report signals must be dict, got {type(data['signals']).__name__}")
    signals: dict[str, SignalResult] = {}
    for name, raw in data["signals"].items():
        if not isinstance(raw, dict) or "passed" not in raw or "details" not in raw:
            raise ValueError(f"Evaluation report signal {name!r} must have passed and details")
        signals[name] = SignalResult(passed=bool(raw["passed"]), details=raw["details"])
    return EvaluationReport(
        evaluation_id=data["evaluation_id"],
        index_version=data["index_version"],
        query_hash=data["query_hash"],
        top_k_requested=int(data["top_k_requested"]),
        hit_count=int(data["hit_count"]),
        signals=signals,
        overall_pass=bool(data["overall_pass"]),
        created_at=data["created_at"],
    )
