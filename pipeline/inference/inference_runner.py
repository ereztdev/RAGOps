"""
Grounded inference runner: produce answers only when supported by evaluated retrieval.
Gates strictly on EvaluationReport.overall_pass; uses only retrieved chunk raw_text as context.
Deterministic for fixed inputs. No promotion, re-ranking, retrieval, or evaluation internally.
Inference path that uses the active index: run_inference_using_active_index; refuses when no active index.
"""
from __future__ import annotations

from pathlib import Path

from core.schema import (
    AnswerWithCitations,
    EvaluationConfig,
    EvaluationReport,
    RetrievalResult,
)
from pipeline.embedding.embedding_engine import EmbeddingBackend
from pipeline.evaluation.evaluation_engine import (
    FixturesMap,
    evaluate_retrieval,
)
from pipeline.inference.backends.base import FakeInferenceBackend, InferenceBackend
from pipeline.promotion.index_registry import resolve_active_index
from pipeline.retrieval.retrieval_engine import retrieve
from storage.vector_index_store import load_index


# ---------------------------------------------------------------------------
# Refusal reason constants (enumerated)
# ---------------------------------------------------------------------------

REFUSAL_NO_RETRIEVAL_HITS = "no_retrieval_hits"
REFUSAL_EVALUATION_FAILED_PREFIX = "evaluation_failed:"
REFUSAL_UNSUPPORTED_BY_CONTEXT = "unsupported_by_context"
REFUSAL_NO_ACTIVE_INDEX = "no_active_index"


# ---------------------------------------------------------------------------
# Gating and context assembly
# ---------------------------------------------------------------------------


def _refusal_reason_evaluation_failed(report: EvaluationReport) -> str:
    """Build refusal_reason from failed signals. Which signals failed must be explained."""
    failed = [name for name, sr in report.signals.items() if not sr.passed]
    failed.sort()
    return f"{REFUSAL_EVALUATION_FAILED_PREFIX} {', '.join(failed)}"


def _assemble_context(retrieval_result: RetrievalResult) -> tuple[str, list[str]]:
    """
    Context = concatenation of hit.raw_text in retrieval order.
    Returns (context_string, list of chunk_ids used).
    No truncation. Citations come from retrieval_result.hits only.
    """
    parts: list[str] = []
    chunk_ids: list[str] = []
    for hit in retrieval_result.hits:
        parts.append(hit.raw_text)
        chunk_ids.append(hit.chunk_id)
    context = "\n\n".join(parts)
    return context, chunk_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_grounded_inference(
    query: str,
    retrieval_result: RetrievalResult,
    evaluation_report: EvaluationReport,
    llm: InferenceBackend,
) -> AnswerWithCitations:
    """
    Produce an answer only when evaluation passes and hits exist. Otherwise refuse with explicit reason.
    - Gate strictly on evaluation_report.overall_pass.
    - Use only retrieved chunk raw_text as context.
    - Citations are chunk_ids from retrieval_result.hits (all used as context).
    - Deterministic for fixed inputs.
    """
    # Empty hits: refuse regardless of evaluation
    if not retrieval_result.hits:
        return AnswerWithCitations(
            answer_text=None,
            citation_chunk_ids=[],
            found=False,
            refusal_reason=REFUSAL_NO_RETRIEVAL_HITS,
        )

    # Evaluation failed: refuse and explain which signals failed
    if not evaluation_report.overall_pass:
        return AnswerWithCitations(
            answer_text=None,
            citation_chunk_ids=[],
            found=False,
            refusal_reason=_refusal_reason_evaluation_failed(evaluation_report),
        )

    # Assemble context and generate; citations = chunk_ids used
    context, citation_chunk_ids = _assemble_context(retrieval_result)
    answer_text = llm.generate(query, context)

    # If backend returns empty, treat as unsupported (no invented text)
    if not answer_text or not answer_text.strip():
        return AnswerWithCitations(
            answer_text=None,
            citation_chunk_ids=[],
            found=False,
            refusal_reason=REFUSAL_UNSUPPORTED_BY_CONTEXT,
        )

    return AnswerWithCitations(
        answer_text=answer_text.strip(),
        citation_chunk_ids=citation_chunk_ids,
        found=True,
        refusal_reason=None,
    )


# ---------------------------------------------------------------------------
# Inference using active index only (enforcement boundary)
# ---------------------------------------------------------------------------


def run_inference_using_active_index(
    query: str,
    indexes_dir: Path | str,
    embedding_backend: EmbeddingBackend,
    top_k: int,
    llm: InferenceBackend,
    evaluation_config: EvaluationConfig,
    fixtures: FixturesMap | None = None,
    evaluations_dir: Path | str | None = None,
) -> AnswerWithCitations:
    """
    Run grounded inference using the promoted active index only.
    Resolves index_version_id and index_path from active_index.json + registry;
    loads index, runs retrieval, evaluation, then run_grounded_inference.
    Missing or invalid active index causes refusal (no_active_index); no defaults, no fallbacks.
    """
    indexes_dir = Path(indexes_dir)
    registry_path = indexes_dir / "registry.json"
    active_index_path = indexes_dir / "active_index.json"
    try:
        _index_version_id, index_path = resolve_active_index(registry_path, active_index_path)
    except (FileNotFoundError, ValueError):
        return AnswerWithCitations(
            answer_text=None,
            citation_chunk_ids=[],
            found=False,
            refusal_reason=REFUSAL_NO_ACTIVE_INDEX,
        )
    path = Path(index_path)
    if not path.is_absolute():
        path = indexes_dir / path
    index = load_index(path)
    retrieval_result = retrieve(query, index, embedding_backend, top_k)
    evaluation_report = evaluate_retrieval(
        retrieval_result,
        query,
        evaluation_config,
        fixtures=fixtures,
        evaluations_dir=evaluations_dir,
    )
    return run_grounded_inference(query, retrieval_result, evaluation_report, llm)
