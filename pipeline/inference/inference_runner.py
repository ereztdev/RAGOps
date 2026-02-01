"""
Grounded inference runner: produce answers only when supported by evaluated retrieval.
Gates strictly on EvaluationReport.overall_pass; uses only retrieved chunk raw_text as context.
Deterministic for fixed inputs. No promotion, re-ranking, retrieval, or evaluation internally.
"""
from __future__ import annotations

import re
from typing import Protocol

from core.schema import (
    AnswerWithCitations,
    EvaluationReport,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Refusal reason constants (enumerated)
# ---------------------------------------------------------------------------

REFUSAL_NO_RETRIEVAL_HITS = "no_retrieval_hits"
REFUSAL_EVALUATION_FAILED_PREFIX = "evaluation_failed:"
REFUSAL_UNSUPPORTED_BY_CONTEXT = "unsupported_by_context"


# ---------------------------------------------------------------------------
# Inference backend abstraction
# ---------------------------------------------------------------------------


class InferenceBackend(Protocol):
    """Interface for generating answers from query and context. No inference in this module beyond backend call."""

    def generate(self, query: str, context: str) -> str:
        """Produce answer text from query and assembled context. Deterministic when backend is deterministic."""
        ...


class FakeInferenceBackend:
    """
    Deterministic, test-only backend. Returns first sentence from context that overlaps query terms.
    MUST NOT hallucinate: only returns text present in context.
    """

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return ""
        query_terms = {t.lower() for t in re.findall(r"\w+", query) if len(t) > 1}
        if not query_terms:
            return _first_sentence(context)
        sentences = re.split(r"[.!?]\s+", context)
        for sent in sentences:
            sent_lower = sent.lower()
            if any(term in sent_lower for term in query_terms):
                sent = sent.strip()
                if not sent.endswith((".", "!", "?")):
                    sent += "."
                return sent
        return _first_sentence(context)


def _first_sentence(text: str) -> str:
    """First sentence of text, or full text if no sentence boundary. Deterministic."""
    text = text.strip()
    if not text:
        return ""
    match = re.search(r"^[^.!?]*[.!?]?", text)
    if match:
        return match.group(0).strip() or text
    return text


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
