"""
Inference backend abstraction: protocol and test-only fake implementation.
"""
from __future__ import annotations

import re
from typing import Protocol


class InferenceBackend(Protocol):
    """Interface for generating answers from query and context. No inference in this module beyond backend call."""

    @property
    def model_id(self) -> str:
        """Identifier for the model/backend (e.g. 'fake', 'llama3.1:8b')."""
        ...

    def generate(self, query: str, context: str) -> str:
        """Produce answer text from query and assembled context. Deterministic when backend is deterministic."""
        ...


def _first_sentence(text: str) -> str:
    """First sentence of text, or full text if no sentence boundary. Deterministic."""
    text = text.strip()
    if not text:
        return ""
    match = re.search(r"^[^.!?]*[.!?]?", text)
    if match:
        return match.group(0).strip() or text
    return text


class FakeInferenceBackend:
    """
    Deterministic, test-only backend. Returns first sentence from context that overlaps query terms.
    MUST NOT hallucinate: only returns text present in context.
    """

    @property
    def model_id(self) -> str:
        return "fake"

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
