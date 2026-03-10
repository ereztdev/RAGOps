"""
Shared tokenizer for chunking and BM25.
Whitespace tokenization; deterministic. BM25 must use same tokenization as chunking.
"""
from __future__ import annotations


def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer. Deterministic."""
    return text.split()
