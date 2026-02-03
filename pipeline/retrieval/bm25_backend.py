"""
BM25 keyword search backend.
Uses TF-IDF with BM25 weighting (k1=1.2, b=0.75).
No external dependencies; built-in Python only.
Deterministic for same corpus and query.
"""
from __future__ import annotations

import math
from collections import Counter


def _simple_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer (same as chunking). Deterministic."""
    return text.split()


class Bm25Backend:
    """
    BM25 scoring for keyword-based retrieval.

    Algorithm:
    - TF-IDF with BM25 weighting
    - k1 = 1.2 (term frequency saturation)
    - b = 0.75 (length normalization)
    - IDF = log((N - df + 0.5) / (df + 0.5) + 1)
    - BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))

    Deterministic: same corpus + query -> same scores.
    """

    def __init__(
        self,
        corpus: list[str],  # List of raw_text from IndexEntry
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """
        Initialize BM25 backend with corpus.

        Args:
            corpus: List of document texts (one per IndexEntry)
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)

        # Tokenize all documents
        self.doc_tokens: list[list[str]] = [_simple_tokenize(doc) for doc in corpus]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.N if self.N > 0 else 0.0

        # Compute document frequencies (df) for all terms
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query: str) -> list[float]:
        """
        Compute BM25 scores for query against all corpus documents.

        Args:
            query: Query string

        Returns:
            List of BM25 scores (one per corpus document, same order)
        """
        if self.N == 0:
            return []

        query_tokens = _simple_tokenize(query)
        if not query_tokens:
            return [0.0] * self.N

        scores = []
        for i, tokens in enumerate(self.doc_tokens):
            doc_len = self.doc_lens[i]
            term_freqs = Counter(tokens)

            score = 0.0
            for term in set(query_tokens):
                if term not in term_freqs:
                    continue

                tf = term_freqs[term]
                df = self.df.get(term, 0)

                # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

                # BM25 term score; avoid div by zero when avg_doc_len is 0
                norm = self.avg_doc_len if self.avg_doc_len > 0 else 1.0
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / norm)
                )
                score += idf * (tf * (self.k1 + 1) / denominator)

            scores.append(score)

        return scores
