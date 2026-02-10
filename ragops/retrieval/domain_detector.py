"""
Domain detector: keyword-based detection for Engine, Generator, Troubleshooting.
Used for post-retrieval boosting (no change to BGE/BM25 scoring).
"""
from __future__ import annotations

from typing import List

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Engine": [
        "engine",
        "starter",
        "oil",
        "coolant",
        "temperature",
        "fuel",
        "rpm",
        "crankcase",
    ],
    "Generator": [
        "generator",
        "voltage",
        "regulator",
        "contactor",
        "rectifier",
        "field",
        "brush",
    ],
    "Troubleshooting": [
        "troubleshoot",
        "fault",
        "symptom",
        "will not",
        "won't",
        "problem",
        "fails",
        "stops",
        "temperature",
    ],
}


def detect_domains(query: str) -> List[str]:
    """
    Case-insensitive keyword matching; return all domains whose keywords appear in query.

    Args:
        query: User query string.

    Returns:
        List of domain names (e.g. ["Engine", "Troubleshooting"]). Order not guaranteed.
    """
    if not query or not query.strip():
        return []
    q = query.lower().strip()
    matched: list[str] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            # Word-boundary style: avoid "engines" matching only "engine" if we want exact;
            # spec says "keyword matching" so we allow substring (e.g. "engine" in "engines")
            if kw in q:
                matched.append(domain)
                break
    return matched
