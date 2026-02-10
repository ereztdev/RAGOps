"""
PDF parser: extract per-page/chunk metadata (chapter, section, page_num, domain_hint).
Graceful degradation: if extraction fails, metadata is None; no crash.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

# Regex patterns for structure (graceful: no match → None)
CHAPTER_PATTERN = re.compile(
    r"Chapter\s+(\d+(?:-\d+)?)",
    re.IGNORECASE,
)
SECTION_PATTERN = re.compile(
    r"Section\s+(\d+)\.\s*(.+?)(?=\n|Section\s+\d+|Chapter\s+\d+|$)",
    re.IGNORECASE | re.DOTALL,
)

# Domain hint inferred from section title keywords (same semantic set as domain_detector)
DOMAIN_HINT_KEYWORDS: dict[str, list[str]] = {
    "Engine": ["engine", "starter", "oil", "coolant", "temperature", "fuel", "rpm", "crankcase"],
    "Generator": ["generator", "voltage", "regulator", "contactor", "rectifier", "field", "brush"],
    "Troubleshooting": ["troubleshoot", "fault", "symptom", "will not", "won't", "problem", "fails", "stops"],
}


@dataclass
class PageMetadata:
    """Metadata extracted from one page (or None if extraction fails)."""
    chapter: Optional[str] = None
    section: Optional[str] = None
    section_title: Optional[str] = None
    domain_hint: Optional[str] = None


def extract_metadata_from_page_text(page_number: int, text: str) -> Optional[PageMetadata]:
    """
    Extract chapter, section, and domain_hint from page text using regex.
    If regex fails or text is empty, return None (graceful degradation).

    Args:
        page_number: 1-based page number.
        text: Raw page text.

    Returns:
        PageMetadata or None if extraction fails or text is empty.
    """
    if not text or not text.strip():
        return None
    try:
        chapter: Optional[str] = None
        section: Optional[str] = None
        section_title: Optional[str] = None

        ch_match = CHAPTER_PATTERN.search(text)
        if ch_match:
            chapter = ch_match.group(1).strip()

        sec_match = SECTION_PATTERN.search(text)
        if sec_match:
            section = sec_match.group(1).strip()
            raw_title = (sec_match.group(2) or "").strip()
            # First line or first 80 chars as section title
            section_title = raw_title.split("\n")[0][:80].strip() if raw_title else None

        domain_hint = _infer_domain_hint_from_text(text)
        return PageMetadata(
            chapter=chapter or None,
            section=section or None,
            section_title=section_title or None,
            domain_hint=domain_hint,
        )
    except Exception:
        return None


def _infer_domain_hint_from_text(text: str) -> Optional[str]:
    """
    Infer single domain from text using keyword match (section title or body).
    If ambiguous or no match, return None.
    """
    if not text or not text.strip():
        return None
    low = text.lower()
    matched: List[str] = []
    for domain, keywords in DOMAIN_HINT_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                matched.append(domain)
                break
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        # Prefer Troubleshooting if present (often section header)
        if "Troubleshooting" in matched:
            return "Troubleshooting"
        return matched[0]
    return None


def chunk_id_from_page_and_index(page_num: int, chunk_index: int) -> str:
    """
    Deterministic human-readable chunk id for display: chunk_052_01.
    Use when page numbers are available.
    """
    return f"chunk_{page_num:03d}_{chunk_index:02d}"


def chunk_id_hash_fallback(document_id: str, source_type: str, page_number: int, chunk_index: int) -> str:
    """
    Hash-based chunk_id when page numbers unavailable (backward compatible).
    """
    canonical = f"{document_id}:{source_type}:{page_number}:{chunk_index}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
