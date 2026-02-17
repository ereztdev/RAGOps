"""Docling-based PDF parser for structure-aware ingestion."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ragops.ingest.docling_parser")

# Lazy import to keep Docling optional
_DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    _DOCLING_AVAILABLE = True
except ImportError:
    pass


def is_docling_available() -> bool:
    return _DOCLING_AVAILABLE


def parse_pdf_with_docling(
    pdf_path: str | Path,
    tokenizer: str = "BAAI/bge-base-en-v1.5",
    max_tokens: int = 400,
) -> list[dict]:
    """
    Parse PDF using Docling. Returns list of chunk dicts with keys:
    - raw_text: str
    - page_number: int | None
    - chapter: str | None
    - section: str | None
    - domain_hint: str | None
    - headings: list[str]  (section hierarchy)
    - content_type: str | None  (e.g. "table", "text", "list")
    """
    if not _DOCLING_AVAILABLE:
        raise ImportError("Docling is not installed. pip install docling")

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Use HuggingFaceTokenizer aligned to BGE embedding model (avoids deprecation)
    hf_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=hf_tokenizer,
        merge_peers=True,
    )
    chunks = list(chunker.chunk(doc))

    parsed_chunks = []
    for chunk in chunks:
        # Extract text via chunker's contextualize (includes heading context)
        text = chunker.contextualize(chunk)

        # Extract metadata
        headings: list[str] = []
        page_number: Optional[int] = None
        content_type: Optional[str] = None

        # Docling chunks have meta with headings, page info
        if hasattr(chunk, "meta") and chunk.meta is not None:
            meta = chunk.meta
            if hasattr(meta, "headings") and meta.headings:
                headings = list(meta.headings)
            if hasattr(meta, "origin") and meta.origin is not None:
                origin = meta.origin
                if hasattr(origin, "page_no"):
                    page_number = getattr(origin, "page_no", None)
                elif hasattr(origin, "page_number"):
                    page_number = getattr(origin, "page_number", None)
            if hasattr(meta, "doc_items") and meta.doc_items:
                for item in meta.doc_items:
                    if hasattr(item, "label"):
                        label = getattr(item, "label", None)
                        if label is not None:
                            content_type = str(label) if not isinstance(label, str) else label
                            break

        # Derive chapter/section from headings
        chapter: Optional[str] = None
        section: Optional[str] = None
        if headings:
            for h in headings:
                h_str = str(h) if not isinstance(h, str) else h
                h_lower = h_str.lower()
                if "chapter" in h_lower:
                    chapter = h_str
                elif section is None:
                    section = h_str

        # Derive domain_hint from headings and text
        domain_hint = _infer_domain_from_headings(headings, text)

        parsed_chunks.append(
            {
                "raw_text": text,
                "page_number": page_number,
                "chapter": chapter,
                "section": section,
                "domain_hint": domain_hint,
                "headings": headings,
                "content_type": content_type,
            }
        )

    logger.info("Docling parsed %s: %d chunks", pdf_path, len(parsed_chunks))
    return parsed_chunks


def _infer_domain_from_headings(headings: list[str], text: str) -> Optional[str]:
    """Infer domain hint from section headings, falling back to text keywords."""
    headings_str = " ".join(str(h) for h in headings).lower()
    combined = headings_str + " " + text[:1200].lower()

    if "troubleshoot" in combined:
        return "Troubleshooting"
    if "removal" in combined or "installation" in combined:
        return "Engine"
    if "generator" in combined:
        return "Generator"
    if "maintenance" in combined or "lubrication" in combined or "filter" in combined:
        return "Maintenance"
    if "engine" in combined or "fuel" in combined or "coolant" in combined:
        return "Engine"
    if "electric" in combined or "circuit" in combined or "wiring" in combined:
        return "Electrical"
    if "repair" in combined or "torque" in combined or "adjustment" in combined:
        return "Maintenance"
    return None
