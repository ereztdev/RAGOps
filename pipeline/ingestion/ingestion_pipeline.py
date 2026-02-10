"""
Ingestion pipeline: responsible for hashing, parsing, and chunking documents.
Emits typed Document and Chunk; serializes output using the canonical schema.
Uses ragops.ingest.pdf_parser for per-page metadata (chapter, section, domain_hint).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.schema import (
    Chunk,
    ChunkingConfig,
    Document,
    chunk_id_from_components,
    chunk_key,
    document_id_from_bytes,
)
from core.serialization import write_ingestion_output
from pypdf import PdfReader

from ragops.ingest.pdf_parser import (
    extract_metadata_from_page_text,
    PageMetadata,
)


def _simple_tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer. Deterministic."""
    return text.split()


def _token_offset_to_page(
    start_token_idx: int,
    page_token_offsets: list[tuple[int, int, int]],
) -> int:
    """
    Map global token index to page number. page_token_offsets is list of
    (page_number, start_token_idx, end_token_idx) in order.
    """
    for page_num, start, end in page_token_offsets:
        if start <= start_token_idx < end:
            return page_num
    if page_token_offsets:
        return page_token_offsets[-1][0]
    return 1


def _sliding_window_chunks(
    text: str,
    config: ChunkingConfig,
    document_id: str,
    source_type: str,
    page_token_offsets: list[tuple[int, int, int]],
    page_metadata: dict[int, Optional[PageMetadata]],
) -> list[Chunk]:
    """
    Split text into overlapping chunks using a sliding window.
    Assigns each chunk to a page via token offsets and attaches chapter/section/domain_hint.

    Args:
        text: Full document text (all pages concatenated)
        config: ChunkingConfig with chunk_size_tokens and overlap_tokens
        document_id: Deterministic document identifier
        source_type: "pdf", "txt", etc.
        page_token_offsets: (page_number, start_idx, end_idx) per page
        page_metadata: page_number -> PageMetadata or None

    Returns:
        List of Chunk objects with deterministic chunk_ids and optional metadata
    """
    tokens = _simple_tokenize(text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    stride = config.chunk_size_tokens - config.overlap_tokens
    chunk_index = 0

    for start_idx in range(0, len(tokens), stride):
        end_idx = min(start_idx + config.chunk_size_tokens, len(tokens))
        window_tokens = tokens[start_idx:end_idx]
        chunk_text = " ".join(window_tokens)

        if not chunk_text.strip():
            continue

        page_number = _token_offset_to_page(start_idx, page_token_offsets)
        key = chunk_key(document_id, page_number)
        cid = chunk_id_from_components(
            document_id, source_type, page_number, chunk_index
        )

        meta = page_metadata.get(page_number)
        chapter = meta.chapter if meta else None
        section = meta.section if meta else None
        domain_hint = meta.domain_hint if meta else None

        chunks.append(
            Chunk(
                chunk_id=cid,
                chunk_key=key,
                document_id=document_id,
                source_type=source_type,
                page_number=page_number,
                chunk_index=chunk_index,
                text=chunk_text,
                chapter=chapter,
                section=section,
                domain_hint=domain_hint,
            )
        )

        chunk_index += 1

        if end_idx >= len(tokens):
            break

    return chunks


def run_ingestion_pipeline(
    pdf_path: str,
    output_path: Optional[str | Path] = None,
) -> Document:
    """
    Ingest a PDF file and return a typed Document with ordered Chunks.
    If output_path is set, write ingestion_output.json there.
    """
    path = os.path.abspath(pdf_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with open(path, "rb") as f:
        raw = f.read()

    document_id = document_id_from_bytes(raw)
    source_type = Path(path).suffix.lower().lstrip(".") or "bin"
    if source_type not in ("pdf", "md", "txt"):
        source_type = "pdf" if path.lower().endswith(".pdf") else "bin"

    try:
        reader = PdfReader(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {pdf_path}") from e

    # Extract text per page, minimal normalization (strip only)
    page_texts: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text is None:
            text = ""
        page_texts.append((i + 1, text.strip()))

    # Concatenate all page texts into a single document string
    # (this enables cross-page window overlap)
    full_text_parts: list[str] = []
    first_nonempty_page: Optional[int] = None

    for page_number, text in page_texts:
        if text and first_nonempty_page is None:
            first_nonempty_page = page_number
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)

    if not full_text.strip():
        # No extractable text; return empty document
        document = Document(
            document_id=document_id,
            source_path=path,
            chunks=[],
        )
        if output_path is not None:
            write_ingestion_output(document, output_path)
        return document

    # Per-page metadata (graceful: None if extraction fails)
    page_metadata: dict[int, Optional[PageMetadata]] = {}
    for page_number, text in page_texts:
        try:
            meta = extract_metadata_from_page_text(page_number, text)
            page_metadata[page_number] = meta
        except Exception:
            page_metadata[page_number] = None

    # Token offsets per page: (page_number, start_token_idx, end_token_idx)
    all_tokens = _simple_tokenize(full_text)
    page_token_offsets: list[tuple[int, int, int]] = []
    offset = 0
    for page_number, text in page_texts:
        page_tokens = _simple_tokenize(text)
        start_idx = offset
        end_idx = offset + len(page_tokens)
        offset = end_idx
        page_token_offsets.append((page_number, start_idx, end_idx))

    config = ChunkingConfig()  # Default: 400 tokens, 100 overlap

    chunks = _sliding_window_chunks(
        full_text,
        config,
        document_id,
        source_type,
        page_token_offsets,
        page_metadata,
    )

    document = Document(
        document_id=document_id,
        source_path=path,
        chunks=chunks,
    )

    if output_path is not None:
        write_ingestion_output(document, output_path)

    return document


if __name__ == "__main__":
    pdf_paths = [
        "data/test_pdfs/ragops_semantic_test_pdf.pdf",
        "data/test_pdfs/ragops_gibberish_test_pdf.pdf",
    ]
    for pdf_path in pdf_paths:
        doc = run_ingestion_pipeline(pdf_path)
        print("---", pdf_path)
        print("document_id:", doc.document_id)
        print("Chunks:", len(doc.chunks))
        if doc.chunks:
            print("First chunk key:", doc.chunks[0].chunk_key)
            print("First chunk preview:", doc.chunks[0].text[:200])
        else:
            print("First chunk preview: (no extractable text)")
        print()
