"""
Ingestion pipeline: responsible for hashing, parsing, and chunking documents.
Emits typed Document and Chunk; serializes output using the canonical schema.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.schema import Chunk, Document, chunk_key, document_id_from_bytes
from core.serialization import write_ingestion_output
from pypdf import PdfReader


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

    # Page-based chunking: one chunk per page (skip empty pages)
    chunks: list[Chunk] = []
    for page_number, text in page_texts:
        if text:
            key = chunk_key(document_id, page_number)
            chunks.append(
                Chunk(
                    chunk_key=key,
                    document_id=document_id,
                    page_number=page_number,
                    text=text,
                )
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
