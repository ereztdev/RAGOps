"""
Ingestion pipeline: responsible for hashing, parsing, and chunking documents.
"""
import hashlib
import os

from pypdf import PdfReader


def run_ingestion_pipeline(pdf_path: str) -> dict:
    """
    Ingest a PDF file and return structured chunks with metadata.
    """
    path = os.path.abspath(pdf_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with open(path, "rb") as f:
        raw = f.read()

    document_hash = hashlib.sha256(raw).hexdigest()

    try:
        reader = PdfReader(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {pdf_path}") from e

    # Extract text per page, minimal normalization (strip only)
    page_texts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text is None:
            text = ""
        page_texts.append((i + 1, text.strip()))

    # Page-based chunking: one chunk per page (skip empty pages)
    chunks = []
    chunk_id = 0
    for page_number, text in page_texts:
        if text:
            chunk_id += 1
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": text,
            })

    return {
        "document_hash": document_hash,
        "source_path": path,
        "chunks": chunks,
    }


if __name__ == "__main__":
    pdf_paths = [
        "data/test_pdfs/ragops_semantic_test_pdf.pdf",
        "data/test_pdfs/ragops_gibberish_test_pdf.pdf",
    ]
    for pdf_path in pdf_paths:
        result = run_ingestion_pipeline(pdf_path)
        print("---", pdf_path)
        print("Hash:", result["document_hash"])
        print("Chunks:", len(result["chunks"]))
        if result["chunks"]:
            print("First chunk preview:", result["chunks"][0]["text"][:200])
        else:
            print("First chunk preview: (no extractable text)")
        print()
