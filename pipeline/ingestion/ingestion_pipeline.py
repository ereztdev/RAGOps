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

    # Build full text and char index -> page_number mapping
    full_parts = []
    char_to_page = []
    for page_num, text in page_texts:
        full_parts.append(text)
        char_to_page.extend([page_num] * len(text))
        full_parts.append("\n")
        char_to_page.append(page_num)  # newline
    full_text = "".join(full_parts).rstrip("\n")
    char_to_page = char_to_page[: len(full_text)]

    if not full_text:
        return {
            "document_hash": document_hash,
            "source_path": path,
            "chunks": [],
        }

    # Chunk by character length ~500-800 with overlap ~100
    chunk_size = 650
    overlap = 100
    step = chunk_size - overlap
    chunks = []
    chunk_id = 0
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        segment = full_text[start:end]
        if segment.strip():
            page_number = char_to_page[start] if start < len(char_to_page) else 1
            chunk_id += 1
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": segment,
            })
        start += step
        if step <= 0:
            break

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
