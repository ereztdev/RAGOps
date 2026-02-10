"""
Run log formatting with retrieval provenance: detected domains, chunk_id, page,
confidence, domain_hint, boost indication, preview. Backward compatible with existing format.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from ragops.config import (
    LOG_CHUNK_PREVIEW_LENGTH,
    LOG_RETRIEVAL_DETAILS,
    LOG_TOP_K_CHUNKS,
)
from ragops.ingest.pdf_parser import chunk_id_from_page_and_index


def format_run_log_block(
    pdf_name: str,
    pdf_size_kb: float,
    pages: int,
    chunks: int,
    tokens_approx: int,
    part1_sec: float,
    part2_sec: float,
    index_version: str = "v1",
    model: str = "",
    question: str = "",
    answer: str = "",
    retrieval_result=None,
    detected_domains: List[str] | None = None,
) -> str:
    """
    Format one run log entry. If LOG_RETRIEVAL_DETAILS and retrieval_result (and optionally
    detected_domains) are provided, include Query Analysis and Retrieved Chunks and Answer Provenance.
    Otherwise use backward-compatible short format (no provenance).
    """
    ts = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M")
    model_line = f"- **Model:** {model}\n" if model else ""
    question_line = f"- **Question:** {question.replace(chr(10), ' ')}\n" if question else ""
    answer_escaped = (answer or "(no answer recorded)").replace(chr(10), " ").strip()

    block = f"""## {ts} — {pdf_name}
- **PDF:** {pdf_name} · {pdf_size_kb:.0f} KB · **Pages:** {pages}
- **Chunks:** {chunks} · **Tokens (approx):** {tokens_approx:,}
- **Index version:** {index_version}
{model_line}{question_line}"""

    if LOG_RETRIEVAL_DETAILS and retrieval_result is not None:
        domains = detected_domains if detected_domains is not None else []
        block += "\n\n### Query Analysis\n"
        block += f"- Detected domains: {domains}\n\n"
        block += "### Retrieved Chunks\n"
        hits = getattr(retrieval_result, "hits", None) or []
        preview_len = LOG_CHUNK_PREVIEW_LENGTH
        top_n = min(LOG_TOP_K_CHUNKS, len(hits))
        for i, hit in enumerate(hits[:top_n], 1):
            # Display id: chunk_PAGE_RANK (rank 1..top_n)
            if getattr(hit, "page_number", None) is not None:
                chunk_id_display = chunk_id_from_page_and_index(hit.page_number, i - 1)
            else:
                chunk_id_display = hit.chunk_id
            page_str = f"Page {hit.page_number}" if getattr(hit, "page_number", None) is not None else "Page ?"
            chapter_section = ""
            if getattr(hit, "chapter", None) or getattr(hit, "section", None):
                parts = []
                if getattr(hit, "chapter", None):
                    parts.append(f"Chapter {hit.chapter}")
                if getattr(hit, "section", None):
                    parts.append(f"Section {hit.section}")
                chapter_section = ", ".join(parts) + " · " if parts else ""
            domain_str = getattr(hit, "domain_hint", None) or "None"
            conf = hit.similarity_score
            boost_note = ""
            if getattr(hit, "boosted", False) and getattr(hit, "original_score", None) is not None:
                boost_note = f" (↑ boosted from {hit.original_score:.2f})"
            block += f"{i}. **{chunk_id_display}** · {page_str} · {chapter_section}Domain: {domain_str} · Confidence: {conf:.2f}{boost_note}\n"
            preview = (hit.raw_text or "").replace("\n", " ")[:preview_len]
            if len((hit.raw_text or "")) > preview_len:
                preview += "..."
            block += f"   Preview: \"{preview}\"\n"
        block += "\n### Answer Provenance\n"
        citation_ids = hits
        primary = [h.chunk_id for h in citation_ids[:2]] if citation_ids else []
        fallback = [h.chunk_id for h in citation_ids[2:]] if len(citation_ids) > 2 else []
        block += f"- Primary sources: {', '.join(primary) if primary else 'none'}\n"
        block += f"- Fallback sources: {', '.join(fallback) if fallback else 'none'}\n"
        block += f"- Answer grounded in: {len(citation_ids)} chunks from {pdf_name}\n\n"

    block += f"- **Part 1 (ingest+build):** {part1_sec:.1f} s\n"
    block += f"- **Part 2 (eval+ask):** {part2_sec:.1f} s\n\n"
    block += f"**Answer:**\n{answer_escaped}\n"
    print(
        f"DEBUG format_run_log: answer={answer[:100] if answer else None!r}, retrieval_result={retrieval_result is not None}, detected_domains={detected_domains!r}",
        file=sys.stderr,
    )
    return block


def append_run_log(
    log_path: Path,
    pdf_name: str,
    pdf_size_kb: float,
    pages: int,
    chunks: int,
    tokens_approx: int,
    part1_sec: float,
    part2_sec: float,
    index_version: str = "v1",
    model: str = "",
    question: str = "",
    answer: str = "",
    retrieval_result=None,
    detected_domains: List[str] | None = None,
) -> None:
    """Append one run entry to run_log.md. Uses enhanced format when LOG_RETRIEVAL_DETAILS and retrieval_result provided."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    block = format_run_log_block(
        pdf_name=pdf_name,
        pdf_size_kb=pdf_size_kb,
        pages=pages,
        chunks=chunks,
        tokens_approx=tokens_approx,
        part1_sec=part1_sec,
        part2_sec=part2_sec,
        index_version=index_version,
        model=model,
        question=question,
        answer=answer,
        retrieval_result=retrieval_result,
        detected_domains=detected_domains or [],
    )
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# RAGOps run log\n\n*Appended by `ragops run`. Part 1 = ingest+build; Part 2 = evaluate+promote+ask.*\n\n")
        f.write(block)
