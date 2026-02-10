"""
Stable JSON serialization for pipeline artifacts.
No runtime-only or transient fields. Output is human-readable and deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.schema import Chunk, Document, IndexVersion


def _stable_json_dump(obj: Any, path: Path | str) -> None:
    """Write JSON with sorted keys for stable, human-readable output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)


def ingestion_output_to_dict(document: Document) -> dict[str, Any]:
    """
    Serialize Document to the ingestion_output schema.
    Fields: document metadata, ordered chunks, page numbers, text content.
    No runtime-only or transient fields.
    """
    return document.to_serializable()


def write_ingestion_output(document: Document, path: Path | str) -> None:
    """Write ingestion_output.json with stable field order."""
    data = ingestion_output_to_dict(document)
    _stable_json_dump(data, path)


def load_ingestion_output(path: Path | str) -> Document:
    """
    Load Document from ingestion_output.json. Fails loudly on missing file or invalid schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ingestion output not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Ingestion output root must be dict, got {type(data).__name__}")
    for key in ("document_id", "source_path", "chunks"):
        if key not in data:
            raise ValueError(f"Ingestion output missing required field: {key!r}")
    if not isinstance(data["chunks"], list):
        raise ValueError(f"Ingestion output chunks must be list, got {type(data['chunks']).__name__}")
    chunks: list[Chunk] = []
    for i, item in enumerate(data["chunks"]):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk at index {i} must be dict, got {type(item).__name__}")
        for field in ("chunk_id", "chunk_key", "document_id", "source_type", "page_number", "chunk_index", "text"):
            if field not in item:
                raise ValueError(f"Chunk at index {i} missing required field: {field!r}")
        chunks.append(
            Chunk(
                chunk_id=item["chunk_id"],
                chunk_key=item["chunk_key"],
                document_id=item["document_id"],
                source_type=item["source_type"],
                page_number=int(item["page_number"]),
                chunk_index=int(item["chunk_index"]),
                text=item["text"],
                chapter=item.get("chapter"),
                section=item.get("section"),
                domain_hint=item.get("domain_hint"),
            )
        )
    return Document(
        document_id=data["document_id"],
        source_path=data["source_path"],
        chunks=chunks,
    )


def index_manifest_to_dict(index_version: IndexVersion) -> dict[str, Any]:
    """
    Serialize IndexVersion to the index_manifest schema.
    Fields: index_version_id, created_at, embedding_model, chunking_strategy, document_ids.
    """
    return index_version.to_serializable()


def write_index_manifest(index_version: IndexVersion, path: Path | str) -> None:
    """Write index_manifest.json with stable field order."""
    data = index_manifest_to_dict(index_version)
    _stable_json_dump(data, path)
