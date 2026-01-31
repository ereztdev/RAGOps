"""
Stable JSON serialization for pipeline artifacts.
No runtime-only or transient fields. Output is human-readable and deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.schema import Document, IndexVersion


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
