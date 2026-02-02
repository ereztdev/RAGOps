"""
Persistent vector index storage (Segment 4).
Save/load index snapshots to local filesystem. JSON only, deterministic ordering.
No side effects outside storage/.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.retrieval.retrieval_engine import IndexSnapshot


def save_index(index_snapshot: IndexSnapshot, path: Path | str) -> None:
    """
    Persist an index snapshot to disk as JSON.
    One file per index_version (path chosen by caller).
    Deterministic: same snapshot produces byte-identical file.
    """
    if not isinstance(index_snapshot, IndexSnapshot):
        raise TypeError(
            f"index_snapshot must be IndexSnapshot, got {type(index_snapshot).__name__}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = index_snapshot.to_serializable()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2, ensure_ascii=False)


def load_index(path: Path | str) -> IndexSnapshot:
    """
    Load an index snapshot from disk.
    Fails loudly on missing fields, invalid index_version, or malformed JSON.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return IndexSnapshot.from_serializable(data)
