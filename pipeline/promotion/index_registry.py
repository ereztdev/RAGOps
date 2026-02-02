"""
Index registry: single source of truth for all known index versions.
Append-only entries; status may change. Deterministic serialization.
Extends the existing promotion/registry mechanism; no second registry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Registry entry status enum (directive: created | evaluated | promoted | rejected)
REGISTRY_STATUS_CREATED = "created"
REGISTRY_STATUS_EVALUATED = "evaluated"
REGISTRY_STATUS_PROMOTED = "promoted"
REGISTRY_STATUS_REJECTED = "rejected"

REGISTRY_STATUSES = frozenset({
    REGISTRY_STATUS_CREATED,
    REGISTRY_STATUS_EVALUATED,
    REGISTRY_STATUS_PROMOTED,
    REGISTRY_STATUS_REJECTED,
})

# Deterministic key order for JSON (directive: sorted keys, stable ordering)
_REGISTRY_ENTRY_KEYS = (
    "index_version_id",
    "created_at",
    "index_path",
    "evaluation_report_id",
    "status",
    "notes",
)


@dataclass
class RegistryEntry:
    """
    One registry record. index_version_id is immutable.
    Append-only: entries are never removed; status may change.
    """
    index_version_id: str
    created_at: str  # ISO 8601
    index_path: str
    evaluation_report_id: str | None
    status: str  # created | evaluated | promoted | rejected
    notes: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        """Stable dict for JSON. Sorted keys for deterministic serialization."""
        out: dict[str, Any] = {
            "index_version_id": self.index_version_id,
            "created_at": self.created_at,
            "index_path": self.index_path,
            "evaluation_report_id": self.evaluation_report_id,
            "status": self.status,
        }
        if self.notes is not None:
            out["notes"] = self.notes
        return {k: out[k] for k in _REGISTRY_ENTRY_KEYS if k in out}

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> RegistryEntry:
        """Deserialize. Fails loudly on missing required fields or invalid status."""
        required = ("index_version_id", "created_at", "index_path", "evaluation_report_id", "status")
        for key in required:
            if key not in data:
                raise ValueError(f"RegistryEntry missing required field: {key!r}")
        index_version_id = data["index_version_id"]
        created_at = data["created_at"]
        index_path = data["index_path"]
        evaluation_report_id = data["evaluation_report_id"]
        status = data["status"]
        if not isinstance(index_version_id, str) or not index_version_id.strip():
            raise ValueError("RegistryEntry index_version_id must be non-empty str")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ValueError("RegistryEntry created_at must be non-empty str")
        if not isinstance(index_path, str) or not index_path.strip():
            raise ValueError("RegistryEntry index_path must be non-empty str")
        if evaluation_report_id is not None and not isinstance(evaluation_report_id, str):
            raise ValueError("RegistryEntry evaluation_report_id must be str or None")
        if status not in REGISTRY_STATUSES:
            raise ValueError(f"RegistryEntry status must be one of {sorted(REGISTRY_STATUSES)}, got {status!r}")
        notes = data.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError("RegistryEntry notes must be str or None")
        return cls(
            index_version_id=index_version_id,
            created_at=created_at,
            index_path=index_path,
            evaluation_report_id=evaluation_report_id,
            status=status,
            notes=notes,
        )


def load_registry(path: Path | str) -> list[RegistryEntry]:
    """
    Load registry from JSON file. Deterministic ordering (entries in file order).
    Fails loudly on missing file, invalid JSON, or invalid entries.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Registry file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Registry root must be a list, got {type(data).__name__}")
    entries: list[RegistryEntry] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Registry entry at index {i} must be dict, got {type(item).__name__}")
        entries.append(RegistryEntry.from_serializable(item))
    return entries


def save_registry(path: Path | str, entries: list[RegistryEntry]) -> None:
    """
    Write registry to JSON. Deterministic: sorted keys per entry, stable ordering.
    Append-only semantics: caller must pass full list; entries are never removed here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [e.to_serializable() for e in entries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, sort_keys=True, indent=2, ensure_ascii=False)


def get_entry(entries: list[RegistryEntry], index_version_id: str) -> RegistryEntry | None:
    """Return the registry entry for the given index_version_id, or None."""
    for e in entries:
        if e.index_version_id == index_version_id:
            return e
    return None


def get_promoted_entry(entries: list[RegistryEntry]) -> RegistryEntry | None:
    """Return the single entry with status=promoted, or None. Fails loudly if more than one promoted."""
    promoted = [e for e in entries if e.status == REGISTRY_STATUS_PROMOTED]
    if len(promoted) > 1:
        raise ValueError(
            "Registry violation: more than one promoted entry: %s"
            % [e.index_version_id for e in promoted]
        )
    return promoted[0] if promoted else None


def register_index(
    path: Path | str,
    index_version_id: str,
    index_path: str,
    created_at: str,
    notes: str | None = None,
) -> None:
    """
    Append a new registry entry (status=created, evaluation_report_id=None).
    Fails if index_version_id already exists. Mutates file on disk.
    """
    path = Path(path)
    entries = load_registry(path) if path.is_file() else []
    if get_entry(entries, index_version_id) is not None:
        raise ValueError(f"Registry already contains index_version_id: {index_version_id!r}")
    entries.append(
        RegistryEntry(
            index_version_id=index_version_id,
            created_at=created_at,
            index_path=index_path,
            evaluation_report_id=None,
            status=REGISTRY_STATUS_CREATED,
            notes=notes,
        )
    )
    save_registry(path, entries)


def update_entry_status(
    path: Path | str,
    index_version_id: str,
    status: str,
    evaluation_report_id: str | None = None,
) -> None:
    """
    Update status (and optionally evaluation_report_id) for an existing entry.
    Entry must exist. Violations (e.g. invalid status) fail loudly.
    """
    if status not in REGISTRY_STATUSES:
        raise ValueError(f"status must be one of {sorted(REGISTRY_STATUSES)}, got {status!r}")
    path = Path(path)
    entries = load_registry(path)
    entry = get_entry(entries, index_version_id)
    if entry is None:
        raise ValueError(f"No registry entry for index_version_id: {index_version_id!r}")
    new_report_id = evaluation_report_id if evaluation_report_id is not None else entry.evaluation_report_id
    new_entries = [
        RegistryEntry(
            index_version_id=e.index_version_id,
            created_at=e.created_at,
            index_path=e.index_path,
            evaluation_report_id=new_report_id if e.index_version_id == index_version_id else e.evaluation_report_id,
            status=status if e.index_version_id == index_version_id else e.status,
            notes=e.notes,
        )
        for e in entries
    ]
    save_registry(path, new_entries)


# ---------------------------------------------------------------------------
# Active index pointer (indexes/active_index.json)
# ---------------------------------------------------------------------------

ACTIVE_INDEX_KEYS = ("index_version_id", "promoted_at", "evaluation_report_id")


@dataclass
class ActiveIndexInfo:
    """Single authoritative active index pointer. Inference resolves index from this only."""
    index_version_id: str
    promoted_at: str  # ISO 8601
    evaluation_report_id: str


def _active_to_serializable(info: ActiveIndexInfo) -> dict[str, Any]:
    """Stable dict for active_index.json. Deterministic key order."""
    return {
        "index_version_id": info.index_version_id,
        "promoted_at": info.promoted_at,
        "evaluation_report_id": info.evaluation_report_id,
    }


def load_active_index(path: Path | str) -> ActiveIndexInfo:
    """
    Load active index pointer from JSON.
    Fails loudly on missing file, invalid JSON, or missing required fields.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Active index file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Active index root must be dict, got {type(data).__name__}")
    for key in ACTIVE_INDEX_KEYS:
        if key not in data:
            raise ValueError(f"Active index missing required field: {key!r}")
    index_version_id = data["index_version_id"]
    promoted_at = data["promoted_at"]
    evaluation_report_id = data["evaluation_report_id"]
    if not isinstance(index_version_id, str) or not index_version_id.strip():
        raise ValueError("Active index index_version_id must be non-empty str")
    if not isinstance(promoted_at, str) or not promoted_at.strip():
        raise ValueError("Active index promoted_at must be non-empty str")
    if not isinstance(evaluation_report_id, str) or not evaluation_report_id.strip():
        raise ValueError("Active index evaluation_report_id must be non-empty str")
    return ActiveIndexInfo(
        index_version_id=index_version_id,
        promoted_at=promoted_at,
        evaluation_report_id=evaluation_report_id,
    )


def save_active_index(path: Path | str, info: ActiveIndexInfo) -> None:
    """Write active index pointer. Deterministic serialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_active_to_serializable(info), f, sort_keys=True, indent=2, ensure_ascii=False)


def resolve_active_index(registry_path: Path | str, active_index_path: Path | str) -> tuple[str, str]:
    """
    Resolve the active index: read active_index.json and registry, return (index_version_id, index_path).
    Inference MUST use this (or equivalent) to obtain the index to use; direct index loading by callers is forbidden.
    Raises FileNotFoundError or ValueError if active index is missing or invalid.
    """
    active = load_active_index(active_index_path)
    entries = load_registry(registry_path)
    entry = get_entry(entries, active.index_version_id)
    if entry is None:
        raise ValueError(
            f"Active index references index_version_id {active.index_version_id!r} which is not in registry"
        )
    return active.index_version_id, entry.index_path
