"""
Promotion gate: explicit, parameterized promote_index.
Evaluation report must exist and pass; only one promoted index at a time.
Index contents are never mutated. Registry and active pointer updated atomically.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.schema import EvaluationReport

from pipeline.promotion.index_registry import (
    REGISTRY_STATUS_EVALUATED,
    REGISTRY_STATUS_PROMOTED,
    ActiveIndexInfo,
    get_entry,
    get_promoted_entry,
    load_registry,
    load_active_index,
    RegistryEntry,
    save_active_index,
    save_registry,
)


# ---------------------------------------------------------------------------
# Typed promotion errors
# ---------------------------------------------------------------------------


class PromotionError(Exception):
    """Base for promotion failures. Raised on validation or state violations."""


class EvaluationReportMissingError(PromotionError):
    """evaluation_report is required and must be provided."""


class EvaluationFailedError(PromotionError):
    """evaluation_report.overall_pass must be True to promote."""


class IndexNotInRegistryError(PromotionError):
    """index_version_id must exist in registry."""


class IndexVersionMismatchError(PromotionError):
    """evaluation_report.index_version must match index_version_id."""


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def promote_index(
    index_version_id: str,
    evaluation_report: EvaluationReport,
    registry_path: Path | str,
    active_index_path: Path | str,
) -> None:
    """
    Promote an index version to active. Explicit, parameterized; no implicit promotion.

    Rules enforced:
    - evaluation_report MUST exist
    - evaluation_report.overall_pass MUST be True
    - evaluation_report.index_version MUST match index_version_id
    - index_version_id MUST exist in registry
    - Only ONE index may have status=promoted at any time
    - Previously promoted index is demoted via status update (never deleted)
    - Index contents MUST NOT be mutated
    - Registry state and active index pointer updated atomically (registry first, then active)

    Raises typed PromotionError on failure.
    """
    if evaluation_report is None:
        raise EvaluationReportMissingError("evaluation_report is required")
    if not evaluation_report.overall_pass:
        raise EvaluationFailedError(
            "evaluation_report.overall_pass must be True to promote"
        )
    if evaluation_report.index_version != index_version_id:
        raise IndexVersionMismatchError(
            f"evaluation_report.index_version {evaluation_report.index_version!r} must match index_version_id {index_version_id!r}"
        )

    registry_path = Path(registry_path)
    active_index_path = Path(active_index_path)
    if not registry_path.is_file():
        raise PromotionError(f"Registry file not found: {registry_path}")

    entries = load_registry(registry_path)
    entry = get_entry(entries, index_version_id)
    if entry is None:
        raise IndexNotInRegistryError(
            f"index_version_id {index_version_id!r} not found in registry"
        )

    promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Demote current promoted (status update only; never delete)
    new_entries: list[RegistryEntry] = []
    for e in entries:
        if e.status == REGISTRY_STATUS_PROMOTED and e.index_version_id != index_version_id:
            new_entries.append(
                RegistryEntry(
                    index_version_id=e.index_version_id,
                    created_at=e.created_at,
                    index_path=e.index_path,
                    evaluation_report_id=e.evaluation_report_id,
                    status=REGISTRY_STATUS_EVALUATED,
                    notes=e.notes,
                )
            )
        elif e.index_version_id == index_version_id:
            new_entries.append(
                RegistryEntry(
                    index_version_id=e.index_version_id,
                    created_at=e.created_at,
                    index_path=e.index_path,
                    evaluation_report_id=evaluation_report.evaluation_id,
                    status=REGISTRY_STATUS_PROMOTED,
                    notes=e.notes,
                )
            )
        else:
            new_entries.append(e)

    save_registry(registry_path, new_entries)
    save_active_index(
        active_index_path,
        ActiveIndexInfo(
            index_version_id=index_version_id,
            promoted_at=promoted_at,
            evaluation_report_id=evaluation_report.evaluation_id,
        ),
    )
