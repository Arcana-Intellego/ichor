"""Strict historical ARIADNE evidence shared by adaptive sampling readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .handoff_manifests import (
    HandoffManifestError,
    read_ariadne_landing_audit,
    read_ariadne_results_manifest,
)
from .layout import active_ariadne_dir


def _bound_result_path(
    iter_dir: Path,
    record: Dict[str, Any],
    *,
    allow_absolute: bool,
) -> Path:
    raw = record.get("result_json")
    if not isinstance(raw, str) or not raw:
        raise HandoffManifestError("historical ARIADNE audit result_json is missing")
    supplied = Path(raw)
    root = active_ariadne_dir(iter_dir)
    if supplied.is_absolute():
        if not allow_absolute:
            raise HandoffManifestError("historical ARIADNE audit result path is unsafe")
        candidate = supplied
    else:
        if ".." in supplied.parts or supplied.as_posix() != raw:
            raise HandoffManifestError("historical ARIADNE audit result path is unsafe")
        candidate = root / supplied
    if candidate.is_symlink() or not candidate.is_file():
        raise HandoffManifestError("historical ARIADNE audit result is missing")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HandoffManifestError("historical ARIADNE audit result escapes its iteration") from exc
    return resolved


def read_historical_ariadne_records(
    iter_dir: Path,
    *,
    expected_iteration: int,
) -> List[Dict[str, Any]]:
    """Return accepted records only after both historical manifests agree."""
    root = Path(iter_dir)
    results = read_ariadne_results_manifest(
        root,
        expected_iteration=int(expected_iteration),
        require_nonempty=False,
        accept_legacy_missing_landing_safety=False,
    )
    audit = read_ariadne_landing_audit(
        root,
        expected_iteration=int(expected_iteration),
    )
    audit_by_seed = {int(record["seed_id"]): dict(record) for record in audit["seeds"]}
    result_seed_ids = {
        int(record["seed_id"])
        for record in list(results["accepted"]) + list(results["rejected"])
    }
    if set(audit_by_seed) != result_seed_ids:
        raise HandoffManifestError(
            "historical ARIADNE audit/results seed membership mismatch"
        )
    accepted: List[Dict[str, Any]] = []
    for result_record in results["accepted"]:
        seed_id = int(result_record["seed_id"])
        audit_record = audit_by_seed[seed_id]
        if str(audit_record.get("seed_uid") or "") != str(
            result_record.get("seed_uid") or ""
        ):
            raise HandoffManifestError(
                "historical ARIADNE audit/results seed UID mismatch"
            )
        if audit_record.get("handoff_accepted") is not True:
            raise HandoffManifestError(
                "historical accepted ARIADNE result is rejected by its audit"
            )
        for safety in (
            audit_record.get("landing_safety"),
            result_record.get("landing_safety"),
        ):
            if not isinstance(safety, dict) or safety.get("accepted") is not True:
                raise HandoffManifestError(
                    "historical accepted ARIADNE result lacks accepted safety evidence"
                )
        if _bound_result_path(
            root,
            audit_record,
            allow_absolute=False,
        ) != _bound_result_path(
            root,
            result_record,
            allow_absolute=True,
        ):
            raise HandoffManifestError(
                "historical ARIADNE audit/results path mismatch"
            )
        merged = dict(result_record)
        merged["landing_audit"] = audit_record
        # Movement metrics are richer in the audit record and are immutable
        # only after the strict result identity above has been established.
        for key, value in audit_record.items():
            if key not in {"result_json", "seed_dir", "seed_id", "seed_uid"}:
                merged[key] = value
        accepted.append(merged)
    return accepted


__all__ = ["read_historical_ariadne_records"]
