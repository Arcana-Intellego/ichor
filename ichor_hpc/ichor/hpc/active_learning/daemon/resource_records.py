"""Immutable resource-resolution records for submitted daemon attempts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from .state import atomic_write_json
from .script_bundles import campaign_owned_path, safe_component
from ..versioning.manifest import sha256_file


RESOURCE_RESOLUTION_SCHEMA_VERSION = 1
RESOURCE_FORMULA_VERSION = "1"


def resolution_path(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    submission_identity: str,
) -> Path:
    path = (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "resource_resolutions"
        / safe_component(phase_name, "phase")
        / ("iteration-" + str(int(iteration)).zfill(6))
        / (safe_component(submission_identity, "submission identity") + ".json")
    )
    return campaign_owned_path(campaign_dir, path)


def resolution_payload(
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    attempt_id: str,
    submission_identity: str,
    resolved: Any,
    evidence: Mapping[str, Any],
    scratch_path_template: str,
) -> Dict[str, Any]:
    resources = (
        resolved.to_dict()
        if hasattr(resolved, "to_dict")
        else dict(resolved)
    )
    return {
        "schema_version": RESOURCE_RESOLUTION_SCHEMA_VERSION,
        "formula_version": RESOURCE_FORMULA_VERSION,
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "campaign_uid": str(campaign_uid),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "attempt_id": str(attempt_id),
        "submission_identity": str(submission_identity),
        "resources": resources,
        "evidence": dict(evidence),
        "scratch_path_template": str(scratch_path_template),
    }


def write_resolution(
    campaign_dir: Union[str, Path],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    path = resolution_path(
        campaign_dir,
        str(payload["phase"]),
        int(payload["iteration"]),
        str(payload["submission_identity"]),
    )
    if path.exists() or path.is_symlink():
        existing = read_resolution(path)
        existing_comparable = dict(existing)
        proposed_comparable = dict(payload)
        existing_comparable.pop("created_at_iso", None)
        proposed_comparable.pop("created_at_iso", None)
        if existing_comparable != proposed_comparable:
            raise ValueError("resource resolution already exists with different content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, dict(payload))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "formula_version": str(payload["formula_version"]),
    }


def read_resolution(path: Union[str, Path]) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("resource resolution is not a regular file: " + str(source))
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("resource resolution is unreadable: " + str(source)) from exc
    if not isinstance(payload, dict):
        raise ValueError("resource resolution must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("resource-resolution schema_version is malformed") from exc
    if schema_version != RESOURCE_RESOLUTION_SCHEMA_VERSION:
        raise ValueError("unsupported resource-resolution schema")
    required_text = (
        "formula_version",
        "campaign_uid",
        "phase",
        "attempt_id",
        "submission_identity",
        "scratch_path_template",
    )
    if any(
        not isinstance(payload.get(key), str) or not str(payload.get(key))
        for key in required_text
    ):
        raise ValueError("resource resolution ownership fields are incomplete")
    try:
        iteration = int(payload.get("iteration"))
    except (TypeError, ValueError) as exc:
        raise ValueError("resource-resolution iteration is malformed") from exc
    if iteration < 0:
        raise ValueError("resource-resolution iteration must be >= 0")
    if not isinstance(payload.get("resources"), dict):
        raise ValueError("resource resolution resources must be an object")
    if not isinstance(payload.get("evidence"), dict):
        raise ValueError("resource resolution evidence must be an object")
    return payload


def verify_resolution(path: Union[str, Path], expected_sha256: str) -> Dict[str, Any]:
    source = Path(path)
    observed = sha256_file(source)
    if observed != str(expected_sha256):
        raise ValueError(
            "resource resolution SHA-256 mismatch: expected "
            + str(expected_sha256)
            + " got "
            + observed
        )
    return read_resolution(source)
