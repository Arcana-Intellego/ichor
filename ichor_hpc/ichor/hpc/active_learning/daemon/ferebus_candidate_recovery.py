"""Durable recovery of FEREBUS candidates whose quality measurement failed."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from .filesystem import campaign_owned_path, operational_path
from .state import _fsync_parent_dir, atomic_write_json


FEREBUS_QUALITY_ATTEMPT_SCHEMA_VERSION = 1
FEREBUS_POSTPROCESS_RECOVERY_SCHEMA_VERSION = 1
FEREBUS_POSTPROCESS_RECOVERY_FILENAME = "ferebus_postprocess_recovery.json"
FEREBUS_RECOVERY_MARKER_FILENAME = "FEREBUS_POSTPROCESS_RECOVERY.json"
FEREBUS_QUALITY_ATTEMPTS_DIRNAME = "ferebus_quality_attempts"

_CANONICAL_QUALITY_FILES = {
    "FEREBUS_QUALITY.json",
    "FEREBUS_QUALITY_DECISION.json",
}
_ACTIVE_RECOVERY_STATUSES = {
    "prepared",
    "measurement_incomplete",
    "materialised",
}


class FerebusCandidateRecoveryError(ValueError):
    """Raised when FEREBUS recovery evidence is ambiguous or untrustworthy."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FerebusCandidateRecoveryError(label + " is missing or symlinked")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusCandidateRecoveryError(label + " is unreadable") from exc
    if not isinstance(payload, dict):
        raise FerebusCandidateRecoveryError(label + " must be a JSON object")
    return payload


def _relative_campaign_path(campaign_dir: Path, path: Path) -> str:
    candidate = campaign_owned_path(campaign_dir, path)
    return candidate.relative_to(Path(campaign_dir).absolute()).as_posix()


def recovery_request_path(campaign_dir: Path) -> Path:
    return operational_path(campaign_dir, FEREBUS_POSTPROCESS_RECOVERY_FILENAME)


def read_recovery_request(
    campaign_dir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
    require_active_source: bool = True,
) -> Optional[Dict[str, Any]]:
    """Read and validate the single campaign-level FEREBUS recovery request."""
    path = recovery_request_path(campaign_dir)
    if not path.exists():
        return None
    payload = _read_object(path, "FEREBUS postprocess recovery request")
    if payload.get("schema_version") != FEREBUS_POSTPROCESS_RECOVERY_SCHEMA_VERSION:
        raise FerebusCandidateRecoveryError(
            "unsupported FEREBUS postprocess recovery schema"
        )
    material = dict(payload)
    declared = str(material.pop("request_sha256", ""))
    if declared != _canonical_sha256(material):
        raise FerebusCandidateRecoveryError(
            "FEREBUS postprocess recovery request digest mismatch"
        )
    campaign_uid = str(payload.get("campaign_uid") or "")
    if not campaign_uid or (
        expected_campaign_uid is not None
        and campaign_uid != str(expected_campaign_uid)
    ):
        raise FerebusCandidateRecoveryError(
            "FEREBUS postprocess recovery campaign UID mismatch"
        )
    phase = str(payload.get("phase") or "")
    if phase not in {"INITIAL_FEREBUS", "FEREBUS"}:
        raise FerebusCandidateRecoveryError(
            "FEREBUS postprocess recovery phase is invalid"
        )
    for field in ("iteration", "reference_data_version"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FerebusCandidateRecoveryError(
                "FEREBUS postprocess recovery " + field + " is invalid"
            )
    if str(payload.get("status") or "") not in (
        _ACTIVE_RECOVERY_STATUSES | {"accepted", "rejected", "failed"}
    ):
        raise FerebusCandidateRecoveryError(
            "FEREBUS postprocess recovery status is invalid"
        )
    source_text = str(payload.get("source_path") or "")
    if not source_text:
        raise FerebusCandidateRecoveryError(
            "FEREBUS postprocess recovery source path is empty"
        )
    source = campaign_owned_path(campaign_dir, source_text)
    if (
        require_active_source
        and str(payload.get("status")) in _ACTIVE_RECOVERY_STATUSES
        and (not source.is_dir() or source.is_symlink())
    ):
        raise FerebusCandidateRecoveryError(
            "FEREBUS postprocess recovery source is missing or symlinked"
        )
    return payload


def _legacy_candidate_summary(
    campaign_dir: Path,
    candidate: Path,
    *,
    expected_campaign_uid: str,
    reference_data_version: int,
) -> Optional[Dict[str, Any]]:
    """Inspect only small authority files when considering an old quarantine."""
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    quality_path = candidate / "FEREBUS_QUALITY.json"
    decision_path = candidate / "FEREBUS_QUALITY_DECISION.json"
    task_manifest_path = candidate / "FEREBUS_TASKS.json"
    task_map_path = candidate / "FEREBUS_TASK_MAP.json"
    if not all(
        path.is_file() and not path.is_symlink()
        for path in (quality_path, decision_path, task_manifest_path, task_map_path)
    ):
        return None
    quality = _read_object(quality_path, "legacy FEREBUS quality evidence")
    if (
        quality.get("schema_version") != 4
        or quality.get("measurement_complete") is not False
        or str(quality.get("campaign_uid") or "") != str(expected_campaign_uid)
        or quality.get("reference_data_version") != int(reference_data_version)
    ):
        return None
    errors = quality.get("measurement_errors")
    if (
        not isinstance(errors, list)
        or not errors
        or any(
            not isinstance(reason, str)
            or not reason.startswith("ferebus_quality_metric_failed:")
            for reason in errors
        )
    ):
        return None
    decision = _read_object(decision_path, "legacy FEREBUS quality decision")
    observed_quality_sha256 = sha256_file(quality_path)
    quality_binding = decision.get("quality")
    if (
        decision.get("schema_version") != 2
        or str(decision.get("campaign_uid") or "") != str(expected_campaign_uid)
        or decision.get("reference_data_version") != int(reference_data_version)
        or not isinstance(quality_binding, dict)
        or str(quality_binding.get("path") or "") != "FEREBUS_QUALITY.json"
        or isinstance(quality_binding.get("size"), bool)
        or quality_binding.get("size") != int(quality_path.stat().st_size)
        or str(quality_binding.get("sha256") or "") != observed_quality_sha256
    ):
        return None
    evaluations = decision.get("evaluations")
    current_digest = str(decision.get("current_evaluation_sha256") or "")
    current = None
    if isinstance(evaluations, list):
        validated = {}
        for item in evaluations:
            if not isinstance(item, dict):
                return None
            material = dict(item)
            declared = str(material.pop("evaluation_sha256", ""))
            material.pop("evaluated_at_iso", None)
            if not declared or declared in validated:
                return None
            if declared != _canonical_sha256(material):
                return None
            validated[declared] = item
        current = validated.get(current_digest)
    if not isinstance(current, dict) or current.get("accepted") is not False:
        return None
    allowed_reasons = set(str(value) for value in errors)
    allowed_reasons.add("ferebus_aggregate_metric_missing")
    reasons = current.get("reasons")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(str(reason) not in allowed_reasons for reason in reasons)
    ):
        return None
    return {
        "status": "recoverable_legacy_candidate",
        "candidate_id": candidate.name,
        "source_path": _relative_campaign_path(campaign_dir, candidate),
        "reference_data_version": int(reference_data_version),
        "quality_sha256": observed_quality_sha256,
        "decision_sha256": sha256_file(decision_path),
        "task_manifest_sha256": sha256_file(task_manifest_path),
        "task_map_file_sha256": sha256_file(task_map_path),
        "measurement_errors": list(errors),
    }


def discover_recovery_candidate(
    campaign_dir: Path,
    *,
    expected_campaign_uid: str,
    reference_data_version: int,
) -> Optional[Dict[str, Any]]:
    """Find one uniquely recoverable legacy candidate without walking its tree."""
    campaign = Path(campaign_dir).absolute()
    existing = read_recovery_request(
        campaign,
        expected_campaign_uid=expected_campaign_uid,
    )
    if existing is not None:
        if str(existing.get("status")) in _ACTIVE_RECOVERY_STATUSES:
            if existing.get("reference_data_version") != int(reference_data_version):
                raise FerebusCandidateRecoveryError(
                    "active FEREBUS recovery request targets a different reference version"
                )
            out = dict(existing)
            out["status"] = "active_recovery_request"
            return out
        return None
    root = campaign / "TRAINED_MODELS" / "rejected-candidates" / (
        "reference-" + f"{int(reference_data_version):06d}"
    )
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise FerebusCandidateRecoveryError(
            "FEREBUS rejected-candidate root is not a regular directory"
        )
    candidates = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        summary = _legacy_candidate_summary(
            campaign,
            child,
            expected_campaign_uid=expected_campaign_uid,
            reference_data_version=reference_data_version,
        )
        if summary is not None:
            candidates.append(summary)
    if len(candidates) > 1:
        raise FerebusCandidateRecoveryError(
            "multiple recoverable FEREBUS candidates exist for reference version "
            + str(reference_data_version)
        )
    return candidates[0] if candidates else None


def _write_request(campaign_dir: Path, material: Dict[str, Any]) -> Path:
    material = dict(material)
    material["request_sha256"] = _canonical_sha256(material)
    path = recovery_request_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, material)
    return path


def prepare_recovery_request(
    campaign_dir: Path,
    *,
    candidate: Mapping[str, Any],
    campaign_uid: str,
    phase: str,
    iteration: int,
    reference_data_version: int,
) -> Dict[str, Any]:
    """Publish an idempotent request for postprocessing existing raw output."""
    campaign = Path(campaign_dir).absolute()
    existing = read_recovery_request(
        campaign,
        expected_campaign_uid=campaign_uid,
    )
    if existing is not None and str(existing.get("status")) in _ACTIVE_RECOVERY_STATUSES:
        if (
            str(existing.get("source_path")) != str(candidate.get("source_path"))
            or existing.get("reference_data_version") != int(reference_data_version)
        ):
            raise FerebusCandidateRecoveryError(
                "a different FEREBUS postprocess recovery is already active"
            )
        return existing
    now = _now_iso()
    material = {
        "schema_version": FEREBUS_POSTPROCESS_RECOVERY_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "phase": str(phase),
        "iteration": int(iteration),
        "reference_data_version": int(reference_data_version),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_path": str(candidate.get("source_path") or ""),
        "source_quality_sha256": candidate.get("quality_sha256"),
        "source_decision_sha256": candidate.get("decision_sha256"),
        "source_task_manifest_sha256": candidate.get("task_manifest_sha256"),
        "source_task_map_file_sha256": candidate.get("task_map_file_sha256"),
        "status": "prepared",
        "created_at_iso": now,
        "updated_at_iso": now,
        "staging_path": None,
        "last_error": None,
    }
    _write_request(campaign, material)
    return read_recovery_request(campaign, expected_campaign_uid=campaign_uid) or {}


def prepare_staging_recovery_request(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    phase: str,
    iteration: int,
    reference_data_version: int,
    staging_dir: Path,
    quality_attempt_path: Path,
) -> Dict[str, Any]:
    """Retain current staging after an incomplete quality measurement."""
    campaign = Path(campaign_dir).absolute()
    source = campaign_owned_path(campaign, staging_dir)
    material = {
        "schema_version": FEREBUS_POSTPROCESS_RECOVERY_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "phase": str(phase),
        "iteration": int(iteration),
        "reference_data_version": int(reference_data_version),
        "candidate_id": sha256_file(source / "FEREBUS_TASK_MAP.json"),
        "source_path": _relative_campaign_path(campaign, source),
        "source_quality_sha256": None,
        "source_decision_sha256": None,
        "source_task_manifest_sha256": sha256_file(source / "FEREBUS_TASKS.json"),
        "source_task_map_file_sha256": sha256_file(source / "FEREBUS_TASK_MAP.json"),
        "status": "measurement_incomplete",
        "created_at_iso": _now_iso(),
        "updated_at_iso": _now_iso(),
        "staging_path": _relative_campaign_path(campaign, source),
        "quality_attempt_path": _relative_campaign_path(
            campaign, quality_attempt_path
        ),
        "last_error": None,
    }
    existing = read_recovery_request(
        campaign,
        expected_campaign_uid=campaign_uid,
    )
    if existing is not None:
        material["created_at_iso"] = existing.get("created_at_iso", material["created_at_iso"])
    _write_request(campaign, material)
    return read_recovery_request(campaign, expected_campaign_uid=campaign_uid) or {}


def update_recovery_status(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    status: str,
    staging_path: Optional[Path] = None,
    last_error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    request = read_recovery_request(
        campaign_dir,
        expected_campaign_uid=campaign_uid,
        require_active_source=False,
    )
    if request is None:
        return None
    if status not in (_ACTIVE_RECOVERY_STATUSES | {"accepted", "rejected", "failed"}):
        raise FerebusCandidateRecoveryError("invalid FEREBUS recovery status")
    material = dict(request)
    material.pop("request_sha256", None)
    material["status"] = status
    material["updated_at_iso"] = _now_iso()
    material["last_error"] = None if last_error is None else str(last_error)[:500]
    if staging_path is not None:
        material["staging_path"] = _relative_campaign_path(
            Path(campaign_dir).absolute(), Path(staging_path)
        )
    _write_request(Path(campaign_dir), material)
    return read_recovery_request(
        campaign_dir,
        expected_campaign_uid=campaign_uid,
    )


def write_quality_attempt(
    campaign_dir: Path,
    *,
    phase: str,
    iteration: int,
    quality: Mapping[str, Any],
) -> Path:
    """Persist non-canonical quality telemetry outside model staging."""
    campaign = Path(campaign_dir).absolute()
    task_execution = quality.get("task_execution")
    task_map_sha = (
        str(task_execution.get("task_map_sha256") or "")
        if isinstance(task_execution, Mapping)
        else ""
    )
    candidate_id = task_map_sha or _canonical_sha256(dict(quality))
    attempt_id = _canonical_sha256(dict(quality))
    root = operational_path(
        campaign,
        FEREBUS_QUALITY_ATTEMPTS_DIRNAME,
        "reference-" + f"{int(quality.get('reference_data_version', -1)):06d}",
        candidate_id[:16],
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / (attempt_id[:24] + ".json")
    if path.exists():
        existing = _read_object(path, "FEREBUS quality attempt")
        if existing.get("quality") != dict(quality):
            raise FerebusCandidateRecoveryError(
                "FEREBUS quality attempt identity collision"
            )
        return path
    payload = {
        "schema_version": FEREBUS_QUALITY_ATTEMPT_SCHEMA_VERSION,
        "campaign_uid": str(quality.get("campaign_uid") or ""),
        "phase": str(phase),
        "iteration": int(iteration),
        "reference_data_version": int(quality.get("reference_data_version", -1)),
        "candidate_id": candidate_id,
        "attempt_id": attempt_id,
        "recorded_at_iso": _now_iso(),
        "quality": dict(quality),
    }
    atomic_write_json(path, payload)
    return path


def _copy_regular_tree(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    with os.scandir(source) as entries:
        for entry in entries:
            if entry.name in _CANONICAL_QUALITY_FILES:
                continue
            source_path = Path(entry.path)
            destination_path = destination / entry.name
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise FerebusCandidateRecoveryError(
                    "FEREBUS recovery source contains a symlink: " + str(source_path)
                )
            if stat.S_ISDIR(info.st_mode):
                _copy_regular_tree(source_path, destination_path)
            elif stat.S_ISREG(info.st_mode):
                shutil.copy2(source_path, destination_path, follow_symlinks=False)
            else:
                raise FerebusCandidateRecoveryError(
                    "FEREBUS recovery source contains a special file: "
                    + str(source_path)
                )


def _validate_raw_candidate(
    campaign_dir: Path,
    source: Path,
    request: Mapping[str, Any],
) -> None:
    from . import input_staging as staging
    from .ferebus_task_runner import validate_task_receipts
    from .live_executor import validate_ferebus_completed

    manifest = staging.read_ferebus_manifest(source, verify_dataset_files=True)
    if (
        str(manifest.get("campaign_uid") or "") != str(request.get("campaign_uid"))
        or manifest.get("reference_data_version")
        != int(request.get("reference_data_version", -1))
    ):
        raise FerebusCandidateRecoveryError(
            "FEREBUS recovery source identity does not match its request"
        )
    expected_manifest_sha = request.get("source_task_manifest_sha256")
    if expected_manifest_sha is not None and str(expected_manifest_sha) != sha256_file(
        source / "FEREBUS_TASKS.json"
    ):
        raise FerebusCandidateRecoveryError(
            "FEREBUS recovery source task manifest changed"
        )
    expected_map_sha = request.get("source_task_map_file_sha256")
    if expected_map_sha is not None and str(expected_map_sha) != sha256_file(
        source / "FEREBUS_TASK_MAP.json"
    ):
        raise FerebusCandidateRecoveryError("FEREBUS recovery source task map changed")
    validate_task_receipts(source)
    ok, reason = validate_ferebus_completed(source)
    if not ok:
        raise FerebusCandidateRecoveryError(
            "FEREBUS recovery source is incomplete: " + str(reason)
        )


def materialise_recovery_candidate(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    phase: str,
    iteration: int,
    reference_data_version: int,
) -> Optional[Path]:
    """Copy an authenticated legacy candidate into clean staging once."""
    campaign = Path(campaign_dir).absolute()
    request = read_recovery_request(
        campaign,
        expected_campaign_uid=campaign_uid,
    )
    if request is None or str(request.get("status")) not in _ACTIVE_RECOVERY_STATUSES:
        return None
    if (
        str(request.get("phase")) != str(phase)
        or request.get("iteration") != int(iteration)
        or request.get("reference_data_version") != int(reference_data_version)
    ):
        raise FerebusCandidateRecoveryError(
            "FEREBUS recovery request does not match the current phase"
        )
    source = campaign_owned_path(campaign, str(request["source_path"]))
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    _validate_raw_candidate(campaign, source, request)
    if source.absolute() == staging.absolute():
        update_recovery_status(
            campaign,
            campaign_uid=campaign_uid,
            status="materialised",
            staging_path=staging,
        )
        return staging
    marker = staging / FEREBUS_RECOVERY_MARKER_FILENAME
    if staging.exists():
        if marker.is_file() and not marker.is_symlink():
            marker_payload = _read_object(marker, "FEREBUS recovery staging marker")
            if marker_payload.get("request_sha256") == request.get("request_sha256"):
                _validate_raw_candidate(campaign, staging, request)
                return staging
        raise FerebusCandidateRecoveryError(
            "FEREBUS iteration-staging is not empty for candidate recovery"
        )
    temporary = staging.parent / (
        ".iteration-staging-recovery-" + str(request["candidate_id"])[:16]
    )
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise FerebusCandidateRecoveryError(
                "invalid stale FEREBUS recovery staging path"
            )
        shutil.rmtree(temporary)
    _copy_regular_tree(source, temporary)
    for filename in _CANONICAL_QUALITY_FILES:
        path = temporary / filename
        if path.exists() or path.is_symlink():
            raise FerebusCandidateRecoveryError(
                "legacy FEREBUS quality evidence leaked into recovery staging"
            )
    _validate_raw_candidate(campaign, temporary, request)
    atomic_write_json(
        temporary / FEREBUS_RECOVERY_MARKER_FILENAME,
        {
            "schema_version": FEREBUS_POSTPROCESS_RECOVERY_SCHEMA_VERSION,
            "request_sha256": str(request["request_sha256"]),
            "source_path": str(request["source_path"]),
            "candidate_id": str(request["candidate_id"]),
            "materialised_at_iso": _now_iso(),
        },
    )
    os.replace(temporary, staging)
    _fsync_parent_dir(staging)
    update_recovery_status(
        campaign,
        campaign_uid=campaign_uid,
        status="materialised",
        staging_path=staging,
    )
    return staging


__all__ = [
    "FEREBUS_POSTPROCESS_RECOVERY_FILENAME",
    "FEREBUS_POSTPROCESS_RECOVERY_SCHEMA_VERSION",
    "FEREBUS_QUALITY_ATTEMPT_SCHEMA_VERSION",
    "FEREBUS_RECOVERY_MARKER_FILENAME",
    "FerebusCandidateRecoveryError",
    "discover_recovery_candidate",
    "materialise_recovery_candidate",
    "prepare_recovery_request",
    "prepare_staging_recovery_request",
    "read_recovery_request",
    "recovery_request_path",
    "update_recovery_status",
    "write_quality_attempt",
]
