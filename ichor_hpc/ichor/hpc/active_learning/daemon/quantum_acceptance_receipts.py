"""Immutable content receipts for AIMAll-accepted point directories."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..strict_json import strict_json as json
from ..strict_json import load_path
from ..versioning.manifest import sha256_file
from ..versioning.provenance import PROVENANCE_FILENAME, read_provenance
from .quantum_quality import read_quantum_quality_manifest
from .state import atomic_write_json


QUANTUM_ACCEPTANCE_RECEIPT = "QUANTUM_ACCEPTANCE_RECEIPT.json"
QUANTUM_ACCEPTANCE_RECEIPT_SCHEMA_VERSION = 1


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(label + " must be an exact non-negative integer")
    return int(value)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_bindings(pointdir: Path) -> list[Dict[str, Any]]:
    root = Path(pointdir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("accepted quantum pointdir is missing or symlinked")
    bindings = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("accepted quantum pointdir contains a symlink: " + str(path))
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {QUANTUM_ACCEPTANCE_RECEIPT, ".provenance.lock"}:
            continue
        bindings.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    required_names = {
        "AIMALL_COMPLETION_RECEIPT.json",
        "AIMALL_TASK.json",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        PROVENANCE_FILENAME,
    }
    observed_names = {Path(binding["path"]).name for binding in bindings}
    missing = sorted(required_names - observed_names)
    if missing:
        raise ValueError(
            "accepted quantum pointdir lacks required evidence: " + repr(missing)
        )
    suffixes = [Path(binding["path"]).suffix.lower() for binding in bindings]
    for suffix, label in ((".gjf", "GJF"), (".wfn", "WFN"), (".int", "INT")):
        if suffixes.count(suffix) < 1:
            raise ValueError("accepted quantum pointdir lacks " + label + " evidence")
    if not any(suffix in {".gau", ".gaussianoutput"} for suffix in suffixes):
        raise ValueError("accepted quantum pointdir lacks Gaussian output evidence")
    return bindings


def write_quantum_acceptance_receipt(
    campaign_dir: Path,
    pointdir: Path,
    *,
    phase_name: str,
    iteration: int,
    quality_manifest: Path,
    quality_record: Mapping[str, Any],
) -> Path:
    """Freeze accepted pointdir bytes before allocation and calibration mutate state."""
    campaign = Path(campaign_dir).resolve()
    root = Path(pointdir)
    provenance = read_provenance(root)
    allocation = provenance.get("point_allocation")
    if not isinstance(allocation, dict):
        raise ValueError("accepted quantum pointdir lacks allocation provenance")
    if quality_record.get("pointdir") != root.name or quality_record.get("accepted") is not True:
        raise ValueError("accepted quantum quality record does not match its pointdir")
    manifest = Path(quality_manifest).resolve()
    try:
        manifest_relative = manifest.relative_to(campaign).as_posix()
    except ValueError as exc:
        raise ValueError("quantum quality manifest is outside the campaign") from exc
    bindings = _current_bindings(root)
    payload = {
        "schema_version": QUANTUM_ACCEPTANCE_RECEIPT_SCHEMA_VERSION,
        "campaign_uid": str(provenance["campaign_uid"]),
        "phase": str(phase_name),
        "iteration": _exact_int(iteration, "quantum acceptance iteration"),
        "pointdir": root.name,
        "candidate_id": str(allocation.get("candidate_id") or ""),
        "allocation_slot_assignment_sha256": str(
            allocation.get("slot_assignment_sha256") or ""
        ),
        "quality_manifest": {
            "path": manifest_relative,
            "sha256": sha256_file(manifest),
        },
        "quality_record_sha256": _canonical_sha256(dict(quality_record)),
        "artefacts": bindings,
        "content_sha256": _canonical_sha256(bindings),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    for key in ("campaign_uid", "candidate_id", "allocation_slot_assignment_sha256"):
        if not payload[key]:
            raise ValueError("quantum acceptance receipt " + key + " is empty")
    target = root / QUANTUM_ACCEPTANCE_RECEIPT
    atomic_write_json(target, payload)
    return target


def bind_quantum_acceptance_receipt_to_commit(
    campaign_dir: Path,
    pointdir: Path,
    *,
    source_pointdir: str,
    committed_pointdir: str,
    quality_manifest_source: Path,
    quality_manifest_published: Path,
    quality_record: Mapping[str, Any],
) -> Path:
    """Rebind a copied receipt to immutable, version-owned quality evidence."""
    campaign = Path(campaign_dir).resolve()
    root = Path(pointdir)
    payload = read_quantum_acceptance_receipt(
        campaign,
        root,
        expected_source_pointdir=str(source_pointdir),
    )
    if root.name != str(committed_pointdir):
        raise ValueError("committed quantum pointdir name mismatch")
    if (
        quality_record.get("pointdir") != str(source_pointdir)
        or quality_record.get("committed_pointdir") != str(committed_pointdir)
        or quality_record.get("accepted") is not True
    ):
        raise ValueError("committed quantum quality record does not match its pointdir")
    source = Path(quality_manifest_source)
    published = Path(quality_manifest_published)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError("committed quantum quality source is missing")
    try:
        published_relative = published.resolve(strict=False).relative_to(
            campaign
        ).as_posix()
    except ValueError as exc:
        raise ValueError("committed quantum quality path is outside the campaign") from exc

    rebound = dict(payload)
    rebound["committed_pointdir"] = str(committed_pointdir)
    rebound["quality_manifest"] = {
        "path": published_relative,
        "sha256": sha256_file(source),
    }
    rebound["quality_record_sha256"] = _canonical_sha256(dict(quality_record))
    rebound["artefacts"] = _current_bindings(root)
    rebound["content_sha256"] = _canonical_sha256(rebound["artefacts"])
    rebound["committed_at_iso"] = datetime.now(timezone.utc).isoformat()
    target = root / QUANTUM_ACCEPTANCE_RECEIPT
    atomic_write_json(target, rebound)
    return target


def read_quantum_acceptance_receipt(
    campaign_dir: Path,
    pointdir: Path,
    *,
    expected_phase: Optional[str] = None,
    expected_iteration: Optional[int] = None,
    expected_candidate_id: Optional[str] = None,
    expected_assignment_sha256: Optional[str] = None,
    expected_source_pointdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify that accepted bytes and their quality decision remain unchanged."""
    campaign = Path(campaign_dir).resolve()
    root = Path(pointdir)
    target = root / QUANTUM_ACCEPTANCE_RECEIPT
    try:
        payload = load_path(target)
    except (OSError, ValueError) as exc:
        raise ValueError("quantum acceptance receipt is unreadable: " + str(target)) from exc
    if not isinstance(payload, dict):
        raise ValueError("quantum acceptance receipt must be a JSON object")
    if _exact_int(payload.get("schema_version"), "quantum acceptance schema") != 1:
        raise ValueError("unsupported quantum acceptance receipt schema")
    iteration = _exact_int(payload.get("iteration"), "quantum acceptance iteration")
    payload_source_pointdir = str(payload.get("pointdir") or "")
    if not payload_source_pointdir:
        raise ValueError("quantum acceptance receipt source pointdir is empty")
    if (
        expected_source_pointdir is not None
        and payload_source_pointdir != str(expected_source_pointdir)
    ):
        raise ValueError("quantum acceptance receipt pointdir mismatch")
    committed_pointdir = payload.get("committed_pointdir")
    if committed_pointdir is not None and committed_pointdir != root.name:
        raise ValueError("quantum acceptance receipt committed pointdir mismatch")
    if (
        expected_source_pointdir is None
        and payload_source_pointdir != root.name
        and committed_pointdir != root.name
    ):
        raise ValueError("quantum acceptance receipt pointdir identity mismatch")
    source_pointdir = payload_source_pointdir
    if expected_phase is not None and payload.get("phase") != str(expected_phase):
        raise ValueError("quantum acceptance receipt phase mismatch")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise ValueError("quantum acceptance receipt iteration mismatch")
    if expected_candidate_id is not None and payload.get("candidate_id") != str(
        expected_candidate_id
    ):
        raise ValueError("quantum acceptance receipt candidate mismatch")
    if expected_assignment_sha256 is not None and payload.get(
        "allocation_slot_assignment_sha256"
    ) != str(expected_assignment_sha256):
        raise ValueError("quantum acceptance receipt allocation mismatch")
    bindings = payload.get("artefacts")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("quantum acceptance receipt artefacts are missing")
    current = _current_bindings(root)
    if current != bindings or payload.get("content_sha256") != _canonical_sha256(current):
        raise ValueError("accepted quantum pointdir bytes have changed")
    quality_binding = payload.get("quality_manifest")
    if not isinstance(quality_binding, dict):
        raise ValueError("quantum acceptance quality binding is invalid")
    relative = Path(str(quality_binding.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("quantum acceptance quality path escapes the campaign")
    quality_path = campaign / relative
    if quality_path.is_symlink() or not quality_path.is_file():
        raise FileNotFoundError("quantum acceptance quality manifest is missing")
    if sha256_file(quality_path) != str(quality_binding.get("sha256") or ""):
        raise ValueError("quantum acceptance quality manifest has changed")
    quality = read_quantum_quality_manifest(
        quality_path.parent,
        expected_phase=str(payload.get("phase") or ""),
        expected_iteration=iteration,
        manifest_path=quality_path,
    )
    matching = [
        record
        for record in quality["records"]
        if record.get("pointdir") == source_pointdir
        and record.get("accepted") is True
        and (
            committed_pointdir is None
            or record.get("committed_pointdir") == committed_pointdir
        )
    ]
    if len(matching) != 1 or _canonical_sha256(matching[0]) != payload.get(
        "quality_record_sha256"
    ):
        raise ValueError("quantum acceptance quality record binding mismatch")
    return payload


__all__ = [
    "bind_quantum_acceptance_receipt_to_commit",
    "QUANTUM_ACCEPTANCE_RECEIPT",
    "QUANTUM_ACCEPTANCE_RECEIPT_SCHEMA_VERSION",
    "read_quantum_acceptance_receipt",
    "write_quantum_acceptance_receipt",
]
