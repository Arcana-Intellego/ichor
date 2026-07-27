"""Content receipts for AIMAll-accepted point directories."""

from __future__ import annotations

import hashlib
import os
import platform
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from ..strict_json import strict_json as json
from ..strict_json import load_path
from ..versioning.manifest import sha256_file
from ..versioning.provenance import PROVENANCE_FILENAME, read_provenance
from .filesystem import campaign_owned_path
from .quantum_quality import read_quantum_quality_manifest
from .state import _fsync_file_descriptor, _fsync_parent_dir, atomic_write_json


QUANTUM_ACCEPTANCE_RECEIPT = "QUANTUM_ACCEPTANCE_RECEIPT.json"
QUANTUM_ACCEPTANCE_RECEIPT_SCHEMA_VERSION = 3
_SUPPORTED_QUANTUM_ACCEPTANCE_RECEIPT_SCHEMAS = frozenset({2, 3})


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


def _scientific_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("accepted quantum pointdir is missing or symlinked")
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("accepted quantum pointdir contains a symlink: " + str(path))
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(
                "accepted quantum pointdir contains a special file: " + str(path)
            )
        relative = path.relative_to(root).as_posix()
        if relative in {QUANTUM_ACCEPTANCE_RECEIPT, ".provenance.lock"}:
            continue
        lower_name = path.name.lower()
        if lower_name.startswith(".nfs") or lower_name.endswith((".tmp", ".lock")):
            raise ValueError(
                "accepted quantum pointdir contains a transient file: " + str(path)
            )
        files.append(path)
    return files


def _validate_required_evidence(bindings: list[Dict[str, Any]]) -> None:
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


def _hash_and_fsync_bindings(root: Path) -> list[Dict[str, Any]]:
    bindings: list[Dict[str, Any]] = []
    for path in _scientific_files(root):
        size, digest = _stream_hash_and_fsync(path)
        bindings.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": digest,
            }
        )
    _validate_required_evidence(bindings)
    return bindings


def _stream_hash_and_fsync(path: Path) -> tuple[int, str]:
    """Read, hash and synchronise one accepted artefact through one handle."""
    digest = hashlib.sha256()
    size = 0
    mode = "rb+" if platform.system() == "Windows" else "rb"
    with path.open(mode) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        if int(os.fstat(handle.fileno()).st_size) != size:
            raise ValueError("accepted quantum artefact changed while hashing: " + str(path))
        _fsync_file_descriptor(handle.fileno())
    return int(size), digest.hexdigest()


def _validate_current_inventory(
    root: Path,
    bindings: list[Dict[str, Any]],
    *,
    verification: str,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
) -> None:
    if verification not in {"receipt", "metadata", "deep"}:
        raise ValueError(
            "quantum acceptance verification must be receipt, metadata or deep"
        )
    if verification == "receipt":
        if root.is_symlink() or not root.is_dir():
            raise ValueError("accepted quantum pointdir is missing or symlinked")
        _validate_required_evidence(bindings)
        return
    current_files = _scientific_files(root)
    current = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": int(path.stat().st_size),
        }
        for path in current_files
    ]
    expected = [
        {"path": str(binding.get("path") or ""), "size": binding.get("size")}
        for binding in bindings
    ]
    if current != expected:
        raise ValueError("accepted quantum pointdir inventory has changed")
    _validate_required_evidence(bindings)
    if verification == "deep":
        for path, binding in zip(current_files, bindings):
            observed_sha = (
                digest_file(path, True)
                if digest_file is not None
                else sha256_file(path)
            )
            if observed_sha != str(binding.get("sha256") or ""):
                raise ValueError(
                    "accepted quantum pointdir file hash has changed: " + str(path)
                )


def write_quantum_acceptance_receipt(
    campaign_dir: Path,
    pointdir: Path,
    *,
    phase_name: str,
    iteration: int,
    quality_manifest: Path,
    quality_record: Mapping[str, Any],
) -> Path:
    """Hash and synchronise one accepted pointdir before allocation publication."""
    campaign = Path(campaign_dir).resolve()
    root = campaign_owned_path(campaign, pointdir)
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
    provenance_lock = root / ".provenance.lock"
    if provenance_lock.exists():
        if provenance_lock.is_symlink() or not provenance_lock.is_file():
            raise ValueError("quantum provenance lock is not a regular file")
        provenance_lock.unlink()
        _fsync_parent_dir(provenance_lock)
    bindings = _hash_and_fsync_bindings(root)
    from .ferebus_row_cache import row_shard_binding

    shard_binding = row_shard_binding(
        campaign,
        root,
        source_bindings=bindings,
    )
    payload = {
        "schema_version": QUANTUM_ACCEPTANCE_RECEIPT_SCHEMA_VERSION,
        "campaign_uid": str(provenance["campaign_uid"]),
        "phase": str(phase_name),
        "iteration": _exact_int(iteration, "quantum acceptance iteration"),
        "source_pointdir": root.name,
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
        "ferebus_row_shard_status": (
            "available" if shard_binding is not None else "repair_required"
        ),
        "ferebus_row_shard": shard_binding,
        "accepted_at_iso": datetime.now(timezone.utc).isoformat(),
        "integrity_policy": "sha256_inventory",
    }
    for key in ("campaign_uid", "candidate_id", "allocation_slot_assignment_sha256"):
        if not payload[key]:
            raise ValueError("quantum acceptance receipt " + key + " is empty")
    target = root / QUANTUM_ACCEPTANCE_RECEIPT
    atomic_write_json(target, payload)
    return target


def read_quantum_acceptance_receipt(
    campaign_dir: Path,
    pointdir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
    expected_phase: Optional[str] = None,
    expected_iteration: Optional[int] = None,
    expected_candidate_id: Optional[str] = None,
    expected_assignment_sha256: Optional[str] = None,
    expected_source_pointdir: Optional[str] = None,
    verification: str = "metadata",
    validate_quality: bool = True,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
) -> Dict[str, Any]:
    """Validate acceptance evidence, rehashing payloads only when requested."""
    campaign = Path(campaign_dir).resolve()
    root = campaign_owned_path(campaign, pointdir)
    target = root / QUANTUM_ACCEPTANCE_RECEIPT
    try:
        payload = load_path(target)
    except (OSError, ValueError) as exc:
        raise ValueError("quantum acceptance receipt is unreadable: " + str(target)) from exc
    if not isinstance(payload, dict):
        raise ValueError("quantum acceptance receipt must be a JSON object")
    schema_version = _exact_int(
        payload.get("schema_version"), "quantum acceptance schema"
    )
    if schema_version not in _SUPPORTED_QUANTUM_ACCEPTANCE_RECEIPT_SCHEMAS:
        raise ValueError("unsupported quantum acceptance receipt schema")
    timestamp_key = "sealed_at_iso" if schema_version == 2 else "accepted_at_iso"
    timestamp = payload.get(timestamp_key)
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("quantum acceptance receipt timestamp is invalid")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("quantum acceptance receipt timestamp is invalid") from exc
    if parsed_timestamp.tzinfo is None:
        raise ValueError("quantum acceptance receipt timestamp lacks a timezone")
    if schema_version == 3 and payload.get("integrity_policy") != "sha256_inventory":
        raise ValueError("quantum acceptance receipt integrity policy is invalid")
    campaign_uid = str(payload.get("campaign_uid") or "")
    if not campaign_uid:
        raise ValueError("quantum acceptance receipt campaign UID is empty")
    if (
        expected_campaign_uid is not None
        and campaign_uid != str(expected_campaign_uid)
    ):
        raise ValueError("quantum acceptance receipt campaign UID mismatch")
    iteration = _exact_int(payload.get("iteration"), "quantum acceptance iteration")
    source_pointdir = str(payload.get("source_pointdir") or "")
    if not source_pointdir:
        raise ValueError("quantum acceptance receipt source pointdir is empty")
    if expected_source_pointdir is not None:
        if source_pointdir != str(expected_source_pointdir):
            raise ValueError("quantum acceptance receipt pointdir mismatch")
    elif source_pointdir != root.name:
        raise ValueError("quantum acceptance receipt pointdir identity mismatch")
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
    if not isinstance(bindings, list) or not bindings or any(
        not isinstance(binding, dict) for binding in bindings
    ):
        raise ValueError("quantum acceptance receipt artefacts are missing")
    if payload.get("content_sha256") != _canonical_sha256(bindings):
        raise ValueError("quantum acceptance receipt content digest is invalid")
    _validate_current_inventory(
        root,
        bindings,
        verification=verification,
        digest_file=digest_file,
    )
    quality_binding = payload.get("quality_manifest")
    if not isinstance(quality_binding, dict):
        raise ValueError("quantum acceptance quality binding is invalid")
    if not validate_quality:
        return payload
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
        if record.get("pointdir") == source_pointdir and record.get("accepted") is True
    ]
    if len(matching) != 1 or _canonical_sha256(matching[0]) != payload.get(
        "quality_record_sha256"
    ):
        raise ValueError("quantum acceptance quality record binding mismatch")
    return payload


__all__ = [
    "QUANTUM_ACCEPTANCE_RECEIPT",
    "QUANTUM_ACCEPTANCE_RECEIPT_SCHEMA_VERSION",
    "read_quantum_acceptance_receipt",
    "write_quantum_acceptance_receipt",
]
