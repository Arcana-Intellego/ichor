"""Fast, resumable publication of accepted QM reference-data deltas."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..layout import COMMITTED_VERSION_NAME_WIDTH, qm_reference_data_dir, staging_root
from ..point_allocation import (
    accepted_attempts,
    allocation_manifest_sha256,
    point_allocation_path,
    read_point_allocation,
)
from ..strict_json import load_path
from ..versioning.manifest import sha256_file
from ..versioning.provenance import PROVENANCE_FILENAME
from ..versioning.reference_data import (
    POINTDIR_NAME_WIDTH,
    REFERENCE_DATA_VERSION_FILENAME,
    ReferenceDataEntry,
    ReferenceDataVersioning,
    ReferenceDataView,
    build_reference_data_version_payload,
    canonical_json_sha256,
)
from .ferebus_row_cache import (
    FEREBUS_ROW_CACHE,
    build_version_row_cache,
    read_feature_contract,
    read_version_row_cache,
    row_cache_path,
)
from .filesystem import campaign_owned_path
from .quantum_acceptance_receipts import (
    QUANTUM_ACCEPTANCE_RECEIPT,
    read_quantum_acceptance_receipt,
)
from .quantum_quality import read_quantum_quality_manifest
from .state import _fsync_parent_dir, atomic_write_json


REFERENCE_COMMIT_TRANSACTION_SCHEMA_VERSION = 1
REFERENCE_COMMIT_RECEIPT = "REFERENCE_COMMIT_RECEIPT.json"
REFERENCE_COMMIT_RECEIPT_SCHEMA_VERSION = 1
REFERENCE_COMMIT_LEDGER_BATCH = 32

ProgressCallback = Optional[Callable[[str, Mapping[str, Any]], None]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transaction_identity_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    bindings = payload.get("point_bindings")
    if not isinstance(bindings, list) or any(
        not isinstance(record, Mapping) for record in bindings
    ):
        raise ValueError("reference-commit transaction bindings are invalid")
    return {
        "campaign_uid": str(payload.get("campaign_uid") or ""),
        "reference_data_version": payload.get("reference_data_version"),
        "context": str(payload.get("context") or ""),
        "iteration": payload.get("iteration"),
        "point_allocation_path": str(payload.get("point_allocation_path") or ""),
        "point_allocation_sha256": str(
            payload.get("point_allocation_sha256") or ""
        ),
        "point_bindings": [
            {
                str(key): value
                for key, value in record.items()
                if str(key) != "move_state"
            }
            for record in bindings
        ],
    }


def transaction_dir(campaign_dir: Path) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "reference_commit_transactions"
    )


def transaction_path(campaign_dir: Path, *, context: str, iteration: int) -> Path:
    if str(context) not in {"bootstrap", "active"}:
        raise ValueError("reference-commit transaction context is invalid")
    return transaction_dir(campaign_dir) / (
        str(context) + "-iteration-" + str(int(iteration)).zfill(6) + ".json"
    )


def _emit(callback: ProgressCallback, event: str, **payload: Any) -> None:
    if callback is not None:
        callback(str(event), payload)


def _atomic_copy_small(source: Path, destination: Path) -> Tuple[str, int]:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError("reference metadata source is missing: " + str(source))
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as handle:
            if handle.write(data) != len(data):
                raise OSError("short reference metadata write: " + str(target))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        _fsync_parent_dir(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    if int(target.stat().st_size) != len(data):
        raise OSError("reference metadata size changed during publication: " + str(target))
    return digest, len(data)


def _allocation(campaign: Path, *, context: str, iteration: int) -> Tuple[Path, Dict[str, Any]]:
    path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    payload = read_point_allocation(path)
    if not bool((payload.get("summary") or {}).get("complete", False)):
        raise ValueError("reference commit requires a complete point allocation")
    attempts = accepted_attempts(payload)
    if len(attempts) != int(payload["targets"]["total"]):
        raise ValueError("reference commit allocation accepted count is invalid")
    return path, payload


def _new_ledger(
    campaign: Path,
    *,
    version: int,
    context: str,
    iteration: int,
    allocation_path: Path,
    allocation: Mapping[str, Any],
    parent_view: Optional[ReferenceDataView],
) -> Dict[str, Any]:
    attempts = sorted(accepted_attempts(allocation), key=lambda row: int(row["slot_id"]))
    first_ordinal = len(parent_view.entries) if parent_view is not None else 0
    canonical_staging = campaign_owned_path(campaign, staging_root(campaign))
    records: List[Dict[str, Any]] = []
    for offset, attempt in enumerate(attempts):
        source = campaign_owned_path(
            campaign,
            Path(str(attempt.get("pointdir") or "")),
        )
        if canonical_staging not in source.parents:
            raise ValueError("accepted pointdir is outside daemon staging: " + str(source))
        if source.is_symlink() or not source.is_dir():
            raise FileNotFoundError("accepted pointdir is missing or symlinked: " + str(source))
        receipt = read_quantum_acceptance_receipt(
            campaign,
            source,
            expected_iteration=int(iteration),
            expected_candidate_id=str(attempt["candidate_id"]),
            expected_assignment_sha256=str(allocation["slot_assignment_sha256"]),
            verification="receipt",
            validate_quality=False,
        )
        receipt_path = source / QUANTUM_ACCEPTANCE_RECEIPT
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha != str(attempt.get("quantum_acceptance_receipt_sha256") or ""):
            raise ValueError("allocation acceptance-receipt SHA mismatch")
        if str(receipt.get("content_sha256") or "") != str(
            attempt.get("accepted_pointdir_content_sha256") or ""
        ):
            raise ValueError("allocation accepted-content digest mismatch")
        quality_binding = receipt.get("quality_manifest")
        if not isinstance(quality_binding, Mapping):
            raise ValueError("acceptance receipt lacks quantum-quality binding")
        quality_relative = Path(str(quality_binding.get("path") or ""))
        if quality_relative.is_absolute() or ".." in quality_relative.parts:
            raise ValueError("acceptance receipt quantum-quality path is invalid")
        quality_path = campaign_owned_path(campaign, campaign / quality_relative)
        attempt_quality = campaign_owned_path(
            campaign,
            Path(str(attempt.get("quality_manifest") or "")),
        )
        if quality_path != attempt_quality:
            raise ValueError("allocation and acceptance quality paths differ")
        quality_sha = str(quality_binding.get("sha256") or "")
        quality_record_sha = str(receipt.get("quality_record_sha256") or "")
        for digest, label in (
            (quality_sha, "quantum-quality manifest"),
            (quality_record_sha, "quantum-quality record"),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(label + " binding is not a lowercase SHA-256")
        ordinal = first_ordinal + offset
        destination_name = (
            "POINT_" + str(ordinal).zfill(POINTDIR_NAME_WIDTH) + ".pointdir"
        )
        records.append(
            {
                "ordinal": int(ordinal),
                "source_path": source.relative_to(campaign.resolve()).as_posix(),
                "source_pointdir": source.name,
                "destination_pointdir": destination_name,
                "candidate_id": str(attempt["candidate_id"]),
                "slot_id": int(attempt["slot_id"]),
                "split": str(attempt["split"]),
                "replacement_round": int(attempt.get("round", 0)),
                "acceptance_receipt_sha256": receipt_sha,
                "accepted_content_sha256": str(receipt["content_sha256"]),
                "quality_manifest_path": quality_relative.as_posix(),
                "quality_manifest_sha256": quality_sha,
                "quality_record_sha256": quality_record_sha,
                "provenance_sha256": next(
                    str(binding["sha256"])
                    for binding in receipt["artefacts"]
                    if str(binding["path"]) == PROVENANCE_FILENAME
                ),
                "pointdir_bytes": int(
                    sum(int(binding["size"]) for binding in receipt["artefacts"])
                    + receipt_path.stat().st_size
                ),
                "move_state": "source",
            }
        )
    static_identity = {
        "campaign_uid": str(allocation["campaign_uid"]),
        "reference_data_version": int(version),
        "context": str(context),
        "iteration": int(iteration),
        "point_allocation_path": allocation_path.relative_to(campaign).as_posix(),
        "point_allocation_sha256": allocation_manifest_sha256(allocation_path),
        "point_bindings": records,
    }
    return {
        "schema_version": REFERENCE_COMMIT_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": canonical_json_sha256(
            _transaction_identity_payload(static_identity)
        ),
        **static_identity,
        "status": "prepared",
        "moved_points": 0,
        "moved_bytes": 0,
        "shards_reused": 0,
        "shards_repaired": 0,
        "created_at_iso": _now_iso(),
        "updated_at_iso": _now_iso(),
    }


def _read_ledger(path: Path) -> Dict[str, Any]:
    payload = load_path(path)
    if not isinstance(payload, dict):
        raise ValueError("reference-commit ledger must be a JSON object")
    if payload.get("schema_version") != REFERENCE_COMMIT_TRANSACTION_SCHEMA_VERSION:
        raise ValueError("unsupported reference-commit transaction schema")
    if not isinstance(payload.get("point_bindings"), list) or not payload["point_bindings"]:
        raise ValueError("reference-commit ledger point bindings are missing")
    if payload.get("status") not in {
        "prepared",
        "moving",
        "points_moved",
        "cache_complete",
        "published",
        "complete",
    }:
        raise ValueError("reference-commit ledger status is invalid")
    if payload.get("context") not in {"bootstrap", "active"}:
        raise ValueError("reference-commit ledger context is invalid")
    for field in (
        "reference_data_version",
        "iteration",
        "moved_points",
        "moved_bytes",
        "shards_reused",
        "shards_repaired",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("reference-commit ledger " + field + " is invalid")
    if int(payload["reference_data_version"]) != int(payload["iteration"]):
        raise ValueError("reference-commit ledger version/iteration mismatch")
    if (int(payload["iteration"]) == 0) != (payload["context"] == "bootstrap"):
        raise ValueError("reference-commit ledger context/iteration mismatch")
    if int(payload["moved_points"]) > len(payload["point_bindings"]):
        raise ValueError("reference-commit ledger moved-point count is invalid")
    if any(
        record.get("move_state") not in {"source", "destination"}
        for record in payload["point_bindings"]
    ):
        raise ValueError("reference-commit ledger move state is invalid")
    if str(payload.get("transaction_id") or "") != canonical_json_sha256(
        _transaction_identity_payload(payload)
    ):
        raise ValueError("reference-commit transaction identity mismatch")
    return payload


def _write_ledger(path: Path, payload: Dict[str, Any], *, status: Optional[str] = None) -> None:
    if status is not None:
        payload["status"] = str(status)
    payload["updated_at_iso"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _reconcile_move_states(campaign: Path, staging: Path, ledger: Dict[str, Any]) -> None:
    moved = 0
    moved_bytes = 0
    for record in ledger["point_bindings"]:
        source = campaign_owned_path(campaign, campaign / str(record["source_path"]))
        destination = campaign_owned_path(
            campaign,
            staging / str(record["destination_pointdir"]),
        )
        source_exists = source.is_dir() and not source.is_symlink()
        destination_exists = destination.is_dir() and not destination.is_symlink()
        if source_exists == destination_exists:
            raise ValueError(
                "reference transaction requires exactly one source/destination pointdir for "
                + str(record["candidate_id"])
            )
        root = destination if destination_exists else source
        receipt = read_quantum_acceptance_receipt(
            campaign,
            root,
            expected_iteration=int(ledger["iteration"]),
            expected_candidate_id=str(record["candidate_id"]),
            expected_source_pointdir=str(record["source_pointdir"]),
            verification="receipt",
            validate_quality=False,
        )
        if str(receipt["content_sha256"]) != str(record["accepted_content_sha256"]):
            raise ValueError("reference transaction accepted-content digest mismatch")
        quality_binding = receipt.get("quality_manifest")
        if (
            not isinstance(quality_binding, Mapping)
            or str(quality_binding.get("path") or "")
            != str(record.get("quality_manifest_path") or "")
            or str(quality_binding.get("sha256") or "")
            != str(record.get("quality_manifest_sha256") or "")
            or str(receipt.get("quality_record_sha256") or "")
            != str(record.get("quality_record_sha256") or "")
        ):
            raise ValueError("reference transaction quantum-quality binding mismatch")
        if sha256_file(root / QUANTUM_ACCEPTANCE_RECEIPT) != str(
            record["acceptance_receipt_sha256"]
        ):
            raise ValueError("reference transaction acceptance-receipt SHA mismatch")
        record["move_state"] = "destination" if destination_exists else "source"
        if destination_exists:
            moved += 1
            moved_bytes += int(record["pointdir_bytes"])
    ledger["moved_points"] = int(moved)
    ledger["moved_bytes"] = int(moved_bytes)


def _move_points(
    campaign: Path,
    staging: Path,
    ledger_path: Path,
    ledger: Dict[str, Any],
    callback: ProgressCallback,
) -> None:
    if str(ledger.get("status")) != "prepared":
        _reconcile_move_states(campaign, staging, ledger)
    _write_ledger(ledger_path, ledger, status="moving")
    total = len(ledger["point_bindings"])
    since_checkpoint = 0
    dirty_source_parents = set()
    for record in ledger["point_bindings"]:
        if record["move_state"] == "destination":
            continue
        source = campaign_owned_path(campaign, campaign / str(record["source_path"]))
        destination = campaign_owned_path(
            campaign,
            staging / str(record["destination_pointdir"]),
        )
        if source.stat().st_dev != staging.stat().st_dev:
            raise OSError(
                "reference commit requires source and destination on one filesystem: "
                + str(source)
            )
        os.replace(str(source), str(destination))
        dirty_source_parents.add(source.parent)
        record["move_state"] = "destination"
        ledger["moved_points"] = int(ledger["moved_points"]) + 1
        ledger["moved_bytes"] = int(ledger["moved_bytes"]) + int(
            record["pointdir_bytes"]
        )
        since_checkpoint += 1
        if since_checkpoint >= REFERENCE_COMMIT_LEDGER_BATCH:
            _fsync_parent_dir(destination)
            for parent in sorted(dirty_source_parents, key=str):
                _fsync_parent_dir(parent / ".reference-commit-move")
            dirty_source_parents.clear()
            _write_ledger(ledger_path, ledger, status="moving")
            _emit(
                callback,
                "reference_commit_move_progress",
                moved_points=int(ledger["moved_points"]),
                total_points=total,
                moved_bytes=int(ledger["moved_bytes"]),
            )
            since_checkpoint = 0
    for parent in sorted(dirty_source_parents, key=str):
        _fsync_parent_dir(parent / ".reference-commit-move")
    _fsync_parent_dir(staging / "pointdir-move")
    _write_ledger(ledger_path, ledger, status="points_moved")
    _emit(
        callback,
        "reference_commit_move_progress",
        moved_points=int(ledger["moved_points"]),
        total_points=total,
        moved_bytes=int(ledger["moved_bytes"]),
        complete=True,
    )


def _copy_allocation_metadata(
    allocation_path: Path,
    staging: Path,
    *,
    version: int,
) -> None:
    _atomic_copy_small(allocation_path, staging / "POINT_ALLOCATION.json")
    _atomic_copy_small(
        allocation_path,
        staging
        / (
            "POINT_ALLOCATION.version-"
            + str(int(version)).zfill(COMMITTED_VERSION_NAME_WIDTH)
            + ".json"
        ),
    )
    history = allocation_path.parent / "history"
    if history.is_dir():
        destination = staging / ".point_allocation_history"
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(history.glob("generation-*.json")):
            if source.is_symlink() or not source.is_file():
                raise ValueError("point-allocation history contains a non-regular file")
            _atomic_copy_small(source, destination / source.name)


def _copy_quality_evidence(
    campaign: Path,
    staging: Path,
    ledger: Mapping[str, Any],
    allocation: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    attempts = sorted(accepted_attempts(allocation), key=lambda row: int(row["slot_id"]))
    bindings = list(ledger["point_bindings"])
    if len(attempts) != len(bindings):
        raise ValueError("allocation and reference transaction point counts differ")
    by_manifest: Dict[str, List[Mapping[str, Any]]] = {}
    for record in bindings:
        relative = str(record.get("quality_manifest_path") or "")
        by_manifest.setdefault(relative, []).append(record)
    evidence: List[Dict[str, Any]] = []
    for index, relative_text in enumerate(sorted(by_manifest)):
        source = campaign_owned_path(campaign, campaign / Path(relative_text))
        if source.is_symlink() or not source.is_file():
            raise ValueError("quantum-quality evidence is missing or symlinked")
        raw = load_path(source)
        if not isinstance(raw, dict):
            raise ValueError("quantum-quality evidence must be a JSON object")
        quality = read_quantum_quality_manifest(
            source.parent,
            expected_phase=str(raw.get("phase") or ""),
            expected_iteration=int(raw.get("iteration")),
            manifest_path=source,
        )
        quality_by_name: Dict[str, List[Mapping[str, Any]]] = {}
        for quality_record in quality["records"]:
            if bool(quality_record["accepted"]):
                quality_by_name.setdefault(
                    str(quality_record["pointdir"]), []
                ).append(quality_record)
        relevant = sorted(
            by_manifest[relative_text],
            key=lambda record: int(record["slot_id"]),
        )
        if not relevant:
            raise ValueError("quantum-quality evidence contains no committed point")
        destination = (
            staging / "quality_evidence" / ("quantum_quality_" + str(index).zfill(4) + ".json")
        )
        digest, _size = _atomic_copy_small(source, destination)
        pointdir_bindings = []
        for record in relevant:
            if digest != str(record["quality_manifest_sha256"]):
                raise ValueError("quantum-quality manifest binding mismatch")
            matching = quality_by_name.get(str(record["source_pointdir"]), [])
            if (
                len(matching) != 1
                or canonical_json_sha256(matching[0])
                != str(record["quality_record_sha256"])
            ):
                raise ValueError("quantum-quality record binding mismatch")
            pointdir_bindings.append(
                {
                    "source_pointdir": str(record["source_pointdir"]),
                    "committed_pointdir": str(record["destination_pointdir"]),
                    "candidate_id": str(record["candidate_id"]),
                }
            )
        evidence.append(
            {
                "path": destination.relative_to(staging).as_posix(),
                "sha256": digest,
                "source_path": source.relative_to(campaign.resolve()).as_posix(),
                "source_sha256": digest,
                "phase": str(quality["phase"]),
                "iteration": int(quality["iteration"]),
                "pointdir_bindings": pointdir_bindings,
            }
        )
    committed_names = {
        binding["committed_pointdir"]
        for record in evidence
        for binding in record["pointdir_bindings"]
    }
    if committed_names != {
        str(record["destination_pointdir"]) for record in ledger["point_bindings"]
    }:
        raise ValueError("quantum-quality evidence does not cover every committed point")
    return evidence


def _prehashed_inventory(
    staging: Path,
    ledger: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    manifest: Dict[str, str] = {}
    sizes: Dict[str, int] = {}
    point_roots = {str(record["destination_pointdir"]) for record in ledger["point_bindings"]}
    for record in ledger["point_bindings"]:
        root = staging / str(record["destination_pointdir"])
        receipt = load_path(root / QUANTUM_ACCEPTANCE_RECEIPT)
        if not isinstance(receipt, dict):
            raise ValueError("committed point acceptance receipt is invalid")
        for binding in receipt["artefacts"]:
            relative = str(record["destination_pointdir"]) + "/" + str(binding["path"])
            manifest[relative] = str(binding["sha256"])
            sizes[relative] = int(binding["size"])
        receipt_relative = str(record["destination_pointdir"]) + "/" + QUANTUM_ACCEPTANCE_RECEIPT
        receipt_path = root / QUANTUM_ACCEPTANCE_RECEIPT
        manifest[receipt_relative] = str(record["acceptance_receipt_sha256"])
        sizes[receipt_relative] = int(receipt_path.stat().st_size)
    for current, directory_names, file_names in os.walk(staging, topdown=True):
        current_path = Path(current)
        retained = []
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink():
                raise ValueError(
                    "reference staging contains a symlink: " + str(directory)
                )
            if current_path == staging and directory_name in point_roots:
                continue
            retained.append(directory_name)
        directory_names[:] = retained
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    "reference staging contains a non-regular file: " + str(path)
                )
            if path.name == ".manifest.json":
                continue
            key = path.relative_to(staging).as_posix()
            manifest[key] = sha256_file(path)
            sizes[key] = int(path.stat().st_size)
    return manifest, sizes


def _entries(
    versioning: ReferenceDataVersioning,
    version: int,
    ledger: Mapping[str, Any],
) -> List[ReferenceDataEntry]:
    final = versioning.iteration_path(version)
    return [
        ReferenceDataEntry(
            global_ordinal=int(record["ordinal"]),
            introduced_in_version=int(version),
            pointdir_name=str(record["destination_pointdir"]),
            pointdir_path=(final / str(record["destination_pointdir"])).resolve(strict=False),
            source_pointdir=str(record["source_pointdir"]),
            candidate_id=str(record["candidate_id"]),
            slot_id=int(record["slot_id"]),
            split=str(record["split"]),
            replacement_round=int(record["replacement_round"]),
            accepted_content_sha256=str(record["accepted_content_sha256"]),
            acceptance_receipt_sha256=str(record["acceptance_receipt_sha256"]),
            provenance_sha256=str(record["provenance_sha256"]),
        )
        for record in ledger["point_bindings"]
    ]


def _finish_existing(
    campaign: Path,
    versioning: ReferenceDataVersioning,
    version: int,
    ledger_path: Path,
) -> Tuple[ReferenceDataView, List[str], bool]:
    view = versioning.resolve(version, verification="index")
    contract = read_feature_contract(campaign)
    read_version_row_cache(campaign, str(contract["contract_sha256"]), version)
    current = versioning.current_version()
    if current is None or int(current) < int(version):
        versioning.ensure_current(version)
    elif int(current) > int(version):
        # Replaying an older completed transaction must never move the
        # authoritative pointer backwards. Validate the newer head instead.
        versioning.resolve(int(current), verification="index")
    ledger = _read_ledger(ledger_path)
    _write_ledger(ledger_path, ledger, status="complete")
    names = [
        entry.pointdir_name
        for entry in view.entries
        if entry.introduced_in_version == int(version)
    ]
    return view, names, False


def commit_reference_data_delta(
    campaign_dir: Path,
    *,
    reference_data_version: int,
    context: str,
    iteration: int,
    progress_callback: ProgressCallback = None,
) -> Tuple[ReferenceDataView, List[str], bool]:
    """Move one accepted allocation into a durable delta and publish its row cache."""
    started = time.monotonic()
    campaign = Path(campaign_dir).resolve()
    version = int(reference_data_version)
    if str(context) not in {"bootstrap", "active"}:
        raise ValueError("reference commit context must be bootstrap or active")
    if (version == 0) != (str(context) == "bootstrap") or int(iteration) != version:
        raise ValueError("reference commit context, iteration and version are inconsistent")
    versioning = ReferenceDataVersioning(qm_reference_data_dir(campaign))
    ledger_path = transaction_path(campaign, context=str(context), iteration=int(iteration))
    if versioning.iteration_path(version).is_dir():
        if not ledger_path.is_file():
            raise ValueError("published reference version lacks its transaction ledger")
        return _finish_existing(campaign, versioning, version, ledger_path)

    allocation_path, allocation = _allocation(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    parent_view = None
    if version > 0:
        parent_view = versioning.resolve(version - 1, verification="index")
    elif versioning.list_committed_versions():
        raise ValueError("bootstrap reference data must be the first commit")
    if ledger_path.is_file():
        ledger = _read_ledger(ledger_path)
        expected = (
            int(ledger["reference_data_version"]),
            str(ledger["context"]),
            int(ledger["iteration"]),
            str(ledger["point_allocation_sha256"]),
        )
        observed = (
            version,
            str(context),
            int(iteration),
            allocation_manifest_sha256(allocation_path),
        )
        if expected != observed:
            raise ValueError("reference-commit ledger identity conflicts with allocation")
    else:
        ledger = _new_ledger(
            campaign,
            version=version,
            context=str(context),
            iteration=int(iteration),
            allocation_path=allocation_path,
            allocation=allocation,
            parent_view=parent_view,
        )
        _write_ledger(ledger_path, ledger, status="prepared")
    _emit(
        progress_callback,
        "reference_commit_started",
        transaction_id=str(ledger["transaction_id"]),
        version=version,
        n_points=len(ledger["point_bindings"]),
    )

    staging = versioning.staging_path(version)
    if not staging.exists():
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=False)
        _fsync_parent_dir(staging)
    elif staging.is_symlink() or not staging.is_dir():
        raise ValueError("reference-commit staging is not a regular directory")
    _move_points(campaign, staging, ledger_path, ledger, progress_callback)
    _copy_allocation_metadata(allocation_path, staging, version=version)
    quality_evidence = _copy_quality_evidence(campaign, staging, ledger, allocation)

    cache_bindings = [
        {
            "pointdir_path": staging / str(record["destination_pointdir"]),
            "pointdir_name": str(record["destination_pointdir"]),
            "source_pointdir": str(record["source_pointdir"]),
            "candidate_id": str(record["candidate_id"]),
        }
        for record in ledger["point_bindings"]
    ]
    repair_root = (
        campaign
        / ".DATA"
        / "CACHE"
        / "FEREBUS_ROWS"
        / "_repair"
        / str(ledger["transaction_id"])[:16]
    )
    repair_root.mkdir(parents=True, exist_ok=True)
    cache = build_version_row_cache(
        campaign,
        version=version,
        point_bindings=cache_bindings,
        repair_root=repair_root,
        progress_callback=progress_callback,
    )
    ledger["shards_reused"] = int(cache.get("shards_reused", 0))
    ledger["shards_repaired"] = int(cache.get("shards_repaired", 0))
    _write_ledger(ledger_path, ledger, status="cache_complete")
    _emit(
        progress_callback,
        "reference_commit_shards_resolved",
        shards_reused=int(ledger["shards_reused"]),
        shards_repaired=int(ledger["shards_repaired"]),
    )
    _emit(
        progress_callback,
        "reference_commit_cache_complete",
        shards_reused=int(ledger["shards_reused"]),
        shards_repaired=int(ledger["shards_repaired"]),
    )

    entries = _entries(versioning, version, ledger)
    payload = build_reference_data_version_payload(
        campaign_uid=str(allocation["campaign_uid"]),
        version=version,
        source_context=str(context),
        source_iteration=int(iteration),
        parent_view=parent_view,
        point_allocation_manifest=allocation_path.relative_to(campaign).as_posix(),
        point_allocation_sha256=allocation_manifest_sha256(allocation_path),
        added_entries=entries,
        quantum_quality_evidence=quality_evidence,
    )
    version_manifest = staging / REFERENCE_DATA_VERSION_FILENAME
    atomic_write_json(version_manifest, payload)
    contract = read_feature_contract(campaign)
    cache_dir = row_cache_path(campaign, str(contract["contract_sha256"]), version)
    cache_manifest = cache_dir / FEREBUS_ROW_CACHE
    commit_receipt = {
        "schema_version": REFERENCE_COMMIT_RECEIPT_SCHEMA_VERSION,
        "transaction_id": str(ledger["transaction_id"]),
        "campaign_uid": str(allocation["campaign_uid"]),
        "reference_data_version": version,
        "source_context": str(context),
        "source_iteration": int(iteration),
        "n_moved_points": int(ledger["moved_points"]),
        "moved_bytes": int(ledger["moved_bytes"]),
        "point_bindings_sha256": canonical_json_sha256(ledger["point_bindings"]),
        "reference_data_version_sha256": sha256_file(version_manifest),
        "ferebus_feature_contract_sha256": str(contract["contract_sha256"]),
        "ferebus_row_cache": {
            "path": cache_manifest.relative_to(campaign).as_posix(),
            "sha256": sha256_file(cache_manifest),
            "shards_reused": int(ledger["shards_reused"]),
            "shards_repaired": int(ledger["shards_repaired"]),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "completed_at_iso": _now_iso(),
    }
    atomic_write_json(staging / REFERENCE_COMMIT_RECEIPT, commit_receipt)
    manifest, sizes = _prehashed_inventory(staging, ledger)
    versioning.commit_prehashed(version, manifest=manifest, expected_sizes=sizes)
    _write_ledger(ledger_path, ledger, status="published")
    versioning.update_current(version)
    _write_ledger(ledger_path, ledger, status="complete")
    _emit(
        progress_callback,
        "reference_commit_published",
        version=version,
        moved_points=int(ledger["moved_points"]),
        moved_bytes=int(ledger["moved_bytes"]),
        shards_reused=int(ledger["shards_reused"]),
        shards_repaired=int(ledger["shards_repaired"]),
        elapsed_seconds=float(time.monotonic() - started),
    )
    all_entries = tuple(parent_view.entries if parent_view is not None else ()) + tuple(entries)
    view = ReferenceDataView(
        version=version,
        campaign_uid=str(allocation["campaign_uid"]),
        entries=all_entries,
        cumulative_view_sha256=str(payload["cumulative_view_sha256"]),
        head_manifest_sha256=sha256_file(versioning.iteration_path(version) / REFERENCE_DATA_VERSION_FILENAME),
    )
    return view, [entry.pointdir_name for entry in entries], True


def classify_reference_commit(campaign_dir: Path, *, context: str, iteration: int) -> Dict[str, Any]:
    path = transaction_path(campaign_dir, context=context, iteration=iteration)
    if not path.is_file():
        return {"state": "absent", "path": str(path)}
    ledger = _read_ledger(path)
    versioning = ReferenceDataVersioning(qm_reference_data_dir(campaign_dir))
    final = versioning.iteration_path(iteration)
    staging = versioning.staging_path(iteration)
    reason = None
    if final.is_symlink() or staging.is_symlink():
        state = "invalid"
        reason = "reference version or transaction staging is symlinked"
    elif final.exists() and staging.exists():
        state = "invalid"
        reason = "published version and transaction staging both exist"
    elif final.is_dir():
        if str(ledger.get("status")) == "complete":
            state = "complete"
        elif versioning.current_version() != int(iteration):
            state = "pointer_incomplete"
        else:
            state = "published"
    elif staging.is_dir():
        try:
            _reconcile_move_states(Path(campaign_dir).resolve(), staging, ledger)
        except Exception as exc:
            return {
                "state": "invalid",
                "path": str(path),
                "reason": type(exc).__name__ + ": " + str(exc),
                "ledger": ledger,
            }
        moved = int(ledger["moved_points"])
        total = len(ledger["point_bindings"])
        if moved == 0:
            state = "prepared"
        elif moved < total:
            state = "partially_moved"
        elif str(ledger.get("status")) == "cache_complete":
            state = "publication_incomplete"
        else:
            state = "cache_incomplete"
    else:
        state = "invalid"
        reason = "transaction has neither staging nor published reference version"
    result = {"state": state, "path": str(path), "ledger": ledger}
    if reason:
        result["reason"] = reason
    return result


def inventory_reference_commits(campaign_dir: Path) -> List[Dict[str, Any]]:
    """Classify every durable reference-commit transaction without mutation."""
    campaign = Path(campaign_dir).resolve()
    root = transaction_dir(campaign)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        return [
            {
                "state": "invalid",
                "path": str(root),
                "reason": "transaction root is not a regular directory",
            }
        ]
    records: List[Dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            records.append(
                {
                    "state": "invalid",
                    "path": str(path),
                    "reason": "transaction entry is not a regular JSON file",
                }
            )
            continue
        try:
            ledger = _read_ledger(path)
            record = classify_reference_commit(
                campaign,
                context=str(ledger["context"]),
                iteration=int(ledger["iteration"]),
            )
        except Exception as exc:
            record = {
                "state": "invalid",
                "path": str(path),
                "reason": type(exc).__name__ + ": " + str(exc),
            }
        records.append(record)
    return records


__all__ = [
    "REFERENCE_COMMIT_RECEIPT",
    "REFERENCE_COMMIT_RECEIPT_SCHEMA_VERSION",
    "REFERENCE_COMMIT_TRANSACTION_SCHEMA_VERSION",
    "classify_reference_commit",
    "commit_reference_data_delta",
    "inventory_reference_commits",
    "transaction_path",
]
