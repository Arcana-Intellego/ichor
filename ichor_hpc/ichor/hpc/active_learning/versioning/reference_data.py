"""Immutable delta-only versioning for committed QM reference data."""

from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..layout import COMMITTED_VERSION_NAME_WIDTH, qm_reference_data_dir
from .manifest import (
    ManifestMismatchError,
    read_manifest,
    sha256_file,
    unmanifested_directories,
    verify_manifest,
)
from .provenance import PROVENANCE_FILENAME, read_provenance
from .versioned_directory import VersionedDirectory


REFERENCE_DATA_VERSION_FILENAME = "REFERENCE_DATA_VERSION.json"
REFERENCE_DATA_VERSION_SCHEMA_VERSION = 1
REFERENCE_DATA_CACHE_FILENAME = "reference_data_view_cache.json"
REFERENCE_DATA_CACHE_SCHEMA_VERSION = 1
POINTDIR_NAME_WIDTH = 6
VALID_SPLITS = frozenset({"train", "int_val", "ext_val"})


class ReferenceDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReferenceDataEntry:
    global_ordinal: int
    introduced_in_version: int
    pointdir_name: str
    pointdir_path: Path
    candidate_id: str
    slot_id: int
    split: str
    replacement_round: int
    pointdir_tree_sha256: str
    provenance_sha256: str

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "global_ordinal": int(self.global_ordinal),
            "introduced_in_version": int(self.introduced_in_version),
            "pointdir_name": str(self.pointdir_name),
            "candidate_id": str(self.candidate_id),
            "slot_id": int(self.slot_id),
            "split": str(self.split),
            "replacement_round": int(self.replacement_round),
            "pointdir_tree_sha256": str(self.pointdir_tree_sha256),
            "provenance_sha256": str(self.provenance_sha256),
        }


@dataclass(frozen=True)
class ReferenceDataView:
    version: int
    campaign_uid: str
    entries: Tuple[ReferenceDataEntry, ...]
    cumulative_view_sha256: str
    head_manifest_sha256: str

    @property
    def pointdirs(self) -> Tuple[Path, ...]:
        return tuple(entry.pointdir_path for entry in self.entries)


class ReferenceDataVersioning(VersionedDirectory):
    def resolve(
        self,
        version: int,
        *,
        verification: str = "metadata",
    ) -> ReferenceDataView:
        return resolve_reference_data_view(
            Path(self.parent).parent,
            int(version),
            verification=verification,
            reference_data_root=self.parent,
        )

    def verify_committed_reference_data_inputs(
        self,
        version: int,
        *,
        verification: str = "metadata",
    ) -> None:
        self.resolve(version, verification=verification)


def reference_data_version_path(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / REFERENCE_DATA_VERSION_FILENAME


def reference_data_cache_path(campaign_dir: Union[str, Path]) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / REFERENCE_DATA_CACHE_FILENAME
    )


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_pointdir_tree(pointdir: Union[str, Path]) -> str:
    root = Path(pointdir)
    if not root.is_dir() or root.is_symlink():
        raise ReferenceDataError("reference pointdir is missing or symlinked: " + str(root))
    records: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ReferenceDataError("reference pointdir contains a symlink: " + str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReferenceDataError("reference pointdir contains a special file: " + str(path))
        if path.name.lower().startswith(".nfs") or path.name.endswith((".lock", ".tmp")):
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ReferenceDataError("reference pointdir contains no committed files: " + str(root))
    return canonical_json_sha256(records)


def seal_reference_data_version(iteration_dir: Union[str, Path]) -> None:
    """Make a committed reference-data delta read-only for its owner."""
    root = Path(iteration_dir)
    if not root.is_dir() or root.is_symlink():
        raise ReferenceDataError("cannot seal missing reference-data version: " + str(root))
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ReferenceDataError("cannot seal a symlinked reference-data entry: " + str(path))
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            os.chmod(path, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        elif path.is_file():
            os.chmod(path, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    root_mode = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, root_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    from ..daemon.state import _fsync_parent_dir

    _fsync_parent_dir(root)


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReferenceDataError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(value, dict):
        raise ReferenceDataError(label + " must be a JSON object: " + str(path))
    return value


def _safe_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReferenceDataError(label + " must be a lowercase SHA-256")
    return text


def _safe_int(value: Any, label: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ReferenceDataError(label + " must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ReferenceDataError(label + " must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceDataError(label + " must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ReferenceDataError(label + " must be >= " + str(minimum))
    return parsed


def _entry_from_record(
    record: Mapping[str, Any],
    *,
    iteration_dir: Path,
    version: int,
) -> ReferenceDataEntry:
    ordinal = _safe_int(record.get("global_ordinal"), "global_ordinal", minimum=0)
    name = str(record.get("pointdir_name") or "")
    expected_name = "POINT_" + str(ordinal).zfill(POINTDIR_NAME_WIDTH) + ".pointdir"
    if name != expected_name:
        raise ReferenceDataError("reference-data pointdir name/ordinal mismatch")
    candidate_id = str(record.get("candidate_id") or "")
    if not candidate_id:
        raise ReferenceDataError("reference-data candidate_id is empty")
    slot_id = _safe_int(record.get("slot_id"), "slot_id", minimum=0)
    split = str(record.get("split") or "")
    replacement_round = _safe_int(
        record.get("replacement_round"),
        "replacement_round",
        minimum=0,
    )
    if split not in VALID_SPLITS:
        raise ReferenceDataError("reference-data allocation identity is invalid")
    pointdir = iteration_dir / name
    if not pointdir.is_dir() or pointdir.is_symlink():
        raise ReferenceDataError("reference-data pointdir is missing or symlinked: " + str(pointdir))
    try:
        pointdir.resolve().relative_to(iteration_dir.resolve())
    except ValueError as exc:
        raise ReferenceDataError("reference-data pointdir escapes its version") from exc
    return ReferenceDataEntry(
        global_ordinal=ordinal,
        introduced_in_version=int(version),
        pointdir_name=name,
        pointdir_path=pointdir.resolve(),
        candidate_id=candidate_id,
        slot_id=slot_id,
        split=split,
        replacement_round=replacement_round,
        pointdir_tree_sha256=_safe_sha(
            record.get("pointdir_tree_sha256"), "pointdir_tree_sha256"
        ),
        provenance_sha256=_safe_sha(
            record.get("provenance_sha256"), "provenance_sha256"
        ),
    )


def _validate_provenance(entry: ReferenceDataEntry) -> None:
    path = entry.pointdir_path / PROVENANCE_FILENAME
    if not path.is_file() or path.is_symlink():
        raise ReferenceDataError("reference-data provenance is missing: " + str(path))
    if sha256_file(path) != entry.provenance_sha256:
        raise ReferenceDataError("reference-data provenance SHA mismatch: " + str(path))
    provenance = read_provenance(entry.pointdir_path)
    allocation = provenance.get("point_allocation")
    if not isinstance(allocation, dict):
        raise ReferenceDataError("reference-data provenance lacks point allocation")
    observed = (
        str(allocation.get("candidate_id") or ""),
        _safe_int(allocation.get("slot_id"), "provenance slot_id", minimum=0),
        str(allocation.get("split") or ""),
        _safe_int(
            allocation.get("replacement_round", 0),
            "provenance replacement_round",
            minimum=0,
        ),
    )
    expected = (
        entry.candidate_id,
        entry.slot_id,
        entry.split,
        entry.replacement_round,
    )
    if observed != expected:
        raise ReferenceDataError("reference-data provenance allocation mismatch")


def _validate_allocation_snapshot(
    iteration_dir: Path,
    payload: Mapping[str, Any],
    added: Sequence[ReferenceDataEntry],
) -> None:
    from ..point_allocation import (
        accepted_attempts,
        point_allocation_path,
        read_point_allocation,
    )

    version = _safe_int(
        payload.get("reference_data_version"),
        "reference_data_version",
        minimum=0,
    )
    snapshot = iteration_dir / (
        "POINT_ALLOCATION.version-"
        + str(version).zfill(COMMITTED_VERSION_NAME_WIDTH)
        + ".json"
    )
    if not snapshot.is_file() or snapshot.is_symlink():
        raise ReferenceDataError(
            "reference-data allocation snapshot is missing: " + str(snapshot)
        )
    expected_sha = _safe_sha(
        payload.get("point_allocation_sha256"),
        "point_allocation_sha256",
    )
    if sha256_file(snapshot) != expected_sha:
        raise ReferenceDataError("reference-data allocation snapshot SHA mismatch")
    try:
        allocation = read_point_allocation(
            snapshot,
            history_dir=iteration_dir / ".point_allocation_history",
        )
    except Exception as exc:
        raise ReferenceDataError(
            "reference-data allocation snapshot is invalid: " + str(snapshot)
        ) from exc
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise ReferenceDataError("reference-data allocation snapshot is incomplete")
    expected_header = (
        str(payload.get("campaign_uid") or ""),
        str(payload.get("source_context") or ""),
        _safe_int(payload.get("source_iteration"), "source_iteration", minimum=0),
    )
    observed_header = (
        str(allocation.get("campaign_uid") or ""),
        str(allocation.get("context") or ""),
        _safe_int(allocation.get("iteration"), "allocation iteration", minimum=0),
    )
    if observed_header != expected_header:
        raise ReferenceDataError("reference-data allocation snapshot header mismatch")
    campaign_dir = iteration_dir.parent.parent
    expected_source_path = point_allocation_path(
        campaign_dir,
        context=expected_header[1],
        iteration=expected_header[2],
    ).relative_to(campaign_dir).as_posix()
    if str(payload.get("point_allocation_manifest") or "") != expected_source_path:
        raise ReferenceDataError("reference-data allocation source path mismatch")
    expected_entries = [
        (
            str(record["candidate_id"]),
            int(record["slot_id"]),
            str(record["split"]),
            int(record.get("round", 0)),
        )
        for record in sorted(
            accepted_attempts(allocation),
            key=lambda record: int(record["slot_id"]),
        )
    ]
    observed_entries = [
        (
            entry.candidate_id,
            entry.slot_id,
            entry.split,
            entry.replacement_round,
        )
        for entry in added
    ]
    if observed_entries != expected_entries:
        raise ReferenceDataError(
            "reference-data delta does not match its allocation snapshot"
        )


def _view_sha(entries: Sequence[ReferenceDataEntry]) -> str:
    return canonical_json_sha256([entry.identity_payload() for entry in entries])


def resolve_reference_data_view(
    campaign_dir: Union[str, Path],
    reference_data_version: int,
    *,
    verification: str = "metadata",
    reference_data_root: Optional[Union[str, Path]] = None,
    expected_campaign_uid: Optional[str] = None,
) -> ReferenceDataView:
    if verification not in {"metadata", "deep"}:
        raise ValueError("reference-data verification must be metadata or deep")
    campaign = Path(campaign_dir)
    root = (
        Path(reference_data_root)
        if reference_data_root is not None
        else qm_reference_data_dir(campaign)
    )
    target_version = _safe_int(
        reference_data_version,
        "reference_data_version",
        minimum=0,
    )
    versioning = ReferenceDataVersioning(root)
    committed = versioning.list_committed_versions()
    expected_versions = list(range(target_version + 1))
    if [version for version in committed if version <= target_version] != expected_versions:
        raise ReferenceDataError(
            "reference-data versions are not contiguous through " + str(target_version)
        )

    entries: List[ReferenceDataEntry] = []
    previous_manifest_sha: Optional[str] = None
    campaign_uid: Optional[str] = None
    head_manifest_sha = ""
    for version in expected_versions:
        iteration_dir = versioning.iteration_path(version)
        directory_manifest = read_manifest(iteration_dir)
        if verification == "deep":
            verify_manifest(iteration_dir, manifest=directory_manifest)
        extras = unmanifested_directories(
            iteration_dir,
            manifest=directory_manifest,
            pattern="POINT_*.pointdir",
        )
        if extras:
            raise ManifestMismatchError(
                "unmanifested reference pointdir(s): " + ", ".join(extras[:5])
            )
        manifest_path = reference_data_version_path(iteration_dir)
        payload = _read_json_object(manifest_path, "reference-data version manifest")
        if _safe_int(payload.get("schema_version"), "schema_version") != REFERENCE_DATA_VERSION_SCHEMA_VERSION:
            raise ReferenceDataError("unsupported reference-data version manifest schema")
        if str(payload.get("storage_mode")) != "delta":
            raise ReferenceDataError("reference-data storage_mode must be delta")
        if _safe_int(payload.get("reference_data_version"), "reference_data_version") != version:
            raise ReferenceDataError("reference-data version number mismatch")
        uid = str(payload.get("campaign_uid") or "")
        if not uid or (campaign_uid is not None and uid != campaign_uid):
            raise ReferenceDataError("reference-data campaign UID mismatch")
        campaign_uid = uid
        parent = payload.get("parent_version")
        parent_sha = payload.get("parent_manifest_sha256")
        if version == 0:
            if parent is not None or parent_sha is not None:
                raise ReferenceDataError("reference-data bootstrap must not have a parent")
        else:
            if _safe_int(parent, "parent_version") != version - 1:
                raise ReferenceDataError("reference-data parent version mismatch")
            if _safe_sha(parent_sha, "parent_manifest_sha256") != previous_manifest_sha:
                raise ReferenceDataError("reference-data parent manifest SHA mismatch")
        records = payload.get("added_pointdirs")
        if (
            not isinstance(records, list)
            or not records
            or len(records) != _safe_int(payload.get("n_added"), "n_added", minimum=1)
        ):
            raise ReferenceDataError("reference-data added-point count mismatch")
        added = [
            _entry_from_record(record, iteration_dir=iteration_dir, version=version)
            for record in records
            if isinstance(record, Mapping)
        ]
        if len(added) != len(records):
            raise ReferenceDataError("reference-data point record must be an object")
        _validate_allocation_snapshot(iteration_dir, payload, added)
        quality_status = str(payload.get("quantum_quality_evidence_status") or "legacy_missing")
        quality_records = payload.get("quantum_quality_evidence", [])
        if not isinstance(quality_records, list):
            raise ReferenceDataError("reference-data quantum-quality evidence must be a list")
        if quality_status == "committed" and not quality_records:
            raise ReferenceDataError("committed quantum-quality evidence is empty")
        if quality_status not in {"committed", "legacy_missing"}:
            raise ReferenceDataError("reference-data quantum-quality evidence status is invalid")
        for record in quality_records:
            if not isinstance(record, Mapping):
                raise ReferenceDataError("quantum-quality evidence record must be an object")
            relative = Path(str(record.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ReferenceDataError("quantum-quality evidence path escapes its version")
            evidence_path = iteration_dir / relative
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise ReferenceDataError("quantum-quality evidence file is missing")
            if sha256_file(evidence_path) != _safe_sha(
                record.get("sha256"), "quantum_quality_evidence.sha256"
            ):
                raise ReferenceDataError("quantum-quality evidence SHA mismatch")
        entries.extend(added)
        if [entry.global_ordinal for entry in entries] != list(range(len(entries))):
            raise ReferenceDataError("reference-data global ordinals are not contiguous")
        if len({entry.candidate_id for entry in entries}) != len(entries):
            raise ReferenceDataError("reference-data candidate IDs are not unique")
        if len({entry.pointdir_name for entry in entries}) != len(entries):
            raise ReferenceDataError("reference-data pointdir names are not unique")
        if _safe_int(
            payload.get("cumulative_point_count"),
            "cumulative_point_count",
            minimum=1,
        ) != len(entries):
            raise ReferenceDataError("reference-data cumulative point count mismatch")
        for entry in added:
            _validate_provenance(entry)
            if (
                verification == "deep"
                and hash_pointdir_tree(entry.pointdir_path) != entry.pointdir_tree_sha256
            ):
                raise ReferenceDataError(
                    "reference-data pointdir tree SHA mismatch: " + str(entry.pointdir_path)
                )
        expected_view_sha = _view_sha(entries)
        if _safe_sha(payload.get("cumulative_view_sha256"), "cumulative_view_sha256") != expected_view_sha:
            raise ReferenceDataError("reference-data cumulative view SHA mismatch")
        head_manifest_sha = sha256_file(manifest_path)
        previous_manifest_sha = head_manifest_sha

    view = ReferenceDataView(
        version=target_version,
        campaign_uid=str(campaign_uid),
        entries=tuple(entries),
        cumulative_view_sha256=_view_sha(entries),
        head_manifest_sha256=head_manifest_sha,
    )
    if expected_campaign_uid is not None and view.campaign_uid != str(
        expected_campaign_uid
    ):
        raise ReferenceDataError("reference-data campaign UID does not match state")
    _write_reference_data_cache(campaign, view)
    return view


def _write_reference_data_cache(campaign_dir: Path, view: ReferenceDataView) -> None:
    from ..daemon.state import atomic_write_json

    path = reference_data_cache_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "schema_version": REFERENCE_DATA_CACHE_SCHEMA_VERSION,
            "reference_data_version": int(view.version),
            "head_manifest_sha256": str(view.head_manifest_sha256),
            "cumulative_view_sha256": str(view.cumulative_view_sha256),
            "entries": [entry.identity_payload() for entry in view.entries],
        },
    )


def build_reference_data_version_payload(
    *,
    campaign_uid: str,
    version: int,
    source_context: str,
    source_iteration: int,
    parent_view: Optional[ReferenceDataView],
    point_allocation_manifest: str,
    point_allocation_sha256: str,
    added_entries: Sequence[ReferenceDataEntry],
    quantum_quality_evidence: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    if source_context not in {"bootstrap", "active"}:
        raise ValueError("reference-data source context must be bootstrap or active")
    all_entries = list(parent_view.entries if parent_view is not None else ()) + list(added_entries)
    return {
        "schema_version": REFERENCE_DATA_VERSION_SCHEMA_VERSION,
        "storage_mode": "delta",
        "campaign_uid": str(campaign_uid),
        "reference_data_version": int(version),
        "source_context": str(source_context),
        "source_iteration": int(source_iteration),
        "parent_version": None if parent_view is None else int(parent_view.version),
        "parent_manifest_sha256": (
            None if parent_view is None else str(parent_view.head_manifest_sha256)
        ),
        "point_allocation_manifest": str(point_allocation_manifest),
        "point_allocation_sha256": _safe_sha(
            point_allocation_sha256, "point_allocation_sha256"
        ),
        "n_added": int(len(added_entries)),
        "cumulative_point_count": int(len(all_entries)),
        "first_global_ordinal": (
            None if not added_entries else int(added_entries[0].global_ordinal)
        ),
        "last_global_ordinal": (
            None if not added_entries else int(added_entries[-1].global_ordinal)
        ),
        "added_pointdirs": [entry.identity_payload() for entry in added_entries],
        "quantum_quality_evidence": [dict(record) for record in quantum_quality_evidence],
        "quantum_quality_evidence_status": (
            "committed" if quantum_quality_evidence else "legacy_missing"
        ),
        "cumulative_view_sha256": _view_sha(all_entries),
    }


__all__ = [
    "REFERENCE_DATA_VERSION_FILENAME",
    "REFERENCE_DATA_VERSION_SCHEMA_VERSION",
    "REFERENCE_DATA_CACHE_FILENAME",
    "POINTDIR_NAME_WIDTH",
    "ReferenceDataError",
    "ReferenceDataEntry",
    "ReferenceDataView",
    "ReferenceDataVersioning",
    "reference_data_version_path",
    "reference_data_cache_path",
    "canonical_json_sha256",
    "hash_pointdir_tree",
    "seal_reference_data_version",
    "resolve_reference_data_view",
    "build_reference_data_version_payload",
]
