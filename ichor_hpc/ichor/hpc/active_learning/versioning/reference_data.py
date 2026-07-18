"""Content-verified delta-only versioning for committed QM reference data."""

from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
REFERENCE_DATA_VERSION_SCHEMA_VERSION = 3
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
    source_pointdir: str
    candidate_id: str
    slot_id: int
    split: str
    replacement_round: int
    accepted_content_sha256: str
    acceptance_receipt_sha256: str
    provenance_sha256: str

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "global_ordinal": int(self.global_ordinal),
            "introduced_in_version": int(self.introduced_in_version),
            "pointdir_name": str(self.pointdir_name),
            "source_pointdir": str(self.source_pointdir),
            "candidate_id": str(self.candidate_id),
            "slot_id": int(self.slot_id),
            "split": str(self.split),
            "replacement_round": int(self.replacement_round),
            "accepted_content_sha256": str(self.accepted_content_sha256),
            "acceptance_receipt_sha256": str(self.acceptance_receipt_sha256),
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
        digest_file: Optional[Callable[[Path, bool], str]] = None,
    ) -> ReferenceDataView:
        return resolve_reference_data_view(
            Path(self.parent).parent,
            int(version),
            verification=verification,
            reference_data_root=self.parent,
            digest_file=digest_file,
        )

    def verify_committed_reference_data_inputs(
        self,
        version: int,
        *,
        verification: str = "metadata",
    ) -> None:
        self.resolve(version, verification=verification)

    def resolve_chain(
        self,
        version: int,
        *,
        verification: str = "metadata",
        digest_file: Optional[Callable[[Path, bool], str]] = None,
        resolved_views_out: Optional[List[ReferenceDataView]] = None,
    ) -> Tuple[ReferenceDataView, ...]:
        return resolve_reference_data_chain(
            Path(self.parent).parent,
            int(version),
            verification=verification,
            reference_data_root=self.parent,
            digest_file=digest_file,
            resolved_views_out=resolved_views_out,
        )


def _digest(
    path: Path,
    *,
    payload: bool,
    digest_file: Optional[Callable[[Path, bool], str]],
) -> str:
    if digest_file is not None:
        return digest_file(path, payload)
    return sha256_file(path)


def reference_data_version_path(iteration_dir: Union[str, Path]) -> Path:
    return Path(iteration_dir) / REFERENCE_DATA_VERSION_FILENAME


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
    validate_payload_path: bool = True,
) -> ReferenceDataEntry:
    ordinal = _safe_int(record.get("global_ordinal"), "global_ordinal", minimum=0)
    name = str(record.get("pointdir_name") or "")
    expected_name = "POINT_" + str(ordinal).zfill(POINTDIR_NAME_WIDTH) + ".pointdir"
    if name != expected_name:
        raise ReferenceDataError("reference-data pointdir name/ordinal mismatch")
    candidate_id = str(record.get("candidate_id") or "")
    if not candidate_id:
        raise ReferenceDataError("reference-data candidate_id is empty")
    source_pointdir = str(record.get("source_pointdir") or "")
    if not source_pointdir or Path(source_pointdir).name != source_pointdir:
        raise ReferenceDataError("reference-data source pointdir is invalid")
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
    if validate_payload_path:
        if not pointdir.is_dir() or pointdir.is_symlink():
            raise ReferenceDataError(
                "reference-data pointdir is missing or symlinked: " + str(pointdir)
            )
        try:
            pointdir.resolve().relative_to(iteration_dir.resolve())
        except ValueError as exc:
            raise ReferenceDataError("reference-data pointdir escapes its version") from exc
    return ReferenceDataEntry(
        global_ordinal=ordinal,
        introduced_in_version=int(version),
        pointdir_name=name,
        pointdir_path=(pointdir.resolve() if validate_payload_path else pointdir.absolute()),
        source_pointdir=source_pointdir,
        candidate_id=candidate_id,
        slot_id=slot_id,
        split=split,
        replacement_round=replacement_round,
        accepted_content_sha256=_safe_sha(
            record.get("accepted_content_sha256"), "accepted_content_sha256"
        ),
        acceptance_receipt_sha256=_safe_sha(
            record.get("acceptance_receipt_sha256"), "acceptance_receipt_sha256"
        ),
        provenance_sha256=_safe_sha(
            record.get("provenance_sha256"), "provenance_sha256"
        ),
    )


def _validate_provenance(
    entry: ReferenceDataEntry,
    *,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
) -> None:
    path = entry.pointdir_path / PROVENANCE_FILENAME
    if not path.is_file() or path.is_symlink():
        raise ReferenceDataError("reference-data provenance is missing: " + str(path))
    if _digest(path, payload=False, digest_file=digest_file) != entry.provenance_sha256:
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
    *,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
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
    if _digest(snapshot, payload=False, digest_file=digest_file) != expected_sha:
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


def resolve_reference_data_chain(
    campaign_dir: Union[str, Path],
    reference_data_version: int,
    *,
    verification: str = "metadata",
    reference_data_root: Optional[Union[str, Path]] = None,
    expected_campaign_uid: Optional[str] = None,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
    resolved_views_out: Optional[List[ReferenceDataView]] = None,
) -> Tuple[ReferenceDataView, ...]:
    if verification not in {"authority", "index", "metadata", "deep"}:
        raise ValueError(
            "reference-data verification must be authority, index, metadata or deep"
        )
    if verification == "deep" and digest_file is None:
        digest_cache: Dict[str, str] = {}

        def cached_digest(path: Path, payload: bool) -> str:
            del payload
            key = str(Path(path).absolute())
            if key not in digest_cache:
                digest_cache[key] = sha256_file(path)
            return digest_cache[key]

        digest_file = cached_digest
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
    views: List[ReferenceDataView] = []
    for version in expected_versions:
        iteration_dir = versioning.iteration_path(version)
        directory_manifest = read_manifest(iteration_dir)
        if verification != "authority":
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
        commit_receipt_path = iteration_dir / "REFERENCE_COMMIT_RECEIPT.json"
        commit_receipt = _read_json_object(
            commit_receipt_path,
            "reference-commit receipt",
        )
        manifest_sha = _digest(
            manifest_path,
            payload=False,
            digest_file=digest_file,
        )
        commit_receipt_sha = _digest(
            commit_receipt_path,
            payload=False,
            digest_file=digest_file,
        )
        if _safe_int(commit_receipt.get("schema_version"), "commit receipt schema") != 1:
            raise ReferenceDataError("unsupported reference-commit receipt schema")
        if (
            _safe_int(
                commit_receipt.get("reference_data_version"),
                "commit receipt reference_data_version",
            )
            != version
            or str(commit_receipt.get("campaign_uid") or "") != uid
            or _safe_sha(
                commit_receipt.get("reference_data_version_sha256"),
                "commit receipt reference-data manifest SHA",
            )
            != manifest_sha
        ):
            raise ReferenceDataError("reference-commit receipt identity mismatch")
        if directory_manifest.get(REFERENCE_DATA_VERSION_FILENAME) != manifest_sha:
            raise ReferenceDataError(
                "reference-data directory manifest does not bind its version manifest"
            )
        if directory_manifest.get("REFERENCE_COMMIT_RECEIPT.json") != commit_receipt_sha:
            raise ReferenceDataError(
                "reference-data directory manifest does not bind its commit receipt"
            )
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
            _entry_from_record(
                record,
                iteration_dir=iteration_dir,
                version=version,
                validate_payload_path=verification != "authority",
            )
            for record in records
            if isinstance(record, Mapping)
        ]
        if len(added) != len(records):
            raise ReferenceDataError("reference-data point record must be an object")
        added_by_identity = {
            (entry.pointdir_name, entry.candidate_id): entry
            for entry in added
        }
        if len(added_by_identity) != len(added):
            raise ReferenceDataError(
                "reference-data added-point identities are not unique"
            )
        if verification == "authority":
            allocation_name = (
                "POINT_ALLOCATION.version-"
                + str(version).zfill(COMMITTED_VERSION_NAME_WIDTH)
                + ".json"
            )
            allocation_sha = _safe_sha(
                payload.get("point_allocation_sha256"),
                "point_allocation_sha256",
            )
            if directory_manifest.get(allocation_name) != allocation_sha:
                raise ReferenceDataError(
                    "reference-data directory manifest does not bind its allocation snapshot"
                )
        else:
            _validate_allocation_snapshot(
                iteration_dir,
                payload,
                added,
                digest_file=digest_file,
            )
        quality_status = str(payload.get("quantum_quality_evidence_status") or "")
        quality_records = payload.get("quantum_quality_evidence", [])
        if not isinstance(quality_records, list):
            raise ReferenceDataError("reference-data quantum-quality evidence must be a list")
        if quality_status == "committed" and not quality_records:
            raise ReferenceDataError("committed quantum-quality evidence is empty")
        if quality_status != "committed":
            raise ReferenceDataError("reference-data quantum-quality evidence status is invalid")
        quality_committed_names = set()
        for record in quality_records:
            if not isinstance(record, Mapping):
                raise ReferenceDataError("quantum-quality evidence record must be an object")
            relative = Path(str(record.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ReferenceDataError("quantum-quality evidence path escapes its version")
            expected_evidence_sha = _safe_sha(
                record.get("sha256"), "quantum_quality_evidence.sha256"
            )
            if verification == "authority":
                if directory_manifest.get(relative.as_posix()) != expected_evidence_sha:
                    raise ReferenceDataError(
                        "reference-data directory manifest does not bind quantum-quality evidence"
                    )
                bindings = record.get("pointdir_bindings")
                if not isinstance(bindings, list) or not bindings:
                    raise ReferenceDataError(
                        "quantum-quality pointdir bindings are missing"
                    )
                for binding in bindings:
                    if not isinstance(binding, Mapping):
                        raise ReferenceDataError(
                            "quantum-quality pointdir binding must be an object"
                        )
                    committed_name = str(binding.get("committed_pointdir") or "")
                    candidate_id = str(binding.get("candidate_id") or "")
                    if (
                        (committed_name, candidate_id) not in added_by_identity
                        or committed_name in quality_committed_names
                    ):
                        raise ReferenceDataError(
                            "quantum-quality pointdir binding does not match committed metadata"
                        )
                    quality_committed_names.add(committed_name)
                continue
            evidence_path = iteration_dir / relative
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise ReferenceDataError("quantum-quality evidence file is missing")
            if _digest(
                evidence_path,
                payload=False,
                digest_file=digest_file,
            ) != expected_evidence_sha:
                raise ReferenceDataError("quantum-quality evidence SHA mismatch")
            try:
                from ..daemon.quantum_quality import read_quantum_quality_manifest

                quality_payload = read_quantum_quality_manifest(
                    evidence_path.parent,
                    expected_phase=str(record.get("phase") or ""),
                    expected_iteration=_safe_int(
                        record.get("iteration"),
                        "quantum_quality_evidence.iteration",
                    ),
                    manifest_path=evidence_path,
                )
            except Exception as exc:
                raise ReferenceDataError(
                    "committed quantum-quality evidence is invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            accepted_source_names = {
                str(item["pointdir"])
                for item in quality_payload["records"]
                if bool(item["accepted"])
            }
            bindings = record.get("pointdir_bindings")
            if not isinstance(bindings, list) or not bindings:
                raise ReferenceDataError(
                    "quantum-quality pointdir bindings are missing"
                )
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    raise ReferenceDataError(
                        "quantum-quality pointdir binding must be an object"
                    )
                source_name = str(binding.get("source_pointdir") or "")
                committed_name = str(binding.get("committed_pointdir") or "")
                candidate_id = str(binding.get("candidate_id") or "")
                if (
                    source_name not in accepted_source_names
                    or (committed_name, candidate_id) not in added_by_identity
                ):
                    raise ReferenceDataError(
                        "quantum-quality pointdir binding does not match its evidence"
                    )
                if committed_name in quality_committed_names:
                    raise ReferenceDataError(
                        "duplicate committed pointdir in quantum-quality evidence"
                    )
                quality_committed_names.add(committed_name)
        if quality_committed_names != {entry.pointdir_path.name for entry in added}:
            raise ReferenceDataError(
                "quantum-quality evidence does not match committed pointdirs"
            )
        if verification not in {"authority", "index"}:
            try:
                from ..daemon.quantum_acceptance_receipts import (
                    read_quantum_acceptance_receipt,
                )

                for entry in added:
                    receipt = read_quantum_acceptance_receipt(
                        campaign,
                        entry.pointdir_path,
                        expected_iteration=_safe_int(
                            payload.get("source_iteration"),
                            "source_iteration",
                            minimum=0,
                        ),
                        expected_candidate_id=entry.candidate_id,
                        expected_source_pointdir=entry.source_pointdir,
                        verification=verification,
                        validate_quality=False,
                        digest_file=digest_file,
                    )
                    receipt_path = (
                        entry.pointdir_path / "QUANTUM_ACCEPTANCE_RECEIPT.json"
                    )
                    if _digest(
                        receipt_path,
                        payload=False,
                        digest_file=digest_file,
                    ) != entry.acceptance_receipt_sha256:
                        raise ReferenceDataError(
                            "reference-data acceptance receipt SHA mismatch: "
                            + str(receipt_path)
                        )
                    if (
                        str(receipt.get("content_sha256") or "")
                        != entry.accepted_content_sha256
                    ):
                        raise ReferenceDataError(
                            "reference-data accepted-content digest mismatch: "
                            + str(entry.pointdir_path)
                        )
            except Exception as exc:
                raise ReferenceDataError(
                    "committed quantum acceptance evidence is invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
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
        if verification not in {"authority", "index"}:
            for entry in added:
                _validate_provenance(entry, digest_file=digest_file)
        expected_view_sha = _view_sha(entries)
        if _safe_sha(payload.get("cumulative_view_sha256"), "cumulative_view_sha256") != expected_view_sha:
            raise ReferenceDataError("reference-data cumulative view SHA mismatch")
        if verification == "deep":
            verify_manifest(
                iteration_dir,
                manifest=directory_manifest,
                digest_file=digest_file,
            )
        head_manifest_sha = _digest(
            manifest_path,
            payload=False,
            digest_file=digest_file,
        )
        previous_manifest_sha = head_manifest_sha

        view = ReferenceDataView(
            version=version,
            campaign_uid=str(campaign_uid),
            entries=tuple(entries),
            cumulative_view_sha256=_view_sha(entries),
            head_manifest_sha256=head_manifest_sha,
        )
        views.append(view)
        if resolved_views_out is not None:
            resolved_views_out.append(view)

    if not views:
        raise ReferenceDataError("reference-data chain could not be resolved")
    if expected_campaign_uid is not None and views[-1].campaign_uid != str(
        expected_campaign_uid
    ):
        raise ReferenceDataError("reference-data campaign UID does not match state")
    return tuple(views)


def resolve_reference_data_view(
    campaign_dir: Union[str, Path],
    reference_data_version: int,
    *,
    verification: str = "metadata",
    reference_data_root: Optional[Union[str, Path]] = None,
    expected_campaign_uid: Optional[str] = None,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
) -> ReferenceDataView:
    return resolve_reference_data_chain(
        campaign_dir,
        reference_data_version,
        verification=verification,
        reference_data_root=reference_data_root,
        expected_campaign_uid=expected_campaign_uid,
        digest_file=digest_file,
    )[-1]


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
        "quantum_quality_evidence_status": "committed",
        "cumulative_view_sha256": _view_sha(all_entries),
    }


__all__ = [
    "REFERENCE_DATA_VERSION_FILENAME",
    "REFERENCE_DATA_VERSION_SCHEMA_VERSION",
    "POINTDIR_NAME_WIDTH",
    "ReferenceDataError",
    "ReferenceDataEntry",
    "ReferenceDataView",
    "ReferenceDataVersioning",
    "resolve_reference_data_chain",
    "reference_data_version_path",
    "canonical_json_sha256",
    "hash_pointdir_tree",
    "resolve_reference_data_view",
    "build_reference_data_version_payload",
]
