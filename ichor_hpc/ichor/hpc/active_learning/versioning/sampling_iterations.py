"""Hash-chained manifests for bootstrap and completed sampling iterations."""

from __future__ import annotations

import concurrent.futures
import hashlib
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..daemon.state import atomic_write_json
from ..strict_json import strict_json as json
from ..handoff_manifests import (
    ariadne_landing_audit_path,
    ariadne_results_path,
    phase_b_selection_path,
    read_ariadne_landing_audit,
    read_ariadne_results_manifest,
    load_seeds_picked,
    read_phase_a_sample_manifest,
    read_phase_b_selection_manifest,
    seeds_picked_path,
)
from ..layout import (
    active_allocation_dir,
    active_iteration_dir,
    active_learning_dir,
    active_protocol_dir,
    active_seed_selection_dir,
    bootstrap_allocation_dir,
    bootstrap_dir,
    bootstrap_selection_dir,
)
from ..point_allocation import point_allocation_path, read_point_allocation
from .manifest import sha256_file
from .reference_data import ReferenceDataVersioning
from .trained_models import resolve_trained_model_set


BOOTSTRAP_MANIFEST_FILENAME = "BOOTSTRAP_MANIFEST.json"
ITERATION_MANIFEST_FILENAME = "ITERATION_MANIFEST.json"
SAMPLING_MANIFEST_SCHEMA_VERSION = 2
_LOCK_FILENAMES = frozenset({".provenance.lock", "POINT_ALLOCATION.lock"})
_BOOTSTRAP_TOP_LEVEL_DIRECTORIES = frozenset({"selection", "allocation"})
_ACTIVE_TOP_LEVEL_DIRECTORIES = frozenset({
    "protocol",
    "seed_selection",
    "ariadne",
    "phase_b",
    "allocation",
    "calibration",
})


class SamplingIterationError(RuntimeError):
    """Raised when a sampling iteration cannot be finalised or trusted."""


def bootstrap_manifest_path(campaign_dir: Path) -> Path:
    return bootstrap_dir(campaign_dir) / BOOTSTRAP_MANIFEST_FILENAME


def active_iteration_manifest_path(campaign_dir: Path, iteration: int) -> Path:
    return active_iteration_dir(campaign_dir, iteration) / ITERATION_MANIFEST_FILENAME


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _role(relative_path: str) -> str:
    if relative_path in {
        "selection/selected.xyz",
        "selection/selected_indices.dat",
        "seed_selection/seeds.xyz",
        "phase_b/selected.xyz",
        "phase_b/considered_candidates.xyz",
    }:
        return "derived_cache"
    return "authoritative"


def _remove_completed_lock_files(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.name not in _LOCK_FILENAMES:
            continue
        if path.is_symlink() or not path.is_file():
            raise SamplingIterationError(
                "completed iteration contains an invalid lock entry: " + str(path)
            )
        path.unlink()


@dataclass(frozen=True)
class _InventoryCapture:
    root: Path
    manifest_name: str
    files: list
    directories: list
    root_entries: Tuple[str, ...]
    fingerprints: Mapping[str, Tuple[int, int, int, int, int, int]]
    bytes_hashed: int

    def recheck(self) -> None:
        observed_entries = tuple(
            sorted(
                path.name
                for path in self.root.iterdir()
                if path.name != self.manifest_name
            )
        )
        if observed_entries != self.root_entries:
            raise SamplingIterationError(
                "sampling iteration root entries changed after inventory"
            )
        for relative, expected in self.fingerprints.items():
            path = self.root.joinpath(*PurePosixPath(relative).parts)
            try:
                observed = _lstat_fingerprint(path)
            except OSError as exc:
                raise SamplingIterationError(
                    "sampling iteration changed after inventory: " + relative
                ) from exc
            if observed != expected:
                raise SamplingIterationError(
                    "sampling iteration changed after inventory: " + relative
                )


def _lstat_fingerprint(path: Path) -> Tuple[int, int, int, int, int, int]:
    value = path.lstat()
    return (
        int(value.st_mode),
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _inventory_capture(
    root: Path,
    manifest_name: str,
    *,
    remove_completed_locks: bool = False,
    progress_callback: Optional[Callable[..., None]] = None,
) -> _InventoryCapture:
    if root.is_symlink() or not root.is_dir():
        raise SamplingIterationError("sampling root is not a regular directory: " + str(root))
    file_paths = []
    directories = []
    fingerprints: Dict[str, Tuple[int, int, int, int, int, int]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == manifest_name:
            continue
        try:
            fingerprint = _lstat_fingerprint(path)
        except OSError as exc:
            raise SamplingIterationError(
                "sampling iteration entry became unreadable: " + relative
            ) from exc
        mode = fingerprint[0]
        if stat.S_ISLNK(mode):
            raise SamplingIterationError("sampling iteration contains a symlink: " + relative)
        if remove_completed_locks and path.name in _LOCK_FILENAMES:
            if not stat.S_ISREG(mode):
                raise SamplingIterationError(
                    "completed iteration contains an invalid lock entry: "
                    + relative
                )
            path.unlink()
            continue
        if (
            path.name.startswith(".tmp-")
            or ".partial-" in path.name
            or path.name.endswith(".tmp")
        ):
            raise SamplingIterationError(
                "sampling iteration contains an incomplete artefact: " + relative
            )
        if stat.S_ISDIR(mode):
            directories.append(relative)
        elif stat.S_ISREG(mode):
            file_paths.append((relative, path, fingerprint))
        else:
            raise SamplingIterationError(
                "sampling iteration contains a special filesystem entry: " + relative
            )
    # Lock cleanup changes parent-directory timestamps. Capture directories
    # only after the complete cleanup/discovery traversal has finished.
    for relative in directories:
        directory = root.joinpath(*PurePosixPath(relative).parts)
        fingerprint = _lstat_fingerprint(directory)
        if not stat.S_ISDIR(fingerprint[0]):
            raise SamplingIterationError(
                "sampling iteration directory changed during discovery: "
                + relative
            )
        fingerprints[relative] = fingerprint
    root_entries = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.name != manifest_name
        )
    )
    total_bytes = sum(int(item[2][3]) for item in file_paths)
    completed_bytes = 0
    files_by_path: Dict[str, Dict[str, Any]] = {}

    def digest_one(item: Tuple[str, Path, Tuple[int, int, int, int, int, int]]):
        relative, path, before = item
        digest = sha256_file(path)
        after = _lstat_fingerprint(path)
        if after != before or not stat.S_ISREG(after[0]):
            raise SamplingIterationError(
                "sampling iteration file changed while hashing: " + relative
            )
        return relative, digest, before

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_by_relative = {
            executor.submit(digest_one, item): item[0] for item in file_paths
        }
        completed_files = 0
        for future in concurrent.futures.as_completed(future_by_relative):
            relative, digest, fingerprint = future.result()
            completed_files += 1
            completed_bytes += int(fingerprint[3])
            fingerprints[relative] = fingerprint
            files_by_path[relative] = {
                "path": relative,
                "size": int(fingerprint[3]),
                "sha256": digest,
                "role": _role(relative),
            }
            if progress_callback is not None:
                progress_callback(
                    completed=int(completed_files),
                    total=int(len(file_paths)),
                    bytes_completed=int(completed_bytes),
                    bytes_total=int(total_bytes),
                )
    files = [files_by_path[key] for key in sorted(files_by_path)]
    capture = _InventoryCapture(
        root=Path(root),
        manifest_name=str(manifest_name),
        files=files,
        directories=directories,
        root_entries=root_entries,
        fingerprints=fingerprints,
        bytes_hashed=int(total_bytes),
    )
    capture.recheck()
    return capture


def _inventory(root: Path, manifest_name: str) -> Tuple[list, list]:
    capture = _inventory_capture(root, manifest_name)
    return capture.files, capture.directories


def _required_paths(root: Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise SamplingIterationError("required sampling artefact is missing: " + relative)


def _validate_top_level(
    root: Path,
    *,
    manifest_name: str,
    allowed_directories: frozenset[str],
) -> None:
    for child in root.iterdir():
        if child.name == manifest_name:
            if child.is_symlink() or not child.is_file():
                raise SamplingIterationError(
                    "sampling manifest is not a regular file: " + str(child)
                )
            continue
        if child.name not in allowed_directories:
            raise SamplingIterationError(
                "unexpected sampling top-level artefact: " + child.name
            )
        if child.is_symlink() or not child.is_dir():
            raise SamplingIterationError(
                "sampling top-level entry is not a regular directory: "
                + child.name
            )


def _head_binding(campaign: Path, version: int) -> Dict[str, Any]:
    reference = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(
        int(version),
        verification="metadata",
    )
    models = resolve_trained_model_set(
        campaign,
        int(version),
        verification="metadata",
    )
    if reference.campaign_uid != models.campaign_uid:
        raise SamplingIterationError("reference-data/model campaign UID mismatch")
    return {
        "version": int(version),
        "reference_data": {
            "head_manifest_sha256": str(reference.head_manifest_sha256),
            "cumulative_view_sha256": str(reference.cumulative_view_sha256),
            "point_count": int(len(reference.entries)),
        },
        "trained_models": {
            "head_manifest_sha256": str(models.head_manifest_sha256),
            "model_set_sha256": str(models.model_set_sha256),
            "reference_data_view_sha256": str(models.reference_data_view_sha256),
        },
    }


def _manifest_payload(
    *,
    campaign_uid: str,
    kind: str,
    iteration: int,
    parent: Optional[Mapping[str, Any]],
    input_head: Optional[Mapping[str, Any]],
    output_head: Mapping[str, Any],
    files: list,
    directories: list,
) -> Dict[str, Any]:
    inventory = {"files": files, "directories": directories}
    return {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "kind": str(kind),
        "campaign_uid": str(campaign_uid),
        "iteration": int(iteration),
        "completed_at_iso": datetime.now(timezone.utc).isoformat(),
        "parent": None if parent is None else dict(parent),
        "input_head": None if input_head is None else dict(input_head),
        "output_head": dict(output_head),
        "inventory_sha256": _canonical_sha256(inventory),
        **inventory,
    }


def _verify_inventory(root: Path, payload: Mapping[str, Any], manifest_name: str) -> None:
    expected_files = payload.get("files")
    expected_directories = payload.get("directories")
    if not isinstance(expected_files, list) or not isinstance(expected_directories, list):
        raise SamplingIterationError("sampling manifest inventory is invalid")
    actual_files, actual_directories = _inventory(root, manifest_name)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise SamplingIterationError("sampling iteration exact inventory mismatch")
    if str(payload.get("inventory_sha256") or "") != _canonical_sha256({
        "files": actual_files,
        "directories": actual_directories,
    }):
        raise SamplingIterationError("sampling iteration inventory SHA mismatch")


def _authority_inventory(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Validate the declared inventory without inspecting its payload tree."""
    raw_files = payload.get("files")
    raw_directories = payload.get("directories")
    if not isinstance(raw_files, list) or not isinstance(raw_directories, list):
        raise SamplingIterationError("sampling manifest inventory is invalid")
    if str(payload.get("inventory_sha256") or "") != _canonical_sha256(
        {"files": raw_files, "directories": raw_directories}
    ):
        raise SamplingIterationError("sampling iteration inventory SHA mismatch")

    files: Dict[str, Dict[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "size",
            "sha256",
            "role",
        }:
            raise SamplingIterationError("sampling manifest file record is invalid")
        relative = str(raw.get("path") or "")
        path = PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative in files
        ):
            raise SamplingIterationError("sampling manifest file path is invalid")
        size = raw.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SamplingIterationError("sampling manifest file size is invalid")
        digest = str(raw.get("sha256") or "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise SamplingIterationError("sampling manifest file SHA-256 is invalid")
        role = str(raw.get("role") or "")
        if role not in {"authoritative", "derived_cache"}:
            raise SamplingIterationError("sampling manifest file role is invalid")
        files[relative] = dict(raw)

    directories = []
    for raw in raw_directories:
        if not isinstance(raw, str):
            raise SamplingIterationError("sampling manifest directory is invalid")
        path = PurePosixPath(raw)
        if (
            not raw
            or path.is_absolute()
            or "\\" in raw
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise SamplingIterationError("sampling manifest directory path is invalid")
        directories.append(raw)
    if len(directories) != len(set(directories)):
        raise SamplingIterationError("sampling manifest directories are not unique")
    return files


def sampling_manifest_file_binding(
    payload: Mapping[str, Any],
    relative_path: str,
) -> Dict[str, Any]:
    """Return one validated file binding from a sampling authority manifest."""
    files = _authority_inventory(payload)
    binding = files.get(str(relative_path))
    if binding is None:
        raise SamplingIterationError(
            "sampling manifest does not bind required artefact: "
            + str(relative_path)
        )
    return dict(binding)


def _authority_head_binding(reference: Any, models: Any) -> Dict[str, Any]:
    if int(reference.version) != int(models.version):
        raise SamplingIterationError("reference-data/model version mismatch")
    if str(reference.campaign_uid) != str(models.campaign_uid):
        raise SamplingIterationError("reference-data/model campaign UID mismatch")
    if int(models.reference_data_version) != int(reference.version):
        raise SamplingIterationError("model/reference-data version binding mismatch")
    if str(models.reference_data_view_sha256) != str(
        reference.cumulative_view_sha256
    ):
        raise SamplingIterationError("model/reference-data view binding mismatch")
    return {
        "version": int(reference.version),
        "reference_data": {
            "head_manifest_sha256": str(reference.head_manifest_sha256),
            "cumulative_view_sha256": str(reference.cumulative_view_sha256),
            "point_count": int(len(reference.entries)),
        },
        "trained_models": {
            "head_manifest_sha256": str(models.head_manifest_sha256),
            "model_set_sha256": str(models.model_set_sha256),
            "reference_data_view_sha256": str(
                models.reference_data_view_sha256
            ),
        },
    }


def resolve_sampling_iteration_authority_chain(
    campaign_dir: Path,
    through_iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
    reference_views: Optional[Sequence[Any]] = None,
    model_sets: Optional[Sequence[Any]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """Validate finalisation authority linearly without walking iteration trees."""
    campaign = Path(campaign_dir)
    target = int(through_iteration)
    if target < 0:
        raise SamplingIterationError("sampling authority target must be >= 0")
    if reference_views is None:
        from .reference_data import resolve_reference_data_chain

        reference_views = resolve_reference_data_chain(
            campaign,
            target,
            verification="authority",
        )
    if model_sets is None:
        from .trained_models import resolve_trained_model_chain

        model_sets = resolve_trained_model_chain(
            campaign,
            target,
            verification="authority",
            reference_views=reference_views,
        )
    references = {int(view.version): view for view in reference_views}
    models = {int(model.version): model for model in model_sets}
    expected_versions = set(range(target + 1))
    if expected_versions - set(references) or expected_versions - set(models):
        raise SamplingIterationError("sampling authority heads are incomplete")

    bootstrap_path = bootstrap_manifest_path(campaign)
    bootstrap = _read_manifest(bootstrap_path, kind="bootstrap", iteration=0)
    campaign_uid = str(bootstrap.get("campaign_uid") or "")
    if expected_campaign_uid is not None and campaign_uid != str(
        expected_campaign_uid
    ):
        raise SamplingIterationError("bootstrap campaign UID mismatch")
    _authority_inventory(bootstrap)
    if bootstrap.get("parent") is not None or bootstrap.get("input_head") is not None:
        raise SamplingIterationError("bootstrap sampling authority has a parent")
    if bootstrap.get("output_head") != _authority_head_binding(
        references[0], models[0]
    ):
        raise SamplingIterationError("bootstrap output-head binding mismatch")

    resolved = [bootstrap]
    previous_path = bootstrap_path
    for iteration in range(1, target + 1):
        path = active_iteration_manifest_path(campaign, iteration)
        payload = _read_manifest(
            path,
            kind="active_iteration",
            iteration=iteration,
        )
        if str(payload.get("campaign_uid") or "") != campaign_uid:
            raise SamplingIterationError("active iteration campaign UID mismatch")
        _authority_inventory(payload)
        if payload.get("input_head") != _authority_head_binding(
            references[iteration - 1], models[iteration - 1]
        ):
            raise SamplingIterationError("active iteration input-head binding mismatch")
        if payload.get("output_head") != _authority_head_binding(
            references[iteration], models[iteration]
        ):
            raise SamplingIterationError("active iteration output-head binding mismatch")
        parent = payload.get("parent")
        expected_parent = previous_path.resolve().relative_to(
            campaign.resolve()
        ).as_posix()
        if not isinstance(parent, Mapping) or parent != {
            "path": expected_parent,
            "sha256": sha256_file(previous_path),
        }:
            raise SamplingIterationError("active iteration parent binding mismatch")
        resolved.append(payload)
        previous_path = path
    return tuple(resolved)


def _read_manifest(path: Path, *, kind: str, iteration: int) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SamplingIterationError(
            "sampling manifest is not a regular file: " + str(path)
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SamplingIterationError("sampling manifest is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise SamplingIterationError("sampling manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    observed_iteration = payload.get("iteration")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or isinstance(observed_iteration, bool)
        or not isinstance(observed_iteration, int)
    ):
        raise SamplingIterationError(
            "sampling manifest schema and iteration must be exact JSON integers"
        )
    if schema_version != SAMPLING_MANIFEST_SCHEMA_VERSION:
        raise SamplingIterationError("unsupported sampling manifest schema")
    if str(payload.get("kind") or "") != str(kind):
        raise SamplingIterationError("sampling manifest kind mismatch")
    if observed_iteration != int(iteration):
        raise SamplingIterationError("sampling manifest iteration mismatch")
    campaign_uid = payload.get("campaign_uid")
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise SamplingIterationError("sampling manifest campaign_uid is invalid")
    completed_at = payload.get("completed_at_iso")
    if not isinstance(completed_at, str) or not completed_at:
        raise SamplingIterationError("sampling manifest completion time is invalid")
    try:
        parsed_completed_at = datetime.fromisoformat(completed_at)
    except ValueError as exc:
        raise SamplingIterationError(
            "sampling manifest completion time is not ISO-8601"
        ) from exc
    if parsed_completed_at.tzinfo is None:
        raise SamplingIterationError("sampling manifest completion time lacks a timezone")
    return payload


def finalise_bootstrap(campaign_dir: Path, campaign_uid: str) -> Path:
    campaign = Path(campaign_dir)
    root = bootstrap_dir(campaign)
    path = bootstrap_manifest_path(campaign)
    if path.is_file():
        verify_bootstrap(campaign, expected_campaign_uid=campaign_uid)
        return path
    read_phase_a_sample_manifest(
        bootstrap_selection_dir(campaign),
        expected_campaign_uid=campaign_uid,
    )
    allocation = read_point_allocation(
        point_allocation_path(campaign, context="bootstrap", iteration=0),
        expected_campaign_uid=campaign_uid,
        expected_context="bootstrap",
        expected_iteration=0,
    )
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise SamplingIterationError("bootstrap allocation is incomplete")
    output_head = _head_binding(campaign, 0)
    if str(output_head["reference_data"].get("head_manifest_sha256") or "") == "":
        raise SamplingIterationError("bootstrap reference-data head is empty")
    _remove_completed_lock_files(root)
    _required_paths(root, (
        "selection/SELECTION.json",
        "selection/selected.xyz",
        "selection/selected_indices.dat",
        "allocation/POINT_ALLOCATION.json",
    ))
    _validate_top_level(
        root,
        manifest_name=BOOTSTRAP_MANIFEST_FILENAME,
        allowed_directories=_BOOTSTRAP_TOP_LEVEL_DIRECTORIES,
    )
    files, directories = _inventory(root, BOOTSTRAP_MANIFEST_FILENAME)
    payload = _manifest_payload(
        campaign_uid=campaign_uid,
        kind="bootstrap",
        iteration=0,
        parent=None,
        input_head=None,
        output_head=output_head,
        files=files,
        directories=directories,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    verify_bootstrap(campaign, expected_campaign_uid=campaign_uid)
    return path


def verify_bootstrap(
    campaign_dir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    root = bootstrap_dir(campaign)
    path = bootstrap_manifest_path(campaign)
    payload = _read_manifest(path, kind="bootstrap", iteration=0)
    if expected_campaign_uid is not None and str(payload.get("campaign_uid")) != str(
        expected_campaign_uid
    ):
        raise SamplingIterationError("bootstrap campaign UID mismatch")
    output_head = _head_binding(campaign, 0)
    if payload.get("output_head") != output_head:
        raise SamplingIterationError("bootstrap output-head binding mismatch")
    _verify_inventory(root, payload, BOOTSTRAP_MANIFEST_FILENAME)
    _validate_top_level(
        root,
        manifest_name=BOOTSTRAP_MANIFEST_FILENAME,
        allowed_directories=_BOOTSTRAP_TOP_LEVEL_DIRECTORIES,
    )
    return payload


_ACTIVE_FINALISATION_RECEIPT_PHASES = (
    "ARIADNE_ARRAY",
    "PHASE_B_DIVERSITY",
    "SPLIT",
    "ALLOCATION_CHECK",
    "REFERENCE_COMMIT",
    "FEREBUS",
)


def _report_finalisation_progress(
    callback: Optional[Callable[..., None]],
    stage: str,
    **fields: Any,
) -> None:
    if callback is not None:
        callback(str(stage), **fields)


def _snapshot_supports_iteration(snapshot: Any, iteration: int) -> bool:
    if snapshot is None:
        return False
    value = int(iteration)
    expected = tuple(range(value + 1))
    try:
        return (
            tuple(int(item.version) for item in snapshot.reference_views)
            == expected
            and tuple(int(item.version) for item in snapshot.model_sets)
            == expected
            and not snapshot.reference_errors
            and not snapshot.model_errors
            and not snapshot.completion_receipt_errors
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _snapshot_iteration_context(
    campaign: Path,
    iteration: int,
    campaign_uid: str,
    snapshot: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Dict[str, Any]]:
    from ..daemon.completion_receipts import validate_completion_reference

    value = int(iteration)
    snapshot.assert_anchors_unchanged(campaign)
    if str(campaign_uid) not in set(str(uid) for uid in snapshot.campaign_uids):
        raise SamplingIterationError("sampling snapshot campaign UID mismatch")
    input_head = _authority_head_binding(
        snapshot.reference_view(value - 1),
        snapshot.model_set(value - 1),
    )
    output_head = _authority_head_binding(
        snapshot.reference_view(value),
        snapshot.model_set(value),
    )
    resolve_sampling_iteration_authority_chain(
        campaign,
        value - 1,
        expected_campaign_uid=campaign_uid,
        reference_views=tuple(snapshot.reference_views[:value]),
        model_sets=tuple(snapshot.model_sets[:value]),
    )
    parent_path = (
        bootstrap_manifest_path(campaign)
        if value == 1
        else active_iteration_manifest_path(campaign, value - 1)
    )
    parent = {
        "path": parent_path.resolve().relative_to(campaign.resolve()).as_posix(),
        "sha256": sha256_file(parent_path),
    }

    receipts_by_phase: Dict[str, list] = {
        phase: [] for phase in _ACTIVE_FINALISATION_RECEIPT_PHASES
    }
    for record in snapshot.completion_receipts:
        payload = record.get("payload") if isinstance(record, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or str(payload.get("campaign_uid") or "") != str(campaign_uid)
            or int(payload.get("iteration", -1)) != value
        ):
            continue
        phase = str(payload.get("phase") or "")
        if phase in receipts_by_phase:
            receipts_by_phase[phase].append(record)
    for phase, records in receipts_by_phase.items():
        if not records:
            raise SamplingIterationError(
                "sampling snapshot lacks " + phase + " completion authority"
            )
        validated = []
        for record in records:
            reference = record.get("reference")
            if not isinstance(reference, Mapping):
                raise SamplingIterationError(
                    "sampling completion receipt reference is invalid for " + phase
                )
            validated.append(
                validate_completion_reference(
                    campaign,
                    reference,
                    expected_campaign_uid=campaign_uid,
                )
            )
        if len({str(item.get("config_sha256") or "") for item in validated}) != 1:
            raise SamplingIterationError(
                "sampling completion receipts conflict for " + phase
            )
    return input_head, output_head, parent_path, parent


def _validate_capture_against_payload(
    capture: _InventoryCapture,
    payload: Mapping[str, Any],
) -> None:
    if payload.get("files") != capture.files or payload.get(
        "directories"
    ) != capture.directories:
        raise SamplingIterationError("sampling iteration exact inventory mismatch")
    if str(payload.get("inventory_sha256") or "") != _canonical_sha256(
        {"files": capture.files, "directories": capture.directories}
    ):
        raise SamplingIterationError("sampling iteration inventory SHA mismatch")


def _finalise_active_iteration_from_snapshot(
    campaign: Path,
    iteration: int,
    campaign_uid: str,
    snapshot: Any,
    *,
    progress_callback: Optional[Callable[..., None]],
) -> Path:
    value = int(iteration)
    root = active_iteration_dir(campaign, value)
    path = active_iteration_manifest_path(campaign, value)
    _report_finalisation_progress(progress_callback, "iteration_authority")
    input_head, output_head, unused_parent_path, parent = (
        _snapshot_iteration_context(
            campaign,
            value,
            campaign_uid,
            snapshot,
        )
    )
    del unused_parent_path
    allocation = read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=value),
        expected_campaign_uid=campaign_uid,
        expected_context="active",
        expected_iteration=value,
    )
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise SamplingIterationError("active point allocation is incomplete")
    _required_paths(root, (
        "protocol/SAMPLING_PROTOCOL_RESOLVED.json",
        "protocol/SAMPLING_PROTOCOL_AUDIT.json",
        "protocol/SAMPLING_SCALE_MODEL.json",
        "seed_selection/SELECTION.json",
        "seed_selection/seeds.xyz",
        "ariadne/TASK_MAP.json",
        "ariadne/RESULTS.json",
        "ariadne/AUDIT.json",
        "phase_b/SELECTION.json",
        "phase_b/selected.xyz",
        "allocation/POINT_ALLOCATION.json",
        "allocation/SPLIT_RECEIPT.json",
    ))
    _validate_top_level(
        root,
        manifest_name=ITERATION_MANIFEST_FILENAME,
        allowed_directories=_ACTIVE_TOP_LEVEL_DIRECTORIES,
    )
    _report_finalisation_progress(progress_callback, "iteration_inventory")

    def inventory_progress(**fields: Any) -> None:
        _report_finalisation_progress(
            progress_callback,
            "iteration_inventory",
            **fields,
        )

    capture = _inventory_capture(
        root,
        ITERATION_MANIFEST_FILENAME,
        remove_completed_locks=True,
        progress_callback=inventory_progress,
    )
    snapshot.assert_anchors_unchanged(campaign)
    if path.is_file():
        payload = _read_manifest(
            path,
            kind="active_iteration",
            iteration=value,
        )
        if str(payload.get("campaign_uid") or "") != str(campaign_uid):
            raise SamplingIterationError("active iteration campaign UID mismatch")
        if payload.get("input_head") != input_head or payload.get(
            "output_head"
        ) != output_head or payload.get("parent") != parent:
            raise SamplingIterationError(
                "active iteration authority binding mismatch"
            )
        _validate_capture_against_payload(capture, payload)
        capture.recheck()
        snapshot.assert_anchors_unchanged(campaign)
        return path

    payload = _manifest_payload(
        campaign_uid=campaign_uid,
        kind="active_iteration",
        iteration=value,
        parent=parent,
        input_head=input_head,
        output_head=output_head,
        files=capture.files,
        directories=capture.directories,
    )
    _report_finalisation_progress(
        progress_callback,
        "iteration_manifest_publication",
        completed=0,
        total=1,
        unit="manifest",
    )
    capture.recheck()
    snapshot.assert_anchors_unchanged(campaign)
    atomic_write_json(path, payload)
    published = _read_manifest(
        path,
        kind="active_iteration",
        iteration=value,
    )
    if published != payload:
        raise SamplingIterationError("published sampling manifest changed")
    _validate_capture_against_payload(capture, published)
    capture.recheck()
    snapshot.assert_anchors_unchanged(campaign)
    _report_finalisation_progress(
        progress_callback,
        "iteration_manifest_publication",
        completed=1,
        total=1,
        unit="manifest",
    )
    return path


def finalise_active_iteration(
    campaign_dir: Path,
    iteration: int,
    campaign_uid: str,
    *,
    artifact_snapshot: Any = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Path:
    campaign = Path(campaign_dir)
    value = int(iteration)
    if value < 1:
        raise SamplingIterationError("active iteration must be >= 1")
    root = active_iteration_dir(campaign, value)
    path = active_iteration_manifest_path(campaign, value)
    if _snapshot_supports_iteration(artifact_snapshot, value):
        return _finalise_active_iteration_from_snapshot(
            campaign,
            value,
            campaign_uid,
            artifact_snapshot,
            progress_callback=progress_callback,
        )
    if path.is_file():
        verify_active_iteration(campaign, value, expected_campaign_uid=campaign_uid)
        return path

    read_ariadne_results_manifest(root, expected_iteration=value)
    read_ariadne_landing_audit(root, expected_iteration=value)
    load_seeds_picked(root, expected_iteration=value)
    read_phase_b_selection_manifest(
        root,
        expected_iteration=value,
        expected_campaign_uid=campaign_uid,
    )
    allocation = read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=value),
        expected_campaign_uid=campaign_uid,
        expected_context="active",
        expected_iteration=value,
    )
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise SamplingIterationError("active point allocation is incomplete")
    input_head = _head_binding(campaign, value - 1)
    output_head = _head_binding(campaign, value)
    if str(input_head["trained_models"]["reference_data_view_sha256"]) != str(
        input_head["reference_data"]["cumulative_view_sha256"]
    ):
        raise SamplingIterationError("input model/reference-data view mismatch")
    if str(output_head["trained_models"]["reference_data_view_sha256"]) != str(
        output_head["reference_data"]["cumulative_view_sha256"]
    ):
        raise SamplingIterationError("output model/reference-data view mismatch")

    if value == 1:
        parent_path = bootstrap_manifest_path(campaign)
        verify_bootstrap(campaign, expected_campaign_uid=campaign_uid)
    else:
        parent_path = active_iteration_manifest_path(campaign, value - 1)
        verify_active_iteration(
            campaign,
            value - 1,
            expected_campaign_uid=campaign_uid,
        )
    parent = {
        "path": parent_path.resolve().relative_to(campaign.resolve()).as_posix(),
        "sha256": sha256_file(parent_path),
    }
    _remove_completed_lock_files(root)
    _required_paths(root, (
        "protocol/SAMPLING_PROTOCOL_RESOLVED.json",
        "protocol/SAMPLING_PROTOCOL_AUDIT.json",
        "protocol/SAMPLING_SCALE_MODEL.json",
        "seed_selection/SELECTION.json",
        "seed_selection/seeds.xyz",
        "ariadne/TASK_MAP.json",
        "ariadne/RESULTS.json",
        "ariadne/AUDIT.json",
        "phase_b/SELECTION.json",
        "phase_b/selected.xyz",
        "allocation/POINT_ALLOCATION.json",
        "allocation/SPLIT_RECEIPT.json",
    ))
    _validate_top_level(
        root,
        manifest_name=ITERATION_MANIFEST_FILENAME,
        allowed_directories=_ACTIVE_TOP_LEVEL_DIRECTORIES,
    )
    files, directories = _inventory(root, ITERATION_MANIFEST_FILENAME)
    payload = _manifest_payload(
        campaign_uid=campaign_uid,
        kind="active_iteration",
        iteration=value,
        parent=parent,
        input_head=input_head,
        output_head=output_head,
        files=files,
        directories=directories,
    )
    atomic_write_json(path, payload)
    verify_active_iteration(campaign, value, expected_campaign_uid=campaign_uid)
    return path


def verify_active_iteration(
    campaign_dir: Path,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    value = int(iteration)
    root = active_iteration_dir(campaign, value)
    path = active_iteration_manifest_path(campaign, value)
    payload = _read_manifest(path, kind="active_iteration", iteration=value)
    if expected_campaign_uid is not None and str(payload.get("campaign_uid")) != str(
        expected_campaign_uid
    ):
        raise SamplingIterationError("active iteration campaign UID mismatch")
    if payload.get("input_head") != _head_binding(campaign, value - 1):
        raise SamplingIterationError("active iteration input-head binding mismatch")
    if payload.get("output_head") != _head_binding(campaign, value):
        raise SamplingIterationError("active iteration output-head binding mismatch")
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise SamplingIterationError("active iteration parent binding is missing")
    expected_parent = (
        bootstrap_manifest_path(campaign)
        if value == 1
        else active_iteration_manifest_path(campaign, value - 1)
    )
    expected_parent_relative = expected_parent.resolve().relative_to(
        campaign.resolve()
    ).as_posix()
    if str(parent.get("path") or "") != expected_parent_relative:
        raise SamplingIterationError("active iteration parent path mismatch")
    if str(parent.get("sha256") or "") != sha256_file(expected_parent):
        raise SamplingIterationError("active iteration parent SHA mismatch")
    _verify_inventory(root, payload, ITERATION_MANIFEST_FILENAME)
    _validate_top_level(
        root,
        manifest_name=ITERATION_MANIFEST_FILENAME,
        allowed_directories=_ACTIVE_TOP_LEVEL_DIRECTORIES,
    )
    return payload


def verify_sampling_chain(
    campaign_dir: Path,
    through_iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    campaign = Path(campaign_dir)
    verify_bootstrap(campaign, expected_campaign_uid=expected_campaign_uid)
    for iteration in range(1, int(through_iteration) + 1):
        verify_active_iteration(
            campaign,
            iteration,
            expected_campaign_uid=expected_campaign_uid,
        )


__all__ = [
    "BOOTSTRAP_MANIFEST_FILENAME",
    "ITERATION_MANIFEST_FILENAME",
    "SamplingIterationError",
    "bootstrap_manifest_path",
    "active_iteration_manifest_path",
    "finalise_bootstrap",
    "verify_bootstrap",
    "finalise_active_iteration",
    "resolve_sampling_iteration_authority_chain",
    "sampling_manifest_file_binding",
    "verify_active_iteration",
    "verify_sampling_chain",
]
