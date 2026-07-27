"""Read-only export of accepted active-iteration model-batch geometries."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import portalocker

from .ariadne_outputs import (
    SEED_OUTPUT_MANIFEST_FILENAME,
    SEED_RESULT_FILENAME,
    validate_seed_output,
)
from .daemon.ferebus_split_ledger import (
    ledger_path as ferebus_split_ledger_path,
    read_split_assignments,
    validate_split_assignments_payload,
)
from .daemon.filesystem import lexical_absolute_path, reject_symlink_components
from .daemon.input_staging import (
    FEREBUS_SPLIT_SNAPSHOT,
    read_ferebus_manifest,
)
from .daemon.quantum_acceptance_receipts import (
    QUANTUM_ACCEPTANCE_RECEIPT,
    read_quantum_acceptance_receipt,
)
from .daemon.reference_commit import classify_reference_commit
from .daemon.state import CampaignPhase, DEFAULT_STATE_FILENAME, read_state
from .handoff_manifests import (
    ARIADNE_RESULTS_SCHEMA_VERSION,
    PHASE_B_SELECTION_SCHEMA_VERSION,
    ariadne_results_path,
    load_seeds_picked,
    read_ariadne_batch_decision,
    resolve_handoff_path,
    validate_ariadne_result,
)
from .layout import (
    active_iteration_dir,
    ariadne_seed_dir,
    qm_reference_data_dir,
    trained_models_dir,
)
from .point_allocation import (
    accepted_attempts,
    read_point_allocation,
    stable_candidate_id,
)
from .seed_identity import read_ariadne_task_map
from .strict_json import strict_json as json
from .versioning.manifest import sha256_file
from .versioning.provenance import PROVENANCE_FILENAME, validate_provenance
from .versioning.reference_data import (
    ReferenceDataEntry,
    ReferenceDataVersioning,
    resolve_reference_data_chain,
)
from .versioning.sampling_iterations import (
    active_iteration_manifest_path,
    resolve_sampling_iteration_authority_chain,
    sampling_manifest_file_binding,
)
from .versioning.trained_models import (
    TrainedModelVersioning,
    resolve_trained_model_chain,
)


EXPORT_ROOT_NAME = "EXPORTED_GEOMETRIES"
FINAL_COORDINATE_TOLERANCE_ANGSTROM = 1.0e-6
_SPLIT_ORDER = {"train": 0, "int_val": 1}
_SHA256_CHARS = frozenset("0123456789abcdef")


class BatchGeometryExportError(RuntimeError):
    """Raised when an iteration cannot be exported without ambiguity."""


@dataclass(frozen=True)
class Geometry:
    atom_types: Tuple[str, ...]
    coordinates: np.ndarray


@dataclass(frozen=True)
class ExportRecord:
    iteration: int
    export_order: int
    split: str
    slot_id: int
    seed_id: int
    seed_uid: str
    candidate_id: str
    replacement_round: int
    pointdir_name: str
    seed_geometry: Geometry
    final_geometry: Geometry
    coordinate_discrepancy_angstrom: float

    @property
    def filename(self) -> str:
        return (
            str(self.export_order).zfill(6)
            + "_"
            + self.split
            + "_slot-"
            + str(self.slot_id).zfill(6)
            + "_seed-"
            + str(self.seed_id).zfill(6)
            + ".xyz"
        )


@dataclass(frozen=True)
class BatchGeometryExportSummary:
    output_path: Path
    iterations: Tuple[int, ...]
    training_count: int
    internal_validation_count: int
    maximum_coordinate_discrepancy_angstrom: float

    @property
    def total_count(self) -> int:
        return int(self.training_count + self.internal_validation_count)


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BatchGeometryExportError(label + " must be an exact JSON integer")
    parsed = int(value)
    if parsed < minimum:
        raise BatchGeometryExportError(label + " must be >= " + str(minimum))
    return parsed


def _sha256(value: Any, label: str) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) != 64 or any(character not in _SHA256_CHARS for character in text):
        raise BatchGeometryExportError(label + " must be a lowercase SHA-256")
    return text


def _safe_relative_path(value: Any, label: str) -> str:
    text = value if isinstance(value, str) else ""
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BatchGeometryExportError(label + " is not a safe relative path")
    return text


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BatchGeometryExportError(label + " is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), source=path)
    except (OSError, ValueError) as exc:
        raise BatchGeometryExportError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise BatchGeometryExportError(label + " must contain a JSON object")
    return payload


def _verify_bound_file(
    iteration_dir: Path,
    sampling_manifest: Mapping[str, Any],
    relative_path: str,
) -> Path:
    binding = sampling_manifest_file_binding(sampling_manifest, relative_path)
    path = iteration_dir / PurePosixPath(relative_path)
    if path.is_symlink() or not path.is_file():
        raise BatchGeometryExportError(
            "completed iteration artefact is missing: " + relative_path
        )
    if int(path.stat().st_size) != int(binding["size"]):
        raise BatchGeometryExportError(
            "completed iteration artefact size changed: " + relative_path
        )
    if sha256_file(path) != str(binding["sha256"]):
        raise BatchGeometryExportError(
            "completed iteration artefact hash changed: " + relative_path
        )
    return path


def _completed_iteration_ceiling(state: Any) -> int:
    if state.phase is CampaignPhase.DONE:
        return int(state.iteration)
    return max(int(state.iteration) - 1, 0)


def _requested_iterations(state: Any, requested: Union[int, str]) -> Tuple[int, ...]:
    ceiling = _completed_iteration_ceiling(state)
    if requested == "all":
        if ceiling < 1:
            raise BatchGeometryExportError(
                "no completed active-learning iterations are available to export"
            )
        return tuple(range(1, ceiling + 1))
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise BatchGeometryExportError("iteration must be a positive integer or 'all'")
    iteration = int(requested)
    if iteration == 0:
        raise BatchGeometryExportError(
            "bootstrap iteration 0 cannot be exported because it has no ARIADNE seeds"
        )
    if iteration < 0:
        raise BatchGeometryExportError("iteration must be a positive integer or 'all'")
    if iteration > ceiling:
        raise BatchGeometryExportError(
            "iteration "
            + str(iteration)
            + " is not finalised; the latest exportable iteration is "
            + str(ceiling)
        )
    return (iteration,)


def _sampling_control_hashes(
    campaign: Path,
    iterations: Sequence[int],
) -> Dict[Path, str]:
    paths = [active_iteration_manifest_path(campaign, iteration) for iteration in iterations]
    paths.extend(
        (
            qm_reference_data_dir(campaign)
            / ("iteration-" + str(iteration).zfill(6))
            / "REFERENCE_DATA_VERSION.json"
        )
        for iteration in iterations
    )
    paths.extend(
        trained_models_dir(campaign)
        / ("iteration-" + str(iteration).zfill(6))
        / "FEREBUS_TASK_ARTEFACTS.json"
        for iteration in iterations
    )
    paths.append(ferebus_split_ledger_path(campaign))
    return {path: sha256_file(path) for path in paths}


def _recheck_control_hashes(expected: Mapping[Path, str]) -> None:
    for path, digest in expected.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise BatchGeometryExportError(
                "campaign authority changed while geometries were being exported; retry the command"
            )


def _validate_model_split_snapshot(
    campaign: Path,
    *,
    iteration: int,
    model_set: Any,
    reference_view: Any,
    allocation: Mapping[str, Any],
    allocation_sha256: str,
    global_ledger: Mapping[str, Any],
) -> None:
    task_manifest = read_ferebus_manifest(
        model_set.root,
        verify_dataset_files=False,
    )
    if (
        str(task_manifest.get("campaign_uid") or "") != str(model_set.campaign_uid)
        or _exact_int(
            task_manifest.get("reference_data_version"),
            "FEREBUS reference-data version",
        )
        != iteration
        or str(task_manifest.get("reference_data_head_manifest_sha256") or "")
        != str(reference_view.head_manifest_sha256)
        or str(task_manifest.get("reference_data_view_sha256") or "")
        != str(reference_view.cumulative_view_sha256)
    ):
        raise BatchGeometryExportError(
            "trained model is not bound to the requested reference-data version"
        )
    row_order = [entry.pointdir_name for entry in reference_view.entries]
    if list(task_manifest.get("pointdir_row_order") or []) != row_order:
        raise BatchGeometryExportError("FEREBUS row order does not match QM reference data")

    binding = task_manifest.get("split_ledger")
    if not isinstance(binding, Mapping) or str(binding.get("path") or "") != FEREBUS_SPLIT_SNAPSHOT:
        raise BatchGeometryExportError("trained model lacks its FEREBUS split snapshot")
    snapshot_path = model_set.root / FEREBUS_SPLIT_SNAPSHOT
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise BatchGeometryExportError("trained-model split snapshot is missing")
    if (
        _exact_int(binding.get("size"), "FEREBUS split snapshot size")
        != int(snapshot_path.stat().st_size)
        or _sha256(binding.get("sha256"), "FEREBUS split snapshot hash")
        != sha256_file(snapshot_path)
    ):
        raise BatchGeometryExportError("trained-model split snapshot binding mismatch")
    snapshot = validate_split_assignments_payload(
        _read_json_object(snapshot_path, "trained-model split snapshot")
    )
    expected_version_keys = {str(value) for value in range(iteration + 1)}
    if set(snapshot["version_allocations"]) != expected_version_keys:
        raise BatchGeometryExportError(
            "trained-model split snapshot does not end at the requested iteration"
        )
    if set(snapshot["assignments"]) != set(row_order):
        raise BatchGeometryExportError("trained-model split snapshot coverage mismatch")
    global_assignments = global_ledger.get("assignments")
    global_versions = global_ledger.get("version_allocations")
    if not isinstance(global_assignments, Mapping) or not isinstance(global_versions, Mapping):
        raise BatchGeometryExportError("FEREBUS split ledger is incomplete")
    for name in row_order:
        if snapshot["assignments"].get(name) != global_assignments.get(name):
            raise BatchGeometryExportError(
                "trained-model and campaign split assignments differ for " + name
            )
    version_record = snapshot["version_allocations"][str(iteration)]
    if version_record != global_versions.get(str(iteration)):
        raise BatchGeometryExportError(
            "trained-model and campaign split histories differ for iteration "
            + str(iteration)
        )
    added = [
        entry
        for entry in reference_view.entries
        if int(entry.introduced_in_version) == iteration
    ]
    if list(version_record["pointdirs"]) != [entry.pointdir_name for entry in added]:
        raise BatchGeometryExportError("FEREBUS iteration row order is invalid")
    if str(version_record["allocation_manifest_sha256"]) != allocation_sha256:
        raise BatchGeometryExportError("FEREBUS split allocation hash mismatch")
    expected_splits = {
        entry.pointdir_name: entry.split for entry in reference_view.entries
    }
    for entry in added:
        assignment = snapshot["assignments"][entry.pointdir_name]
        if (
            str(assignment["split"]) != str(entry.split)
            or int(assignment["first_seen_reference_data_version"]) != iteration
            or str(assignment["allocation_manifest_sha256"]) != allocation_sha256
            or str(assignment["provenance_sha256"]) != str(entry.provenance_sha256)
        ):
            raise BatchGeometryExportError(
                "FEREBUS split identity mismatch for " + entry.pointdir_name
            )
    expected_allocation_path = (
        active_iteration_dir(campaign, iteration)
        / "allocation"
        / "POINT_ALLOCATION.json"
    ).resolve().relative_to(campaign.resolve()).as_posix()
    if (
        str(binding.get("source_path") or "")
        != ferebus_split_ledger_path(campaign).resolve().relative_to(
            campaign.resolve()
        ).as_posix()
        or str(binding.get("allocation_manifest") or "") != expected_allocation_path
        or str(binding.get("allocation_manifest_sha256") or "") != allocation_sha256
        or _exact_int(binding.get("allocation_iteration"), "FEREBUS allocation iteration")
        != iteration
        or str(binding.get("slot_assignment_sha256") or "")
        != str(allocation.get("slot_assignment_sha256") or "")
        or dict(binding.get("version_allocation") or {}) != dict(version_record)
        or dict(binding.get("forced_splits") or {}) != expected_splits
    ):
        raise BatchGeometryExportError("FEREBUS task split binding is inconsistent")


def _validate_reference_commit(
    campaign: Path,
    *,
    iteration: int,
    reference_entries: Sequence[ReferenceDataEntry],
    allocation_sha256: str,
) -> None:
    classification = classify_reference_commit(
        campaign,
        context="active",
        iteration=iteration,
        verification="authority",
    )
    if str(classification.get("state") or "") != "complete":
        raise BatchGeometryExportError(
            "reference commit for iteration " + str(iteration) + " is not complete"
        )
    ledger = classification.get("ledger")
    if not isinstance(ledger, Mapping):
        raise BatchGeometryExportError("reference-commit ledger is missing")
    if (
        str(ledger.get("context") or "") != "active"
        or _exact_int(ledger.get("iteration"), "reference-commit iteration")
        != iteration
        or _exact_int(
            ledger.get("reference_data_version"),
            "reference-commit version",
        )
        != iteration
        or str(ledger.get("point_allocation_sha256") or "") != allocation_sha256
    ):
        raise BatchGeometryExportError("reference-commit identity is inconsistent")
    bindings = ledger.get("point_bindings")
    if not isinstance(bindings, list):
        raise BatchGeometryExportError("reference-commit point bindings are missing")
    expected = [
        (
            entry.pointdir_name,
            entry.source_pointdir,
            entry.candidate_id,
            entry.slot_id,
            entry.split,
            entry.replacement_round,
            entry.acceptance_receipt_sha256,
            entry.accepted_content_sha256,
            entry.provenance_sha256,
        )
        for entry in reference_entries
    ]
    observed = [
        (
            str(record.get("destination_pointdir") or ""),
            str(record.get("source_pointdir") or ""),
            str(record.get("candidate_id") or ""),
            _exact_int(record.get("slot_id"), "reference-commit slot"),
            str(record.get("split") or ""),
            _exact_int(record.get("replacement_round"), "replacement round"),
            str(record.get("acceptance_receipt_sha256") or ""),
            str(record.get("accepted_content_sha256") or ""),
            str(record.get("provenance_sha256") or ""),
        )
        for record in bindings
        if isinstance(record, Mapping)
    ]
    if len(observed) != len(bindings) or observed != expected:
        raise BatchGeometryExportError(
            "reference-commit bindings do not match the committed iteration delta"
        )


def _read_phase_b_candidate_records(
    iteration_dir: Path,
    *,
    iteration: int,
    campaign_uid: str,
    sampling_manifest: Mapping[str, Any],
    allocation: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    path = _verify_bound_file(
        iteration_dir,
        sampling_manifest,
        "phase_b/SELECTION.json",
    )
    payload = _read_json_object(path, "Phase B selection manifest")
    if (
        _exact_int(payload.get("schema_version"), "Phase B schema")
        != PHASE_B_SELECTION_SCHEMA_VERSION
        or _exact_int(payload.get("iteration"), "Phase B iteration") != iteration
        or str(payload.get("campaign_uid") or "") != campaign_uid
        or str(payload.get("status") or "") != "complete"
    ):
        raise BatchGeometryExportError("Phase B selection identity is invalid")
    source_path = _verify_bound_file(
        iteration_dir,
        sampling_manifest,
        "ariadne/RESULTS.json",
    )
    if (
        str(payload.get("source_ariadne_manifest") or "") != "ariadne/RESULTS.json"
        or str(payload.get("source_ariadne_manifest_sha256") or "")
        != sha256_file(source_path)
    ):
        raise BatchGeometryExportError("Phase B ARIADNE-results binding mismatch")
    allocation_binding = payload.get("point_allocation")
    if not isinstance(allocation_binding, Mapping):
        raise BatchGeometryExportError("Phase B point-allocation binding is missing")
    if (
        str(allocation_binding.get("manifest") or "")
        != "allocation/POINT_ALLOCATION.json"
        or str(allocation_binding.get("slot_assignment_sha256") or "")
        != str(allocation.get("slot_assignment_sha256") or "")
        or dict(allocation_binding.get("targets") or {})
        != dict(allocation.get("targets") or {})
    ):
        raise BatchGeometryExportError("Phase B point-allocation binding mismatch")
    final = payload.get("final")
    reserve = allocation_binding.get("reserve")
    if not isinstance(final, list) or not isinstance(reserve, list):
        raise BatchGeometryExportError("Phase B candidate lists are invalid")
    if _exact_int(payload.get("n_kept"), "Phase B kept count") != len(final):
        raise BatchGeometryExportError("Phase B final count mismatch")
    if _exact_int(
        allocation_binding.get("reserve_count"),
        "Phase B reserve count",
    ) != len(reserve):
        raise BatchGeometryExportError("Phase B reserve count mismatch")
    candidates: Dict[str, Dict[str, Any]] = {}
    for source, label in ((final, "final"), (reserve, "reserve")):
        for raw in source:
            if not isinstance(raw, Mapping):
                raise BatchGeometryExportError("Phase B " + label + " record is invalid")
            record = dict(raw)
            candidate_id = str(record.get("candidate_id") or "")
            if not candidate_id or candidate_id in candidates:
                raise BatchGeometryExportError("Phase B candidate IDs are invalid")
            seed_id = _exact_int(record.get("seed_id"), "Phase B seed ID", minimum=1)
            seed_uid = _sha256(record.get("seed_uid"), "Phase B seed UID")
            result_sha = _sha256(record.get("result_sha256"), "Phase B result hash")
            expected_id = stable_candidate_id(
                campaign_uid=campaign_uid,
                context="active",
                iteration=iteration,
                source_identity={
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_frame_id": record.get("seed_frame_id"),
                    "result_sha256": result_sha,
                },
            )
            if candidate_id != expected_id:
                raise BatchGeometryExportError("Phase B candidate identity mismatch")
            candidates[candidate_id] = record
    return candidates


def _read_target_ariadne_results(
    iteration_dir: Path,
    *,
    iteration: int,
    campaign_uid: str,
    task_map: Mapping[str, Any],
    target_seed_ids: Iterable[int],
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    path = ariadne_results_path(iteration_dir)
    payload = _read_json_object(path, "ARIADNE results manifest")
    if (
        _exact_int(payload.get("schema_version"), "ARIADNE results schema")
        != ARIADNE_RESULTS_SCHEMA_VERSION
        or _exact_int(payload.get("iteration"), "ARIADNE results iteration")
        != iteration
        or str(payload.get("campaign_uid") or "") != campaign_uid
    ):
        raise BatchGeometryExportError("ARIADNE results identity is invalid")
    accepted = payload.get("accepted")
    rejected = payload.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise BatchGeometryExportError("ARIADNE result dispositions are invalid")
    if (
        _exact_int(payload.get("n_accepted"), "ARIADNE accepted count")
        != len(accepted)
        or _exact_int(payload.get("n_rejected"), "ARIADNE rejected count")
        != len(rejected)
        or _exact_int(payload.get("expected_n"), "ARIADNE expected count")
        != len(accepted) + len(rejected)
    ):
        raise BatchGeometryExportError("ARIADNE result counts are inconsistent")
    tasks = task_map.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(accepted) + len(rejected):
        raise BatchGeometryExportError("ARIADNE results/task-map count mismatch")
    tasks_by_seed = {
        _exact_int(task.get("seed_id"), "ARIADNE task seed ID", minimum=1): dict(task)
        for task in tasks
        if isinstance(task, Mapping)
    }
    if len(tasks_by_seed) != len(tasks):
        raise BatchGeometryExportError("ARIADNE task-map seed IDs are invalid")
    targets = {int(seed_id) for seed_id in target_seed_ids}
    target_records: Dict[int, Dict[str, Any]] = {}
    seen = set()
    for records, accepted_status in ((accepted, True), (rejected, False)):
        for raw in records:
            if not isinstance(raw, Mapping):
                raise BatchGeometryExportError("ARIADNE result record is invalid")
            record = dict(raw)
            seed_id = _exact_int(record.get("seed_id"), "ARIADNE seed ID", minimum=1)
            task = tasks_by_seed.get(seed_id)
            if task is None or seed_id in seen:
                raise BatchGeometryExportError("ARIADNE result task coverage is invalid")
            seen.add(seed_id)
            if str(record.get("seed_uid") or "") != str(task.get("seed_uid") or ""):
                raise BatchGeometryExportError("ARIADNE result seed UID mismatch")
            if record.get("array_task_id") is not None and _exact_int(
                record.get("array_task_id"),
                "ARIADNE array task ID",
            ) != int(task["array_task_id"]):
                raise BatchGeometryExportError("ARIADNE result task ID mismatch")
            expected_seed_dir = ariadne_seed_dir(iteration_dir, seed_id)
            canonical = {
                "seed_dir": expected_seed_dir,
                "result_json": expected_seed_dir / SEED_RESULT_FILENAME,
                "provenance_json": expected_seed_dir / PROVENANCE_FILENAME,
                "output_manifest": expected_seed_dir / SEED_OUTPUT_MANIFEST_FILENAME,
            }
            for key, expected in canonical.items():
                if key not in record or not record.get(key):
                    if accepted_status:
                        raise BatchGeometryExportError(
                            "accepted ARIADNE record lacks " + key
                        )
                    continue
                resolved = resolve_handoff_path(
                    path.parent,
                    record[key],
                    kind="ARIADNE " + key,
                    must_exist=accepted_status and seed_id in targets,
                    directory=(key == "seed_dir"),
                )
                if resolved != expected.resolve(strict=False):
                    raise BatchGeometryExportError(
                        "ARIADNE record path does not match its task map"
                    )
            if seed_id in targets:
                if not accepted_status:
                    raise BatchGeometryExportError(
                        "allocated candidate refers to a rejected ARIADNE task"
                    )
                target_records[seed_id] = record
    if seen != set(tasks_by_seed):
        raise BatchGeometryExportError("ARIADNE results do not cover every task")
    if set(target_records) != targets:
        raise BatchGeometryExportError("allocated ARIADNE results are incomplete")
    return payload, target_records


def _geometry_from_atoms(atoms: Sequence[Any], label: str) -> Geometry:
    atom_types = tuple(str(atom.type) for atom in atoms)
    coordinates = np.asarray(
        [[float(atom.x), float(atom.y), float(atom.z)] for atom in atoms],
        dtype=np.float64,
    )
    if not atom_types or coordinates.shape != (len(atom_types), 3):
        raise BatchGeometryExportError(label + " has an invalid geometry shape")
    if not np.isfinite(coordinates).all():
        raise BatchGeometryExportError(label + " contains non-finite coordinates")
    return Geometry(atom_types=atom_types, coordinates=coordinates)


def _read_committed_geometry(
    campaign: Path,
    *,
    campaign_uid: str,
    entry: ReferenceDataEntry,
    iteration: int,
    attempt: Mapping[str, Any],
    assignment_sha256: str,
) -> Geometry:
    pointdir = entry.pointdir_path
    if pointdir.is_symlink() or not pointdir.is_dir():
        raise BatchGeometryExportError(
            "committed QM point directory is missing: " + str(pointdir)
        )
    receipt_path = pointdir / QUANTUM_ACCEPTANCE_RECEIPT
    if sha256_file(receipt_path) != str(entry.acceptance_receipt_sha256):
        raise BatchGeometryExportError("committed acceptance receipt hash mismatch")
    receipt = read_quantum_acceptance_receipt(
        campaign,
        pointdir,
        expected_campaign_uid=str(campaign_uid),
        expected_iteration=iteration,
        expected_candidate_id=str(attempt["candidate_id"]),
        expected_assignment_sha256=assignment_sha256,
        expected_source_pointdir=entry.source_pointdir,
        verification="receipt",
        validate_quality=False,
    )
    if str(receipt.get("content_sha256") or "") != str(
        entry.accepted_content_sha256
    ):
        raise BatchGeometryExportError("committed point-directory content binding mismatch")
    artefacts = receipt.get("artefacts")
    if not isinstance(artefacts, list):
        raise BatchGeometryExportError("acceptance receipt artefacts are missing")
    xyz = [record for record in artefacts if Path(str(record.get("path") or "")).suffix.lower() == ".xyz"]
    wfn = [record for record in artefacts if Path(str(record.get("path") or "")).suffix.lower() == ".wfn"]
    if len(xyz) == 1:
        binding = xyz[0]
        kind = "xyz"
    elif len(xyz) > 1:
        raise BatchGeometryExportError("committed point directory contains ambiguous XYZ geometries")
    elif len(wfn) == 1:
        binding = wfn[0]
        kind = "wfn"
    else:
        raise BatchGeometryExportError("committed point directory lacks one geometry source")
    relative = _safe_relative_path(binding.get("path"), "accepted geometry path")
    path = pointdir / PurePosixPath(relative)
    if path.is_symlink() or not path.is_file():
        raise BatchGeometryExportError("committed geometry source is missing: " + str(path))
    if (
        _exact_int(binding.get("size"), "accepted geometry size")
        != int(path.stat().st_size)
        or _sha256(binding.get("sha256"), "accepted geometry hash")
        != sha256_file(path)
    ):
        raise BatchGeometryExportError("committed geometry source binding mismatch")
    if kind == "xyz":
        from ichor.core.files.xyz.strict_xyz import read_xyz_frames

        frames = read_xyz_frames(path)
        if len(frames) != 1:
            raise BatchGeometryExportError("committed point XYZ must contain one frame")
        atoms = frames[0]
    else:
        from ichor.core.files.gaussian import WFN

        atoms = WFN(path).atoms.to_angstroms()
    return _geometry_from_atoms(atoms, "committed QM geometry")


def _write_export_xyz(path: Path, record: ExportRecord) -> None:
    comments = (
        "iteration="
        + str(record.iteration)
        + " split="
        + record.split
        + " slot="
        + str(record.slot_id)
        + " seed="
        + str(record.seed_id)
        + " replacement_round="
        + str(record.replacement_round)
        + " candidate="
        + record.candidate_id
    )
    lines: List[str] = []
    for geometry, source in (
        (record.seed_geometry, "ariadne_seed"),
        (record.final_geometry, "committed_qm:" + record.pointdir_name),
    ):
        lines.append(str(len(geometry.atom_types)))
        lines.append(comments + " source=" + source)
        for atom_type, coordinates in zip(geometry.atom_types, geometry.coordinates):
            lines.append(
                "{atom} {x:.12f} {y:.12f} {z:.12f}".format(
                    atom=atom_type,
                    x=float(coordinates[0]),
                    y=float(coordinates[1]),
                    z=float(coordinates[2]),
                )
            )
    data = ("\n".join(lines) + "\n").encode("ascii")
    with path.open("xb") as handle:
        if handle.write(data) != len(data):
            raise OSError("short geometry-export write: " + str(path))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_export_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise BatchGeometryExportError("export work path is not a regular directory: " + str(path))
    for root, directories, files in os.walk(path, topdown=True, followlinks=False):
        current = Path(root)
        for name in directories:
            child = current / name
            if child.is_symlink():
                raise BatchGeometryExportError("export work tree contains a symlink")
        for name in files:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise BatchGeometryExportError("export work tree contains an invalid file")
    shutil.rmtree(path)


def _normalise_output_path(
    campaign: Path,
    output_dir: Optional[Union[str, Path]],
    *,
    all_iterations: bool,
    iteration: int,
) -> Tuple[Path, bool]:
    if output_dir is None:
        name = (
            "all-model-batches"
            if all_iterations
            else "iteration-" + str(iteration).zfill(6) + "-model-batch"
        )
        return lexical_absolute_path(campaign / EXPORT_ROOT_NAME / name), True
    expanded = os.path.expandvars(os.fspath(Path(output_dir).expanduser()))
    target = lexical_absolute_path(expanded)
    try:
        relative = target.relative_to(campaign)
    except ValueError:
        relative = None
    if relative is not None and (
        len(relative.parts) < 2 or relative.parts[0] != EXPORT_ROOT_NAME
    ):
        raise BatchGeometryExportError(
            "an output inside the campaign must be under " + EXPORT_ROOT_NAME
        )
    return target, False


def _publish_records(
    target: Path,
    records_by_iteration: Mapping[int, Sequence[ExportRecord]],
    *,
    all_iterations: bool,
    replace_existing: bool,
) -> None:
    parent = target.parent
    reject_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(parent)
    if parent.is_symlink() or not parent.is_dir():
        raise BatchGeometryExportError("export parent is not a regular directory")
    if target.is_symlink():
        raise BatchGeometryExportError("export destination is a symlink")
    building = parent / ("." + target.name + ".building")
    previous = parent / ("." + target.name + ".previous")
    lock_path = parent / ("." + target.name + ".export.lock")
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise BatchGeometryExportError("export lock path is not a regular file")
    try:
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            timeout=0,
        ):
            if previous.exists() or previous.is_symlink():
                if target.exists() or target.is_symlink():
                    _remove_export_tree(previous)
                else:
                    if previous.is_symlink() or not previous.is_dir():
                        raise BatchGeometryExportError(
                            "prior export recovery path is invalid"
                        )
                    os.replace(previous, target)
                    _fsync_directory(parent)
            _remove_export_tree(building)
            if target.exists() and not replace_existing:
                raise FileExistsError(
                    "export destination already exists: " + str(target)
                )
            if target.exists() and (target.is_symlink() or not target.is_dir()):
                raise BatchGeometryExportError(
                    "export destination is not a regular directory"
                )

            building.mkdir()
            try:
                for iteration, records in records_by_iteration.items():
                    output = (
                        building
                        / ("iteration-" + str(iteration).zfill(6) + "-model-batch")
                        if all_iterations
                        else building
                    )
                    if output != building:
                        output.mkdir()
                    expected_names = []
                    for record in records:
                        destination = output / record.filename
                        _write_export_xyz(destination, record)
                        expected_names.append(record.filename)
                    observed = sorted(path.name for path in output.iterdir())
                    if observed != sorted(expected_names):
                        raise BatchGeometryExportError(
                            "completed export inventory is invalid"
                        )
                    _fsync_directory(output)
                _fsync_directory(building)
                if target.exists():
                    if previous.exists() or previous.is_symlink():
                        raise BatchGeometryExportError(
                            "prior export destination is ambiguous"
                        )
                    os.replace(target, previous)
                    _fsync_directory(parent)
                try:
                    os.replace(building, target)
                    _fsync_directory(parent)
                except Exception:
                    if previous.is_dir() and not target.exists():
                        os.replace(previous, target)
                        _fsync_directory(parent)
                    raise
                if previous.exists():
                    _remove_export_tree(previous)
                    _fsync_directory(parent)
            except Exception:
                if building.exists():
                    _remove_export_tree(building)
                raise
    except (portalocker.LockException, portalocker.AlreadyLocked) as exc:
        raise BatchGeometryExportError(
            "another geometry export is already publishing this destination"
        ) from exc


def _records_for_iteration(
    campaign: Path,
    *,
    iteration: int,
    campaign_uid: str,
    reference_view: Any,
    model_set: Any,
    sampling_manifest: Mapping[str, Any],
    global_ledger: Mapping[str, Any],
) -> Tuple[ExportRecord, ...]:
    iteration_dir = active_iteration_dir(campaign, iteration)
    allocation_path = _verify_bound_file(
        iteration_dir,
        sampling_manifest,
        "allocation/POINT_ALLOCATION.json",
    )
    reference_root = qm_reference_data_dir(campaign) / (
        "iteration-" + str(iteration).zfill(6)
    )
    immutable_allocation_path = reference_root / (
        "POINT_ALLOCATION.version-" + str(iteration).zfill(6) + ".json"
    )
    if immutable_allocation_path.is_symlink() or not immutable_allocation_path.is_file():
        raise BatchGeometryExportError("committed allocation snapshot is missing")
    allocation_sha = sha256_file(immutable_allocation_path)
    if allocation_sha != sha256_file(allocation_path):
        raise BatchGeometryExportError(
            "finalised and committed point allocations do not match"
        )
    allocation = read_point_allocation(
        immutable_allocation_path,
        history_dir=reference_root / ".point_allocation_history",
        expected_campaign_uid=campaign_uid,
        expected_context="active",
        expected_iteration=int(iteration),
    )
    if (
        str(allocation.get("context") or "") != "active"
        or int(allocation.get("iteration", -1)) != iteration
        or not bool((allocation.get("summary") or {}).get("complete", False))
    ):
        raise BatchGeometryExportError("requested point allocation is not complete and active")
    attempts = [
        attempt
        for attempt in accepted_attempts(allocation)
        if str(attempt.get("split") or "") in _SPLIT_ORDER
    ]
    attempts.sort(key=lambda item: (_SPLIT_ORDER[str(item["split"])], int(item["slot_id"])))
    if len(attempts) != int(allocation["targets"]["train"]) + int(
        allocation["targets"]["int_val"]
    ):
        raise BatchGeometryExportError("accepted model-batch allocation count is invalid")

    entries = [
        entry
        for entry in reference_view.entries
        if int(entry.introduced_in_version) == iteration
    ]
    if len(entries) != int(allocation["targets"]["total"]):
        raise BatchGeometryExportError("reference-data iteration delta count is invalid")
    entries_by_candidate = {entry.candidate_id: entry for entry in entries}
    if len(entries_by_candidate) != len(entries):
        raise BatchGeometryExportError("reference-data candidate identities are not unique")
    expected_attempt_identities = [
        (
            str(attempt["candidate_id"]),
            int(attempt["slot_id"]),
            str(attempt["split"]),
            int(attempt.get("round", 0)),
        )
        for attempt in accepted_attempts(allocation)
    ]
    observed_entry_identities = [
        (
            entry.candidate_id,
            entry.slot_id,
            entry.split,
            entry.replacement_round,
        )
        for entry in entries
    ]
    if observed_entry_identities != expected_attempt_identities:
        raise BatchGeometryExportError("reference-data delta/allocation identity mismatch")
    _validate_reference_commit(
        campaign,
        iteration=iteration,
        reference_entries=entries,
        allocation_sha256=allocation_sha,
    )
    _validate_model_split_snapshot(
        campaign,
        iteration=iteration,
        model_set=model_set,
        reference_view=reference_view,
        allocation=allocation,
        allocation_sha256=allocation_sha,
        global_ledger=global_ledger,
    )

    _verify_bound_file(iteration_dir, sampling_manifest, "seed_selection/SELECTION.json")
    seeds_path = _verify_bound_file(
        iteration_dir,
        sampling_manifest,
        "seed_selection/seeds.xyz",
    )
    _verify_bound_file(iteration_dir, sampling_manifest, "ariadne/TASK_MAP.json")
    _verify_bound_file(iteration_dir, sampling_manifest, "ariadne/ARIADNE_BATCH_DECISION.json")
    selection = load_seeds_picked(iteration_dir, expected_iteration=iteration)
    if (
        str(selection.get("campaign_uid") or "") != campaign_uid
        or int(selection.get("models_version", -1)) != iteration - 1
        or str(selection.get("model_manifest_sha256") or "")
        != str(sampling_manifest["input_head"]["trained_models"]["head_manifest_sha256"])
        or str(selection.get("model_set_sha256") or "")
        != str(sampling_manifest["input_head"]["trained_models"]["model_set_sha256"])
    ):
        raise BatchGeometryExportError("seed selection is not bound to the input model")
    task_map = read_ariadne_task_map(iteration_dir, expected_iteration=iteration)
    if str(task_map.get("campaign_uid") or "") != campaign_uid:
        raise BatchGeometryExportError("ARIADNE task map campaign identity mismatch")
    read_ariadne_batch_decision(
        iteration_dir,
        expected_iteration=iteration,
        expected_campaign_uid=campaign_uid,
        require_accepted=True,
        verify_current_config=False,
    )
    candidates = _read_phase_b_candidate_records(
        iteration_dir,
        iteration=iteration,
        campaign_uid=campaign_uid,
        sampling_manifest=sampling_manifest,
        allocation=allocation,
    )
    target_seed_ids = {
        _exact_int(attempt.get("seed_id"), "allocation seed ID", minimum=1)
        for attempt in attempts
    }
    results_payload, result_records = _read_target_ariadne_results(
        iteration_dir,
        iteration=iteration,
        campaign_uid=campaign_uid,
        task_map=task_map,
        target_seed_ids=target_seed_ids,
    )
    if str(results_payload.get("trajectory_sha256") or "") != str(
        selection.get("trajectory_sha256") or ""
    ):
        raise BatchGeometryExportError("ARIADNE result trajectory identity mismatch")

    from ichor.core.files.xyz.strict_xyz import read_xyz_frames

    seed_frames = read_xyz_frames(seeds_path)
    seed_records = list(selection.get("seed_records") or [])
    if len(seed_frames) != len(seed_records):
        raise BatchGeometryExportError("seed geometry/selection count mismatch")
    selection_by_seed = {int(record["seed_id"]): dict(record) for record in seed_records}
    tasks_by_seed = {int(task["seed_id"]): dict(task) for task in task_map["tasks"]}

    records: List[ExportRecord] = []
    assignment_sha = str(allocation["slot_assignment_sha256"])
    for export_order, attempt in enumerate(attempts, start=1):
        candidate_id = str(attempt["candidate_id"])
        entry = entries_by_candidate.get(candidate_id)
        phase_candidate = candidates.get(candidate_id)
        if entry is None or phase_candidate is None:
            raise BatchGeometryExportError("accepted candidate lacks committed source evidence")
        seed_id = _exact_int(attempt.get("seed_id"), "allocation seed ID", minimum=1)
        seed_uid = _sha256(attempt.get("seed_uid"), "allocation seed UID")
        result_sha = _sha256(attempt.get("result_sha256"), "allocation result hash")
        expected_candidate = stable_candidate_id(
            campaign_uid=campaign_uid,
            context="active",
            iteration=iteration,
            source_identity={
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "seed_frame_id": attempt.get("seed_frame_id"),
                "result_sha256": result_sha,
            },
        )
        if expected_candidate != candidate_id:
            raise BatchGeometryExportError("allocation candidate identity mismatch")
        for field in ("seed_id", "seed_uid", "seed_frame_id", "result_sha256"):
            if phase_candidate.get(field) != attempt.get(field):
                raise BatchGeometryExportError(
                    "allocation and Phase B candidate " + field + " differ"
                )
        selection_record = selection_by_seed.get(seed_id)
        task = tasks_by_seed.get(seed_id)
        result_record = result_records.get(seed_id)
        if selection_record is None or task is None or result_record is None:
            raise BatchGeometryExportError("accepted candidate seed identity is incomplete")
        if (
            str(selection_record.get("seed_uid") or "") != seed_uid
            or str(task.get("seed_uid") or "") != seed_uid
            or result_record.get("result_sha256") != result_sha
        ):
            raise BatchGeometryExportError("accepted candidate seed/result binding mismatch")
        seed_geometry = _geometry_from_atoms(
            seed_frames[seed_id - 1],
            "ARIADNE seed geometry",
        )
        seed_dir = ariadne_seed_dir(iteration_dir, seed_id)
        output = validate_seed_output(
            seed_dir,
            expected_campaign_uid=campaign_uid,
            expected_iteration=iteration,
            expected_seed_id=seed_id,
            expected_seed_uid=seed_uid,
            expected_array_task_id=int(task["array_task_id"]),
        )
        if not bool(output.get("task_success")) or int(output.get("task_exit_code", -1)) != 0:
            raise BatchGeometryExportError("allocated ARIADNE seed output was not successful")
        result_path = seed_dir / SEED_RESULT_FILENAME
        if sha256_file(result_path) != result_sha:
            raise BatchGeometryExportError("ARIADNE result hash mismatch")
        result = _read_json_object(result_path, "ARIADNE result")
        validated = validate_ariadne_result(
            result,
            expected_iteration=iteration,
            seed_record={
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "frame_id": attempt.get("seed_frame_id"),
            },
            expected_atom_types=seed_geometry.atom_types,
            expected_initial_coordinates=seed_geometry.coordinates.tolist(),
            expected_trajectory_sha256=str(selection["trajectory_sha256"]),
        )
        provenance_path = entry.pointdir_path / PROVENANCE_FILENAME
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise BatchGeometryExportError("committed provenance is missing")
        if sha256_file(provenance_path) != str(entry.provenance_sha256):
            raise BatchGeometryExportError("committed provenance hash mismatch")
        provenance = validate_provenance(
            entry.pointdir_path,
            campaign_uid=campaign_uid,
            iteration=iteration,
            trajectory_sha256=str(selection["trajectory_sha256"]),
            seed_frame_id=attempt.get("seed_frame_id"),
            seed_id=seed_id,
            seed_uid=seed_uid,
            array_task_id_zero_based=int(task["array_task_id"]),
            allocation_split=str(attempt["split"]),
            allocation_slot_id=int(attempt["slot_id"]),
            allocation_candidate_id=candidate_id,
            allocation_context="active",
            allocation_slot_assignment_sha256=assignment_sha,
        )
        phase_b = provenance.get("phase_b")
        if (
            not isinstance(phase_b, Mapping)
            or str(phase_b.get("candidate_id") or "") != candidate_id
            or not (
                bool(phase_b.get("selected_after_fps", False))
                or bool(phase_b.get("reserve_candidate", False))
            )
        ):
            raise BatchGeometryExportError("committed provenance lacks Phase B selection identity")
        final_geometry = _read_committed_geometry(
            campaign,
            campaign_uid=campaign_uid,
            entry=entry,
            iteration=iteration,
            attempt=attempt,
            assignment_sha256=assignment_sha,
        )
        if final_geometry.atom_types != tuple(validated["atom_types"]):
            raise BatchGeometryExportError("committed and ARIADNE atom orders differ")
        ariadne_final = np.asarray(validated["final_coordinates"], dtype=np.float64)
        if ariadne_final.shape != final_geometry.coordinates.shape:
            raise BatchGeometryExportError("committed and ARIADNE geometry shapes differ")
        discrepancy = float(np.max(np.abs(final_geometry.coordinates - ariadne_final)))
        if not math.isfinite(discrepancy) or discrepancy > FINAL_COORDINATE_TOLERANCE_ANGSTROM:
            raise BatchGeometryExportError(
                "committed geometry differs from the ARIADNE final geometry by "
                + format(discrepancy, ".12g")
                + " A"
            )
        records.append(
            ExportRecord(
                iteration=iteration,
                export_order=export_order,
                split=str(attempt["split"]),
                slot_id=int(attempt["slot_id"]),
                seed_id=seed_id,
                seed_uid=seed_uid,
                candidate_id=candidate_id,
                replacement_round=int(attempt.get("round", 0)),
                pointdir_name=entry.pointdir_name,
                seed_geometry=seed_geometry,
                final_geometry=final_geometry,
                coordinate_discrepancy_angstrom=discrepancy,
            )
        )
    return tuple(records)


def export_batch_geometries(
    campaign_dir: Union[str, Path],
    iteration: Union[int, str],
    *,
    output_dir: Optional[Union[str, Path]] = None,
) -> BatchGeometryExportSummary:
    """Validate and atomically export one or all completed active batches."""
    campaign = Path(campaign_dir).resolve()
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    state = read_state(state_path)
    iterations = _requested_iterations(state, iteration)
    maximum = max(iterations)
    target, is_default = _normalise_output_path(
        campaign,
        output_dir,
        all_iterations=(iteration == "all"),
        iteration=iterations[0],
    )
    if not is_default and (target.exists() or target.is_symlink()):
        raise FileExistsError("export destination already exists: " + str(target))

    reference_versioning = ReferenceDataVersioning(qm_reference_data_dir(campaign))
    model_versioning = TrainedModelVersioning(trained_models_dir(campaign))
    reference_current = reference_versioning.current_version()
    model_current = model_versioning.current_version()
    if reference_current is None or reference_current < maximum:
        raise BatchGeometryExportError("QM reference data is not committed through the requested iteration")
    if model_current is None or model_current < maximum:
        raise BatchGeometryExportError("trained models are not committed through the requested iteration")
    reference_views = resolve_reference_data_chain(
        campaign,
        maximum,
        verification="authority",
        expected_campaign_uid=str(state.campaign_uid),
    )
    model_sets = resolve_trained_model_chain(
        campaign,
        maximum,
        verification="authority",
        reference_views=reference_views,
    )
    sampling_manifests = resolve_sampling_iteration_authority_chain(
        campaign,
        maximum,
        expected_campaign_uid=str(state.campaign_uid),
        reference_views=reference_views,
        model_sets=model_sets,
    )
    global_ledger = read_split_assignments(campaign)
    controls = _sampling_control_hashes(campaign, iterations)

    records_by_iteration: Dict[int, Tuple[ExportRecord, ...]] = {}
    for value in iterations:
        records_by_iteration[value] = _records_for_iteration(
            campaign,
            iteration=value,
            campaign_uid=str(state.campaign_uid),
            reference_view=reference_views[value],
            model_set=model_sets[value],
            sampling_manifest=sampling_manifests[value],
            global_ledger=global_ledger,
        )
    _recheck_control_hashes(controls)

    _publish_records(
        target,
        records_by_iteration,
        all_iterations=(iteration == "all"),
        replace_existing=is_default,
    )
    flattened = [record for records in records_by_iteration.values() for record in records]
    return BatchGeometryExportSummary(
        output_path=target,
        iterations=iterations,
        training_count=sum(record.split == "train" for record in flattened),
        internal_validation_count=sum(record.split == "int_val" for record in flattened),
        maximum_coordinate_discrepancy_angstrom=max(
            (record.coordinate_discrepancy_angstrom for record in flattened),
            default=0.0,
        ),
    )


__all__ = [
    "BatchGeometryExportError",
    "BatchGeometryExportSummary",
    "export_batch_geometries",
]
