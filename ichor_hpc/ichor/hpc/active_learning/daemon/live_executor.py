"""LiveBackendsPhaseExecutor -- the configured Slurm deployment target.

Inherits the file system layout management from :class: DryRunPhaseExecutor
but overrides the SBATCH phase submission to:

    1. Write a real submission script with SLURM directives + module loads +
       the actual backend invocation (Gaussian / AIMAll / FEREBUS / ARIADNE).
    2. Shell out to "sbatch --parsable" via subprocess.run and capture the
       returned JobID.
    3. On terminal sacct outcome, parse the real output files into the
       canonical ICHOR data structures (PointsDirectory, Models, etc.).

The per-phase output parsers below are real: they read the Gaussian /
AIMAll / FEREBUS / ARIADNE / diversity results into the canonical ICHOR data
structures.

The wiring is verified via the smoke tests ('pytest -m live'), which
skip cleanly when the required binaries are absent. 
On configured Slurm clusters they exercise sbatch + sacct against tiny one-shot
jobs that take seconds, not hours.
"""
from __future__ import annotations

import os
import hashlib
import shlex
import shutil
import subprocess
import sys
import math
import re
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..config import (
    CampaignConfig,
    VALID_AIMALL_BOAQ_VALUES,
    VALID_AIMALL_IASMESH_VALUES,
    normalise_gaussian_route_keywords,
)
from ..versioning.provenance import (
    PROVENANCE_FILENAME,
    upsert_index_records,
    ensure_index,
    enrich_with_anti_overlap,
    enrich_with_ariadne,
    enrich_with_error_calibration_input,
    validate_provenance,
    write_seed_provenance,
)
from ..versioning.trained_models import (
    TRAINED_MODEL_AUXILIARY_SUFFIXES,
    TRAINED_MODEL_SET_FILENAME,
)
from .dry_run_executor import (
    DryRunPhaseExecutor,
    anti_overlap_whitened_distance_bounds,
)
from .phase_executor import (
    BackendSubmissionError,
    FailureAction,
    INLINE_PHASES,
    PhaseResult,
    PostprocessRetryDisposition,
    SBATCH_PHASES,
    SubmissionCancelledBeforeSchedulerAcceptance,
)
from .resource_solver import (
    ResolvedPhaseResources,
    gaussian_mdef,
    resolve_phase_resources,
    slurm_memory_mib,
    validate_partition_walltime,
)
from .resource_records import (
    capture_implementation_identity,
    resolution_payload,
    write_resolution,
)
from .script_bundles import (
    AttemptBundle,
    bundle_root,
    prepare_attempt_bundle,
    read_source_array_task_ids,
    scheduler_log_paths,
    write_attempt_script,
    write_script_binding,
)
from .scratch import scratch_path_template
from .preflight import BackendAvailability, check_backends, missing_backend_message
from .cluster_profile import (
    active_machine,
    expanded_profile_value,
    profile_value,
)
from .state import CampaignPhase, atomic_write_json
from .job_names import live_job_name
from .scheduler_recovery import (
    SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION,
    scheduler_terminal_recoveries,
    write_phase_recovery_ledger,
)
from .environment_equivalence import assess_recovery_environment
from .array_recovery import (
    archive_existing_array_task_outputs,
    compact_array_recovery_summary,
    prepare_retry_submission,
    supports_partial_array_recovery,
)
from .runtime_environment import (
    DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES,
    DEFAULT_DAEMON_PYTHON_MODULES,
    DEFAULT_DAEMON_RUNTIME_MODULES,
    ariadne_runtime_command_prefix,
    configured_daemon_runtime_modules,
    configured_python_library_paths,
    module_initialisation_lines,
    native_runtime_setup_lines,
    normalise_module_list,
    python_library_path_export_lines,
)
from ..submit.scheduler_backend import get_scheduler_backend
from ..submit.sge import sge_safe_job_name


__all__ = [
    "LiveBackendsPhaseExecutor",
    "LiveBackendNotAvailableError",
    "build_scheduler_script",
    "build_sbatch_script",
    "live_job_name",
    "make_live_job_finder",
    "make_live_job_accounting_finder",
    "make_live_job_liveness_checker",
    "LIVE_POSTPROCESS_IMPLEMENTED",
]

def _format_slurm_walltime_hours(hours: Any) -> str:
    try:
        total_seconds = int(math.ceil(float(hours) * 3600.0))
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError("walltime_hours must be a positive number") from exc
    if total_seconds <= 0:
        raise BackendSubmissionError("walltime_hours must be > 0")
    days, rem = divmod(total_seconds, 24 * 3600)
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    clock = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return str(days) + "-" + clock if days else clock


def _format_sge_walltime_hours(hours: Any) -> str:
    try:
        total_seconds = int(math.ceil(float(hours) * 3600.0))
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError("walltime_hours must be a positive number") from exc
    if total_seconds <= 0:
        raise BackendSubmissionError("walltime_hours must be > 0")
    hh, rem = divmod(total_seconds, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

_SHEBANG_RE = re.compile(r"^#![A-Za-z0-9_./ -]+$")
_SHELL_PATH_FRAGMENT_RE = re.compile(r"^[A-Za-z0-9_./${}:+-]+$")
FEREBUS_TASK_ARTEFACTS_MANIFEST = TRAINED_MODEL_SET_FILENAME
FEREBUS_TASK_AUXILIARY_SUFFIXES = TRAINED_MODEL_AUXILIARY_SUFFIXES


def _iteration_active_learning_dir(campaign_dir: Path, iteration: int) -> Path:
    from ..layout import active_iteration_dir

    return active_iteration_dir(campaign_dir, int(iteration))


def _object_with_overrides(default_obj: Any, overrides: Any) -> SimpleNamespace:
    """Return an attribute object using manifest values over current defaults."""
    values: Dict[str, Any] = {}
    if default_obj is not None:
        try:
            values.update(vars(default_obj))
        except TypeError:
            pass
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            values[str(key)] = value
    return SimpleNamespace(**values)


def _is_relative_to_path(path: Path, root: Path) -> bool:
    resolved = Path(path).resolve(strict=False)
    resolved_root = Path(root).resolve(strict=False)
    return resolved == resolved_root or resolved_root in resolved.parents


def _reject_symlinked_path(path: Path, root: Path, label: str) -> None:
    probe = Path(path)
    resolved_root = Path(root).resolve(strict=False)
    while True:
        if probe.exists() and probe.is_symlink():
            raise BackendSubmissionError(
                "refusing symlinked FEREBUS "
                + label
                + ": "
                + str(probe)
            )
        if probe.resolve(strict=False) == resolved_root:
            return
        parent = probe.parent
        if parent == probe:
            return
        probe = parent


def _resolve_ferebus_staging_path(staging: Path, raw_path: Any, label: str) -> Path:
    raw = Path(str(raw_path))
    candidate = raw if raw.is_absolute() else Path(staging) / raw
    if not _is_relative_to_path(candidate, staging):
        raise BackendSubmissionError(
            "FEREBUS " + label + " escapes iteration-staging: " + str(raw_path)
        )
    _reject_symlinked_path(candidate, staging, label)
    return candidate


def _resolve_ferebus_destination(root: Path, dest: Path, label: str) -> Path:
    target = Path(dest)
    if not _is_relative_to_path(target, root):
        raise BackendSubmissionError(
            "FEREBUS " + label + " destination escapes committed model directory: "
            + str(dest)
        )
    _reject_symlinked_path(target.parent, root, label + " parent")
    return target


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for path in paths:
        key = str(Path(path).resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(Path(path))
    return ordered


def _find_ferebus_auxiliary_file(
    staging: Path,
    model_file: Path,
    search_dirs: Sequence[Path],
    suffix: str,
) -> Optional[Path]:
    stem = model_file.stem
    for directory in search_dirs:
        candidate = directory / (stem + "." + suffix)
        if candidate.is_file():
            return _resolve_ferebus_staging_path(
                staging,
                candidate,
                "auxiliary ." + suffix + " file",
            )
    model_dir = model_file.parent
    if model_dir != Path(staging):
        matches = sorted(model_dir.glob("*." + suffix))
        if len(matches) == 1 and matches[0].is_file():
            return _resolve_ferebus_staging_path(
                staging,
                matches[0],
                "auxiliary ." + suffix + " file",
            )
        if len(matches) > 1:
            raise BackendSubmissionError(
                "ambiguous FEREBUS auxiliary ." + suffix + " files in " + str(model_dir)
            )
    return None


def _copy_regular_file_no_symlink(src: Path, dest: Path, root: Path) -> None:
    if not src.is_file() or src.is_symlink():
        raise BackendSubmissionError("refusing non-regular FEREBUS artefact: " + str(src))
    target = _resolve_ferebus_destination(root, dest, "artefact")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(target))


def _write_ferebus_task_artefact_layout(
    staging: Path,
    committed_dir: Path,
    manifest: Dict[str, Any],
    *,
    models_version: int,
    parent_model_set: Any,
) -> Path:
    """Build one complete hierarchical committed FEREBUS model snapshot."""
    from . import input_staging as _stg
    from .ferebus_quality import (
        FEREBUS_QUALITY_DECISION_MANIFEST,
        FEREBUS_QUALITY_MANIFEST,
    )
    from .ferebus_task_runner import (
        FEREBUS_TASK_MAP_FILENAME,
        FEREBUS_TASK_RECEIPT_FILENAME,
    )
    from ..versioning.trained_models import (
        build_trained_model_set_payload,
        file_record,
    )

    staging = Path(staging)
    committed_dir = Path(committed_dir)
    if any(committed_dir.iterdir()):
        raise BackendSubmissionError("trained-model version staging is not empty")
    task_records: List[Dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        prop = str(task.get("property"))
        atom = str(task.get("atom"))
        try:
            _stg.validate_safe_path_token("FEREBUS property", prop)
            _stg.validate_safe_path_token("FEREBUS atom label", atom)
        except ValueError as exc:
            raise BackendSubmissionError(str(exc)) from exc
        model_file = _resolve_ferebus_staging_path(
            staging,
            task["expected_model_path"],
            "model path",
        )
        config_path = _resolve_ferebus_staging_path(
            staging,
            task.get("config_path", ""),
            "config path",
        )
        search_dirs = _dedupe_paths([
            directory
            for directory in (model_file.parent, config_path.parent, staging)
            if _is_relative_to_path(directory, staging)
        ])
        task_dir = _resolve_ferebus_destination(
            committed_dir,
            committed_dir / prop / atom,
            "task artefact directory",
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        committed_model = task_dir / model_file.name
        committed_config = task_dir / "ferebus.config"
        _copy_regular_file_no_symlink(model_file, committed_model, committed_dir)
        _copy_regular_file_no_symlink(config_path, committed_config, committed_dir)
        raw_datasets = task.get("datasets")
        if not isinstance(raw_datasets, dict) or set(raw_datasets) != {
            "train",
            "int_val",
            "ext_val",
        }:
            raise BackendSubmissionError(
                "FEREBUS task dataset identities are invalid for " + prop + "/" + atom
            )
        committed_datasets: Dict[str, Dict[str, Any]] = {}
        dataset_dir = task_dir / "datasets"
        for split in ("train", "int_val", "ext_val"):
            dataset_record = raw_datasets[split]
            if not isinstance(dataset_record, dict):
                raise BackendSubmissionError(
                    "FEREBUS " + split + " dataset identity is invalid"
                )
            dataset_source = _resolve_ferebus_staging_path(
                staging,
                dataset_record.get("path", ""),
                split + " dataset path",
            )
            dataset_destination = dataset_dir / dataset_source.name
            _copy_regular_file_no_symlink(
                dataset_source,
                dataset_destination,
                committed_dir,
            )
            committed_datasets[split] = file_record(
                dataset_destination,
                committed_dir,
            )
        receipt_source = _resolve_ferebus_staging_path(
            staging,
            Path(str(task.get("output_dir") or ""))
            / FEREBUS_TASK_RECEIPT_FILENAME,
            "task execution receipt",
        )
        committed_receipt = task_dir / FEREBUS_TASK_RECEIPT_FILENAME
        _copy_regular_file_no_symlink(
            receipt_source,
            committed_receipt,
            committed_dir,
        )
        auxiliary: Dict[str, Optional[Dict[str, Any]]] = {}
        for suffix in FEREBUS_TASK_AUXILIARY_SUFFIXES:
            source = _find_ferebus_auxiliary_file(
                staging,
                model_file,
                search_dirs,
                suffix,
            )
            if source is None:
                auxiliary[suffix] = None
                continue
            dest = task_dir / source.name
            _copy_regular_file_no_symlink(source, dest, committed_dir)
            auxiliary[suffix] = file_record(dest, committed_dir)
        task_records.append(
            {
                "task_index": int(task.get("task_index")),
                "property": prop,
                "atom": atom,
                "alf_1_indexed": [int(value) for value in task.get("alf_1_indexed", [])],
                "directory": task_dir.relative_to(committed_dir).as_posix(),
                "model": file_record(committed_model, committed_dir),
                "config": file_record(committed_config, committed_dir),
                "execution_receipt": file_record(
                    committed_receipt,
                    committed_dir,
                ),
                "datasets": committed_datasets,
                "auxiliary": auxiliary,
            }
        )
    sidecar_names = (
        _stg.FEREBUS_TASK_MANIFEST,
        _stg.FEREBUS_ROW_IDENTITIES,
        _stg.FEREBUS_SPLIT_SNAPSHOT,
        FEREBUS_TASK_MAP_FILENAME,
        _stg.FEREBUS_JOB_DETAILS,
        "runFerebus.sh",
        "ATOMS.txt",
        "PROPERTIES.txt",
        FEREBUS_QUALITY_MANIFEST,
        FEREBUS_QUALITY_DECISION_MANIFEST,
        "MODEL_BOOTSTRAP.json",
    )
    root_records: List[Dict[str, Any]] = []
    for sidecar_name in sidecar_names:
        source = staging / sidecar_name
        if not source.is_file():
            if sidecar_name in {
                _stg.FEREBUS_TASK_MANIFEST,
                _stg.FEREBUS_ROW_IDENTITIES,
                _stg.FEREBUS_SPLIT_SNAPSHOT,
                FEREBUS_TASK_MAP_FILENAME,
                FEREBUS_QUALITY_MANIFEST,
                FEREBUS_QUALITY_DECISION_MANIFEST,
            } or (
                sidecar_name == "MODEL_BOOTSTRAP.json"
                and isinstance(manifest.get("model_bootstrap"), dict)
            ):
                raise BackendSubmissionError(
                    "required FEREBUS sidecar is missing: " + str(source)
                )
            continue
        destination = committed_dir / sidecar_name
        _copy_regular_file_no_symlink(source, destination, committed_dir)
        root_records.append(file_record(destination, committed_dir))
    root_records.sort(key=lambda record: str(record["path"]))
    root_record_by_path = {str(record["path"]): record for record in root_records}
    payload = build_trained_model_set_payload(
        campaign_uid=str(manifest.get("campaign_uid") or ""),
        version=int(models_version),
        system=str(manifest.get("system") or ""),
        reference_data_head_manifest_sha256=str(
            manifest.get("reference_data_head_manifest_sha256") or ""
        ),
        reference_data_view_sha256=str(
            manifest.get("reference_data_view_sha256") or ""
        ),
        parent=parent_model_set,
        source_task_manifest=root_record_by_path[_stg.FEREBUS_TASK_MANIFEST],
        quality_manifest=root_record_by_path[FEREBUS_QUALITY_MANIFEST],
        quality_decision_manifest=root_record_by_path[
            FEREBUS_QUALITY_DECISION_MANIFEST
        ],
        properties=[str(value) for value in manifest.get("properties", [])],
        atoms=[str(value) for value in manifest.get("atoms", [])],
        tasks=task_records,
        root_files=root_records,
    )
    manifest_path = committed_dir / FEREBUS_TASK_ARTEFACTS_MANIFEST
    atomic_write_json(manifest_path, payload)
    return manifest_path


def _with_trained_models_commit_lock(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        from ..versioning.trained_models import trained_models_commit_lock

        with trained_models_commit_lock(self.campaign_dir):
            return method(self, *args, **kwargs)

    return locked


def _without_keys(payload: Any, *keys: str) -> Dict[str, Any]:
    data = dict(payload) if isinstance(payload, dict) else {}
    for key in keys:
        data.pop(str(key), None)
    return data


def _campaign_manifest_path(
    campaign_dir: Path,
    iter_dir: Path,
    raw_path: Any,
    *,
    field_name: str,
) -> Path:
    if raw_path in (None, ""):
        raise ValueError(field_name + " is empty")
    path = Path(str(raw_path))
    campaign = Path(campaign_dir).resolve()
    if not path.is_absolute():
        candidates = [iter_dir / path, campaign / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    resolved = path.resolve()
    try:
        resolved.relative_to(campaign)
    except ValueError as exc:
        raise ValueError(field_name + " points outside campaign directory: " + str(resolved)) from exc
    if not resolved.is_file():
        raise ValueError(field_name + " does not exist: " + str(resolved))
    return resolved


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    from ..strict_json import strict_json as json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(label + " is unreadable: " + type(exc).__name__ + ": " + str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError(label + " must contain a JSON object")
    return payload


def _diversity_publication_is_incomplete(
    *,
    phase_name: str,
    output_dir: Path,
    manifest_path: Path,
) -> bool:
    """Return whether a scalar diversity publication stopped before completion.

    A missing manifest or a valid manifest whose fixed derived files were not
    all published is recoverably incomplete.  Malformed, non-canonical or
    contradictory evidence remains a hard failure for the strict reader.
    """
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return True
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BackendSubmissionError(
            "diversity publication manifest is not a regular file: "
            + str(manifest_path)
        )
    try:
        payload = _read_json_object(
            manifest_path,
            "diversity publication manifest",
        )
    except ValueError as exc:
        raise BackendSubmissionError(str(exc)) from exc

    if phase_name == "PHASE_A_DIVERSITY":
        from ..handoff_manifests import PHASE_A_SAMPLE_SCHEMA_VERSION

        if (
            int(payload.get("schema_version", -1))
            != int(PHASE_A_SAMPLE_SCHEMA_VERSION)
            or str(payload.get("phase") or "") != phase_name
        ):
            raise BackendSubmissionError(
                "Phase A diversity publication manifest is contradictory"
            )
        bindings = (
            ("sample_xyz", output_dir / "selected.xyz"),
            ("index_path", output_dir / "selected_indices.dat"),
        )
    elif phase_name == "PHASE_B_DIVERSITY":
        from ..handoff_manifests import PHASE_B_SELECTION_SCHEMA_VERSION

        if (
            int(payload.get("schema_version", -1))
            != int(PHASE_B_SELECTION_SCHEMA_VERSION)
            or int(payload.get("iteration", -1)) < 1
        ):
            raise BackendSubmissionError(
                "Phase B diversity publication manifest is contradictory"
            )
        if str(payload.get("status") or "") != "complete":
            raise BackendSubmissionError(
                "Phase B diversity published a terminal non-success result: "
                + str(payload.get("failure_reason") or "unknown failure")
            )
        bindings = (
            (
                "considered_candidates_xyz",
                output_dir / "considered_candidates.xyz",
            ),
            ("selected_xyz", output_dir / "selected.xyz"),
        )
    else:  # pragma: no cover - guarded by the caller
        raise BackendSubmissionError(
            "unsupported scalar diversity phase " + str(phase_name)
        )

    for field_name, expected_path in bindings:
        raw_binding = payload.get(field_name)
        if phase_name == "PHASE_B_DIVERSITY":
            if not isinstance(raw_binding, Mapping):
                raise BackendSubmissionError(
                    "Phase B diversity " + field_name + " binding is malformed"
                )
            raw_path = raw_binding.get("path")
        else:
            raw_path = raw_binding
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BackendSubmissionError(
                "diversity publication " + field_name + " path is missing"
            )
        declared = Path(raw_path)
        if not declared.is_absolute():
            declared = output_dir.parent / declared
        if declared.resolve(strict=False) != expected_path.resolve(strict=False):
            raise BackendSubmissionError(
                "diversity publication "
                + field_name
                + " path is non-canonical"
            )
        if not expected_path.exists() and not expected_path.is_symlink():
            return True
        if expected_path.is_symlink() or not expected_path.is_file():
            raise BackendSubmissionError(
                "diversity publication "
                + field_name
                + " is not a regular file"
            )
    return False


def _sampling_protocol_for_ariadne_result(
    *,
    campaign_dir: Path,
    iter_dir: Path,
    fallback_protocol: Any,
    result_dict: Dict[str, Any],
    iteration: int,
) -> tuple[SimpleNamespace, Dict[str, Any]]:
    """Load the exact sampling protocol recorded by an ARIADNE task.

    Live postprocess must not reinterpret a seed result with mutable sidecars.
    Every required protocol artefact is therefore named and SHA-bound by the
    per-seed result.
    """
    from ..sampling_protocol import (
        SAMPLING_PROTOCOL_SCHEMA_VERSION,
    )
    from ..sampling_scale_model import (
        SAMPLING_SCALE_MODEL_SCHEMA_VERSION,
    )
    from ..versioning.manifest import sha256_file

    sampling = result_dict.get("sampling_protocol")
    if not isinstance(sampling, dict):
        raise ValueError("ARIADNE result sampling_protocol binding is missing")

    def bound_manifest(field_name: str) -> Path:
        raw_path = sampling.get(field_name)
        if raw_path in (None, ""):
            raise ValueError("sampling_protocol." + field_name + " is missing")
        path = _campaign_manifest_path(
            campaign_dir,
            iter_dir,
            raw_path,
            field_name="sampling_protocol." + field_name,
        )
        expected_sha = str(sampling.get(field_name + "_sha256") or "")
        if len(expected_sha) != 64:
            raise ValueError(
                "sampling_protocol." + field_name + "_sha256 is invalid"
            )
        if sha256_file(path) != expected_sha:
            raise ValueError(
                "sampling_protocol." + field_name + " SHA-256 mismatch"
            )
        return path

    protocol_manifest = bound_manifest("resolved_manifest")
    scale_manifest = bound_manifest("scale_model_manifest")
    audit_manifest = bound_manifest("audit_manifest")

    protocol_payload = _read_json_object(
        protocol_manifest,
        "SAMPLING_PROTOCOL_RESOLVED.json",
    )
    if int(protocol_payload.get("schema_version", -1)) != int(
        SAMPLING_PROTOCOL_SCHEMA_VERSION
    ):
        raise ValueError("unsupported sampling protocol resolved schema")
    if int(protocol_payload.get("iteration", -1)) != int(iteration):
        raise ValueError("sampling protocol resolved iteration mismatch")
    result_level = sampling.get("sampling_aggressiveness")
    if result_level is None or int(result_level) != int(
        protocol_payload.get("sampling_aggressiveness", -1)
    ):
        raise ValueError(
            "result sampling aggressiveness does not match resolved manifest"
        )

    scale_payload = _read_json_object(
        scale_manifest,
        "SAMPLING_SCALE_MODEL.json",
    )
    if int(scale_payload.get("schema_version", -1)) != int(
        SAMPLING_SCALE_MODEL_SCHEMA_VERSION
    ):
        raise ValueError("unsupported sampling scale model schema")
    if int(scale_payload.get("iteration", -1)) != int(iteration):
        raise ValueError("sampling scale model iteration mismatch")
    audit_payload = _read_json_object(
        audit_manifest,
        "SAMPLING_PROTOCOL_AUDIT.json",
    )
    if int(audit_payload.get("iteration", -1)) != int(iteration):
        raise ValueError("sampling protocol audit iteration mismatch")

    level = int(protocol_payload.get("sampling_aggressiveness"))
    quality_gate_values = protocol_payload.get("resolved_quality_gates")
    if not isinstance(quality_gate_values, dict):
        raise ValueError("resolved sampling protocol quality-gates block is missing")
    quality_gate_overrides = dict(quality_gate_values)
    for persisted_name, campaign_name in (
        (
            "ariadne_max_displacement_ang",
            "ariadne_max_displacement_angstrom",
        ),
        (
            "ariadne_min_pair_distance_ang",
            "ariadne_min_pair_distance_angstrom",
        ),
    ):
        if persisted_name in quality_gate_overrides:
            quality_gate_overrides[campaign_name] = quality_gate_overrides.pop(
                persisted_name
            )
    quality_gates = _object_with_overrides(
        getattr(fallback_protocol, "quality_gates", None),
        quality_gate_overrides,
    )
    adversarial_safety = _object_with_overrides(
        getattr(fallback_protocol, "adversarial_safety", None),
        protocol_payload.get("resolved_adversarial_safety"),
    )
    anti_overlap_values = protocol_payload.get("resolved_anti_overlap")
    if not isinstance(anti_overlap_values, dict):
        raise ValueError("resolved sampling protocol anti-overlap block is missing")
    import copy

    effective_config = copy.copy(fallback_protocol.effective_config)
    effective_config.anti_overlap = _object_with_overrides(
        getattr(fallback_protocol.effective_config, "anti_overlap", None),
        anti_overlap_values,
    )
    diagnostics = {
        "sampling_protocol_source": "result_manifest",
        "sampling_scale_model_source": "result_manifest",
        "used_exact_sampling_protocol": True,
        "sampling_protocol_manifest": str(protocol_manifest),
        "sampling_scale_model_manifest": str(scale_manifest),
        "sampling_protocol_audit_manifest": str(audit_manifest),
    }
    return SimpleNamespace(
        schema_version=int(protocol_payload.get("schema_version")),
        iteration=int(iteration),
        sampling_aggressiveness=level,
        effective_config=effective_config,
        adversarial_safety=adversarial_safety,
        quality_gates=quality_gates,
        scale_model_payload=dict(scale_payload),
        manifest_path=protocol_manifest,
        scale_model_path=scale_manifest,
        replay_diagnostics=diagnostics,
    ), diagnostics


def clean_stale_ariadne_seed_outputs(
    campaign_dir,
    iteration: int,
    *,
    retry_array_task_ids: Optional[Sequence[int]] = None,
) -> List[str]:
    """Quarantine incomplete task directories before a bounded retry."""
    from datetime import datetime, timezone

    from ..layout import (
        active_iteration_dir,
        active_iteration_name,
        ariadne_seed_dir,
        ariadne_seeds_dir,
    )
    from ..seed_identity import read_ariadne_task_map, task_for_array_task_id
    from .ariadne_quarantine import (
        ensure_quarantine_capacity,
        prepare_quarantine_manifest,
        quarantine_root,
        write_quarantine_manifest,
    )

    campaign = Path(campaign_dir)
    iter_dir = active_iteration_dir(campaign, int(iteration))
    seeds_dir = ariadne_seeds_dir(iter_dir)
    if not seeds_dir.exists():
        return []
    if seeds_dir.is_symlink() or not seeds_dir.is_dir():
        raise BackendSubmissionError(
            "refusing to clean ARIADNE seeds path that is not a real directory: "
            + str(seeds_dir)
        )
    task_map = read_ariadne_task_map(iter_dir, expected_iteration=int(iteration))
    task_ids = (
        [int(value) for value in retry_array_task_ids]
        if retry_array_task_ids is not None
        else []
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    quarantine = (
        quarantine_root(campaign)
        / active_iteration_name(int(iteration))
        / stamp
    )
    moved: List[str] = []
    moved_sources: List[Path] = []
    candidates: List[Path] = []
    for task_id in task_ids:
        task = task_for_array_task_id(task_map, task_id)
        candidates.append(ariadne_seed_dir(iter_dir, int(task["seed_id"])))
    candidates.extend(sorted(seeds_dir.glob(".seed-*.partial-*")))
    retained_candidates: List[Path] = []
    seen_candidates = set()
    for candidate in candidates:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise BackendSubmissionError(
                "refusing to quarantine non-directory ARIADNE output: "
                + str(candidate)
            )
        try:
            candidate.resolve().relative_to(seeds_dir.resolve())
        except ValueError as exc:
            raise BackendSubmissionError(
                "refusing to quarantine ARIADNE output outside seeds directory: "
                + str(candidate)
            ) from exc
        identity = str(candidate.resolve())
        if identity not in seen_candidates:
            seen_candidates.add(identity)
            retained_candidates.append(candidate)
    if not retained_candidates:
        return []
    ensure_quarantine_capacity(campaign, retained_candidates)
    quarantine.mkdir(parents=True, exist_ok=False)
    targets = [quarantine / candidate.name for candidate in retained_candidates]
    prepare_quarantine_manifest(
        campaign,
        quarantine,
        iteration=int(iteration),
        source_paths=retained_candidates,
        target_paths=targets,
    )
    for candidate, target in zip(retained_candidates, targets):
        shutil.move(str(candidate), str(target))
        moved_sources.append(candidate)
        moved.append(str(target))
    if moved:
        write_quarantine_manifest(
            campaign,
            quarantine,
            iteration=int(iteration),
            source_paths=moved_sources,
            target_paths=[Path(value) for value in moved],
        )
    return moved


def archive_stale_ariadne_publication(
    campaign_dir,
    state,
    *,
    retry_task_ids: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Retire derived batch files before retry or postprocess replay."""
    from .ariadne_publication import (
        AriadnePublicationError,
        archive_ariadne_publication,
        classify_ariadne_publication,
    )
    from .submission_intent import load_intent

    campaign = Path(campaign_dir)
    iteration = int(getattr(state, "iteration", 0))
    try:
        classification = classify_ariadne_publication(
            campaign,
            iteration,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if str(classification.get("state") or "") == "invalid":
            raise AriadnePublicationError(
                str(classification.get("reason") or "invalid publication")
            )
        publication_state = str(classification.get("state") or "")
        if publication_state == "complete" and not bool(
            classification.get("accepted", False)
        ):
            raise AriadnePublicationError(
                "complete rejected ARIADNE batch decision is preserved for user review"
            )
        force = bool(classification.get("files")) and (
            bool(retry_task_ids)
            or (
                publication_state == "complete"
                and bool(classification.get("accepted", False))
            )
        )
        if not bool(classification.get("archive_required", False)) and not force:
            return None
        intent = load_intent(
            campaign,
            CampaignPhase.ARIADNE_ARRAY.value,
            iteration,
            expected_campaign_uid=str(state.campaign_uid),
        )
        submission_identity = (
            str(intent.get("submission_identity"))
            if isinstance(intent, dict) and intent.get("submission_identity")
            else None
        )
        return archive_ariadne_publication(
            campaign,
            iteration,
            reason=(
                "ariadne_retry_preparation"
                if retry_task_ids
                else "ariadne_postprocess_only_recovery"
            ),
            campaign_uid=str(state.campaign_uid),
            submission_identity=submission_identity,
            classification=classification,
            force=force,
        )
    except AriadnePublicationError as exc:
        raise BackendSubmissionError(
            "ARIADNE derived publication could not be archived safely: " + str(exc)
        ) from exc

#  SBATCH-phase postprocess refusal guard.
#
#
# Each parser registers itself by adding its phase name
# to this frozenset. Until then, postprocess() raises NotImplementedError
# with a hint pointing the user at a separate dry-run campaign.
LIVE_POSTPROCESS_IMPLEMENTED: frozenset = frozenset({
    "INITIAL_GAUSSIAN", "GAUSSIAN",
    "INITIAL_AIMALL", "AIMALL",
    "INITIAL_REPLACEMENT_GAUSSIAN", "REPLACEMENT_GAUSSIAN",
    "INITIAL_REPLACEMENT_AIMALL", "REPLACEMENT_AIMALL",
    "INITIAL_FEREBUS", "FEREBUS",
    "ARIADNE_ARRAY",
    "PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY",
})


#Validator functions for live parser pointdir inspection.
#Each takes a PointDirectory and returns (ok: bool, reason: str). The
#reason is a short tag like "scf_nonconvergence" or "missing_wfn" used
#in the quantum_output_rejected journal event.


def validate_gaussian_completed(pdir) -> tuple:
    """Validate one complete, fixed-geometry Gaussian force/WFN task."""
    from ichor.core.files.gaussian.gaussian_output import GaussianOutput
    from ichor.core.files.gaussian.gjf import GJF
    from ichor.core.files.gaussian.wfn import WFN

    root = Path(getattr(pdir, "path", pdir))
    if root.is_symlink() or not root.is_dir():
        return False, "gaussian_pointdir_missing_or_symlinked"

    def regular_matches(pattern):
        return sorted(
            path
            for path in root.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )

    output_paths = regular_matches("*.gau") + regular_matches("*.gaussianoutput")
    if len(output_paths) != 1:
        return False, "missing_or_ambiguous_gaussian_output"
    output_path = output_paths[0]
    try:
        output_text = output_path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return False, "gaussian_output_unreadable"
    meaningful_lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if (
        not meaningful_lines
        or "Normal termination of Gaussian" not in meaningful_lines[-1]
        or output_text.rfind("Error termination") > output_text.rfind("Normal termination")
    ):
        return False, "scf_nonconvergence_or_crash"

    gjf_paths = regular_matches("*.gjf")
    if len(gjf_paths) != 1:
        return False, "missing_or_ambiguous_gjf"
    wfn_paths = regular_matches("*.wfn")
    if len(wfn_paths) != 1:
        return False, "missing_or_ambiguous_wfn"
    gjf_path = gjf_paths[0]
    wfn_path = wfn_paths[0]
    try:
        gjf = GJF(gjf_path)
        gjf_atoms = list(gjf.atoms)
        keywords = [str(value).lower() for value in gjf.keywords]
        mandatory = {"nosymm", "output=wfn", "force", "geom=notest"}
        if not mandatory.issubset(set(keywords)):
            return False, "gjf_route_contract_mismatch"
        extras = [
            value
            for value in gjf.keywords
            if str(value).lower() not in mandatory
        ]
        normalise_gaussian_route_keywords(extras)
        nonblank = [
            line.strip()
            for line in gjf_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not nonblank or Path(nonblank[-1]).name != wfn_path.name:
            return False, "gjf_wfn_destination_mismatch"
    except Exception:
        return False, "gjf_parse_or_route_failure"

    try:
        wfn = WFN(wfn_path)
        wfn_atoms = list(wfn.atoms.to_angstroms())
        if not math.isfinite(float(wfn.total_energy)) or not math.isfinite(
            float(wfn.virial_ratio)
        ):
            return False, "wfn_energy_or_virial_nonfinite"
    except Exception:
        return False, "wfn_parse_failure"

    try:
        gaussian_output = GaussianOutput(output_path)
        output_atoms = list(gaussian_output.atoms)
        output_forces = dict(gaussian_output.global_forces)
        output_charge = int(gaussian_output.charge)
        output_multiplicity = int(gaussian_output.multiplicity)
    except Exception:
        return False, "gaussian_output_semantic_parse_failure"

    if output_charge != int(gjf.charge) or output_multiplicity != int(
        gjf.spin_multiplicity
    ):
        return False, "gaussian_output_charge_or_multiplicity_mismatch"
    if not gjf_atoms or len(output_atoms) != len(gjf_atoms):
        return False, "gaussian_output_geometry_count_mismatch"
    if len(output_forces) != len(gjf_atoms):
        return False, "gaussian_output_force_count_mismatch"
    if set(output_forces) != {str(atom.name) for atom in output_atoms}:
        return False, "gaussian_output_force_identity_mismatch"
    if any(
        not np.all(np.isfinite(np.asarray(force, dtype=float)))
        for force in output_forces.values()
    ):
        return False, "gaussian_output_force_nonfinite"

    gjf_types = [str(atom.type).capitalize() for atom in gjf_atoms]
    wfn_types = [str(atom.type).capitalize() for atom in wfn_atoms]
    output_types = [str(atom.type).capitalize() for atom in output_atoms]
    if gjf_types != wfn_types:
        return False, "wfn_gjf_atom_order_mismatch"
    if gjf_types != output_types:
        return False, "gaussian_output_gjf_atom_order_mismatch"
    gjf_coords = np.asarray([atom.coordinates for atom in gjf_atoms], dtype=float)
    wfn_coords = np.asarray([atom.coordinates for atom in wfn_atoms], dtype=float)
    output_coords = np.asarray(
        [atom.coordinates for atom in output_atoms],
        dtype=float,
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (gjf_coords, wfn_coords, output_coords)
    ):
        return False, "gaussian_geometry_nonfinite"
    if float(np.max(np.abs(gjf_coords - wfn_coords))) > 1.0e-4:
        return False, "wfn_gjf_geometry_mismatch"
    if float(np.max(np.abs(gjf_coords - output_coords))) > 1.0e-4:
        return False, "gaussian_output_gjf_geometry_mismatch"
    return True, ""


def validate_aimall_completed(pdir) -> tuple:
    """Return (True, "") if the pointdir contains a complete AIMAll output;
    (False, reason_tag) otherwise. Validates: *_atomicfiles/ directory
    exists; each .int file parses without raising."""
    ints = getattr(pdir, "ints", None)
    if ints is None or not getattr(ints, "path", None):
        return False, "missing_atomicfiles_dir"
    int_path = Path(ints.path)
    if not int_path.is_dir():
        return False, "atomicfiles_not_a_dir"
    #IntDirectory auto discovers .int files; iterate + trigger parse on each.
    try:
        n_int = 0
        for int_file in ints.ints:
         #int_file is an Int instance; touching net_charge triggers the
         # lazy parse and raises on malformed input.
            _ = int_file.net_charge
            n_int += 1
    except Exception:
        return False, "int_parse_failure"
    if n_int == 0:
        return False, "no_int_files"
    # one .int per atom. a partial AIMAll (crashed after a few atoms, or one atom whose integration
    # failed) leaves fewer .int than there are atoms -- the old n_int>=1 check waved that through and
    # then the FEREBUS feature export later choked on the point missing IQA for some atoms (A36). so
    # demand one per atom and reject unreadable geometry rather than accepting a point we cannot
    # count.
    try:
        n_atoms = len(pdir.atoms)
    except Exception:
        return False, "aimall_geometry_unreadable"
    if n_int != n_atoms:
        return False, "aimall_partial_" + str(n_int) + "_of_" + str(n_atoms) + "_int"
    return True, ""


def _aimall_visibility_issue(pointdir: Path) -> Optional[str]:
    """Return a non-mutating shared-filesystem readiness problem, if any."""
    root = Path(pointdir)
    try:
        from ichor.core.files.point_directory import PointDirectory

        n_atoms = len(PointDirectory(root).atoms)
    except Exception:
        return "geometry_unreadable"
    try:
        atomic_directories = [
            child
            for child in root.iterdir()
            if child.name.endswith("_atomicfiles")
        ]
    except OSError:
        return "atomicfiles_directory_unreadable"
    if len(atomic_directories) != 1:
        return "missing_or_ambiguous_atomicfiles_directory"
    atomic_directory = atomic_directories[0]
    if atomic_directory.is_symlink() or not atomic_directory.is_dir():
        return "atomicfiles_directory_missing_or_symlinked"
    int_paths = sorted(
        path
        for path in atomic_directory.glob("*.int")
        if "_" not in path.name
    )
    if len(int_paths) != int(n_atoms):
        return (
            "missing_or_partial_int_set_"
            + str(len(int_paths))
            + "_of_"
            + str(n_atoms)
        )
    for int_path in int_paths:
        if int_path.is_symlink() or not int_path.is_file():
            return "int_file_missing_or_symlinked:" + int_path.name
        try:
            text = int_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            return "int_file_unreadable:" + int_path.name
        if "Total time" not in text:
            return "int_file_incomplete:" + int_path.name
    return None


def _pointdir_index(name: str) -> Optional[int]:
    if not (name.startswith("POINT_") and name.endswith(".pointdir")):
        return None
    try:
        return int(name[len("POINT_"):-len(".pointdir")])
    except ValueError:
        return None


def _next_pointdir_index(root) -> int:
    indexes = []
    for child in Path(root).glob("POINT_*.pointdir"):
        idx = _pointdir_index(child.name)
        if idx is not None:
            indexes.append(idx)
    return (max(indexes) + 1) if indexes else 0


def _ferebus_model_data_rows_ok(model_path) -> tuple:
    """spot a truncated .model. FEREBUS declares number_of_training_points N in the header then
    writes an [training_data.x] block of N rows. if it died mid-write the block is short, and the
    core reader fills the gap with uninitialised np.empty memory instead of erroring (A56) -- so
    the GP would quietly train on garbage. that reader lives in ichor_core/models which we are not
    allowed to touch, so we catch it here: parse the declared count, count the rows ourselves,
    reject anything short.
    """
    ntrain = None
    rows = None
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            it = iter(f)
            for line in it:
                if "number_of_training_points" in line:
                    try:
                        ntrain = int(line.split()[1])
                    except (IndexError, ValueError):
                        return False, "model_ntrain_unparseable"
                if "[training_data.x]" in line:
                    rows = 0
                    for row in it:
                        if row.strip() == "":
                            break
                        rows += 1
                    break  # the x block is enough to spot a truncation
    except OSError:
        return False, "model_file_unreadable"
    if ntrain is None or rows is None:
        # no FEREBUS header/data block we recognise. that is NOT the A56 case (which is a header that
        # claims N rows followed by fewer than N) -- we just cannot assess truncation here, so do not
        # block on it. size>0 already passed and a genuinely corrupt model fails loudly at load.
        return True, ""
    if rows < ntrain:
        return False, "model_truncated"
    return True, ""


def _read_ferebus_model_metadata(model_path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("["):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0]
                if key == "name":
                    metadata["system"] = parts[1]
                elif key == "atom":
                    metadata["atom"] = parts[1]
                elif key == "property":
                    metadata["property"] = parts[1]
                elif key == "ALF":
                    try:
                        metadata["alf_1_indexed"] = [int(x) for x in parts[1:4]]
                    except ValueError:
                        metadata["alf_1_indexed"] = []
                elif key == "number_of_training_points":
                    try:
                        metadata["ntrain"] = int(parts[1])
                    except ValueError:
                        metadata["ntrain"] = None
    except OSError:
        metadata["unreadable"] = True
    return metadata


def validate_ferebus_completed(staging_dir) -> tuple:
    """Validate the exact pyferebus task manifest and expected model set."""
    staging = Path(staging_dir)
    try:
        from .model_contract import (
            ModelContractError,
            validate_ferebus_model_contract,
        )
        validate_ferebus_model_contract(staging, committed=False)
        from .ferebus_task_runner import validate_task_receipts

        validate_task_receipts(staging)
    except FileNotFoundError as exc:
        return False, "ferebus_manifest_invalid: " + type(exc).__name__ + ": " + str(exc)
    except ModelContractError as exc:
        if str(exc).startswith("ferebus_model_root_missing"):
            return False, "ferebus_staging_missing"
        return False, str(exc)
    except Exception as exc:
        return False, "ferebus_manifest_invalid: " + type(exc).__name__ + ": " + str(exc)
    return True, ""




def _high_quantile(vals, q=0.9):
    """Return the nearest-rank observed high quantile (p90 by default)."""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    if not all(math.isfinite(value) for value in s):
        raise ValueError("high-quantile inputs must be finite")
    quantile = float(q)
    if not math.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("high quantile must be in (0, 1]")
    rank = max(1, int(math.ceil(quantile * len(s))))
    return s[rank - 1]


def _ariadne_geometry_quality(result_dict: Dict[str, Any], validated: Dict[str, Any], gates: Any) -> Dict[str, Any]:
    import numpy as _np

    reasons = []
    final = _np.asarray(validated.get("final_coordinates"), dtype=float)
    metrics: Dict[str, Any] = {
        "max_displacement_ang": None,
        "min_pair_distance_ang": None,
        "pair_distance_applicable": bool(final.ndim == 2 and final.shape[0] >= 2),
    }
    if final.ndim != 2 or final.shape[1] != 3 or not _np.all(_np.isfinite(final)):
        reasons.append("ariadne_final_geometry_nonfinite")
        return {"accepted": False, "reasons": reasons, "metrics": metrics}
    if final.shape[0] >= 2:
        dmins = []
        for i in range(final.shape[0]):
            for j in range(i + 1, final.shape[0]):
                dmins.append(float(_np.linalg.norm(final[i] - final[j])))
        metrics["min_pair_distance_ang"] = min(dmins) if dmins else None
    initial_raw = result_dict.get("initial_coordinates")
    try:
        initial = _np.asarray(initial_raw, dtype=float)
        if initial.shape == final.shape and _np.all(_np.isfinite(initial)):
            displacements = _np.linalg.norm(final - initial, axis=1)
            metrics["max_displacement_ang"] = float(_np.max(displacements))
    except Exception:
        metrics["max_displacement_ang"] = None
    max_disp = getattr(gates, "ariadne_max_displacement_angstrom", None)
    if max_disp is None:
        reasons.append("ariadne_max_displacement_gate_unresolved")
    elif metrics["max_displacement_ang"] is None:
        reasons.append("ariadne_max_displacement_metric_unavailable")
    elif float(metrics["max_displacement_ang"]) > float(max_disp):
        reasons.append("ariadne_max_displacement_threshold_exceeded")
    min_pair = getattr(gates, "ariadne_min_pair_distance_angstrom", None)
    if min_pair is None:
        reasons.append("ariadne_min_pair_distance_gate_unresolved")
    elif not metrics["pair_distance_applicable"]:
        pass
    elif metrics["min_pair_distance_ang"] is None:
        reasons.append("ariadne_min_pair_distance_metric_unavailable")
    elif float(metrics["min_pair_distance_ang"]) < float(min_pair):
        reasons.append("ariadne_min_pair_distance_threshold_exceeded")
    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics}


def _ariadne_landing_audit_summary(seed_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "accepted": 0,
        "salvaged": 0,
        "backtracked": 0,
        "rejected": 0,
        "handoff_accepted": 0,
        "handoff_rejected": 0,
        "rejection_reasons": {},
        "handoff_rejection_reasons": {},
        "policies": {},
    }
    unique_records: Dict[str, Dict[str, Any]] = {}
    for pos, rec in enumerate(seed_records):
        key = str(rec.get("seed_id", "__pos_" + str(pos)))
        unique_records[key] = rec
    for rec in unique_records.values():
        safety = rec.get("landing_safety")
        if not isinstance(safety, dict):
            summary["rejected"] += 1
            reasons = [str(rec.get("reason", "missing_landing_safety"))]
            policy = "missing_landing_safety"
        else:
            accepted = bool(safety.get("accepted", False))
            policy = str(safety.get("policy", "unknown"))
            reasons = [str(r) for r in safety.get("reasons", [])]
            if accepted:
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
            if policy == "salvaged_iterate":
                summary["salvaged"] += 1
            if policy == "backtracked":
                summary["backtracked"] += 1
        policies = summary["policies"]
        policies[policy] = int(policies.get(policy, 0)) + 1
        if reasons:
            bucket = summary["rejection_reasons"]
            for reason in reasons:
                bucket[reason] = int(bucket.get(reason, 0)) + 1
        handoff_accepted = rec.get("handoff_accepted")
        if handoff_accepted is True:
            summary["handoff_accepted"] += 1
        elif handoff_accepted is False:
            summary["handoff_rejected"] += 1
            reason = str(
                rec.get(
                    "handoff_rejection_reason",
                    rec.get("reason", "handoff_rejected"),
                )
            )
            bucket = summary["handoff_rejection_reasons"]
            bucket[reason] = int(bucket.get(reason, 0)) + 1
    return summary


def _ariadne_optional_diagnostic_warnings(result_dict: Dict[str, Any]) -> List[str]:
    """Return warnings for optional ARIADNE diagnostics.

    These fields are telemetry, not handoff contract. Reconcile must keep old
    result.json files readable and must not reject an otherwise safe landing
    because an optional scale diagnostic was malformed.
    """
    diagnostics = result_dict.get("optimiser_diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    numeric_fields = (
        "trqn_objective_scale",
        "trqn_target_initial_grad_norm",
        "trqn_initial_raw_grad_norm",
        "trqn_initial_scaled_grad_norm",
        "trqn_retry_objective_scale",
        "trqn_retry_target_initial_grad_norm",
        "trqn_retry_raw_grad_norm",
        "trqn_retry_scaled_grad_norm",
    )
    warnings: List[str] = []
    for field in numeric_fields:
        if field not in diagnostics or diagnostics.get(field) is None:
            continue
        try:
            value = float(diagnostics.get(field))
        except (TypeError, ValueError):
            warnings.append(field + "_not_numeric")
            continue
        if not math.isfinite(value):
            warnings.append(field + "_not_finite")
    mode = diagnostics.get("trqn_scale_mode")
    if mode is not None and str(mode) not in {
        "not_applicable",
        "off",
        "fixed",
        "adaptive_initial_gradient",
        "adaptive_initial_gradient_rms",
    }:
        warnings.append("trqn_scale_mode_unknown")
    return warnings


class LiveBackendNotAvailableError(RuntimeError):
    """Raised when live mode is requested but a required backend is missing."""


def locate_gaussian_sample_xyz(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
    *,
    replacement_round: int = 0,
    campaign_uid: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the producer-owned geometry sample for a Gaussian phase."""
    camp = Path(campaign_dir)
    if phase_name in {
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_GAUSSIAN",
    }:
        from ..replacement_sampling import ensure_replacement_sample_strict

        context = (
            "bootstrap"
            if phase_name == "INITIAL_REPLACEMENT_GAUSSIAN"
            else "active"
        )
        manifest = ensure_replacement_sample_strict(
            camp,
            context=context,
            iteration=0 if context == "bootstrap" else int(iteration),
            replacement_round=int(replacement_round),
            expected_campaign_uid=(
                str(campaign_uid).strip() if campaign_uid is not None else None
            ),
        )
        return Path(str(manifest["sample_xyz"]))
    if phase_name == "INITIAL_GAUSSIAN":
        from ..handoff_manifests import read_phase_a_sample_manifest
        from ..layout import bootstrap_selection_dir
        from .state import DEFAULT_STATE_FILENAME, read_state

        outdir = bootstrap_selection_dir(camp)
        try:
            expected_campaign_uid = str(campaign_uid or "").strip()
            if not expected_campaign_uid:
                current_state = read_state(
                    camp / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
                )
                expected_campaign_uid = str(current_state.campaign_uid)
            manifest = read_phase_a_sample_manifest(
                outdir,
                expected_campaign_uid=expected_campaign_uid,
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "phase_a_sample_manifest_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        return Path(str(manifest["sample_xyz"]))
    from ..layout import active_iteration_dir, active_phase_b_dir

    iter_dir = active_iteration_dir(camp, int(iteration))
    candidate = active_phase_b_dir(iter_dir) / "selected.xyz"
    return candidate if candidate.is_file() else None


@dataclass
class LiveBackendsPhaseExecutor(DryRunPhaseExecutor):
    """Production PhaseExecutor.

    Compared with :class: DryRunPhaseExecutor:

    * "submit_or_run" for SBATCH phases writes a real sbatch script
      (with module loads + backend invocation) and submits it via
      "sbatch --parsable".
    * "postprocess" for SBATCH phases parses real output files (Gaussian
      .log + .wfn, AIMAll .int, FEREBUS .model) into ICHOR data
      structures and runs the atomic append pipeline.

    Inline phases (SEED_SELECT / SPLIT / REFERENCE_COMMIT / STOP_CHECK) are inherited
    from the dry-run executor since they do not change.
    """

    sbatch_runner: Callable[..., Any] = subprocess.run
    walltime_hours: Optional[float] = None
    partition: Optional[str] = None
    backend_check: bool = True
    strict_committed_artifact_verification: bool = True
    scheduler_identity_kind: str = "slurm"
    strict_completion_receipt_evidence: bool = True

    def __post_init__(self) -> None:
        DryRunPhaseExecutor.__post_init__(self)
        self.scheduler_identity_kind = _configured_scheduler()
        self._scheduler_backend = get_scheduler_backend(
            self.scheduler_identity_kind
        )
        if self.backend_check:
            avail = check_backends()
            if not avail.all_present:
                raise LiveBackendNotAvailableError(missing_backend_message(avail))

    def handle_failure(self, state, phase, observations) -> FailureAction:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        if phase_name in ("ARIADNE_ARRAY", "PHASE_B_DIVERSITY"):
            return FailureAction.HALT
        return super().handle_failure(state, phase, observations)

    # --- SBATCH submission ---------------------------------------------

    def _locate_sample_xyz(self, phase_name, iteration, replacement_round=0):
        from .filesystem import operational_path
        from .state import read_state

        state = read_state(operational_path(self.campaign_dir, "state.json"))
        return locate_gaussian_sample_xyz(
            self.campaign_dir,
            str(phase_name),
            int(iteration),
            replacement_round=int(replacement_round),
            campaign_uid=str(state.campaign_uid),
        )

    def _count_seeds(self, iteration):
        from ..handoff_manifests import load_seeds_picked

        from ..layout import active_iteration_dir

        iter_dir = active_iteration_dir(self.campaign_dir, int(iteration))
        try:
            data = load_seeds_picked(iter_dir, expected_iteration=int(iteration))
        except Exception as exc:
            raise BackendSubmissionError(
                "seed_selection/SELECTION.json unreadable: " + str(exc)
            )
        return int(data.get("n_picked", 0))

    def _seed_dir_for_record(self, iteration: int, seed_record: Dict[str, Any]) -> Path:
        from ..layout import active_iteration_dir, ariadne_seed_dir

        return ariadne_seed_dir(
            active_iteration_dir(self.campaign_dir, int(iteration)),
            int(seed_record["seed_id"]),
        )

    @staticmethod
    def _seed_record_frame_id(seed_record: Dict[str, Any]) -> Optional[int]:
        raw = seed_record.get("frame_id")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE seed record frame_id is not an integer: " + repr(raw)
            ) from exc

    @staticmethod
    def _seed_record_variance(seed_record: Dict[str, Any]) -> Optional[float]:
        raw = seed_record.get("variance_at_selection")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE seed record variance_at_selection is not numeric: "
                + repr(raw)
            ) from exc
        return value if math.isfinite(value) else None

    @staticmethod
    def _seed_record_subspace_payload(seed_record: Dict[str, Any]) -> tuple:
        neighbours = (
            seed_record.get("subspace_neighbour_frame_ids")
            if isinstance(seed_record.get("subspace_neighbour_frame_ids"), list)
            else seed_record.get("neighbour_frame_ids")
        )
        if not isinstance(neighbours, list):
            neighbours = []
        eigenvalues = seed_record.get("subspace_eigenvalues")
        if not isinstance(eigenvalues, list):
            eigenvalues = []
        raw_dimension = seed_record.get("subspace_dimension")
        if raw_dimension is None:
            raw_dimension = len(eigenvalues) if eigenvalues else 0
        try:
            dimension = int(raw_dimension)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE seed record subspace_dimension is not an integer: "
                + repr(raw_dimension)
            ) from exc
        return neighbours, dimension, eigenvalues

    def _ensure_ariadne_seed_provenance(
        self,
        state,
        picked: Dict[str, Any],
        seed_record: Dict[str, Any],
    ) -> tuple:
        """Ensure one live ARIADNE seed directory has its provenance sidecar.

        Dry-run has always written this before enriching ARIADNE/Phase-B
        blocks. Live mode must do the same because downstream Gaussian staging
        validates and copies the sidecar into the labelled pointdir.
        """
        iteration = int(state.iteration)
        seed_dir = self._seed_dir_for_record(iteration, seed_record)
        prov_path = seed_dir / PROVENANCE_FILENAME
        seed_frame_id = self._seed_record_frame_id(seed_record)
        seed_id = int(seed_record["seed_id"])
        seed_uid = str(seed_record["seed_uid"])
        array_task_id = seed_id - 1
        trajectory_sha = str(picked.get("trajectory_sha256", "") or "")
        trajectory_sha_for_validation = trajectory_sha if trajectory_sha else None
        if prov_path.is_file():
            try:
                validate_provenance(
                    seed_dir,
                    campaign_uid=str(getattr(state, "campaign_uid", "")),
                    iteration=iteration,
                    trajectory_sha256=trajectory_sha_for_validation,
                    seed_frame_id=seed_frame_id,
                    seed_id=seed_id,
                    seed_uid=seed_uid,
                    array_task_id_zero_based=array_task_id,
                )
            except Exception as exc:
                raise BackendSubmissionError(
                    "ARIADNE seed provenance invalid for "
                    + seed_dir.name
                    + ": "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            else:
                return prov_path, False

        neighbours, dimension, eigenvalues = self._seed_record_subspace_payload(seed_record)
        try:
            write_seed_provenance(
                seed_dir,
                campaign_uid=str(getattr(state, "campaign_uid", "")),
                iteration=iteration,
                trajectory_sha256=trajectory_sha,
                seed_frame_id=seed_frame_id,
                seed_id=seed_id,
                seed_uid=seed_uid,
                array_task_id_zero_based=array_task_id,
                seed_selection_origin=str(seed_record.get("selection_origin", "unknown")),
                seed_variance_at_selection=self._seed_record_variance(seed_record),
                subspace_neighbour_frame_ids=neighbours,
                subspace_dimension=dimension,
                subspace_eigenvalues=eigenvalues,
                mode_weighting_policy=self._mode_weighting_policy_or_default(),
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "ARIADNE seed provenance write failed for "
                + seed_dir.name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        return prov_path, True

    def _ensure_ariadne_seed_provenance_for_iteration(self, state) -> int:
        from ..handoff_manifests import load_seeds_picked
        from ..layout import active_iteration_dir

        iteration = int(state.iteration)
        iter_dir = active_iteration_dir(self.campaign_dir, iteration)
        try:
            picked = load_seeds_picked(iter_dir, expected_iteration=iteration)
        except Exception as exc:
            raise BackendSubmissionError(
                "seed_selection/SELECTION.json unreadable: " + str(exc)
            )
        created = 0
        for seed_record in list(picked.get("seed_records", [])):
            _path, was_created = self._ensure_ariadne_seed_provenance(
                state,
                picked,
                seed_record,
            )
            if was_created:
                created += 1
        if created:
            self._journal_event(
                "ariadne_seed_provenance_staged",
                iteration=iteration,
                n_created=int(created),
            )
        return int(picked.get("n_picked", 0))

    def _array_size_after_staging(self, phase_name, state):
        """Stage this phase's per-point inputs and return the SLURM array size.
        None for single-job phases (FEREBUS / diversity) so no --array is emitted."""
        import inspect

        from . import input_staging as _stg

        def progress_kwargs(staging_callable: Callable[..., Any]) -> Dict[str, Any]:
            try:
                parameters = inspect.signature(staging_callable).parameters
            except (TypeError, ValueError):
                return {}
            if "progress_callback" not in parameters:
                return {}
            return {"progress_callback": self._report_runtime_progress}

        camp = Path(self.campaign_dir)
        it = int(state.iteration)
        effective_partition = (
            str(self.partition)
            if self.partition is not None
            else str(self.config.resources.partition_for(phase_name))
        )
        if phase_name == "PHASE_A_DIVERSITY":
            from .pool_feasibility import require_pool_feasibility

            feasibility = require_pool_feasibility(camp, self.config)
            self._journal_event(
                "pool_feasibility_checked",
                phase=phase_name,
                **feasibility.to_dict(),
            )
            return None
        if "GAUSSIAN" in phase_name:
            sample = self._locate_sample_xyz(
                phase_name,
                it,
                replacement_round=int(getattr(state, "replacement_round", 0)),
            )
            if sample is None:
                raise BackendSubmissionError(
                    "no diversity sample to stage for " + phase_name
                )
            _, n = _stg.stage_gaussian_inputs(
                camp,
                self.config,
                phase_name,
                it,
                sample,
                campaign_uid=str(getattr(state, "campaign_uid", "")),
                **progress_kwargs(_stg.stage_gaussian_inputs),
            )
            return n
        if "AIMALL" in phase_name:
            is_replacement = "REPLACEMENT" in phase_name
            if is_replacement:
                from ..replacement_sampling import replacement_round_dir

                context = "bootstrap" if phase_name.startswith("INITIAL_") else "active"
                staging_override = replacement_round_dir(
                    camp,
                    context=context,
                    iteration=0 if context == "bootstrap" else it,
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                )
                expected_gaussian_phase = (
                    "INITIAL_REPLACEMENT_GAUSSIAN"
                    if context == "bootstrap"
                    else "REPLACEMENT_GAUSSIAN"
                )
            else:
                staging_override = None
                expected_gaussian_phase = None
            _, n = _stg.stage_aimall_inputs(
                camp,
                self.config,
                phase_name,
                it,
                partition_override=effective_partition,
                staging_override=staging_override,
                expected_gaussian_phase=expected_gaussian_phase,
                **progress_kwargs(_stg.stage_aimall_inputs),
            )
            return n
        if phase_name == "ARIADNE_ARRAY":
            from ..layout import active_iteration_dir
            from ..seed_identity import read_ariadne_task_map

            task_map = read_ariadne_task_map(
                active_iteration_dir(self.campaign_dir, it),
                expected_iteration=it,
            )
            n = int(task_map["n_tasks"])
            if n < 1:
                raise BackendSubmissionError("ARIADNE task map is empty")
            try:
                from ..sampling_protocol import resolve_or_load_sampling_protocol

                resolved_protocol = resolve_or_load_sampling_protocol(
                    self.campaign_dir,
                    self.config,
                    iteration=it,
                )
                payload = dict(resolved_protocol.geometry_scale_payload)
            except Exception as exc:
                raise BackendSubmissionError(
                    "sampling protocol resolution failed before ARIADNE_ARRAY "
                    + "submission: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            self._journal_event(
                "sampling_protocol_resolved",
                iteration=it,
                sampling_aggressiveness=int(
                    resolved_protocol.sampling_aggressiveness
                ),
                sampling_policy_version=int(
                    resolved_protocol.sampling_policy_version
                ),
                sampling_policy_table_sha256=str(
                    resolved_protocol.sampling_policy_table_sha256
                ),
                target_motion_ratio=(
                    resolved_protocol.scale_model_payload.get(
                        "trust_radius_policy",
                        {},
                    ).get("target_motion_ratio")
                ),
                initial_trust_multiplier=(
                    resolved_protocol.scale_model_payload.get(
                        "trust_radius_policy",
                        {},
                    ).get("aggressiveness_multiplier")
                ),
                history_records=(
                    resolved_protocol.scale_model_payload.get("history", {}).get(
                        "n_records"
                    )
                ),
                history_baseline_source=(
                    resolved_protocol.scale_model_payload.get(
                        "geometry_motion_scale",
                        {},
                    ).get("source")
                ),
                scale_angstrom=payload.get("scale_angstrom"),
                scale_resolution_mode=payload.get("scale_resolution_mode"),
                n_values=payload.get("n_values"),
                resolved_manifest=(
                    None if resolved_protocol.manifest_path is None
                    else str(resolved_protocol.manifest_path)
                ),
                scale_model_manifest=(
                    None if resolved_protocol.scale_model_path is None
                    else str(resolved_protocol.scale_model_path)
                ),
                audit_manifest=(
                    None if resolved_protocol.audit_manifest_path is None
                    else str(resolved_protocol.audit_manifest_path)
                ),
            )
            return n
        return None

    def _scheduler_cancel_recovery_sources(
        self,
        state,
        phase_name: str,
    ) -> Sequence[Dict[str, Any]]:
        try:
            return scheduler_terminal_recoveries(
                self.campaign_dir,
                campaign_uid=str(state.campaign_uid),
                phase=phase_name,
                iteration=int(state.iteration),
                replacement_round=int(getattr(state, "replacement_round", 0)),
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "scheduler cancellation evidence is invalid for "
                + str(phase_name)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

    def _raise_if_immediate_cancel_before_submission(self, state) -> None:
        """Close the PRE_SUBMIT race immediately before scheduler acceptance."""
        from .stop_control import read_stop_request

        request = read_stop_request(
            self.campaign_dir,
            expected_campaign_uid=str(state.campaign_uid),
        )
        if (
            isinstance(request, Mapping)
            and str(request.get("mode")) == "immediate"
            and bool(request.get("cancel_jobs_requested", False))
            and str(request.get("status")) in {"cancelling", "requested"}
        ):
            raise SubmissionCancelledBeforeSchedulerAcceptance(
                "immediate stop prevented scheduler submission"
            )

    def _scheduler_recovery_environment_assessments(
        self,
        recoveries: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Assess each producer attempt without turning uncertainty into reuse."""
        assessments: Dict[str, Dict[str, Any]] = {}
        for recovery in recoveries:
            intent = recovery["intent"]
            identity = str(intent.get("submission_identity") or "")
            if (
                recovery.get("receipt", {}).get("schema_version")
                != SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION
            ):
                assessments[identity] = {
                    "equivalent": False,
                    "reasons": ["legacy_scheduler_identity_unproven"],
                    "proof": None,
                    "path": None,
                    "sha256": None,
                }
                continue
            recorded_scheduler = str(
                intent.get("scheduler_identity_kind") or "slurm"
            ).strip().lower()
            if recorded_scheduler != str(self.scheduler_identity_kind):
                raise BackendSubmissionError(
                    "scheduler kind cannot change during "
                    + str(intent.get("phase") or "scheduler-backed phase")
                    + " recovery: recorded "
                    + recorded_scheduler
                    + ", active "
                    + str(self.scheduler_identity_kind)
                )
            try:
                assessments[identity] = assess_recovery_environment(
                    self.campaign_dir,
                    intent=intent,
                    current_scheduler_kind=self.scheduler_identity_kind,
                )
            except Exception as exc:
                assessments[identity] = {
                    "equivalent": False,
                    "reasons": [
                        "environment_equivalence_unproven:"
                        + type(exc).__name__
                        + ":"
                        + str(exc)[:180]
                    ],
                    "proof": None,
                    "path": None,
                    "sha256": None,
                }
        return assessments

    @staticmethod
    def _latest_scheduler_recovery_by_task(
        recoveries: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Mapping[str, Any]]:
        latest: Dict[int, Mapping[str, Any]] = {}
        for recovery in recoveries:
            for record in recovery["receipt"].get("outcomes", []):
                latest[int(record["logical_task_id"])] = recovery
        return latest

    @staticmethod
    def _environment_proof_records(
        assessments: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "submission_identity": str(identity),
                "equivalent": bool(assessment.get("equivalent", False)),
                "reasons": list(assessment.get("reasons") or []),
                "path": assessment.get("path"),
                "sha256": assessment.get("sha256"),
            }
            for identity, assessment in sorted(assessments.items())
        ]

    @staticmethod
    def _task_recovery_lineage(
        task_ids: Sequence[int],
        latest: Mapping[int, Mapping[str, Any]],
        assessments: Mapping[str, Mapping[str, Any]],
        *,
        reuse_basis: str,
    ) -> List[Dict[str, Any]]:
        records = []
        for task_id in sorted(int(value) for value in task_ids):
            recovery = latest.get(task_id)
            if not isinstance(recovery, Mapping):
                continue
            intent = recovery["intent"]
            identity = str(intent.get("submission_identity") or "")
            assessment = assessments.get(identity, {})
            records.append(
                {
                    "logical_task_id": int(task_id),
                    "reuse_basis": str(reuse_basis),
                    "attempt_id": str(intent.get("attempt_id") or ""),
                    "submission_identity": identity,
                    "job_id": str(intent.get("job_id") or ""),
                    "environment_generation": intent.get(
                        "environment_generation"
                    ),
                    "environment_generation_digest_sha256": intent.get(
                        "environment_generation_digest_sha256"
                    ),
                    "environment_equivalence_path": assessment.get("path"),
                    "environment_equivalence_sha256": assessment.get("sha256"),
                }
            )
        return records

    def _reconstruct_cancelled_quantum_receipts(
        self,
        state,
        phase_name: str,
        recoveries: Sequence[Mapping[str, Any]],
        assessments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Validate scheduler-completed quantum outputs and bind receipts."""
        from ichor.core.files.point_directory import PointDirectory

        from .quantum_task_contracts import quantum_task_contract
        from .quantum_task_receipts import (
            read_quantum_task_receipt,
            write_quantum_task_receipt_from_scheduler_terminal_receipt,
        )

        replacement_round = int(getattr(state, "replacement_round", 0))
        contract = quantum_task_contract(
            self.campaign_dir,
            phase_name,
            int(state.iteration),
            replacement_round=replacement_round,
            validate_points_file=True,
        )
        latest = self._latest_scheduler_recovery_by_task(recoveries)

        validators = self._validators_for(phase_name)
        reusable = []
        invalid = []
        for logical_id in sorted(latest):
            recovery = latest[logical_id]
            receipt = recovery["receipt"]
            identity = str(
                recovery["intent"].get("submission_identity") or ""
            )
            if not bool(
                assessments.get(identity, {}).get("equivalent", False)
            ):
                invalid.append(
                    {
                        "task_id": logical_id,
                        "reason": "environment_equivalence_unproven",
                    }
                )
                continue
            if logical_id not in {
                int(value)
                for value in receipt.get("completed_logical_task_ids", [])
            }:
                continue
            if logical_id >= len(contract.tasks):
                raise BackendSubmissionError(
                    "scheduler terminal evidence references a quantum task "
                    "outside the canonical task contract"
                )
            pointdir = Path(contract.tasks[logical_id].pointdir)
            valid = bool(pointdir.is_dir() and not pointdir.is_symlink())
            reason = "pointdir_missing_or_invalid"
            if valid:
                try:
                    candidate = PointDirectory(pointdir)
                    for validator in validators:
                        ok, observed_reason = validator(candidate)
                        if not ok:
                            valid = False
                            reason = str(
                                observed_reason or "structural_validation_failed"
                            )
                            break
                except Exception as exc:
                    valid = False
                    reason = type(exc).__name__ + ": " + str(exc)[:160]
            if not valid:
                invalid.append({"task_id": logical_id, "reason": reason})
                continue
            try:
                read_quantum_task_receipt(
                    pointdir,
                    phase_name=phase_name,
                    iteration=int(state.iteration),
                    logical_task_id=logical_id,
                    expected_campaign_uid=str(
                        recovery["intent"]["campaign_uid"]
                    ),
                    expected_attempt_id=str(
                        recovery["intent"]["attempt_id"]
                    ),
                    expected_submission_identity=str(
                        recovery["intent"]["submission_identity"]
                    ),
                    expected_job_id=str(
                        recovery["intent"]["job_id"]
                    ),
                )
            except (OSError, ValueError):
                write_quantum_task_receipt_from_scheduler_terminal_receipt(
                    self.campaign_dir,
                    pointdir,
                    phase_name=phase_name,
                    iteration=int(state.iteration),
                    logical_task_id=logical_id,
                    intent=recovery["intent"],
                    terminal_receipt=receipt,
                )
            reusable.append(logical_id)
        return {
            "scheduler_completed_candidates": len(
                {
                    int(record["logical_task_id"])
                    for recovery in recoveries
                    for record in recovery["receipt"].get("outcomes", [])
                    if record.get("status") == "COMPLETED"
                    and record.get("exit_code") == [0, 0]
                }
            ),
            "validated_reusable_task_ids": reusable,
            "invalid_completed_tasks": invalid,
        }

    def _prepare_scheduler_cancelled_array_recovery(
        self,
        state,
        phase_name: str,
    ) -> Optional[Dict[str, Any]]:
        recoveries = self._scheduler_cancel_recovery_sources(state, phase_name)
        if not recoveries:
            return None
        assessments = self._scheduler_recovery_environment_assessments(
            recoveries
        )
        latest = self._latest_scheduler_recovery_by_task(recoveries)
        forced_retry_ids = [
            int(task_id)
            for task_id, recovery in latest.items()
            if not bool(
                assessments.get(
                    str(
                        recovery["intent"].get("submission_identity") or ""
                    ),
                    {},
                ).get("equivalent", False)
            )
        ]
        validation: Dict[str, Any] = {
            "scheduler_completed_candidates": sum(
                int(item["receipt"].get("n_completed", 0))
                for item in recoveries
            ),
            "validated_reusable_task_ids": [],
            "invalid_completed_tasks": [],
        }
        if "GAUSSIAN" in phase_name or "AIMALL" in phase_name:
            validation = self._reconstruct_cancelled_quantum_receipts(
                state,
                phase_name,
                recoveries,
                assessments,
            )
        source_digest = hashlib.sha256(
            ",".join(
                str(item["receipt"]["receipt_sha256"]) for item in recoveries
            ).encode("ascii")
        ).hexdigest()
        result = prepare_retry_submission(
            self.campaign_dir,
            phase_name,
            int(state.iteration),
            forced_retry_task_ids=forced_retry_ids,
        )
        reusable_ids = [
            int(task["task_id"])
            for task in result.get("tasks", [])
            if bool(task.get("complete", False))
        ]
        retry_ids = [int(value) for value in result.get("retry_task_ids", [])]
        ledger = write_phase_recovery_ledger(
            self.campaign_dir,
            {
                "campaign_uid": str(state.campaign_uid),
                "phase": phase_name,
                "iteration": int(state.iteration),
                "replacement_round": int(
                    getattr(state, "replacement_round", 0)
                ),
                "source_receipt_sha256": source_digest,
                "source_terminal_receipts": [
                    {
                        "path": str(item["path"]),
                        "receipt_sha256": str(
                            item["receipt"]["receipt_sha256"]
                        ),
                        "job_id": str(item["receipt"].get("job_id") or ""),
                        "submission_identity": str(
                            item["receipt"]["submission_identity"]
                        ),
                    }
                    for item in recoveries
                ],
                "environment_equivalences": (
                    self._environment_proof_records(assessments)
                ),
                "scheduler_completed_candidates": int(
                    validation["scheduler_completed_candidates"]
                ),
                "reusable_logical_task_ids": reusable_ids,
                "retry_logical_task_ids": retry_ids,
                "invalid_completed_tasks": list(
                    validation["invalid_completed_tasks"]
                ),
                "recovery_lineage": self._task_recovery_lineage(
                    reusable_ids,
                    latest,
                    assessments,
                    reuse_basis=(
                        "self_authenticating_output"
                        if phase_name == "ARIADNE_ARRAY"
                        else "scheduler_completed_validated_output"
                    ),
                ),
            },
        )
        result["_scheduler_cancel_recovery"] = ledger
        return result

    def _prepare_scheduler_cancelled_ferebus_recovery(
        self,
        state,
        phase_name: str,
        staging: Path,
        *,
        n_tasks: int,
    ) -> Optional[Dict[str, Any]]:
        """Classify reusable FEREBUS tasks after an explicit cancellation."""
        from .ferebus_task_runner import (
            FerebusTaskRunnerError,
            validate_task_receipt,
        )

        recoveries = self._scheduler_cancel_recovery_sources(state, phase_name)
        if not recoveries:
            return None
        assessments = self._scheduler_recovery_environment_assessments(
            recoveries
        )
        latest = self._latest_scheduler_recovery_by_task(recoveries)
        scheduler_completed_candidates = {
            int(task_id)
            for task_id, recovery in latest.items()
            if int(task_id)
            in {
                int(value)
                for value in recovery["receipt"].get(
                    "completed_logical_task_ids",
                    [],
                )
            }
        }
        completed_candidates = {
            int(task_id)
            for task_id in scheduler_completed_candidates
            if bool(
                assessments.get(
                    str(
                        latest[int(task_id)]["intent"].get(
                            "submission_identity"
                        )
                        or ""
                    ),
                    {},
                ).get("equivalent", False)
            )
        }
        if any(
            not 0 <= task_id < int(n_tasks)
            for task_id in scheduler_completed_candidates
        ):
            raise BackendSubmissionError(
                "scheduler terminal evidence references a FEREBUS task outside "
                "the canonical task map"
            )
        reusable = []
        invalid = [
            {
                "task_id": int(task_id),
                "reason": "environment_equivalence_unproven",
            }
            for task_id in sorted(
                scheduler_completed_candidates.difference(
                    completed_candidates
                )
            )
        ]
        for logical_task_id in sorted(completed_candidates):
            try:
                validate_task_receipt(staging, logical_task_id)
            except (FerebusTaskRunnerError, OSError, ValueError) as exc:
                invalid.append(
                    {
                        "task_id": int(logical_task_id),
                        "reason": type(exc).__name__ + ": " + str(exc)[:160],
                    }
                )
                continue
            reusable.append(int(logical_task_id))
        reusable_set = set(reusable)
        retry = [
            logical_task_id
            for logical_task_id in range(int(n_tasks))
            if logical_task_id not in reusable_set
        ]
        source_digest = hashlib.sha256(
            ",".join(
                str(item["receipt"]["receipt_sha256"]) for item in recoveries
            ).encode("ascii")
        ).hexdigest()
        ledger = write_phase_recovery_ledger(
            self.campaign_dir,
            {
                "campaign_uid": str(state.campaign_uid),
                "phase": phase_name,
                "iteration": int(state.iteration),
                "replacement_round": int(
                    getattr(state, "replacement_round", 0)
                ),
                "source_receipt_sha256": source_digest,
                "source_terminal_receipts": [
                    {
                        "path": str(item["path"]),
                        "receipt_sha256": str(
                            item["receipt"]["receipt_sha256"]
                        ),
                        "job_id": str(item["receipt"].get("job_id") or ""),
                        "submission_identity": str(
                            item["receipt"]["submission_identity"]
                        ),
                    }
                    for item in recoveries
                ],
                "environment_equivalences": (
                    self._environment_proof_records(assessments)
                ),
                "scheduler_completed_candidates": len(
                    scheduler_completed_candidates
                ),
                "reusable_logical_task_ids": reusable,
                "retry_logical_task_ids": retry,
                "invalid_completed_tasks": invalid,
                "recovery_lineage": self._task_recovery_lineage(
                    reusable,
                    latest,
                    assessments,
                    reuse_basis="scheduler_completed_task_receipt",
                ),
            },
        )
        return {
            "reusable_logical_task_ids": reusable,
            "retry_logical_task_ids": retry,
            "ledger": ledger,
        }

    def _recover_cancelled_diversity_phase(
        self,
        state,
        phase,
        phase_name: str,
    ) -> Optional[PhaseResult]:
        """Adopt a complete scalar publication or retire incomplete output."""
        recoveries = self._scheduler_cancel_recovery_sources(state, phase_name)
        if not recoveries:
            return None
        assessments = self._scheduler_recovery_environment_assessments(
            recoveries
        )
        latest = self._latest_scheduler_recovery_by_task(recoveries)
        latest_recovery = latest.get(0)
        latest_identity = (
            ""
            if not isinstance(latest_recovery, Mapping)
            else str(
                latest_recovery["intent"].get("submission_identity") or ""
            )
        )
        publication_environment_equivalent = bool(
            assessments.get(latest_identity, {}).get("equivalent", False)
        )
        source_digest = hashlib.sha256(
            ",".join(
                str(item["receipt"]["receipt_sha256"]) for item in recoveries
            ).encode("ascii")
        ).hexdigest()
        parsed = self._parse_diversity_postprocess(
            state,
            phase,
            [],
            emit_success_events=False,
        )
        if (
            parsed.failure_reason is None
            and publication_environment_equivalent
        ):
            write_phase_recovery_ledger(
                self.campaign_dir,
                {
                    "campaign_uid": str(state.campaign_uid),
                    "phase": phase_name,
                    "iteration": int(state.iteration),
                    "replacement_round": int(
                        getattr(state, "replacement_round", 0)
                    ),
                    "source_receipt_sha256": source_digest,
                    "source_terminal_receipts": [
                        {
                            "path": str(item["path"]),
                            "receipt_sha256": str(
                                item["receipt"]["receipt_sha256"]
                            ),
                            "job_id": str(
                                item["receipt"].get("job_id") or ""
                            ),
                            "submission_identity": str(
                                item["receipt"]["submission_identity"]
                            ),
                        }
                        for item in recoveries
                    ],
                    "environment_equivalences": (
                        self._environment_proof_records(assessments)
                    ),
                    "publication_disposition": "adopted",
                    "reusable_logical_task_ids": [0],
                    "retry_logical_task_ids": [],
                    "invalid_completed_tasks": [],
                    "recovery_lineage": self._task_recovery_lineage(
                        [0],
                        latest,
                        assessments,
                        reuse_basis="authority_valid_publication",
                    ),
                },
            )
            for event_payload in list(
                parsed.journal_events
            ):
                payload = dict(event_payload)
                event_name = str(payload.pop("event"))
                self._journal_event(event_name, **payload)
            parsed.journal_events = []
            self._journal_event(
                "partial_array_recovery_postprocess_only",
                phase=phase_name,
                iteration=int(state.iteration),
                logical_total=1,
                n_complete=1,
                n_retry=0,
                recovery_source="scheduler_cancellation",
                scalar_publication="adopted",
            )
            return parsed

        from ..handoff_manifests import (
            phase_a_sample_manifest_path,
            phase_b_selection_path,
        )
        from ..layout import (
            active_iteration_dir,
            active_phase_b_dir,
            bootstrap_selection_dir,
        )

        if phase_name == "PHASE_A_DIVERSITY":
            output_dir = bootstrap_selection_dir(self.campaign_dir)
            manifest_path = phase_a_sample_manifest_path(output_dir)
        else:
            iteration_dir = active_iteration_dir(
                self.campaign_dir,
                int(state.iteration),
            )
            output_dir = active_phase_b_dir(iteration_dir)
            manifest_path = phase_b_selection_path(iteration_dir)
        publication_incomplete = _diversity_publication_is_incomplete(
            phase_name=phase_name,
            output_dir=output_dir,
            manifest_path=manifest_path,
        )
        if parsed.failure_reason is not None and not publication_incomplete:
            raise BackendSubmissionError(
                "completed diversity publication is invalid after scheduler "
                "cancellation: "
                + str(parsed.failure_reason)
            )

        archive = (
            Path(self.campaign_dir)
            / ".DATA"
            / "ACTIVE_LEARNING"
            / "diversity_retry_quarantine"
            / (
                phase_name
                + "-"
                + f"{int(state.iteration):06d}"
                + "-"
                + source_digest
            )
        )
        if output_dir.exists() or output_dir.is_symlink():
            if output_dir.is_symlink() or not output_dir.is_dir():
                raise BackendSubmissionError(
                    "partial diversity output is not a regular directory: "
                    + str(output_dir)
                )
            if archive.exists() or archive.is_symlink():
                raise BackendSubmissionError(
                    "partial diversity output and its recovery archive both exist"
                )
            archive.parent.mkdir(parents=True, exist_ok=True)
            output_dir.replace(archive)
        elif archive.exists():
            if archive.is_symlink() or not archive.is_dir():
                raise BackendSubmissionError(
                    "diversity recovery archive is invalid"
                )
        ledger = write_phase_recovery_ledger(
            self.campaign_dir,
            {
                "campaign_uid": str(state.campaign_uid),
                "phase": phase_name,
                "iteration": int(state.iteration),
                "replacement_round": int(
                    getattr(state, "replacement_round", 0)
                ),
                "source_receipt_sha256": source_digest,
                "source_terminal_receipts": [
                    {
                        "path": str(item["path"]),
                        "receipt_sha256": str(
                            item["receipt"]["receipt_sha256"]
                        ),
                        "job_id": str(item["receipt"].get("job_id") or ""),
                        "submission_identity": str(
                            item["receipt"]["submission_identity"]
                        ),
                    }
                    for item in recoveries
                ],
                "environment_equivalences": (
                    self._environment_proof_records(assessments)
                ),
                "publication_disposition": "rerun",
                "publication_archive": (
                    str(archive) if archive.exists() else None
                ),
                "reusable_logical_task_ids": [],
                "retry_logical_task_ids": [0],
                "invalid_completed_tasks": [
                    {
                        "task_id": 0,
                        "reason": (
                            str(parsed.failure_reason)
                            if parsed.failure_reason is not None
                            else "environment_equivalence_unproven"
                        ),
                    }
                ],
            },
        )
        self._journal_event(
            "partial_array_recovery_prepared",
            phase=phase_name,
            iteration=int(state.iteration),
            logical_tasks=1,
            reusable_tasks=0,
            retry_tasks=1,
            recovery_source="scheduler_cancellation",
            scalar_publication="rerun",
            recovery_ledger_sha256=str(ledger["ledger_sha256"]),
        )
        return None

    def _submit_ferebus_phase(self, state, phase_name: str) -> PhaseResult:
        from . import input_staging as _stg
        from .ferebus_candidate_recovery import (
            materialise_recovery_candidate,
            read_recovery_request,
            update_recovery_status,
        )
        from ..submit.pyferebus_wrap import FerebusSubmissionError, submit_ferebus

        try:
            tv = int(getattr(state, "reference_data_version", 0))
            is_initial = phase_name == "INITIAL_FEREBUS"
            state_updates: Dict[str, Any] = {}
            if not is_initial:
                from ..versioning.reference_data import ReferenceDataVersioning

                v_train = ReferenceDataVersioning(
                    Path(self.campaign_dir) / self.reference_data_dir_name
                )
                committed = v_train.list_committed_versions()
                committed_max = max(committed) if committed else -1
                if committed_max > tv:
                    tv = committed_max
                    v_train.ensure_current(committed_max)
                    state_updates["reference_data_version"] = int(committed_max)
                elif committed_max < tv:
                    raise BackendSubmissionError(
                        "state.reference_data_version "
                        + str(tv)
                        + " is ahead of committed reference-data versions "
                        + repr(committed)
                    )
            recovery = read_recovery_request(
                self.campaign_dir,
                expected_campaign_uid=str(state.campaign_uid),
            )
            if recovery is not None and str(recovery.get("status")) in {
                "prepared",
                "measurement_incomplete",
                "materialised",
            }:
                recovered_staging = materialise_recovery_candidate(
                    self.campaign_dir,
                    campaign_uid=str(state.campaign_uid),
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    reference_data_version=int(tv),
                )
                if recovered_staging is None:
                    raise BackendSubmissionError(
                        "active FEREBUS recovery request could not be materialised"
                    )
                self._journal_event(
                    "ferebus_candidate_recovery_materialised",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    reference_data_version=int(tv),
                    candidate_id=str(recovery.get("candidate_id") or ""),
                    staging=str(recovered_staging),
                    scheduler_jobs_submitted=0,
                )
                result = self._parse_ferebus_postprocess(state, phase_name, [])
                quality_disposition = str(
                    (result.submission_metadata or {}).get(
                        "ferebus_quality_disposition", ""
                    )
                )
                if result.failure_reason is None:
                    recovery_status = "accepted"
                elif quality_disposition == "measurement_incomplete":
                    recovery_status = "measurement_incomplete"
                elif quality_disposition == "quality_rejected":
                    recovery_status = "rejected"
                else:
                    recovery_status = "failed"
                update_recovery_status(
                    self.campaign_dir,
                    campaign_uid=str(state.campaign_uid),
                    status=recovery_status,
                    staging_path=(
                        recovered_staging if recovered_staging.exists() else None
                    ),
                    last_error=result.failure_reason,
                )
                self._journal_event(
                    "ferebus_candidate_reprocessed",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    reference_data_version=int(tv),
                    candidate_id=str(recovery.get("candidate_id") or ""),
                    outcome=recovery_status,
                    scheduler_jobs_submitted=0,
                )
                return result
            import inspect

            try:
                stage_parameters = inspect.signature(
                    _stg.stage_ferebus_inputs
                ).parameters
            except (TypeError, ValueError):
                stage_parameters = {}
            stage_kwargs = (
                {"progress_callback": self._report_runtime_progress}
                if "progress_callback" in stage_parameters
                else {}
            )
            staging, staged_task_count = _stg.stage_ferebus_inputs(
                self.campaign_dir,
                self.config,
                tv,
                **stage_kwargs,
            )
            f = self.config.ferebus
            ferebus_manifest = _stg.read_ferebus_manifest(staging)
            expected_ferebus_tasks = int(ferebus_manifest.get("n_tasks", 0))
            if expected_ferebus_tasks <= 0:
                raise BackendSubmissionError(
                    "FEREBUS task manifest contains no scheduler tasks"
                )
            if int(staged_task_count) != expected_ferebus_tasks:
                raise BackendSubmissionError(
                    "FEREBUS staging task count does not match its immutable "
                    "task manifest: staged "
                    + str(int(staged_task_count))
                    + ", manifest "
                    + str(expected_ferebus_tasks)
                )
            from ..ferebus_prior import contract_from_payload

            prior_contract = contract_from_payload(
                ferebus_manifest.get("prior_mean_contract")
            )
            if is_initial and isinstance(
                ferebus_manifest.get("model_bootstrap"), dict
            ):
                imported_manifest = _stg.prepare_imported_model_bootstrap(staging)
                self._journal_event(
                    "model_bootstrap_staged",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    n_models=int(imported_manifest.get("n_tasks", 0)),
                    historical_training_rows=int(
                        imported_manifest["model_bootstrap"][
                            "historical_training_rows"
                        ]
                    ),
                )
                result = self._parse_ferebus_postprocess(
                    state,
                    phase_name,
                    [],
                )
                if result.failure_reason is None:
                    self._journal_event(
                        "model_bootstrap_committed",
                        phase=phase_name,
                        iteration=int(getattr(state, "iteration", 0)),
                        models_version=int(
                            (result.state_updates or {}).get("models_version", 0)
                        ),
                        scheduler_jobs_submitted=0,
                    )
                return result
            ferebus_recovery = (
                self._prepare_scheduler_cancelled_ferebus_recovery(
                    state,
                    phase_name,
                    staging,
                    n_tasks=expected_ferebus_tasks,
                )
            )
            ferebus_retry_ids = (
                list(range(expected_ferebus_tasks))
                if ferebus_recovery is None
                else [
                    int(value)
                    for value in ferebus_recovery["retry_logical_task_ids"]
                ]
            )
            if ferebus_recovery is not None:
                self._journal_event(
                    "partial_array_recovery_prepared",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    logical_tasks=expected_ferebus_tasks,
                    reusable_tasks=len(
                        ferebus_recovery["reusable_logical_task_ids"]
                    ),
                    retry_tasks=len(ferebus_retry_ids),
                    scheduler_completed_candidates=int(
                        ferebus_recovery["ledger"].get(
                            "scheduler_completed_candidates",
                            0,
                        )
                    ),
                    recovery_source="scheduler_cancellation",
                )
            if not ferebus_retry_ids:
                return self._parse_ferebus_postprocess(
                    state,
                    phase_name,
                    [],
                )
            if ferebus_recovery is not None:
                from .ferebus_task_runner import quarantine_task_outputs

                quarantine_root = (
                    Path(self.campaign_dir)
                    / ".DATA"
                    / "ACTIVE_LEARNING"
                    / "ferebus_retry_quarantine"
                    / (
                        phase_name
                        + "-"
                        + f"{int(state.iteration):06d}"
                        + "-r"
                        + f"{int(getattr(state, 'replacement_round', 0)):04d}"
                    )
                    / str(
                        ferebus_recovery["ledger"]["source_receipt_sha256"]
                    )
                )
                quarantine_task_outputs(
                    staging,
                    ferebus_retry_ids,
                    quarantine_root,
                )
            resources = getattr(self.config, "resources", None)
            effective_partition = (
                str(self.partition)
                if self.partition is not None
                else str(resources.partition_for(phase_name))
            )
            from . import submission_intent as _submission_intent

            active_intent = _submission_intent.load_active_intent(
                self.campaign_dir,
                phase_name,
                int(getattr(state, "iteration", 0)),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if not isinstance(active_intent, dict) or str(
                active_intent.get("status")
            ) != "PRE_SUBMIT":
                raise BackendSubmissionError(
                    "live FEREBUS submission requires an active PRE_SUBMIT intent"
            )
            identity = str(active_intent["submission_identity"])
            bundle = prepare_attempt_bundle(
                self.campaign_dir,
                phase_name,
                int(getattr(state, "iteration", 0)),
                identity,
                array_size=len(ferebus_retry_ids),
                max_log_files_per_directory=(
                    _configured_max_job_log_files_per_directory()
                ),
                logical_task_ids=ferebus_retry_ids,
            )
            scratch_template = scratch_path_template(
                self.campaign_dir,
                phase_name,
                int(getattr(state, "iteration", 0)),
                identity,
                scheduler_kind=self.scheduler_identity_kind,
            )
            ferebus_path = _configured_backend_path("ferebus", "ferebus")
            allow_bare_ferebus = (
                os.environ.get("ICHOR_ALLOW_BARE_FEREBUS", "") == "1"
                or self.sbatch_runner is not subprocess.run
            )
            if ferebus_path == "ferebus" and not allow_bare_ferebus:
                raise BackendSubmissionError(
                    "live FEREBUS requires software.ferebus.executable_path "
                    "in ichor_config.yaml; set ICHOR_ALLOW_BARE_FEREBUS=1 only for "
                    "development tests"
                )
            path_to_executable = None if ferebus_path == "ferebus" else ferebus_path
            ferebus_platform = _configured_ferebus_platform()
            expected_job_name = _current_submission_job_name(
                self.campaign_dir,
                phase_name,
                int(getattr(state, "iteration", 0)),
                campaign_uid=getattr(state, "campaign_uid", None),
            )
            effective_walltime = (
                self.walltime_hours
                if self.walltime_hours is not None
                else resources.walltime_for(phase_name) if resources is not None else 24
            )
            prepared: Dict[str, Any] = {}

            def _prepared_ferebus_runtime(
                _working_dir: Path,
                _generated_script: Path,
                _generated_configs: Sequence[Mapping[str, Any]],
            ) -> Mapping[str, Any]:
                resolved = resolve_phase_resources(
                    phase_name=phase_name,
                    config=self.config,
                    partition=effective_partition,
                    campaign_dir=self.campaign_dir,
                    iteration=int(getattr(state, "iteration", 0)),
                    array_size=len(ferebus_retry_ids),
                    expected_reference_data_version=int(
                        0 if phase_name == "INITIAL_FEREBUS" else tv
                    ),
                    require_evidence=True,
                )
                resolution_evidence = dict(
                    resolved.extra.get("evidence") or {}
                )
                from ..versioning.manifest import sha256_file

                if bundle.array_task_map is None:
                    raise ValueError("FEREBUS attempt has no immutable task map")
                resolution_evidence["submitted_array_task_map"] = {
                    "path": str(bundle.array_task_map.resolve()),
                    "size": int(bundle.array_task_map.stat().st_size),
                    "sha256": sha256_file(bundle.array_task_map),
                }
                resource_payload = resolution_payload(
                    campaign_uid=str(state.campaign_uid),
                    phase_name=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    attempt_id=str(active_intent["attempt_id"]),
                    submission_identity=identity,
                    resolved=resolved,
                    evidence=resolution_evidence,
                    scratch_path_template=scratch_template,
                    implementation_identity=capture_implementation_identity(
                        self.campaign_dir,
                        backend="ferebus",
                        backend_executable_path=ferebus_path,
                        require_environment_generation=False,
                    ),
                )
                resolution_binding = write_resolution(
                    self.campaign_dir, resource_payload
                )
                _submission_intent.bind_resource_resolution(
                    self.campaign_dir,
                    phase_name,
                    int(getattr(state, "iteration", 0)),
                    path=str(resolution_binding["path"]),
                    sha256=str(resolution_binding["sha256"]),
                    formula_version=str(resolution_binding["formula_version"]),
                    scratch_path_template=scratch_template,
                    expected_tasks=len(ferebus_retry_ids),
                )
                self._journal_event(
                    "resolved_phase_resources",
                    **resolved.journal_payload(phase_name=phase_name),
                    resource_resolution_path=str(resolution_binding["path"]),
                    resource_resolution_sha256=str(
                        resolution_binding["sha256"]
                    ),
                )
                prepared.update(
                    resolved=resolved,
                    resolution_binding=resolution_binding,
                )
                return {
                    "partition": str(resolved.partition),
                    "mem_per_cpu": str(resolved.mem_per_cpu),
                    "cpus_per_task": int(resolved.cpus_per_task),
                    "ntasks": int(resolved.ntasks),
                    "submission_script_path": bundle.script,
                    "output_path": scheduler_log_paths(
                        bundle,
                        scheduler_kind=self.scheduler_identity_kind,
                        is_array=True,
                    )[
                        "output"
                    ],
                    "error_path": scheduler_log_paths(
                        bundle,
                        scheduler_kind=self.scheduler_identity_kind,
                        is_array=True,
                    )[
                        "error"
                    ],
                    "path_to_executable": path_to_executable,
                    "array_concurrency_limit": getattr(
                        resources, "array_concurrency_limit", None
                    ),
                    "scheduler_queue": resolved.extra.get("scheduler_queue"),
                    "parallel_environment": resolved.extra.get(
                        "parallel_environment"
                    ),
                    "runtime_preamble": [
                        *(
                            module_initialisation_lines()
                            if self.scheduler_identity_kind == "sge"
                            else []
                        ),
                        "module purge",
                        *[
                            "module load " + module
                            for module in _configured_daemon_runtime_modules()
                        ],
                        *native_runtime_setup_lines(),
                        "export ICHOR_ACTIVE_WORKERS="
                        + str(
                            int(
                                resolved.extra.get(
                                    "active_workers", f.nagents
                                )
                            )
                        ),
                        "export ICHOR_MEMORY_ONLY_CPUS="
                        + str(
                            int(resolved.extra.get("memory_only_cpus", 0))
                        ),
                        "export OMP_NUM_THREADS="
                        + str(
                            int(
                                resolved.extra.get(
                                    "active_workers", f.nagents
                                )
                            )
                        ),
                        *_job_scratch_preamble(
                            campaign_dir=self.campaign_dir,
                            phase_name=phase_name,
                            iteration=int(getattr(state, "iteration", 0)),
                            submission_intent=active_intent,
                            resource_resolution_binding=resolution_binding,
                            script_binding_path=bundle.script_binding,
                        ),
                    ],
                }

            def _bind_ferebus_script(
                script_path: Path,
                script_binding: Mapping[str, Any],
            ) -> None:
                _submission_intent.bind_submission_script(
                    self.campaign_dir,
                    phase_name,
                    int(getattr(state, "iteration", 0)),
                    script_path=str(script_path.resolve()),
                    script_sha256=str(script_binding["script_sha256"]),
                    binding_path=str(script_binding["path"]),
                    binding_sha256=str(script_binding["sha256"]),
                )

            from ..ferebus_prior import backend_kernel_token

            self._raise_if_immediate_cancel_before_submission(state)
            submission = submit_ferebus(
                staging / _stg.FEREBUS_JOB_DETAILS,
                staging,
                platform=ferebus_platform,
                walltime_hours=effective_walltime,
                ncores=max(1, int(f.nagents)),
                partition=effective_partition,
                kernel=backend_kernel_token(f.kernel),
                loss="huber",
                is_constant_noise=True,
                nagents=int(f.nagents),
                maxiter=int(f.maxiter),
                full_ARD=True,
                prior_mean_type=int(prior_contract.mean_type),
                prior_mean_level_of_theory=str(
                    prior_contract.level_of_theory or "not_applicable"
                ),
                prior_mean_iqa_deviation_factor=float(
                    prior_contract.iqa_deviation_factor
                ),
                feature_scaling=bool(prior_contract.feature_scaling),
                property_scaling=bool(prior_contract.property_scaling),
                overwrite_workdir=False,
                move_dataset_files=True,
                path_to_executable=path_to_executable,
                expected_tasks=expected_ferebus_tasks,
                submitted_tasks=len(ferebus_retry_ids),
                scheduler_task_map=bundle.array_task_map,
                reuse_prepared_inputs=(ferebus_recovery is not None),
                require_existing_task_map_match=bool(
                    ferebus_recovery is not None
                    and ferebus_recovery[
                        "reusable_logical_task_ids"
                    ]
                ),
                expected_job_name=expected_job_name,
                submit_runner=self.sbatch_runner,
                prepared_callback=_prepared_ferebus_runtime,
                pre_submit_hook=_bind_ferebus_script,
                scheduler_timeout_seconds=int(
                    self.config.runtime.scheduler_command_timeout_seconds
                ),
                scheduler_kind=self.scheduler_identity_kind,
            )
        except BackendSubmissionError:
            raise
        except (FerebusSubmissionError, OSError, ValueError) as exc:
            raise BackendSubmissionError(
                "pyferebus submission failed for "
                + phase_name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        except Exception as exc:
            raise BackendSubmissionError(
                "pre-submit staging failed for "
                + phase_name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        self.artefact_log.append(str(submission.submission_script))
        resolved = prepared.get("resolved")
        resolution_binding = prepared.get("resolution_binding")
        if not isinstance(resolved, ResolvedPhaseResources) or not isinstance(
            resolution_binding, dict
        ):
            raise BackendSubmissionError(
                "FEREBUS submission bypassed immutable resource preparation"
            )
        return PhaseResult(
            is_complete=False,
            submitted_job_id=str(submission.job_id),
            expected_tasks=len(ferebus_retry_ids),
            state_updates=state_updates,
            submission_metadata={
                "resource_resolution_path": str(resolution_binding["path"]),
                "resource_resolution_sha256": str(resolution_binding["sha256"]),
                "resource_formula_version": str(
                    resolution_binding["formula_version"]
                ),
                "scratch_path_template": scratch_template,
                "script_bundle": str(bundle.root.resolve()),
                "submitted_script_sha256": submission.script_binding.get(
                    "script_sha256"
                ),
                "script_binding_path": submission.script_binding.get("path"),
                "script_binding_sha256": submission.script_binding.get(
                    "sha256"
                ),
                "scheduler_recovery_reusable_tasks": (
                    0
                    if ferebus_recovery is None
                    else len(
                        ferebus_recovery["reusable_logical_task_ids"]
                    )
                ),
                "scheduler_recovery_retry_tasks": len(ferebus_retry_ids),
            },
        )

    def _complete_empty_aimall_phase(self, state, phase_name: str) -> PhaseResult:
        """Complete an AIMAll handoff when Gaussian accepted no candidates."""
        from . import input_staging as _stg

        staging = self._quantum_staging_path(state, phase_name)
        _stg.write_quantum_acceptance_manifest(
            staging,
            phase_name=phase_name,
            iteration=int(state.iteration),
            accepted=[],
            rejected=[],
        )
        context = "bootstrap" if phase_name.startswith("INITIAL_") else "active"
        gaussian_phase = (
            "INITIAL_REPLACEMENT_GAUSSIAN"
            if phase_name == "INITIAL_REPLACEMENT_AIMALL"
            else "REPLACEMENT_GAUSSIAN"
            if phase_name == "REPLACEMENT_AIMALL"
            else "INITIAL_GAUSSIAN"
            if phase_name == "INITIAL_AIMALL"
            else "GAUSSIAN"
        )
        allocation = _stg.record_allocation_quantum_results(
            self.campaign_dir,
            context=context,
            iteration=0 if context == "bootstrap" else int(state.iteration),
            staging_dir=staging,
            gaussian_phase=gaussian_phase,
            aimall_phase=phase_name,
            expected_campaign_uid=str(state.campaign_uid),
            replacement_round=int(getattr(state, "replacement_round", 0)),
        )
        self._journal_event(
            "aimall_skipped_no_gaussian_acceptances",
            phase=phase_name,
            iteration=int(state.iteration),
            allocation_summary=dict(allocation.get("summary") or {}),
        )
        override = (
            "INITIAL_ALLOCATION_CHECK"
            if phase_name == "INITIAL_REPLACEMENT_AIMALL"
            else "ALLOCATION_CHECK"
            if phase_name == "REPLACEMENT_AIMALL"
            else None
        )
        return PhaseResult(
            is_complete=True,
            next_phase_override=override,
        )

    def _seed_selection_pool(self, state):
        context = getattr(self, "_active_seed_selection_context", None)
        if isinstance(context, dict) and context.get("pool") is not None:
            return context["pool"]
        from ..acquisition.trajectory_pool import TrajectoryPool

        return TrajectoryPool.load(self.campaign_dir)

    def _seed_selection_prepare_context(self, _state, pool) -> None:
        context = getattr(self, "_active_seed_selection_context", None)
        if not isinstance(context, dict) or context.get("pool") is not pool:
            return
        cache = context.get("cache")
        if cache is not None and not bool(context.get("features_ready", False)):
            reporter = context.get("progress")
            if reporter is not None:
                reporter.update(
                    "features",
                    completed=0,
                    total=int(pool.n_frames()),
                )
            cache.ensure_features()
            if reporter is not None:
                reporter.update(
                    "features",
                    completed=int(pool.n_frames()),
                    total=int(pool.n_frames()),
                )
            context["features_ready"] = True

    def _seed_selection_population(self, pool):
        context = getattr(self, "_active_seed_selection_context", None)
        if isinstance(context, dict) and context.get("pool") is pool:
            return pool
        return super()._seed_selection_population(pool)

    def _seed_selection_progress_callback(self, _state):
        context = getattr(self, "_active_seed_selection_context", None)
        reporter = context.get("progress") if isinstance(context, dict) else None
        return reporter.callback if reporter is not None else None

    def _seed_selection_model_set(self, state):
        context = getattr(self, "_active_seed_selection_context", None)
        if isinstance(context, dict) and context.get("model_set") is not None:
            return context["model_set"]
        return super()._seed_selection_model_set(state)

    def _seed_selection_assert_publication_authority(self, state, pool) -> None:
        context = getattr(self, "_active_seed_selection_context", None)
        if not isinstance(context, dict) or context.get("pool") is not pool:
            return
        snapshot = context.get("artifact_snapshot")
        if snapshot is not None:
            snapshot.assert_anchors_unchanged(self.campaign_dir)
        from ..versioning.manifest import sha256_file
        from ..versioning.trained_models import (
            assert_current_model_payloads_unchanged,
        )

        if sha256_file(pool.canonical_path) != str(pool.sha256):
            raise BackendSubmissionError(
                "trajectory pool changed during seed selection"
            )
        bindings = context.get("current_model_bindings")
        if bindings is not None:
            assert_current_model_payloads_unchanged(
                self.campaign_dir,
                context["model_set"],
                bindings,
            )

    def _seed_selection_finished(self, state, *, selection_published: bool) -> None:
        context = getattr(self, "_active_seed_selection_context", None)
        if not isinstance(context, dict):
            return
        cache = context.get("cache")
        indexed = context.get("indexed_posterior")
        if cache is not None:
            try:
                if bool(selection_published) and indexed is not None:
                    close = getattr(indexed, "close", None)
                    if callable(close):
                        close()
                cache.finalise(selection_published=bool(selection_published))
                if (
                    bool(selection_published)
                    and indexed is not None
                    and bool(getattr(indexed, "resolved", True))
                ):
                    cache.prune_superseded(
                        active_feature_cache_id=indexed.feature_cache_id,
                        active_variance_cache_id=indexed.variance_cache_id,
                    )
            except Exception as exc:
                self._journal_event(
                    "seed_selection_cache",
                    phase="SEED_SELECT",
                    iteration=int(state.iteration),
                    cache_kind="derived_cleanup",
                    cache_status="failed",
                    error=type(exc).__name__ + ": " + str(exc)[:240],
                )
        reporter = context.get("progress")
        if reporter is not None and bool(selection_published):
            reporter.finish(selection_published=True)

    def _seed_selection_posterior(
        self,
        state,
        training_atoms,
        *,
        eligible_indices=None,
    ):
        """Live seed selection ranks the exploit half by the real GP posterior
        variance, so the most uncertain pool frames get attacked."""
        models_version = int(getattr(state, "models_version", -1))
        if models_version < 0:
            return super()._seed_selection_posterior(
                state,
                training_atoms,
                eligible_indices=eligible_indices,
            )
        context = getattr(self, "_active_seed_selection_context", None)
        if isinstance(context, dict) and context.get("posterior") is not None:
            try:
                from .seed_selection_runtime import SeedSelectionRuntimeCache

                cache = context.get("cache")
                if cache is None:
                    model_set = context["model_set"]
                    cache = SeedSelectionRuntimeCache(
                        self.campaign_dir,
                        pool=context["pool"],
                        posterior=context["posterior"],
                        model_set_sha256=str(model_set.model_set_sha256),
                        model_manifest_sha256=str(model_set.head_manifest_sha256),
                        iteration=int(state.iteration),
                        progress=context.get("progress"),
                        model_file_sha256_by_atom=context.get(
                            "model_file_sha256_by_atom"
                        ),
                    )
                    context["cache"] = cache
                indexed = cache.ensure_indexed_posterior(
                    eligible_indices=(
                        list(context["pool"].frame_ids())
                        if eligible_indices is None
                        else list(eligible_indices)
                    ),
                    chunk_size=int(self.config.seed_selection.variance_chunk_size),
                    defer_variances=True,
                )
                context["indexed_posterior"] = indexed
                return indexed
            except Exception as exc:
                raise BackendSubmissionError(
                    "live seed-selection cache preparation failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
        try:
            from pathlib import Path as _Path
            from .model_contract import smoke_total_energy_posterior
            from .artifact_contracts import verify_committed_model_version
            from ..versioning.trained_models import TrainedModelVersioning

            models_dir = TrainedModelVersioning(
                _Path(self.campaign_dir) / self.models_dir_name
            ).iteration_path(models_version)
            if not models_dir.is_dir():
                raise BackendSubmissionError(
                    "committed models directory missing for seed selection: "
                    + str(models_dir)
                )
            verify_committed_model_version(
                self.campaign_dir,
                models_version,
                models_dir_name=self.models_dir_name,
                verification="metadata",
            )
            return smoke_total_energy_posterior(
                models_dir,
                property_name="iqa",
                probe_frames=list(training_atoms)[:3],
            )
        except Exception as exc:
            if isinstance(exc, BackendSubmissionError):
                raise
            raise BackendSubmissionError(
                "live seed selection requires loadable committed models: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

    def _inline_seed_select(self, state):
        """Live SEED_SELECT must never use the dry-run no-pool placeholder."""
        from ..acquisition.trajectory_pool import TrajectoryPool
        from ..handoff_manifests import seeds_picked_path

        iteration_dir = self._iter_dir(state.iteration)
        if seeds_picked_path(iteration_dir).is_file():
            try:
                from .seed_selection_runtime import (
                    finalise_seed_selection_workspace,
                )

                finalise_seed_selection_workspace(
                    self.campaign_dir,
                    iteration=int(state.iteration),
                )
            except Exception as exc:
                self._journal_event(
                    "seed_selection_cache",
                    phase="SEED_SELECT",
                    iteration=int(state.iteration),
                    cache_kind="shortlist_projections",
                    cache_status="cleanup_failed",
                    error=type(exc).__name__ + ": " + str(exc)[:240],
                )
            return super()._inline_seed_select(state)

        progress = None
        try:
            from ichor.core.adversarial.posterior import TotalEnergyPosterior
            from ..versioning.trained_models import (
                load_trained_models,
                load_trained_models_from_snapshot,
            )
            from .seed_selection_runtime import (
                SeedSelectionProgressReporter,
                SeedSelectionRuntimeCache,
            )

            progress = SeedSelectionProgressReporter(
                self.campaign_dir,
                campaign_uid=str(state.campaign_uid),
                iteration=int(state.iteration),
                journal_event=self._journal_event,
            )
            pool = TrajectoryPool.load(
                self.campaign_dir,
                progress_callback=progress.callback,
            )
            artifact_snapshot = getattr(
                self, "_committed_artifact_snapshot", None
            )
            current_model_bindings = None
            if artifact_snapshot is None:
                progress.update("model_authority", completed=0, total=1)
                model_set, models = load_trained_models(
                    self.campaign_dir,
                    int(state.models_version),
                    verification="metadata",
                    reference_verification="metadata",
                )
                progress.update("model_authority", completed=1, total=1)
                progress.update("models", completed=1, total=1)
            else:
                (
                    model_set,
                    models,
                    current_model_bindings,
                ) = load_trained_models_from_snapshot(
                    self.campaign_dir,
                    int(state.models_version),
                    snapshot=artifact_snapshot,
                    expected_campaign_uid=str(state.campaign_uid),
                    progress_callback=progress.callback,
                )
            posterior = TotalEnergyPosterior(
                models,
                property_name="iqa",
                scaled=True,
            )
            model_file_sha256_by_atom = {
                str(task.atom): str(task.model.sha256)
                for task in model_set.tasks
                if str(task.property) == "iqa"
            }
            progress.bind_inputs(
                trajectory_sha256=str(pool.sha256),
                model_set_sha256=str(model_set.model_set_sha256),
                model_manifest_sha256=str(model_set.head_manifest_sha256),
            )
            cache = SeedSelectionRuntimeCache(
                self.campaign_dir,
                pool=pool,
                posterior=posterior,
                model_set_sha256=str(model_set.model_set_sha256),
                model_manifest_sha256=str(model_set.head_manifest_sha256),
                iteration=int(state.iteration),
                progress=progress,
                model_file_sha256_by_atom=model_file_sha256_by_atom,
            )
            progress.update(
                "model_factors",
                completed=0,
                total=int(len(posterior._property_models)),
            )
            cache.ensure_model_factors()
        except Exception as exc:
            if progress is not None:
                progress.update(
                    "failed",
                    force=True,
                    status="failed",
                    error=type(exc).__name__ + ": " + str(exc)[:240],
                )
            raise BackendSubmissionError(
                "live seed selection requires an imported trajectory pool and loadable "
                "committed models: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        self._active_seed_selection_context = {
            "pool": pool,
            "model_set": model_set,
            "models": models,
            "posterior": posterior,
            "progress": progress,
            "cache": cache,
            "indexed_posterior": None,
            "features_ready": False,
            "artifact_snapshot": artifact_snapshot,
            "current_model_bindings": current_model_bindings,
            "model_file_sha256_by_atom": model_file_sha256_by_atom,
        }
        try:
            return super()._inline_seed_select(state)
        except Exception as exc:
            progress.update(
                "failed",
                force=True,
                status="failed",
                error=type(exc).__name__ + ": " + str(exc)[:240],
            )
            raise
        finally:
            self._active_seed_selection_context = None

    def _inline_reference_commit(self, state):
        """Run the shared transactional reference publication contract."""
        return super()._inline_reference_commit(state)

    def submit_or_run(self, state, phase) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        self._set_journal_state(state, phase_name)
        if phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
            self._report_runtime_progress("row_cache_validation")
            return self._submit_ferebus_phase(state, phase_name)
        if phase_name not in SBATCH_PHASES:
            if phase_name in INLINE_PHASES:
                try:
                    return super().submit_or_run(state, phase)
                except BackendSubmissionError:
                    raise
                except Exception as exc:
                    raise BackendSubmissionError(
                        "inline phase failed before completion for " + phase_name
                        + ": " + type(exc).__name__ + ": " + str(exc)
                    ) from exc
            #defensive: a phase that is neither inline nor sbatch is a bug.
            raise RuntimeError(
                "phase " + phase_name + " classified as neither INLINE nor SBATCH"
            )
        if phase_name in {"PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"}:
            diversity_recovery = self._recover_cancelled_diversity_phase(
                state,
                phase,
                phase_name,
            )
            if diversity_recovery is not None:
                return diversity_recovery
        if "AIMALL" in phase_name or "GAUSSIAN" in phase_name:
            from . import submission_intent as _submission_intent

            postprocess_intent = _submission_intent.load_active_intent(
                self.campaign_dir,
                phase_name,
                int(getattr(state, "iteration", 0)),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if (
                isinstance(postprocess_intent, dict)
                and isinstance(
                    postprocess_intent.get("postprocess_source"),
                    Mapping,
                )
            ):
                resolver = (
                    _submission_intent.resolve_aimall_postprocess_source
                    if "AIMALL" in phase_name
                    else _submission_intent.resolve_gaussian_postprocess_source
                )
                source = resolver(
                        self.campaign_dir,
                        campaign_uid=str(state.campaign_uid),
                        phase_name=phase_name,
                        iteration=int(getattr(state, "iteration", 0)),
                        replacement_round=int(
                            getattr(state, "replacement_round", 0)
                        ),
                        intent=postprocess_intent,
                    )
                self._journal_event(
                    "partial_array_recovery_postprocess_only",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    logical_total=int(source["logical_total"]),
                    n_complete=int(source["logical_total"]),
                    n_retry=0,
                    producer_job_id=str(source["job_id"]),
                )
                self._report_runtime_progress(
                    "output_visibility",
                    completed=0,
                    total=int(source["logical_total"]),
                    unit="tasks",
                )
                return self.postprocess(state, phase, [])
        try:
            self._report_runtime_progress("handoff_validation")
            self._report_runtime_progress("input_staging")
            array_size = self._array_size_after_staging(phase_name, state)
            self._report_runtime_progress(
                "input_staging",
                completed=(int(array_size) if array_size is not None else 1),
                total=(int(array_size) if array_size is not None else 1),
                unit="tasks",
            )
            if array_size is not None and array_size <= 0:
                if "AIMALL" in phase_name:
                    return self._complete_empty_aimall_phase(state, phase_name)
                raise BackendSubmissionError(
                    "nothing to submit for " + phase_name + ": staged 0 points/seeds"
                )
            submission_metadata: Dict[str, Any] = {}
            array_task_map: Optional[Path] = None
            if supports_partial_array_recovery(phase_name):
                recovery = (
                    self._prepare_scheduler_cancelled_array_recovery(
                        state,
                        phase_name,
                    )
                    or prepare_retry_submission(
                        self.campaign_dir,
                        phase_name,
                        int(getattr(state, "iteration", 0)),
                    )
                )
                recovery_summary = compact_array_recovery_summary(recovery)
                submission_metadata["array_recovery"] = recovery_summary
                recovery_journal_payload = _without_keys(
                    recovery_summary,
                    "phase",
                    "iteration",
                )
                scheduler_recovery = recovery.get(
                    "_scheduler_cancel_recovery"
                )
                if isinstance(scheduler_recovery, Mapping):
                    recovery_journal_payload.update(
                        {
                            "scheduler_completed_candidates": int(
                                scheduler_recovery.get(
                                    "scheduler_completed_candidates",
                                    0,
                                )
                            ),
                            "scheduler_terminal_receipts": len(
                                scheduler_recovery.get(
                                    "source_terminal_receipts",
                                    [],
                                )
                            ),
                            "recovery_ledger_sha256": str(
                                scheduler_recovery.get(
                                    "ledger_sha256",
                                    "",
                                )
                            ),
                        }
                    )
                self._journal_event(
                    "partial_array_recovery_prepared",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    **recovery_journal_payload,
                )
                retry_ids = list(recovery.get("retry_task_ids") or [])
                submission_metadata["logical_task_set_sha256"] = hashlib.sha256(
                    (",".join(str(int(task_id)) for task_id in retry_ids)).encode(
                        "ascii"
                    )
                ).hexdigest()
                if phase_name == "ARIADNE_ARRAY":
                    publication_archive = archive_stale_ariadne_publication(
                        self.campaign_dir,
                        state,
                        retry_task_ids=retry_ids,
                    )
                    if isinstance(publication_archive, dict) and bool(
                        publication_archive.get("changed", False)
                    ):
                        self._journal_event(
                            "ariadne_publication_archived",
                            phase=phase_name,
                            iteration=int(getattr(state, "iteration", 0)),
                            reason=(
                                "ariadne_retry_preparation"
                                if retry_ids
                                else "ariadne_postprocess_only_recovery"
                            ),
                            archive_id=str(
                                publication_archive.get("archive_id") or ""
                            ),
                            archive_manifest=str(
                                publication_archive.get("manifest_path") or ""
                            ),
                            n_files=int(
                                len(publication_archive.get("archived_paths") or [])
                            ),
                        )
                if not retry_ids and int(recovery.get("logical_total") or 0) > 0:
                    self._journal_event(
                        "partial_array_recovery_postprocess_only",
                        phase=phase_name,
                        iteration=int(getattr(state, "iteration", 0)),
                        **recovery_journal_payload,
                    )
                    return self.postprocess(state, phase, [])
                if retry_ids:
                    if (
                        isinstance(scheduler_recovery, Mapping)
                        and (
                            "GAUSSIAN" in phase_name
                            or "AIMALL" in phase_name
                        )
                    ):
                        source_digest = str(
                            scheduler_recovery.get(
                                "source_receipt_sha256",
                                "",
                            )
                        )
                        archived_retry_outputs = (
                            archive_existing_array_task_outputs(
                                self.campaign_dir,
                                phase_name,
                                int(getattr(state, "iteration", 0)),
                                task_ids=retry_ids,
                                archive_identity=(
                                    "scheduler-cancel-"
                                    + source_digest[:16]
                                ),
                            )
                        )
                        submission_metadata[
                            "scheduler_recovery_archived_outputs"
                        ] = len(archived_retry_outputs)
                    array_size = len(retry_ids)
                    retry_file = recovery.get("retry_task_file")
                    if retry_file:
                        array_task_map = Path(str(retry_file))
            if phase_name == "ARIADNE_ARRAY":
                removed = clean_stale_ariadne_seed_outputs(
                    self.campaign_dir,
                    int(getattr(state, "iteration", 0)),
                    retry_array_task_ids=(
                        None
                        if array_task_map is None
                        else [int(x) for x in (recovery.get("retry_task_ids") or [])]
                    ),
                )
                if removed:
                    self._journal_event(
                        "ariadne_stale_outputs_quarantined",
                        iteration=int(getattr(state, "iteration", 0)),
                        retained=int(len(removed)),
                        sample=[str(p) for p in removed[:5]],
                    )
            self._report_runtime_progress(
                "script_rendering",
                completed=0,
                total=(int(array_size) if array_size is not None else 1),
                unit="tasks",
            )
            script = self._write_real_script(
                phase_name,
                state,
                array_size,
                array_task_map=array_task_map,
            )
            from . import submission_intent as _submission_intent

            bound_intent = _submission_intent.load_active_intent(
                self.campaign_dir,
                phase_name,
                int(getattr(state, "iteration", 0)),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if isinstance(bound_intent, dict):
                submission_metadata.update({
                    "resource_resolution_path": bound_intent.get(
                        "resource_resolution_path"
                    ),
                    "resource_resolution_sha256": bound_intent.get(
                        "resource_resolution_sha256"
                    ),
                    "resource_formula_version": bound_intent.get(
                        "resource_formula_version"
                    ),
                    "scratch_path_template": bound_intent.get(
                        "scratch_path_template"
                    ),
                    "script_bundle": str(script.parent.resolve()),
                    "submitted_script_sha256": bound_intent.get(
                        "submitted_script_sha256"
                    ),
                    "script_binding_path": bound_intent.get(
                        "script_binding_path"
                    ),
                    "script_binding_sha256": bound_intent.get(
                        "script_binding_sha256"
                    ),
                })
        except BackendSubmissionError:
            raise
        except Exception as exc:
            raise BackendSubmissionError(
                "pre-submit staging failed for " + phase_name + ": "
                + type(exc).__name__ + ": " + str(exc)
            ) from exc
        binding_sha = str(
            (bound_intent or {}).get("script_binding_sha256") or ""
        )
        if len(binding_sha) != 64:
            raise BackendSubmissionError(
                "final submitted script has no immutable binding"
            )
        submission_stage = (
            "slurm_submission"
            if self.scheduler_identity_kind == "slurm"
            else "sge_submission"
        )
        self._report_runtime_progress(
            submission_stage,
            completed=0,
            total=(int(array_size) if array_size is not None else 1),
            unit="tasks",
        )
        try:
            self._raise_if_immediate_cancel_before_submission(state)
            submission = self._scheduler_backend.submit(
                script,
                binding_sha256=binding_sha,
                runner=self.sbatch_runner,
                timeout_seconds=int(
                    self.config.runtime.scheduler_command_timeout_seconds
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise BackendSubmissionError(
                self._scheduler_backend.display_name
                + " submission failed for phase "
                + phase_name
                + ": "
                + str(exc)
            ) from exc
        job_id = submission.job_id
        self.artefact_log.append(str(script))
        expected_tasks = int(array_size) if array_size is not None else 1
        self._report_runtime_progress(
            submission_stage,
            completed=expected_tasks,
            total=expected_tasks,
            unit="tasks",
        )
        return PhaseResult(
            is_complete=False,
            submitted_job_id=job_id,
            expected_tasks=expected_tasks,
            submission_metadata=submission_metadata,
        )

    # --- real script bodies --------------------------------------------

    def _write_real_script(
        self,
        phase_name: str,
        state,
        array_size=None,
        *,
        array_task_map: Optional[Path] = None,
    ) -> Path:
        try:
            from . import submission_intent as _submission_intent

            active_intent = _submission_intent.load_active_intent(
                self.campaign_dir,
                phase_name,
                int(state.iteration),
                expected_campaign_uid=str(state.campaign_uid),
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "cannot load PRE_SUBMIT intent for resource resolution: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        if not isinstance(active_intent, dict) or str(
            active_intent.get("status")
        ) != "PRE_SUBMIT":
            raise BackendSubmissionError(
                "live submission requires an active PRE_SUBMIT intent"
            )
        identity = str(active_intent["submission_identity"])
        effective_partition = (
            str(self.partition)
            if self.partition is not None
            else str(self.config.resources.partition_for(phase_name))
        )
        try:
            submitted_task_ids = (
                list(read_source_array_task_ids(array_task_map))
                if array_task_map is not None
                else None
            )
        except ValueError as exc:
            raise BackendSubmissionError(str(exc)) from exc
        evidence_override = None
        resource_task_ids = submitted_task_ids
        resource_evidence_mode = "computed"
        resource_evidence_source = None
        if phase_name == "ARIADNE_ARRAY" and array_size is not None:
            from .ariadne_resource_reuse import (
                resolve_reusable_ariadne_resource_evidence,
            )

            logical_task_ids = (
                list(submitted_task_ids)
                if submitted_task_ids is not None
                else list(range(int(array_size)))
            )
            try:
                resource_evidence_source = (
                    resolve_reusable_ariadne_resource_evidence(
                        self.campaign_dir,
                        self.config,
                        active_intent,
                        expected_campaign_uid=str(state.campaign_uid),
                        iteration=int(state.iteration),
                        replacement_round=int(
                            getattr(state, "replacement_round", 0)
                        ),
                        expected_scheduler_kind=self.scheduler_identity_kind,
                        expected_models_version=int(state.models_version),
                        submitted_task_ids=logical_task_ids,
                    )
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise BackendSubmissionError(
                    "ARIADNE resource-evidence reuse is unsafe: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            if resource_evidence_source is not None:
                evidence_override = dict(resource_evidence_source.evidence)
                evidence_override["resource_evidence_reuse"] = {
                    "schema_version": 1,
                    "fingerprint_algorithm": (
                        resource_evidence_source.fingerprint_algorithm
                    ),
                    "equivalence_basis": (
                        resource_evidence_source.equivalence_basis
                    ),
                    "source_submission_identity": (
                        resource_evidence_source.source_submission_identity
                    ),
                    "source_attempt_id": (
                        resource_evidence_source.source_attempt_id
                    ),
                    "source_resolution_sha256": (
                        resource_evidence_source.source_resolution_sha256
                    ),
                }
                resource_task_ids = (
                    None
                    if resource_evidence_source.already_filtered
                    else logical_task_ids
                )
                resource_evidence_mode = "reused"
                self._report_runtime_progress(
                    "ariadne_resource_reuse",
                    completed=len(logical_task_ids),
                    total=len(logical_task_ids),
                    unit="retry tasks",
                    source_submission_identity=(
                        resource_evidence_source.source_submission_identity
                    ),
                    source_attempt_id=(
                        resource_evidence_source.source_attempt_id
                    ),
                )
            else:
                self._report_runtime_progress("resource_resolution")
        else:
            self._report_runtime_progress("resource_resolution")
        resolved = resolve_phase_resources(
            phase_name=phase_name,
            config=self.config,
            partition=effective_partition,
            campaign_dir=self.campaign_dir,
            iteration=int(state.iteration),
            array_size=array_size,
            replacement_round=int(getattr(state, "replacement_round", 0)),
            staging_dir=(
                self._quantum_staging_path(state, phase_name)
                if "GAUSSIAN" in phase_name or "AIMALL" in phase_name
                else None
            ),
            expected_models_version=(
                int(state.models_version)
                if phase_name == "ARIADNE_ARRAY"
                else None
            ),
            submitted_task_ids=resource_task_ids,
            evidence_override=evidence_override,
            require_evidence=True,
            progress_callback=self._report_runtime_progress,
        )
        self._report_runtime_progress("script_rendering")
        try:
            bundle = prepare_attempt_bundle(
                self.campaign_dir,
                phase_name,
                int(state.iteration),
                identity,
                array_size=array_size,
                max_log_files_per_directory=(
                    _configured_max_job_log_files_per_directory()
                ),
                source_array_task_map=array_task_map,
            )
            resolution_evidence = dict(resolved.extra.get("evidence") or {})
            if bundle.array_task_map is not None:
                from .script_bundles import read_array_task_map
                from ..versioning.manifest import sha256_file

                copied_ids = [
                    int(task_id)
                    for task_id in read_array_task_map(bundle.array_task_map)
                ]
                if copied_ids != list(submitted_task_ids or []):
                    raise ValueError(
                        "attempt array task map does not match resolved retry tasks"
                    )
                resolution_evidence["submitted_array_task_map"] = {
                    "path": str(bundle.array_task_map.resolve()),
                    "size": int(bundle.array_task_map.stat().st_size),
                    "sha256": sha256_file(bundle.array_task_map),
                }
        except ValueError as exc:
            raise BackendSubmissionError(str(exc)) from exc
        scratch_template = scratch_path_template(
            self.campaign_dir,
            phase_name,
            int(state.iteration),
            identity,
            scheduler_kind=self.scheduler_identity_kind,
        )
        payload = resolution_payload(
            campaign_uid=str(state.campaign_uid),
            phase_name=phase_name,
            iteration=int(state.iteration),
            attempt_id=str(active_intent["attempt_id"]),
            submission_identity=identity,
            resolved=resolved,
            evidence=resolution_evidence,
            scratch_path_template=scratch_template,
            implementation_identity=capture_implementation_identity(
                self.campaign_dir,
                backend=str(resolved.backend),
                backend_executable_path=(
                    _configured_backend_path("gaussian", "g16")
                    if str(resolved.backend) == "gaussian"
                    else _configured_backend_path("aimall", "aimqb.ish")
                    if str(resolved.backend) == "aimall"
                    else None
                ),
                require_environment_generation=False,
            ),
        )
        try:
            resolution_binding = write_resolution(self.campaign_dir, payload)
            _submission_intent.bind_resource_resolution(
                self.campaign_dir,
                phase_name,
                int(state.iteration),
                path=str(resolution_binding["path"]),
                sha256=str(resolution_binding["sha256"]),
                formula_version=str(resolution_binding["formula_version"]),
                scratch_path_template=scratch_template,
                expected_tasks=(
                    int(array_size) if array_size is not None else 1
                ),
            )
        except (OSError, ValueError) as exc:
            raise BackendSubmissionError(
                "cannot persist immutable resource resolution: " + str(exc)
            ) from exc
        self._journal_event(
            "resolved_phase_resources",
            **resolved.journal_payload(phase_name=phase_name),
            resource_resolution_path=str(resolution_binding["path"]),
            resource_resolution_sha256=str(resolution_binding["sha256"]),
            resource_evidence_mode=resource_evidence_mode,
            resource_evidence_source_submission_identity=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.source_submission_identity
            ),
            resource_evidence_source_attempt_id=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.source_attempt_id
            ),
            resource_evidence_source_resolution_path=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.source_resolution_path
            ),
            resource_evidence_source_resolution_sha256=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.source_resolution_sha256
            ),
            resource_evidence_source_tasks=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.source_task_count
            ),
            resource_evidence_fingerprint_algorithm=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.fingerprint_algorithm
            ),
            resource_evidence_equivalence_basis=(
                None
                if resource_evidence_source is None
                else resource_evidence_source.equivalence_basis
            ),
        )
        body = build_scheduler_script(
            phase_name=phase_name,
            iteration=state.iteration,
            campaign_dir=self.campaign_dir,
            config=self.config,
            array_size=array_size,
            array_task_map=bundle.array_task_map,
            walltime_hours=self.walltime_hours,
            partition=self.partition,
            campaign_uid=getattr(state, "campaign_uid", None),
            replacement_round=int(getattr(state, "replacement_round", 0)),
            resolved_resources=resolved,
            attempt_bundle=bundle,
            submission_intent=active_intent,
            resource_resolution_binding=resolution_binding,
            scheduler_kind=self.scheduler_identity_kind,
        )
        script = write_attempt_script(bundle, body)
        script_binding = write_script_binding(bundle)
        try:
            _submission_intent.bind_submission_script(
                self.campaign_dir,
                phase_name,
                int(state.iteration),
                script_path=str(script.resolve()),
                script_sha256=str(script_binding["script_sha256"]),
                binding_path=str(script_binding["path"]),
                binding_sha256=str(script_binding["sha256"]),
            )
        except (OSError, ValueError) as exc:
            raise BackendSubmissionError(
                "cannot bind final submitted script: " + str(exc)
            ) from exc
        return script

    # --- postprocess (CSF4-only implementation) -------------------------

    def _parse_staged_pointdirs(self, staging_root, *, validators, allowed_pointdir_names=None):
        """Walk staging_root for POINT_*.pointdir/ children, run
        every validator on each, return (kept, rejected).

        Parameters
        ----------
        staging_root
            Directory expected to contain POINT_NNNN.pointdir subdirectories
            (typically ".DATA/STAGING/initial/" or ".DATA/STAGING/iter_<N>/").
        validators
            Sequence of callables, each "validator(pdir) -> (ok, reason)".
            A pointdir passes only if EVERY validator returns ok=True.

        Returns
        -------
        kept : List[PointDirectory]
            Pointdirs that pass every validator.
        rejected : List[Tuple[str, str]]
            "(pointdir_name, first_failure_reason)" for every reject.
        """
        from pathlib import Path as _Path
        from ichor.core.files.point_directory import PointDirectory

        staging_root = _Path(staging_root)
        kept = []
        rejected = []
        if not staging_root.is_dir():
            return kept, rejected
        allowed = None
        if allowed_pointdir_names is not None:
            allowed = {str(name) for name in allowed_pointdir_names}
            actual = {
                child.name
                for child in staging_root.iterdir()
                if child.is_dir() and PointDirectory.check_path(child)
            }
            extras = sorted(actual - allowed)
            if extras:
                raise ValueError(
                    "staging contains unsubmitted pointdirs: " + ", ".join(extras[:5])
                )
        children = sorted(staging_root.iterdir()) if allowed is None else [
            staging_root / name for name in sorted(allowed)
        ]
        total_children = len(children)
        for child_index, child in enumerate(children, start=1):
            if child_index == 1 or (child_index - 1) % 16 == 0:
                self._report_runtime_progress(
                    "structural_parsing",
                    completed=int(child_index - 1),
                    total=int(total_children),
                    unit="point directories",
                )
            if allowed is not None and not child.exists():
                rejected.append((child.name, "submitted_pointdir_missing"))
                continue
            if not (child.is_dir() and PointDirectory.check_path(child)):
                continue
            pdir = PointDirectory(child)
            failure_reason = None
            for validator in validators:
                ok, reason = validator(pdir)
                if not ok:
                    failure_reason = reason
                    break
            if failure_reason is None:
                kept.append(pdir)
            else:
                rejected.append((child.name, failure_reason))
            if child_index == total_children or child_index % 16 == 0:
                self._report_runtime_progress(
                    "structural_parsing",
                    completed=int(child_index),
                    total=int(total_children),
                    unit="point directories",
                )
        if total_children:
            self._report_runtime_progress(
                "structural_parsing",
                completed=int(total_children),
                total=int(total_children),
                unit="point directories",
            )
        return kept, rejected

    def postprocess(self, state, phase, observations: Sequence[Any]) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        self._set_journal_state(state, phase_name)
        if phase_name == CampaignPhase.ARIADNE_ARRAY.value:
            archived = archive_stale_ariadne_publication(
                self.campaign_dir,
                state,
                retry_task_ids=[],
            )
            if isinstance(archived, dict) and bool(archived.get("changed", False)):
                self._journal_event(
                    "ariadne_publication_archived",
                    phase=phase_name,
                    iteration=int(getattr(state, "iteration", 0)),
                    reason="ariadne_postprocess_only_recovery",
                    archive_id=str(archived.get("archive_id") or ""),
                    archive_manifest=str(archived.get("manifest_path") or ""),
                    n_files=int(len(archived.get("archived_paths") or [])),
                )
        handler = self._live_postprocess_handlers().get(phase_name)
        if handler is not None:
            stage = (
                "model_parsing"
                if "FEREBUS" in phase_name
                else "result_parsing"
                if phase_name == "ARIADNE_ARRAY"
                else "descriptor_construction"
                if "DIVERSITY" in phase_name
                else "structural_parsing"
            )
            self._report_runtime_progress(stage)
            #handlers take (state, phase, observations) so the shared
            #handler can dispatch by phase name to the right validator.
            return handler(state, phase, observations)
        # if this is a registered SBATCH phase, refuse rather than
        #silently call super() -- the dry-run postprocess writes stub
        # artefacts that would overwrite real backend output.
        if phase_name in SBATCH_PHASES and phase_name not in LIVE_POSTPROCESS_IMPLEMENTED:
            self._journal_event(
                "live_postprocess_refused",
                phase=phase_name,
                iteration=int(getattr(state, "iteration", -1)),
            )
            raise NotImplementedError(
                "Live postprocess for " + phase_name + " is not yet implemented. "
                "Refusing to overwrite real backend output with dry-run stub "
                "artefacts. Register a real parser in LIVE_POSTPROCESS_IMPLEMENTED "
                "+ _live_postprocess_handlers, or use a separate --mode dry_run campaign"
            )
        # Inline equivalent bookkeeping (manifests / versioning) defers to the
        #dry-run executor so artefact handling is identical between executors.
        return super().postprocess(state, phase, observations)

    def _live_postprocess_handlers(self):
        """Return phase name -> handler dict for postprocess dispatch.

        Four ab initio phases (Gaussian + AIMAll, INITIAL + iter)
        all share _parse_quantum_postprocess. The handler reads its phase
        argument and picks the right validator set internally.
        """
        return {
            "INITIAL_GAUSSIAN": self._parse_quantum_postprocess,
            "GAUSSIAN":         self._parse_quantum_postprocess,
            "INITIAL_AIMALL":   self._parse_quantum_postprocess,
            "AIMALL":           self._parse_quantum_postprocess,
            "INITIAL_REPLACEMENT_GAUSSIAN": self._parse_quantum_postprocess,
            "REPLACEMENT_GAUSSIAN": self._parse_quantum_postprocess,
            "INITIAL_REPLACEMENT_AIMALL": self._parse_quantum_postprocess,
            "REPLACEMENT_AIMALL": self._parse_quantum_postprocess,
            "INITIAL_FEREBUS":  self._parse_ferebus_postprocess,
            "FEREBUS":          self._parse_ferebus_postprocess,
            "ARIADNE_ARRAY":    self._parse_ariadne_array_postprocess,
            "PHASE_A_DIVERSITY":    self._parse_diversity_postprocess,
            "PHASE_B_DIVERSITY":    self._parse_diversity_postprocess,
        }

    # --- quantum-phase parser body ----------------------------

    def _quantum_staging_path(self, state, phase_name):
        """Return the canonical staging root for the given quantum phase.

        INITIAL_GAUSSIAN / INITIAL_AIMALL share ".DATA/STAGING/initial/";
        GAUSSIAN / AIMALL share ".DATA/STAGING/iter_<N>/" (per-iteration).
        """
        if "REPLACEMENT" in str(phase_name):
            from ..replacement_sampling import replacement_round_dir

            context = "bootstrap" if str(phase_name).startswith("INITIAL_") else "active"
            return replacement_round_dir(
                self.campaign_dir,
                context=context,
                iteration=0 if context == "bootstrap" else int(state.iteration),
                replacement_round=int(getattr(state, "replacement_round", 0)),
            )
        from ..layout import staging_phase_dir

        return staging_phase_dir(
            self.campaign_dir,
            str(phase_name),
            int(state.iteration),
        )

    def _validators_for(self, phase_name):
        """Pick the right validator tuple for the phase. Gaussian / AIMAll
        validators are independent; iterative-phase validation only checks
        the phase-specific output not the prior phase's output (the prior
        phase already had its own postprocess call to validate)."""
        if "GAUSSIAN" in phase_name:
            return (validate_gaussian_completed,)
        if "AIMALL" in phase_name:
            return (validate_aimall_completed,)
        raise ValueError("unknown quantum phase: " + phase_name)

    def _parse_quantum_postprocess(self, state, phase, observations):
        """Shared postprocess for the four quantum phases.

        Reads the canonical staging root (via _quantum_staging_path),
        applies the per-phase validator(s) to each POINT_*.pointdir/, and:
          - journals quantum_output_rejected per rejected pointdir.
          - journals phase_succeeded_live on overall success.
          - records the combined Gaussian/AIMAll outcome in the exact point
            allocation after AIMAll.

        Does NOT commit anything to QM_REFERENCE_DATA / TRAINED_MODELS itself --
        that lives in the subsequent REFERENCE_COMMIT / INITIAL_FEREBUS / FEREBUS
        phases. Rejection counts are allocation-managed: failed slots proceed
        to bounded reserve replacement instead of tripping a batch-level
        percentage threshold here.
        """
        from .phase_executor import PhaseResult
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        staging_root = self._quantum_staging_path(state, phase_name)
        validators = self._validators_for(phase_name)
        from . import input_staging as _stg

        if not Path(staging_root).is_dir():
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "no_pointdirs_in_staging: " + str(staging_root)
                ),
                retry_disposition=(
                    PostprocessRetryDisposition.FILESYSTEM_SETTLE
                ),
            )
        try:
            point_names = _stg._points_file_names(staging_root)
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "quantum_task_membership_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:180]
                ),
            )
        task_id_by_name = {name: index for index, name in enumerate(point_names)}

        def publish_task_receipts(candidates):
            from .quantum_task_receipts import write_quantum_task_receipt

            total_candidates = len(candidates)
            self._report_runtime_progress(
                "receipt_publication",
                completed=0,
                total=int(total_candidates),
                unit="task receipts",
            )
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_path = Path(getattr(candidate, "path", candidate))
                if candidate_path.name not in task_id_by_name:
                    raise ValueError(
                        "validated quantum task is outside canonical POINTS.txt"
                    )
                write_quantum_task_receipt(
                    self.campaign_dir,
                    candidate_path,
                    phase_name=phase_name,
                    iteration=int(state.iteration),
                    logical_task_id=task_id_by_name[candidate_path.name],
                )
                if candidate_index == total_candidates or candidate_index % 16 == 0:
                    self._report_runtime_progress(
                        "receipt_publication",
                        completed=int(candidate_index),
                        total=int(total_candidates),
                        unit="task receipts",
                    )

        gaussian_previous_manifest = None
        gaussian_membership_archive = None
        gaussian_aimall_phase = None
        if "AIMALL" in phase_name:
            from ichor.core.files.point_directory import PointDirectory
            from .quantum_quality import (
                evaluate_aimall_pointdir,
                read_quantum_quality_manifest,
                write_quantum_quality_manifest,
            )

            expected_phase = (
                "INITIAL_REPLACEMENT_GAUSSIAN"
                if phase_name == "INITIAL_REPLACEMENT_AIMALL"
                else "REPLACEMENT_GAUSSIAN"
                if phase_name == "REPLACEMENT_AIMALL"
                else "INITIAL_GAUSSIAN"
                if phase_name.startswith("INITIAL_")
                else "GAUSSIAN"
            )
            try:
                gaussian_accepted, _gaussian_manifest = _stg.read_quantum_acceptance_manifest(
                    staging_root,
                    expected_phase=expected_phase,
                    expected_iteration=int(state.iteration),
                    points_membership=(
                        _stg.POINTS_MEMBERSHIP_PRODUCER_OR_ACCEPTED
                    ),
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "prior_gaussian_acceptance_manifest_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            unsettled = []
            total_accepted = len(gaussian_accepted)
            self._report_runtime_progress(
                "output_visibility",
                completed=0,
                total=int(total_accepted),
                unit="point directories",
            )
            for candidate_index, candidate in enumerate(gaussian_accepted, start=1):
                issue = _aimall_visibility_issue(Path(candidate))
                if issue is not None:
                    unsettled.append(Path(candidate).name + ":" + issue)
                if candidate_index == total_accepted or candidate_index % 16 == 0:
                    self._report_runtime_progress(
                        "output_visibility",
                        completed=int(candidate_index),
                        total=int(total_accepted),
                        unit="point directories",
                    )
            if unsettled:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "aimall_outputs_not_settled_missing_or_unreadable: "
                        + "; ".join(unsettled[:8])
                    ),
                    retry_disposition=(
                        PostprocessRetryDisposition.FILESYSTEM_SETTLE
                    ),
                )

            structurally_complete = []
            structural_failures = {}
            pointdirs = []
            self._report_runtime_progress(
                "structural_parsing",
                completed=0,
                total=int(total_accepted),
                unit="point directories",
            )
            for candidate_index, candidate in enumerate(gaussian_accepted, start=1):
                pdir = PointDirectory(candidate)
                pointdirs.append(pdir)
                failure_reason = None
                for validator in validators:
                    ok, reason = validator(pdir)
                    if not ok:
                        failure_reason = reason
                        break
                if failure_reason is None:
                    structurally_complete.append(pdir)
                else:
                    structural_failures[Path(candidate).name] = str(failure_reason)
                if candidate_index == total_accepted or candidate_index % 16 == 0:
                    self._report_runtime_progress(
                        "structural_parsing",
                        completed=int(candidate_index),
                        total=int(total_accepted),
                        unit="point directories",
                    )

            quality_records = []
            quality_kept = []
            quality_rejected = []
            try:
                publish_task_receipts(structurally_complete)
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "quantum_task_receipt_failed: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            total_pointdirs = len(pointdirs)
            self._report_runtime_progress(
                "scientific_quality",
                completed=0,
                total=int(total_pointdirs),
                unit="point directories",
            )
            for point_index, pdir in enumerate(pointdirs, start=1):
                record = evaluate_aimall_pointdir(
                    pdir,
                    getattr(self.config, "quality_gates", None),
                    expected_method=str(self.config.gaussian.method),
                )
                pointdir_name = str(
                    record.get("pointdir", Path(getattr(pdir, "path", pdir)).name)
                )
                structural_reason = structural_failures.get(pointdir_name)
                if structural_reason is not None:
                    reasons = sorted(
                        set(record.get("reasons") or []) | {structural_reason}
                    )
                    record["accepted"] = False
                    record["reasons"] = reasons
                quality_records.append(record)
                if bool(record.get("accepted")):
                    quality_kept.append(pdir)
                else:
                    quality_rejected.append(
                        (
                            str(record.get("pointdir", Path(getattr(pdir, "path", pdir)).name)),
                            ";".join(record.get("reasons") or ["quantum_quality_rejected"]),
                        )
                    )
                if point_index == total_pointdirs or point_index % 16 == 0:
                    self._report_runtime_progress(
                        "scientific_quality",
                        completed=int(point_index),
                        total=int(total_pointdirs),
                        unit="point directories",
                        accepted=int(len(quality_kept)),
                        rejected=int(len(quality_rejected)),
                    )
            kept = quality_kept
            rejected = quality_rejected
            quality_path = write_quantum_quality_manifest(
                staging_root,
                phase_name=phase_name,
                iteration=int(state.iteration),
                records=quality_records,
                gates=getattr(self.config, "quality_gates", None),
            )
            read_quantum_quality_manifest(
                staging_root,
                expected_phase=phase_name,
                expected_iteration=int(state.iteration),
                expected_pointdirs=[str(record["pointdir"]) for record in quality_records],
                expected_method=str(self.config.gaussian.method),
            )
            try:
                from .quantum_acceptance_receipts import (
                    write_quantum_acceptance_receipt,
                )

                quality_by_name = {
                    str(record["pointdir"]): record
                    for record in quality_records
                    if bool(record.get("accepted"))
                }
                total_kept = len(kept)
                self._report_runtime_progress(
                    "acceptance_publication",
                    completed=0,
                    total=int(total_kept),
                    unit="point directories",
                )
                for kept_index, pdir in enumerate(kept, start=1):
                    pointdir_path = Path(getattr(pdir, "path", pdir))
                    write_quantum_acceptance_receipt(
                        self.campaign_dir,
                        pointdir_path,
                        phase_name=phase_name,
                        iteration=int(state.iteration),
                        quality_manifest=quality_path,
                        quality_record=quality_by_name[pointdir_path.name],
                    )
                    if kept_index == total_kept or kept_index % 16 == 0:
                        self._report_runtime_progress(
                            "acceptance_publication",
                            completed=int(kept_index),
                            total=int(total_kept),
                            unit="point directories",
                        )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "quantum_acceptance_receipt_failed: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            self._journal_event(
                "quantum_quality_summary",
                phase=phase_name,
                iteration=int(state.iteration),
                manifest=str(quality_path),
                n_total=int(len(quality_records)),
                n_rejected=int(sum(1 for r in quality_records if not bool(r.get("accepted")))),
            )
        else:
            from .quantum_task_contracts import aimall_phase_for_gaussian

            gaussian_aimall_phase = aimall_phase_for_gaussian(phase_name)
            try:
                _stg.resume_aimall_membership_changes(
                    self.campaign_dir,
                    staging_dir=staging_root,
                    gaussian_phase=phase_name,
                    aimall_phase=gaussian_aimall_phase,
                    iteration=int(state.iteration),
                )
                previous_path = _stg.quantum_acceptance_manifest_path(
                    staging_root,
                    phase_name=phase_name,
                )
                if previous_path.exists() or previous_path.is_symlink():
                    if previous_path.is_symlink():
                        raise ValueError(
                            "previous Gaussian acceptance must not be a symlink"
                        )
                    _unused, gaussian_previous_manifest = (
                        _stg.read_quantum_acceptance_manifest(
                            staging_root,
                            expected_phase=phase_name,
                            expected_iteration=int(state.iteration),
                            require_nonempty=False,
                            points_membership=_stg.POINTS_MEMBERSHIP_NONE,
                            require_accepted_payloads=False,
                        )
                    )
                    dispositions = list(
                        gaussian_previous_manifest["accepted_pointdirs"]
                    ) + [
                        str(record["pointdir"])
                        for record in list(
                            gaussian_previous_manifest["rejected"]
                        )
                    ]
                    if (
                        int(gaussian_previous_manifest["n_total"])
                        != len(point_names)
                        or set(dispositions) != set(point_names)
                        or len(dispositions) != len(set(dispositions))
                    ):
                        raise ValueError(
                            "previous Gaussian acceptance does not exactly "
                            "cover POINTS.txt"
                        )
                    _stg.publish_completed_aimall_sibling_receipts(
                        self.campaign_dir,
                        gaussian_phase=phase_name,
                        aimall_phase=gaussian_aimall_phase,
                        iteration=int(state.iteration),
                        replacement_round=int(
                            getattr(state, "replacement_round", 0)
                        ),
                    )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "gaussian_membership_recovery_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            try:
                allowed_names = _stg._points_file_names(staging_root)
                kept, rejected = self._parse_staged_pointdirs(
                    staging_root,
                    validators=validators,
                    allowed_pointdir_names=allowed_names,
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "quantum_task_membership_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            try:
                publish_task_receipts(kept)
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "quantum_task_receipt_failed: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
        try:
            if gaussian_previous_manifest is not None:
                gaussian_membership_archive = (
                    _stg.archive_aimall_membership_change(
                        self.campaign_dir,
                        staging_dir=staging_root,
                        gaussian_phase=phase_name,
                        aimall_phase=str(gaussian_aimall_phase),
                        iteration=int(state.iteration),
                        replacement_round=int(
                            getattr(state, "replacement_round", 0)
                        ),
                        previous_manifest=gaussian_previous_manifest,
                        replacement_accepted=[
                            Path(getattr(item, "path", item)).name
                            for item in kept
                        ],
                        replacement_rejected=rejected,
                    )
                )
            _stg.write_quantum_acceptance_manifest(
                staging_root,
                phase_name=phase_name,
                iteration=int(state.iteration),
                accepted=kept,
                rejected=rejected,
            )
            if gaussian_membership_archive is not None:
                _stg.finalise_aimall_membership_change(
                    self.campaign_dir,
                    staging_dir=staging_root,
                    archive_root=gaussian_membership_archive,
                    gaussian_phase=phase_name,
                    aimall_phase=str(gaussian_aimall_phase),
                    iteration=int(state.iteration),
                )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "quantum_acceptance_publication_failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:180]
                ),
            )
        self._report_runtime_progress(
            "acceptance_publication",
            completed=int(len(kept) + len(rejected)),
            total=int(len(kept) + len(rejected)),
            unit="point directories",
        )

        allocation_payload = None
        if "AIMALL" in phase_name:
            self._report_runtime_progress("allocation_join")
            context = "bootstrap" if phase_name.startswith("INITIAL_") else "active"
            gaussian_phase = (
                "INITIAL_REPLACEMENT_GAUSSIAN"
                if phase_name == "INITIAL_REPLACEMENT_AIMALL"
                else "REPLACEMENT_GAUSSIAN"
                if phase_name == "REPLACEMENT_AIMALL"
                else "INITIAL_GAUSSIAN"
                if phase_name == "INITIAL_AIMALL"
                else "GAUSSIAN"
            )
            try:
                allocation_payload = _stg.record_allocation_quantum_results(
                    self.campaign_dir,
                    context=context,
                    iteration=0 if context == "bootstrap" else int(state.iteration),
                    staging_dir=staging_root,
                    gaussian_phase=gaussian_phase,
                    aimall_phase=phase_name,
                    expected_method=str(self.config.gaussian.method),
                    expected_campaign_uid=str(state.campaign_uid),
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "point_allocation_quantum_join_failed: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            self._journal_event(
                "point_allocation_quantum_recorded",
                phase=phase_name,
                context=context,
                iteration=int(state.iteration),
                summary=dict(allocation_payload.get("summary") or {}),
            )
            allocation_slots = list(allocation_payload.get("slots") or [])
            self._report_runtime_progress(
                "allocation_join",
                completed=int(len(allocation_slots)),
                total=int(len(allocation_slots)),
                unit="slots",
            )
            if phase_name in ("AIMALL", "REPLACEMENT_AIMALL"):
                joined_names = {
                    Path(str(attempt.get("pointdir") or "")).name
                    for slot in allocation_slots
                    if isinstance(slot, dict)
                    for attempt in list(slot.get("attempts") or [])
                    if isinstance(attempt, dict)
                    and str(attempt.get("status") or "") == "accepted"
                }
                calibration_pointdirs = [
                    pdir
                    for pdir in kept
                    if Path(getattr(pdir, "path", pdir)).name in joined_names
                ]
                if len(calibration_pointdirs) != len(kept):
                    return PhaseResult(
                        is_complete=True,
                        failure_reason=(
                            "error_calibration_allocation_join_mismatch: accepted AIMAll "
                            "pointdirs are not exactly represented by the allocation update"
                        ),
                    )
                try:
                    self._report_runtime_progress(
                        "calibration",
                        completed=0,
                        total=int(len(calibration_pointdirs)),
                        unit="point directories",
                    )
                    from .error_calibration import (
                        ERROR_CALIBRATION_MODEL_FILENAME,
                        audit_path as error_calibration_audit_path,
                        update_from_aimall_acceptance,
                    )

                    iter_dir = self._iter_dir(state.iteration)
                    audit = update_from_aimall_acceptance(
                        campaign_dir=self.campaign_dir,
                        iter_dir=iter_dir,
                        config=self.config,
                        iteration=int(state.iteration),
                        models_version=int(getattr(state, "models_version", -1)),
                        accepted_pointdirs=calibration_pointdirs,
                        quality_records=quality_records,
                    )
                    self.artefact_log.append(
                        str(error_calibration_audit_path(iter_dir).resolve())
                    )
                    self.artefact_log.append(
                        str(
                            (
                                Path(self.campaign_dir)
                                / ".DATA" / "ACTIVE_LEARNING"
                                / ERROR_CALIBRATION_MODEL_FILENAME
                            ).resolve()
                        )
                    )
                    self._journal_event(
                        "error_calibration_summary",
                        phase=phase_name,
                        iteration=int(state.iteration),
                        n_added_records=int(audit.get("n_added_records", 0)),
                        n_total_records=int(audit.get("n_total_records", 0)),
                        usable_for_acquisition=bool(
                            audit.get("usable_for_acquisition", False)
                        ),
                    )
                    self._report_runtime_progress(
                        "calibration",
                        completed=int(len(calibration_pointdirs)),
                        total=int(len(calibration_pointdirs)),
                        unit="point directories",
                    )
                except Exception as exc:
                    try:
                        from .error_calibration import mark_calibration_model_stale

                        mark_calibration_model_stale(
                            self.campaign_dir,
                            reason=type(exc).__name__ + ": " + str(exc)[:240],
                            iteration=int(state.iteration),
                        )
                    except Exception:
                        pass
                    self._journal_event(
                        "error_calibration_failed",
                        phase=phase_name,
                        iteration=int(state.iteration),
                        reason=type(exc).__name__ + ": " + str(exc)[:240],
                    )

        for pdir_name, reason in rejected:
            self._journal_event(
                "quantum_quality_rejected" if "quality" in str(reason) or "iqa_" in str(reason) or "integration_" in str(reason) else "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=pdir_name,
                reason=reason,
            )

        if "AIMALL" in phase_name:
            override = (
                "INITIAL_ALLOCATION_CHECK"
                if phase_name == "INITIAL_REPLACEMENT_AIMALL"
                else "ALLOCATION_CHECK"
                if phase_name == "REPLACEMENT_AIMALL"
                else None
            )
            self._journal_event(
                "phase_succeeded_live",
                phase=phase_name,
                iteration=int(state.iteration),
                n_kept=int(len(kept)),
                n_rejected=int(len(rejected)),
                allocation_managed=True,
            )
            return PhaseResult(
                is_complete=True,
                state_updates={},
                next_phase_override=override,
            )
        if "GAUSSIAN" in phase_name:
            self._journal_event(
                "phase_succeeded_live",
                phase=phase_name,
                iteration=int(state.iteration),
                n_kept=int(len(kept)),
                n_rejected=int(len(rejected)),
                allocation_managed=True,
            )
            return PhaseResult(is_complete=True, state_updates={})
        raise AssertionError("unhandled quantum phase: " + phase_name)

    # ---helpers ---------------------------------------------

    def _models_staging_path(self):
        """Path to the FEREBUS staging directory.

        FEREBUS writes a .model file (single artefact per training run)
        to this canonical location; the parser validates and atomically
        renames it into TRAINED_MODELS/iteration-NNNNNN/ via the
        VersionedDirectory helper.
        """
        from pathlib import Path as _Path
        return (
            _Path(self.campaign_dir)
            / self.models_dir_name
            / "iteration-staging"
        )

    def _initial_quantum_staging_path(self):
        """Path the initial diversity sample staging dir lives at."""
        from ..layout import staging_phase_dir

        return staging_phase_dir(self.campaign_dir, "INITIAL_GAUSSIAN", 0)


    # --- reference scales from a real GP posterior ----------------------

    def _maybe_refresh_reference_scales(self, state) -> bool:
        """Compute per-iteration reference scales from the trained
        FEREBUS models, persist to a sidecar, update state.

        Reference scales are the five per-property anchors the adversarial
        acquisition uses to normalise its energy, force, frequency,
        anharmonicity and anharmonic-std contributions before combining
        them with the configured lambda weights. Without real scales the
        lambda weights operate on mixed-unit quantities and the result is
        nonsense -- which is exactly the dry-run synthetic stub case.

        Placement: inline at SEED_SELECT time (on the login node). costs
        roughly 480 GP evals per iteration for a 12-atom system, which is
        5-30 seconds. larger systems or higher subspace.max_subspace_dim
        push that toward a minute; if it ever becomes painful, promote
        the work to a dedicated REFERENCE_SCALES sbatch phase. for now
        the cost is small enough that the login node is the cheapest
        right place.

        Honours acquisition.references.refresh_policy exactly the same way
        the dry-run path does -- the only difference is what the cache
        gets populated with.
        """
        from ichor.core.adversarial.acquisition import compute_reference_scales
        from pathlib import Path as _Path
        from .model_contract import validate_reference_scales
        from .artifact_contracts import verify_committed_model_version

        # if we have not committed any models yet (pre-INITIAL_FEREBUS),
        # there is no posterior to sample. leave state alone -- the dry
        # synthetic path will not fire either at this point, and the
        # next call after INITIAL_FEREBUS commits will populate the cache.
        models_version = int(getattr(state, "models_version", -1))
        if models_version < 0:
            return False
        # same policy decision as the dry-run path -- decide whether to
        # recompute. we duplicate the small block here rather than calling
        # the _dry method, because the dry method would clobber the real
        # scales with its synthetic dict if we let it through.
        refs = self.config.acquisition.references
        policy = refs.refresh_policy
        prev_iter = int(getattr(state, "reference_scales_iteration", -1))
        prev_scales = getattr(state, "reference_scales", None)
        should_refresh = False
        if prev_scales is None:
            should_refresh = True
        elif prev_iter == int(state.iteration):
            should_refresh = False
        elif policy == "every_iteration":
            should_refresh = True
        elif policy == "every_n_iterations":
            period = max(1, int(refs.refresh_period))
            should_refresh = (int(state.iteration) - prev_iter) >= period
        elif policy == "never":
            should_refresh = False
        if not should_refresh:
            try:
                from ..layout import active_iteration_dir, active_protocol_dir
                from ..reference_scale_snapshot import (
                    build_reference_scale_snapshot,
                    write_reference_scale_snapshot,
                )

                source_models_version = int(
                    getattr(state, "reference_scales_models_version", -1)
                )
                source_manifest_sha = getattr(
                    state,
                    "reference_scales_model_manifest_sha256",
                    None,
                )
                snapshot = build_reference_scale_snapshot(
                    iteration=int(state.iteration),
                    source_iteration=prev_iter,
                    models_version=source_models_version,
                    model_set_manifest_sha256=source_manifest_sha,
                    values=prev_scales,
                )
                protocol_dir = active_protocol_dir(
                    active_iteration_dir(
                        self.campaign_dir,
                        int(state.iteration),
                    )
                )
                write_reference_scale_snapshot(
                    protocol_dir / "reference_scales.json",
                    snapshot,
                )
            except Exception as exc:
                raise BackendSubmissionError(
                    "cached reference scales cannot be materialised for this "
                    "iteration: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            return False

        # Load once when this method is called outside the shared live
        # SEED_SELECT context; normal daemon execution reuses the objects that
        # were verified at command entry.
        context = getattr(self, "_active_seed_selection_context", None)
        from ..versioning.trained_models import (
            TrainedModelVersioning,
            load_trained_models,
            trained_model_set_path,
        )
        from ..versioning.manifest import sha256_file
        from ..acquisition.trajectory_pool import TrajectoryPool

        try:
            if isinstance(context, dict):
                model_set = context["model_set"]
                models = context["models"]
                pool = context["pool"]
                posterior = context["posterior"]
                model_manifest_sha256 = str(model_set.head_manifest_sha256)
            else:
                models_dir = TrainedModelVersioning(
                    _Path(self.campaign_dir) / self.models_dir_name
                ).iteration_path(models_version)
                if not models_dir.is_dir():
                    raise FileNotFoundError(
                        "reference scales require committed models: "
                        + str(models_dir)
                    )
                verify_committed_model_version(
                    self.campaign_dir,
                    models_version,
                    models_dir_name=self.models_dir_name,
                    verification="metadata",
                )
                _, models = load_trained_models(
                    self.campaign_dir,
                    models_version,
                    verification="metadata",
                    reference_verification="metadata",
                )
                pool = TrajectoryPool.load(_Path(self.campaign_dir))
                posterior = None
                model_manifest_sha256 = sha256_file(
                    trained_model_set_path(models_dir)
                )
        except Exception as exc:
            self._journal_event(
                "reference_scales_computed",
                iteration=int(state.iteration),
                policy=str(policy),
                error="load_failed: " + str(exc)[:80],
            )
            raise BackendSubmissionError(
                "reference scale model/pool load failed: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

        from ..layout import active_iteration_dir, active_protocol_dir
        from ..reference_scale_snapshot import read_reference_scale_snapshot

        iter_dir = active_iteration_dir(self.campaign_dir, int(state.iteration))
        protocol_dir = active_protocol_dir(iter_dir)
        sidecar = protocol_dir / "reference_scales.json"
        if sidecar.is_file() and not sidecar.is_symlink():
            try:
                existing_snapshot = read_reference_scale_snapshot(
                    sidecar,
                    expected_iteration=int(state.iteration),
                )
                if (
                    int(existing_snapshot["models_version"]) == models_version
                    and str(existing_snapshot["model_set_manifest_sha256"])
                    == str(model_manifest_sha256)
                ):
                    scales = validate_reference_scales(
                        dict(existing_snapshot["values"])
                    )
                    state.reference_scales = scales
                    state.reference_scales_iteration = int(
                        existing_snapshot["source_iteration"]
                    )
                    state.reference_scales_models_version = models_version
                    state.reference_scales_model_manifest_sha256 = str(
                        model_manifest_sha256
                    )
                    reporter = (
                        context.get("progress")
                        if isinstance(context, dict)
                        else None
                    )
                    if reporter is not None:
                        reporter.cache(
                            "reference_scales",
                            "adopted",
                            source_iteration=int(
                                existing_snapshot["source_iteration"]
                            ),
                        )
                        reporter.update(
                            "reference_scales",
                            force=True,
                            completed=1,
                            total=1,
                            cache_status="adopted",
                        )
                    self._journal_event(
                        "reference_scales_computed",
                        iteration=int(state.iteration),
                        policy=str(policy),
                        n_keys=int(len(scales)),
                        models_version=models_version,
                        adopted_existing=True,
                    )
                    return False
            except Exception:
                # A conflicting immutable sidecar remains a publication error;
                # the normal write below will reject it after recomputation.
                pass

        # pick a representative anchor geometry to fit the local subspace
        # around. the first frame in the trajectory pool is a fine choice;
        # using a seed from seed_selection/SELECTION.json would be more principled but
        # requires SEED_SELECT to have already written that file which it
        # has not at the call site.
        anchor_atoms = pool.frame(0)

        try:
            from ..sampling_protocol import resolve_or_load_sampling_protocol

            resolved_protocol = resolve_or_load_sampling_protocol(
                self.campaign_dir,
                self.config,
                iteration=int(state.iteration),
                trajectory_pool=pool,
            )
            acquisition_config = resolved_protocol.acquisition_config
        except Exception as exc:
            self._journal_event(
                "reference_scales_computed",
                iteration=int(state.iteration),
                policy=str(policy),
                error="sampling_protocol_failed: " + str(exc)[:80],
            )
            raise BackendSubmissionError(
                "sampling protocol resolution failed for reference scale computation: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

        reporter = context.get("progress") if isinstance(context, dict) else None
        runtime_cache = context.get("cache") if isinstance(context, dict) else None
        cached_neighbours = None
        neighbour_cache_id = None
        if reporter is not None:
            reporter.update(
                "reference_neighbours",
                force=True,
                completed=0,
                total=int(pool.n_frames()),
            )
        if runtime_cache is not None:
            cached_neighbours, neighbour_cache_id = (
                runtime_cache.load_reference_neighbours(
                    anchor_frame_id=0,
                    max_neighbours=int(
                        acquisition_config.subspace.neighbour_count
                    ),
                    deduplicate_rmsd=float(
                        acquisition_config.subspace.neighbour_deduplicate_rmsd
                    ),
                )
            )
            if reporter is not None and cached_neighbours is not None:
                reporter.update(
                    "reference_neighbours",
                    force=True,
                    completed=int(pool.n_frames()),
                    total=int(pool.n_frames()),
                    cache_status="reused",
                )

        def _reference_progress(payload):
            if reporter is not None:
                values = dict(payload)
                stage = str(values.pop("stage", "reference_scales"))
                reporter.update(stage, **values)

        try:
            # The reference-scale service computes only the state needed for
            # the iteration snapshot.
            acq = compute_reference_scales(
                models=models,
                seed=anchor_atoms,
                trajectory=pool,
                config=acquisition_config,
                seed_frame_id=0,
                posterior_override=posterior,
                preselected_neighbours=cached_neighbours,
                reference_progress=_reference_progress,
                use_prepared_reference_stencils=True,
            )
            scales = validate_reference_scales(dict(acq.reference_scales))
            if runtime_cache is not None and cached_neighbours is None:
                runtime_cache.store_reference_neighbours(
                    cache_id=str(neighbour_cache_id),
                    anchor_frame_id=0,
                    max_neighbours=int(
                        acquisition_config.subspace.neighbour_count
                    ),
                    deduplicate_rmsd=float(
                        acquisition_config.subspace.neighbour_deduplicate_rmsd
                    ),
                    neighbours=acq.subspace.neighbours,
                )
            if reporter is not None:
                reporter.update(
                    "reference_scales",
                    force=True,
                    completed=1,
                    total=1,
                    cache_status=(
                        "neighbours_reused"
                        if cached_neighbours is not None
                        else "neighbours_built"
                    ),
                )
        except Exception as exc:
            self._journal_event(
                "reference_scales_computed",
                iteration=int(state.iteration),
                policy=str(policy),
                error="compute_failed: " + str(exc)[:80],
            )
            raise BackendSubmissionError(
                "reference scale computation failed: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

        # persist to the per-iteration sidecar that ARIADNE_ARRAY tasks
        # read at the top of their main. saves them ~480 GP evaluations
        # per seed.
        from ..layout import active_iteration_dir, active_protocol_dir
        from ..reference_scale_snapshot import (
            build_reference_scale_snapshot,
            write_reference_scale_snapshot,
        )

        iter_dir = active_iteration_dir(self.campaign_dir, int(state.iteration))
        protocol_dir = active_protocol_dir(iter_dir)
        sidecar = protocol_dir / "reference_scales.json"
        snapshot = build_reference_scale_snapshot(
            iteration=int(state.iteration),
            source_iteration=int(state.iteration),
            models_version=models_version,
            model_set_manifest_sha256=model_manifest_sha256,
            values=scales,
        )
        write_reference_scale_snapshot(sidecar, snapshot)

        state.reference_scales = scales
        state.reference_scales_iteration = int(state.iteration)
        state.reference_scales_models_version = models_version
        state.reference_scales_model_manifest_sha256 = model_manifest_sha256
        self._journal_event(
            "reference_scales_computed",
            iteration=int(state.iteration),
            policy=str(policy),
            n_keys=int(len(scales)),
            models_version=models_version,
        )
        return True

    # --- FEREBUS parser body -------------------------------------------

    @_with_trained_models_commit_lock
    def _parse_ferebus_postprocess(self, state, phase, observations):
        """Parse FEREBUS output, validate the .model file, commit
        a new TRAINED_MODELS/iteration-NNNNNN/ via VersionedDirectory.

        REFERENCE_COMMIT must already have published the matching immutable QM
        reference-data version. This parser commits TRAINED_MODELS only.

        """
        from .phase_executor import PhaseResult

        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        self._report_runtime_progress("model_parsing")
        staging = self._models_staging_path()
        v_models = self._versioning("models")
        committed = v_models.list_committed_versions()
        is_initial = phase_name == "INITIAL_FEREBUS"
        if is_initial:
            expected_next = 0
            if int(getattr(state, "reference_data_version", -1)) != 0:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "initial_ferebus_requires_reference_data_version_zero"
                    ),
                )
        else:
            expected_next = int(getattr(state, "reference_data_version", -1))
            if expected_next < 0:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "ferebus_reference_data_version_invalid: "
                        + repr(getattr(state, "reference_data_version", None))
                    ),
                )
        if expected_next in committed:
            next_version = int(expected_next)
            committed_dir = v_models.iteration_path(next_version)
            try:
                from .model_contract import validate_ferebus_model_contract

                resolved_model_set = v_models.resolve(
                    next_version,
                    verification="metadata",
                )
                validate_ferebus_model_contract(
                    committed_dir,
                    committed=True,
                    expected_version=next_version,
                    trained_model_set=resolved_model_set,
                )
                newest_version = max(committed)
                if newest_version != next_version:
                    v_models.resolve(
                        newest_version,
                        verification="metadata",
                    )
                v_models.ensure_current(newest_version)
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "committed_model_contract_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
            self._journal_event(
                "models_committed",
                phase=phase_name,
                iteration=int(state.iteration),
                models_version=int(next_version),
                model_set_sha256=str(resolved_model_set.model_set_sha256),
                model_set_manifest_sha256=str(
                    resolved_model_set.head_manifest_sha256
                ),
                idempotent_skip=True,
            )
            state_updates = {"models_version": int(next_version), "validation_set_version": int(next_version)}
            if is_initial:
                from ..versioning.sampling_iterations import finalise_bootstrap

                try:
                    finalise_bootstrap(
                        self.campaign_dir,
                        str(state.campaign_uid),
                    )
                except Exception as exc:
                    return PhaseResult(
                        is_complete=True,
                        failure_reason=(
                            "bootstrap_finalisation_failed: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                        ),
                    )
            return PhaseResult(is_complete=True, state_updates=state_updates)

        ok, reason = validate_ferebus_completed(staging)
        if not ok:
            self._journal_event(
                "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=str(staging),
                reason=reason,
            )
            return PhaseResult(
                is_complete=True,
                failure_reason="ferebus_staging_invalid: " + reason,
                retry_disposition=(
                    PostprocessRetryDisposition.FILESYSTEM_SETTLE
                    if reason == "ferebus_staging_missing"
                    or reason.startswith("expected_model_missing")
                    or "task receipt is missing" in reason
                    else PostprocessRetryDisposition.NONE
                ),
            )

        try:
            from .ferebus_quality import (
                FEREBUS_QUALITY_MANIFEST,
                evaluate_ferebus_quality,
                read_ferebus_quality_decision,
                write_ferebus_quality_decision,
                write_ferebus_quality_manifest,
            )
            from .config_lock import canonical_config, config_fingerprint

            self._report_runtime_progress("quality_metrics")
            quality = evaluate_ferebus_quality(
                staging,
                getattr(self.config, "quality_gates", None),
            )
            quality_summary = dict(quality.get("summary") or {})
            measured_models = int(quality_summary.get("n_measured") or 0)
            total_models = int(
                quality_summary.get("n_total")
                or quality_summary.get("n_tasks")
                or measured_models
            )
            self._report_runtime_progress(
                "quality_metrics",
                completed=measured_models,
                total=total_models,
                unit="models",
            )
            if quality.get("measurement_complete") is not True:
                from .ferebus_candidate_recovery import (
                    prepare_staging_recovery_request,
                    write_quality_attempt,
                )

                attempt_path = write_quality_attempt(
                    self.campaign_dir,
                    phase=phase_name,
                    iteration=int(state.iteration),
                    quality=quality,
                )
                recovery = prepare_staging_recovery_request(
                    self.campaign_dir,
                    campaign_uid=str(state.campaign_uid),
                    phase=phase_name,
                    iteration=int(state.iteration),
                    reference_data_version=int(expected_next),
                    staging_dir=staging,
                    quality_attempt_path=attempt_path,
                )
                errors = [
                    str(value) for value in quality.get("measurement_errors", [])
                ]
                self._journal_event(
                    "ferebus_quality_measurement_incomplete",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    reference_data_version=int(expected_next),
                    n_measurement_failures=int(
                        (quality.get("summary") or {}).get(
                            "n_measurement_failures", 0
                        )
                    ),
                    quality_attempt=str(attempt_path),
                    recovery_request=str(
                        recovery.get("request_sha256") or ""
                    ),
                    errors=errors[:8],
                )
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "ferebus_quality_measurement_incomplete: "
                        + ";".join(errors)[:300]
                    ),
                    submission_metadata={
                        "ferebus_quality_disposition": "measurement_incomplete",
                        "quality_attempt_path": str(attempt_path),
                    },
                )
            quality_path = write_ferebus_quality_manifest(staging, quality)
            config_sha = config_fingerprint(canonical_config(self.config))
            self._report_runtime_progress("incumbent_comparison")
            decision_path = write_ferebus_quality_decision(
                staging,
                config_sha256=config_sha,
                gates=getattr(self.config, "quality_gates", None),
            )
            decision = read_ferebus_quality_decision(
                staging,
                expected_config_sha256=config_sha,
                require_accepted=False,
            )
            current_decision = dict(decision.get("current_evaluation") or {})
            self._report_runtime_progress(
                "promotion_decision",
                completed=int(current_decision.get("n_tasks") or 0),
                total=int(current_decision.get("n_tasks") or 0),
                unit="models",
            )
            self._journal_event(
                "ferebus_quality_summary",
                phase=phase_name,
                iteration=int(state.iteration),
                manifest=str(quality_path),
                **dict(quality.get("summary") or {}),
                decision_manifest=str(decision_path),
                accepted=bool(current_decision.get("accepted")),
                n_total=int(current_decision.get("n_tasks", 0)),
            )
            if not bool(current_decision.get("accepted")):
                from ..layout import trained_models_dir
                from ..versioning.manifest import sha256_file

                evaluation_digest = str(
                    current_decision.get("evaluation_sha256") or "rejected"
                )
                quality_digest = sha256_file(quality_path)
                candidate_digest = hashlib.sha256(
                    (quality_digest + ":" + evaluation_digest).encode("ascii")
                ).hexdigest()
                quarantine = (
                    trained_models_dir(self.campaign_dir)
                    / "rejected-candidates"
                    / ("reference-" + f"{int(expected_next):06d}")
                    / candidate_digest
                )
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                if quarantine.exists() or quarantine.is_symlink():
                    raise ValueError(
                        "rejected FEREBUS candidate quarantine already exists: "
                        + str(quarantine)
                    )
                os.replace(staging, quarantine)
                from .state import _fsync_parent_dir

                _fsync_parent_dir(quarantine)
                self._journal_event(
                    "ferebus_candidate_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    reference_data_version=int(expected_next),
                    quarantine=str(quarantine),
                    candidate_sha256=candidate_digest,
                    quality_sha256=quality_digest,
                    evaluation_sha256=evaluation_digest,
                    reasons=list(current_decision.get("reasons", [])),
                )
                return PhaseResult(
                    is_complete=True,
                    failure_reason="ferebus_quality_failed: "
                    + ";".join(
                        str(r) for r in current_decision.get("reasons", [])
                    )[:300],
                    submission_metadata={
                        "ferebus_quality_disposition": "quality_rejected",
                    },
                )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "ferebus_quality_failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
                submission_metadata={
                    "ferebus_quality_disposition": "measurement_failed",
                },
            )

        self._report_runtime_progress("model_commit")
        v_models.recover_dangling_staging()
        next_version = int(expected_next)
        try:
            if next_version == 0:
                if committed:
                    raise ValueError("bootstrap model snapshot is not the first commit")
                parent_model_set = None
            else:
                if committed != list(range(next_version)):
                    raise ValueError(
                        "committed model versions are not contiguous before "
                        + str(next_version)
                    )
                parent_model_set = v_models.resolve(
                    next_version - 1,
                    verification="metadata",
                )
            staged = v_models.stage(
                source_version=None,
                target_version=next_version,
            )
            if int(staged.stat().st_dev) != int(Path(v_models.parent).stat().st_dev):
                raise OSError("trained-model staging and final root are on different filesystems")
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "trained_model_staging_failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        from . import input_staging as _stg
        manifest = _stg.read_ferebus_manifest(staging)
        try:
            _write_ferebus_task_artefact_layout(
                staging,
                staged,
                manifest,
                models_version=next_version,
                parent_model_set=parent_model_set,
            )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "ferebus_model_snapshot_build_failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        try:
            from .model_contract import validate_ferebus_model_contract
            from ..versioning.trained_models import (
                validate_trained_model_snapshot,
            )
            from ..versioning.reference_data import ReferenceDataVersioning

            reference_view = ReferenceDataVersioning(
                Path(self.campaign_dir) / self.reference_data_dir_name
            ).resolve(next_version, verification="metadata")

            staged_model_set = validate_trained_model_snapshot(
                self.campaign_dir,
                staged,
                next_version,
                parent=parent_model_set,
                verification="deep",
                reference_view=reference_view,
            )
            validate_ferebus_model_contract(
                staged,
                committed=True,
                expected_version=next_version,
                trained_model_set=staged_model_set,
            )
        except Exception as exc:
            self._journal_event(
                "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=str(staged),
                reason="staged_model_contract_invalid: " + str(exc)[:160],
            )
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "staged_model_contract_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        v_models.commit(next_version)
        committed_dir = v_models.iteration_path(next_version)
        try:
            committed_model_set = v_models.resolve(
                next_version,
                verification="deep",
                reference_verification="metadata",
            )
            validate_ferebus_model_contract(
                committed_dir,
                committed=True,
                expected_version=next_version,
                trained_model_set=committed_model_set,
            )
            v_models.update_current(next_version)
            self._report_runtime_progress(
                "model_commit",
                completed=int(len(manifest.get("tasks") or [])),
                total=int(len(manifest.get("tasks") or [])),
                unit="models",
            )
        except Exception as exc:
            self._journal_event(
                "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=str(committed_dir),
                reason="committed_model_contract_invalid: " + str(exc)[:160],
            )
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "committed_model_contract_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )

        try:
            from ichor.core.adversarial.posterior import TotalEnergyPosterior
            from ichor.core.models import Models
            from .seed_selection_runtime import SeedSelectionRuntimeCache

            factor_models = Models.from_model_files(
                committed_model_set.root,
                committed_model_set.model_paths,
            )
            factor_posterior = TotalEnergyPosterior(
                factor_models,
                property_name="iqa",
                scaled=True,
            )
            factor_cache = SeedSelectionRuntimeCache(
                self.campaign_dir,
                pool=None,
                posterior=factor_posterior,
                model_set_sha256=str(committed_model_set.model_set_sha256),
                model_manifest_sha256=str(
                    committed_model_set.head_manifest_sha256
                ),
                iteration=int(next_version) + 1,
                model_file_sha256_by_atom={
                    str(task.atom): str(task.model.sha256)
                    for task in committed_model_set.tasks
                    if str(task.property) == "iqa"
                },
            )
            factor_statuses = factor_cache.ensure_model_factors()
            self._journal_event(
                "seed_selection_cache",
                phase=phase_name,
                iteration=int(state.iteration),
                cache_kind="model_factors",
                cache_status="prewarmed",
                n_models=int(len(factor_statuses)),
            )
        except Exception as exc:
            self._journal_event(
                "seed_selection_cache",
                phase=phase_name,
                iteration=int(state.iteration),
                cache_kind="model_factors",
                cache_status="prewarm_failed",
                error=type(exc).__name__ + ": " + str(exc)[:240],
            )

        state_updates = {"models_version": int(next_version), "validation_set_version": int(next_version)}
        if is_initial:
            from ..versioning.sampling_iterations import finalise_bootstrap

            try:
                finalise_bootstrap(
                    self.campaign_dir,
                    str(state.campaign_uid),
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "bootstrap_finalisation_failed: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )

        self._journal_event(
            "models_committed",
            phase=phase_name,
            iteration=int(state.iteration),
            models_version=int(next_version),
            model_set_sha256=str(committed_model_set.model_set_sha256),
            evidence_set_sha256=str(committed_model_set.evidence_set_sha256),
            model_set_manifest_sha256=str(
                committed_model_set.head_manifest_sha256
            ),
            n_models=int(len(committed_model_set.tasks)),
            n_properties=int(len(committed_model_set.properties)),
            n_atoms=int(len(committed_model_set.atoms)),
            reference_data_view_sha256=str(
                committed_model_set.reference_data_view_sha256
            ),
            idempotent_skip=False,
        )
        self._journal_event(
            "phase_succeeded_live",
            phase=phase_name,
            iteration=int(state.iteration),
            models_version=int(next_version),
        )
        return PhaseResult(is_complete=True, state_updates=state_updates)


    # --- ARIADNE_ARRAY parser body -------------------------------------

    def _parse_ariadne_array_postprocess(self, state, phase, observations):
        """Validate per-seed ARIADNE results and publish ariadne/RESULTS.json."""
        from pathlib import Path as _Path
        from ..strict_json import strict_json as _json
        from ..handoff_manifests import (
            ARIADNE_RESULTS_SCHEMA_VERSION,
            acquisition_maturity_audit_payload,
            load_seeds_picked,
            validate_ariadne_result,
            write_acquisition_maturity_audit,
            write_ariadne_landing_audit,
            write_ariadne_batch_decision,
            write_ariadne_results_manifest,
        )
        from ..acquisition.ariadne_runner import ariadne_result_usability_payload
        from ..ariadne_outputs import (
            SEED_OUTPUT_MANIFEST_FILENAME,
            validate_seed_output,
        )
        from ..layout import active_ariadne_dir, ariadne_seed_dir, ariadne_seeds_dir
        from ..seed_identity import read_ariadne_task_map
        from ..versioning.manifest import sha256_file
        from .phase_executor import PhaseResult
        from ..sampling_protocol import preview_sampling_protocol

        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        self._report_runtime_progress("handoff_validation")
        iter_dir = self._iter_dir(state.iteration)
        ariadne_root = active_ariadne_dir(iter_dir)
        seeds_root = ariadne_seeds_dir(iter_dir)
        geometry_scale_payload = None
        try:
            from ..geometry_novelty import read_geometry_novelty_scale

            geometry_scale_payload = read_geometry_novelty_scale(
                iter_dir,
                expected_iteration=int(state.iteration),
            )
        except Exception:
            geometry_scale_payload = None
        try:
            fallback_protocol = preview_sampling_protocol(
                self.config,
                campaign_dir=self.campaign_dir,
                iteration=int(state.iteration),
                geometry_scale_payload=geometry_scale_payload,
            )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "sampling_protocol_invalid_for_ariadne_postprocess: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        try:
            from ..ferebus_prior import contract_from_payload
            from ..versioning.trained_models import TrainedModelVersioning
            from .input_staging import read_ferebus_manifest

            model_root = TrainedModelVersioning(
                _Path(self.campaign_dir) / self.models_dir_name
            ).iteration_path(int(state.models_version))
            prior_contract_hash = contract_from_payload(
                read_ferebus_manifest(
                    model_root,
                    verify_dataset_files=False,
                ).get("prior_mean_contract")
            ).contract_sha256
        except Exception as exc:
            if not bool(self.backend_check):
                from ..ferebus_prior import resolve_ferebus_prior_contract

                prior_contract_hash = resolve_ferebus_prior_contract(
                    self.config
                ).contract_sha256
            else:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "ferebus_prior_contract_unavailable_for_ariadne_postprocess: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
        try:
            picked = load_seeds_picked(iter_dir, expected_iteration=int(state.iteration))
            task_map = read_ariadne_task_map(
                iter_dir,
                expected_iteration=int(state.iteration),
            )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "seeds_picked_invalid_for_ariadne: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )

        try:
            from .error_calibration_contract import (
                calibration_context_sha256,
            )
            from .submission_intent import (
                ariadne_producer_environment_binding,
                load_intent,
            )
            from .scheduler_recovery import (
                phase_recovery_ledger_path,
                read_phase_recovery_ledger,
            )
            from ..execution_identity import read_environment_generation

            postprocess_intent = load_intent(
                self.campaign_dir,
                "ARIADNE_ARRAY",
                int(state.iteration),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if not isinstance(postprocess_intent, dict):
                raise ValueError(
                    "submission intent is unavailable for ARIADNE provenance"
                )
            default_calibration_environment = (
                ariadne_producer_environment_binding(
                self.campaign_dir,
                postprocess_intent,
                expected_campaign_uid=str(state.campaign_uid),
                expected_iteration=int(state.iteration),
            )
            )
            calibration_environments_by_task: Dict[int, Dict[str, Any]] = {}
            recovery_ledger_path = phase_recovery_ledger_path(
                self.campaign_dir,
                phase="ARIADNE_ARRAY",
                iteration=int(state.iteration),
                replacement_round=int(
                    getattr(state, "replacement_round", 0)
                ),
            )
            if recovery_ledger_path.exists() or recovery_ledger_path.is_symlink():
                # This lineage is consumed only after the ARIADNE publication
                # exists.  Pending recovery cannot reach this path until the
                # executor has written a v2 ledger, while historical v1
                # publications must remain readable by existing campaigns.
                recovery_ledger = read_phase_recovery_ledger(
                    recovery_ledger_path
                )
                recovery_identity = (
                    str(recovery_ledger.get("campaign_uid") or ""),
                    str(recovery_ledger.get("phase") or ""),
                    int(recovery_ledger.get("iteration", -1)),
                    int(recovery_ledger.get("replacement_round", -1)),
                )
                expected_recovery_identity = (
                    str(state.campaign_uid),
                    "ARIADNE_ARRAY",
                    int(state.iteration),
                    int(getattr(state, "replacement_round", 0)),
                )
                if recovery_identity != expected_recovery_identity:
                    raise ValueError(
                        "ARIADNE recovery lineage identity mismatch"
                    )
                for lineage in recovery_ledger.get(
                    "recovery_lineage",
                    [],
                ):
                    if not isinstance(lineage, Mapping):
                        raise ValueError(
                            "ARIADNE recovery lineage record is invalid"
                        )
                    task_id = int(lineage["logical_task_id"])
                    generation = int(lineage["environment_generation"])
                    generation_digest = str(
                        lineage[
                            "environment_generation_digest_sha256"
                        ]
                    )
                    historical = read_environment_generation(
                        self.campaign_dir,
                        generation=generation,
                        expected_campaign_uid=str(state.campaign_uid),
                    )
                    if str(historical["digest_sha256"]) != generation_digest:
                        raise ValueError(
                            "ARIADNE recovery lineage environment digest mismatch"
                        )
                    calibration_environments_by_task[task_id] = {
                        "generation": generation,
                        "generation_digest_sha256": generation_digest,
                    }
            calibration_contexts: Dict[str, str] = {}

            def calibration_environment_for_task(
                logical_task_id: int,
            ) -> Dict[str, Any]:
                return dict(
                    calibration_environments_by_task.get(
                        int(logical_task_id),
                        default_calibration_environment,
                    )
                )

            def calibration_context_for_environment(
                environment: Mapping[str, Any],
            ) -> str:
                digest = str(
                    environment["generation_digest_sha256"]
                )
                if digest not in calibration_contexts:
                    calibration_contexts[digest] = (
                        calibration_context_sha256(
                            self.config,
                            prior_mean_contract_sha256=prior_contract_hash,
                            environment_generation_digest_sha256=digest,
                        )
                    )
                return calibration_contexts[digest]
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason="calibration_context_unavailable: " + str(exc),
            )
        seed_records = list(picked["seed_records"])
        task_records = list(task_map["tasks"])
        expected_n = int(task_map["n_tasks"])
        try:
            decision_contract = self._decision_contract_for_submission(
                state,
                "ARIADNE_ARRAY",
            )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "ariadne_submission_decision_contract_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        config_sha256 = str(decision_contract["config_sha256"])
        failure_threshold_fraction = float(
            decision_contract["failure_threshold_fraction"]
        )

        def publish_batch_decision(
            *,
            n_accepted: int,
            n_rejected: int,
            accepted_batch: bool,
            reasons,
        ):
            path = write_ariadne_batch_decision(
                iter_dir,
                campaign_uid=str(state.campaign_uid),
                iteration=int(state.iteration),
                config_sha256=str(config_sha256),
                failure_threshold_fraction=failure_threshold_fraction,
                expected_n=int(expected_n),
                n_accepted=int(n_accepted),
                n_rejected=int(n_rejected),
                accepted=bool(accepted_batch),
                reasons=[str(reason) for reason in reasons],
            )
            self.artefact_log.append(str(path))
            return path
        if len(seed_records) != expected_n:
            return PhaseResult(
                is_complete=True,
                failure_reason="ARIADNE task-map/selection count mismatch",
            )
        if int(task_map["models_version"]) != int(state.models_version):
            return PhaseResult(
                is_complete=True,
                failure_reason="ARIADNE task-map/state model version mismatch",
            )
        try:
            from ..acquisition.trajectory_pool import TrajectoryPool

            trajectory_pool = TrajectoryPool.load(self.campaign_dir)
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "trajectory_pool_unavailable_for_ariadne_validation: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        if str(trajectory_pool.sha256) != str(picked["trajectory_sha256"]):
            return PhaseResult(
                is_complete=True,
                failure_reason="trajectory pool/seed selection SHA mismatch",
            )
        kept_alphas = []
        flagged_count = 0
        accepted = []
        rejected = []
        landing_audit_records = []

        if not seeds_root.is_dir():
            for task in task_records:
                seed_id = int(task["seed_id"])
                seed_dir = ariadne_seed_dir(iter_dir, seed_id)
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": str(task["seed_uid"]),
                    "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
                    "reason": "ariadne_seeds_directory_missing",
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": str(task["seed_uid"]),
                    "seed_dir": seed_dir.relative_to(ariadne_root).as_posix(),
                    "reason": "ariadne_seeds_directory_missing",
                    "handoff_accepted": False,
                    "handoff_rejection_reason": "ariadne_seeds_directory_missing",
                })
            write_ariadne_landing_audit(iter_dir, {
                "iteration": int(state.iteration),
                "summary": _ariadne_landing_audit_summary(landing_audit_records),
                "seeds": landing_audit_records,
            })
            write_acquisition_maturity_audit(
                iter_dir,
                acquisition_maturity_audit_payload(
                    iteration=int(state.iteration),
                    seed_records=landing_audit_records,
                ),
            )
            write_ariadne_results_manifest(iter_dir, {
                "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
                "campaign_uid": str(state.campaign_uid),
                "iteration": int(state.iteration),
                "trajectory_sha256": str(picked.get("trajectory_sha256", "")),
                "task_map": {
                    "path": "TASK_MAP.json",
                    "sha256": sha256_file(ariadne_root / "TASK_MAP.json"),
                },
                "expected_n": int(expected_n),
                "n_accepted": 0,
                "n_rejected": int(len(rejected)),
                "accepted": [],
                "rejected": rejected,
            })
            try:
                from ..sampling_history import prewarm_sampling_history_cache

                prewarm_sampling_history_cache(
                    iter_dir,
                    iteration=int(state.iteration),
                )
            except Exception:
                pass
            publish_batch_decision(
                n_accepted=0,
                n_rejected=len(rejected),
                accepted_batch=False,
                reasons=["ariadne_seeds_directory_missing"],
            )
            return PhaseResult(
                is_complete=True,
                failure_reason="ariadne_seeds_directory_missing: " + str(seeds_root),
            )

        self._report_runtime_progress(
            "result_parsing",
            completed=0,
            total=int(expected_n),
            unit="seed results",
        )
        for result_index, (task, seed_record) in enumerate(
            zip(task_records, seed_records),
            start=1,
        ):
            if result_index == 1 or result_index % 8 == 0:
                self._report_runtime_progress(
                    "result_parsing",
                    completed=int(result_index - 1),
                    total=int(expected_n),
                    unit="seed results",
                    accepted=int(len(accepted)),
                    rejected=int(len(rejected)),
                )
            seed_id = int(task["seed_id"])
            seed_uid = str(task["seed_uid"])
            array_task_id = int(task["array_task_id"])
            seed_dir = ariadne_seed_dir(iter_dir, seed_id)
            result_path = seed_dir / "result.json"
            output_manifest_path = seed_dir / SEED_OUTPUT_MANIFEST_FILENAME
            seed_dir_rel = seed_dir.relative_to(ariadne_root).as_posix()
            result_path_rel = result_path.relative_to(ariadne_root).as_posix()
            provenance_path_rel = (
                seed_dir / PROVENANCE_FILENAME
            ).relative_to(ariadne_root).as_posix()
            output_manifest_rel = output_manifest_path.relative_to(
                ariadne_root
            ).as_posix()
            try:
                output_payload = validate_seed_output(
                    seed_dir,
                    expected_campaign_uid=str(state.campaign_uid),
                    expected_iteration=int(state.iteration),
                    expected_seed_id=seed_id,
                    expected_seed_uid=seed_uid,
                    expected_array_task_id=array_task_id,
                )
            except Exception as exc:
                reason = "seed_output_invalid: " + type(exc).__name__ + ": " + str(exc)
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "array_task_id": array_task_id,
                    "seed_dir": seed_dir_rel,
                    "reason": reason,
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "reason": reason,
                    "handoff_accepted": False,
                    "handoff_rejection_reason": reason,
                })
                self._journal_event(
                    "ariadne_task_rejected_invalid_output",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_id=seed_id,
                    seed_uid=seed_uid,
                    seed_dir=seed_dir.name,
                    reason=reason[:320],
                )
                continue
            if not bool(output_payload["task_success"]) or int(
                output_payload["task_exit_code"]
            ) != 0:
                failure_detail = str(
                    output_payload.get("task_failure_reason")
                    or (
                        "exit_code="
                        + str(output_payload["task_exit_code"])
                    )
                )
                reason = "ariadne_unusable:" + failure_detail
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "array_task_id": array_task_id,
                    "seed_dir": seed_dir_rel,
                    "reason": reason,
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "reason": reason,
                    "handoff_accepted": False,
                    "handoff_rejection_reason": reason,
                })
                self._journal_event(
                    "ariadne_task_rejected_invalid_output",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_id=seed_id,
                    seed_uid=seed_uid,
                    seed_dir=seed_dir.name,
                    reason=reason,
                )
                continue
            if not result_path.is_file():
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": "missing_result_json",
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": "missing_result_json",
                    "handoff_accepted": False,
                    "handoff_rejection_reason": "missing_result_json",
                })
                self._journal_event(
                    "ariadne_task_rejected_missing_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason="missing_result_json",
                )
                continue
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result_dict = _json.load(f)
            except (OSError, ValueError):
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": "result_json_parse_failure",
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": "result_json_parse_failure",
                    "handoff_accepted": False,
                    "handoff_rejection_reason": "result_json_parse_failure",
                })
                self._journal_event(
                    "ariadne_task_rejected_malformed_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    reason="result_json_parse_failure",
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason="result_json_parse_failure",
                )
                continue

            try:
                seed_frame_id = seed_record.get("frame_id")
                expected_atom_types = None
                expected_initial_coordinates = None
                if trajectory_pool is not None and seed_frame_id is not None:
                    seed_atoms = trajectory_pool.frame(int(seed_frame_id))
                    expected_atom_types = [str(atom.type) for atom in seed_atoms]
                    expected_initial_coordinates = [
                        [float(atom.x), float(atom.y), float(atom.z)]
                        for atom in seed_atoms
                    ]
                validated = validate_ariadne_result(
                    result_dict,
                    expected_iteration=int(state.iteration),
                    seed_record=seed_record,
                    expected_atom_types=expected_atom_types,
                    expected_initial_coordinates=expected_initial_coordinates,
                    expected_trajectory_sha256=str(picked.get("trajectory_sha256", "")),
                )
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                    "handoff_accepted": False,
                    "handoff_rejection_reason": reason,
                })
                self._journal_event(
                    "ariadne_task_rejected_malformed_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    reason=reason,
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue

            optional_diag_warnings = _ariadne_optional_diagnostic_warnings(result_dict)
            if optional_diag_warnings:
                self._journal_event(
                    "ariadne_optional_diagnostics_warning",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    warnings=list(optional_diag_warnings[:8]),
                    n_warnings=int(len(optional_diag_warnings)),
                )

            try:
                result_protocol, protocol_replay = _sampling_protocol_for_ariadne_result(
                    campaign_dir=self.campaign_dir,
                    iter_dir=iter_dir,
                    fallback_protocol=fallback_protocol,
                    result_dict=result_dict,
                    iteration=int(state.iteration),
                )
            except Exception as exc:
                reason = (
                    "sampling_protocol_replay_failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                )
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                    "handoff_accepted": False,
                    "handoff_rejection_reason": reason,
                })
                self._journal_event(
                    "ariadne_sampling_protocol_replay_failed",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    reason=reason,
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue
            if not bool(protocol_replay.get("used_exact_sampling_protocol", False)):
                reason = "ARIADNE result does not bind the exact sampling protocol"
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                })
                landing_audit_records.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                    "handoff_accepted": False,
                    "handoff_rejection_reason": reason,
                })
                continue

            usability = ariadne_result_usability_payload(result_dict)

            landing_safety = result_dict.get("landing_safety")
            if not isinstance(landing_safety, dict):
                landing_safety = {
                    "accepted": False,
                    "policy": "missing_safety",
                    "selected_origin": None,
                    "selected_candidate_index": None,
                    "reasons": ["missing_landing_safety"],
                    "record_only_reasons": [],
                    "metrics": {},
                    "raw_final": {},
                    "n_candidates_evaluated": 0,
                    "n_safe_candidates": 0,
                }
            audit_record = {
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "seed_dir": seed_dir_rel,
                "result_json": result_path_rel,
                "output_manifest": output_manifest_rel,
                "landing_safety": dict(landing_safety),
                "landing_candidates": list(result_dict.get("landing_candidates") or []),
                "task_success": bool(usability.get("usable", False)),
                "task_success_reason": str(usability.get("reason", "")),
                "sampling_protocol_replay": dict(protocol_replay),
            }
            if optional_diag_warnings:
                audit_record["optional_diagnostic_warnings"] = list(
                    optional_diag_warnings
                )
            landing_audit_records.append(audit_record)
            if not bool(usability.get("usable", False)):
                reason = str(usability.get("reason", "ariadne_result_unusable"))
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                    "landing_safety": dict(landing_safety),
                    "task_success": False,
                    "task_success_reason": reason,
                })
                self._journal_event(
                    "ariadne_task_rejected_unusable_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    return_code=int(validated["return_code"]),
                    reason=reason,
                    policy=str(landing_safety.get("policy", "unknown")),
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue
            if not bool(landing_safety.get("accepted", False)):
                reasons = landing_safety.get("reasons") or ["unsafe_landing"]
                reason = ";".join(str(r) for r in reasons)
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                    "landing_safety": dict(landing_safety),
                })
                self._journal_event(
                    "ariadne_task_rejected_unsafe_landing",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    reason=reason,
                    policy=str(landing_safety.get("policy", "unknown")),
                )
                self._journal_event(
                    "ariadne_landing_rejected",
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    reason=reason,
                    policy=str(landing_safety.get("policy", "unknown")),
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue

            if int(validated["return_code"]) != 0:
                self._journal_event(
                    "ariadne_task_salvaged_from_nonzero_exit",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=result_path_rel,
                    return_code=int(validated["return_code"]),
                    reason=str(usability.get("reason", "")),
                    policy=str(landing_safety.get("policy", "unknown")),
                )

            geometry_quality = _ariadne_geometry_quality(
                result_dict,
                validated,
                result_protocol.quality_gates,
            )
            if not bool(geometry_quality.get("accepted")):
                reason = ";".join(str(r) for r in geometry_quality.get("reasons", []))
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "reason": reason,
                    "geometry_quality": dict(geometry_quality.get("metrics") or {}),
                })
                audit_record["geometry_quality"] = dict(
                    geometry_quality.get("metrics") or {}
                )
                audit_record["geometry_rejection_reason"] = reason
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue

            try:
                prov_path, provenance_reconstructed = self._ensure_ariadne_seed_provenance(
                    state,
                    picked,
                    seed_record,
                )
            except BackendSubmissionError as exc:
                reason = "ariadne_provenance_missing: " + str(exc)
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "provenance_json": provenance_path_rel,
                    "reason": reason,
                })
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue
            audit_record["provenance_json"] = provenance_path_rel
            audit_record["provenance_created"] = bool(provenance_reconstructed)
            if provenance_reconstructed:
                self._journal_event(
                    "ariadne_provenance_reconstructed",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    provenance_json=provenance_path_rel,
                )

            enrich_with_ariadne(
                seed_dir,
                alpha_initial=float(validated["alpha_initial"]),
                alpha_final=float(validated["alpha_final"]),
                n_evaluations=int(validated["n_evaluations"]),
                fell_back_to_ds=bool(validated["fell_back_to_ds"]),
                wall_seconds=float(validated["wall_seconds"]),
                return_code=int(validated["return_code"]),
            )
            selection_diagnostics = result_dict.get("selection_diagnostics")
            if isinstance(selection_diagnostics, dict):
                calibration_environment = (
                    calibration_environment_for_task(array_task_id)
                )
                calibration_context = (
                    calibration_context_for_environment(
                        calibration_environment
                    )
                )
                diag_payload = dict(selection_diagnostics)
                diag_payload["model_version"] = int(getattr(state, "models_version", -1))
                diag_payload["model_set_sha256"] = str(picked["model_set_sha256"])
                diag_payload["prior_mean_contract_sha256"] = prior_contract_hash
                diag_payload["environment_generation"] = int(
                    calibration_environment["generation"]
                )
                diag_payload["environment_generation_digest_sha256"] = str(
                    calibration_environment["generation_digest_sha256"]
                )
                diag_payload["calibration_context_sha256"] = calibration_context
                diag_payload["sampling_protocol_sha256"] = str(
                    dict(result_dict.get("sampling_protocol") or {}).get(
                        "resolved_manifest_sha256"
                    )
                    or ""
                )
                diag_payload["seed_id"] = seed_id
                diag_payload["seed_uid"] = seed_uid
                diag_payload["array_task_id"] = array_task_id
                diag_payload["seed_frame_id"] = seed_record.get("frame_id")
                diag_payload["result_json"] = result_path.resolve().relative_to(
                    Path(self.campaign_dir).resolve()
                ).as_posix()
                diag_payload["landing_policy"] = str(
                    landing_safety.get(
                        "policy",
                        diag_payload.get("landing_policy", "unknown"),
                    )
                )
                diag_payload["safety_metrics"] = dict(
                    landing_safety.get(
                        "metrics",
                        diag_payload.get("safety_metrics", {}),
                    )
                    or {}
                )
                audit_record["selection_diagnostics"] = dict(diag_payload)
                enrich_with_error_calibration_input(seed_dir, diag_payload)

            class _ResultShim:
                def __init__(self, ai, af, alpha_trajectory):
                    self.alpha_initial = ai
                    self.alpha_final = af
                    self.alpha_trajectory = alpha_trajectory

            shim = _ResultShim(
                float(validated["alpha_initial"]),
                float(validated["alpha_final"]),
                list(validated.get("alpha_trajectory") or []),
            )
            d_w_is_synthetic = False
            if validated.get("whitened_distance_final") is not None:
                d_w = float(validated["whitened_distance_final"])
            else:
                d_w = self._synthetic_whitened_distance(shim)
                d_w_is_synthetic = True
            flag = None
            if d_w is not None:
                min_d, max_d = anti_overlap_whitened_distance_bounds(
                    result_protocol.effective_config
                )
                if d_w < min_d:
                    flag = "moved_too_little"
                elif d_w > max_d:
                    flag = "moved_too_far"
            enrich_with_anti_overlap(
                seed_dir,
                min_whitened_distance_to_training=d_w,
                passed=(flag is None),
                flag=flag,
            )

            rejected_by_anti_overlap = False
            if flag is not None:
                flagged_count += 1
                self._journal_event(
                    "anti_overlap_flagged",
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    whitened_distance=float(d_w if d_w is not None else 0.0),
                    flag=str(flag),
                    synthetic_distance=bool(d_w_is_synthetic),
                )
                # only ENFORCE a drop on a REAL whitened distance. the synthetic |delta-alpha| proxy
                # is a different physical quantity (a hartree-scale alpha magnitude, not a feature-
                # space std), so discarding real work because it trips the whitened thresholds is
                # meaningless and would spuriously starve the batch (A45). a missing real distance
                # therefore keeps the seed -- it stays a diagnostic flag, never a silent filter.
                # (enforcement is off by default now anyway, see A43.)
                if (
                    not d_w_is_synthetic
                    and getattr(
                        result_protocol.effective_config.anti_overlap,
                        "enforce_post_ariadne",
                        False,
                    )
                ):
                    rejected_by_anti_overlap = True

            if rejected_by_anti_overlap:
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = str(flag)
                rejected.append({
                    "seed_id": seed_id,
                    "seed_uid": seed_uid,
                    "seed_dir": seed_dir_rel,
                    "result_json": result_path_rel,
                    "provenance_json": provenance_path_rel,
                    "reason": str(flag),
                })
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=str(flag),
                )
                continue

            audit_record["handoff_accepted"] = True
            kept_alphas.append(float(validated["alpha_final"]))
            accepted.append({
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "array_task_id": array_task_id,
                "seed_dir": seed_dir_rel,
                "result_json": result_path_rel,
                "provenance_json": provenance_path_rel,
                "output_manifest": output_manifest_rel,
                "seed_frame_id": seed_record.get("frame_id"),
                "pool_row_index_zero_based": int(
                    seed_record["pool_row_index_zero_based"]
                ),
                "selection_origin": str(seed_record.get("selection_origin", "unknown")),
                "variance_at_selection": seed_record.get("variance_at_selection"),
                "alpha_initial": float(validated["alpha_initial"]),
                "alpha_final": float(validated["alpha_final"]),
                "trajectory_sha256": str(validated.get("trajectory_sha256", "")),
                "whitened_distance_final": d_w,
                "geometry_quality": dict(geometry_quality.get("metrics") or {}),
                "landing_safety": dict(landing_safety),
                "landing_policy": str(landing_safety.get("policy", "unknown")),
                "selection_diagnostics": (
                    dict(selection_diagnostics)
                    if isinstance(selection_diagnostics, dict)
                    else None
                ),
                "sampling_protocol_replay": dict(protocol_replay),
                "return_code": int(validated["return_code"]),
                "task_success": bool(usability.get("usable", False)),
                "task_success_reason": str(usability.get("reason", "")),
                "result_sha256": sha256_file(result_path),
                "output_manifest_sha256": sha256_file(output_manifest_path),
            })

        self._report_runtime_progress(
            "result_parsing",
            completed=int(expected_n),
            total=int(expected_n),
            unit="seed results",
            accepted=int(len(accepted)),
            rejected=int(len(rejected)),
        )
        self._report_runtime_progress("landing_classification")
        n_kept = len(accepted)
        n_rejected = len(rejected)
        rejection_counts: Dict[str, int] = {}
        for record in rejected:
            reason = str(record.get("reason") or "unknown_rejection")[:180]
            rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1
        dominant_rejection_reasons = [
            {"reason": reason, "count": int(count)}
            for reason, count in sorted(
                rejection_counts.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[:8]
        ]
        audit_summary = _ariadne_landing_audit_summary(landing_audit_records)
        self._report_runtime_progress(
            "audit_publication",
            completed=0,
            total=3,
            unit="manifests",
        )
        audit_path = write_ariadne_landing_audit(iter_dir, {
            "iteration": int(state.iteration),
            "summary": audit_summary,
            "seeds": landing_audit_records,
        })
        self.artefact_log.append(str(audit_path))
        maturity_path = write_acquisition_maturity_audit(
            iter_dir,
            acquisition_maturity_audit_payload(
                iteration=int(state.iteration),
                seed_records=landing_audit_records,
            ),
        )
        self.artefact_log.append(str(maturity_path))
        manifest_path = write_ariadne_results_manifest(iter_dir, {
            "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
            "campaign_uid": str(state.campaign_uid),
            "iteration": int(state.iteration),
            "trajectory_sha256": str(picked.get("trajectory_sha256", "")),
            "task_map": {
                "path": "TASK_MAP.json",
                "sha256": sha256_file(ariadne_root / "TASK_MAP.json"),
            },
            "expected_n": int(expected_n),
            "n_accepted": int(n_kept),
            "n_rejected": int(n_rejected),
            "accepted": accepted,
            "rejected": rejected,
        })
        self.artefact_log.append(str(manifest_path))
        try:
            from ..sampling_history import prewarm_sampling_history_cache

            prewarm_sampling_history_cache(
                iter_dir,
                iteration=int(state.iteration),
            )
        except Exception:
            pass
        self._report_runtime_progress(
            "audit_publication",
            completed=3,
            total=3,
            unit="manifests",
        )
        self._journal_event(
            "ariadne_landing_summary",
            iteration=int(state.iteration),
            n_expected=int(expected_n),
            n_accepted=int(n_kept),
            n_rejected=int(n_rejected),
            accepted=int(audit_summary.get("accepted", 0)),
            salvaged=int(audit_summary.get("salvaged", 0)),
            backtracked=int(audit_summary.get("backtracked", 0)),
            rejected=int(audit_summary.get("rejected", 0)),
            dominant_rejection_reasons=dominant_rejection_reasons,
        )

        decision_reasons = []
        if n_kept == 0:
            decision_reasons.append(
                "ariadne_no_usable_seed_results: " + str(n_rejected)
            )
        if expected_n and (
            n_rejected / float(expected_n)
        ) > failure_threshold_fraction:
            decision_reasons.append(
                "too_many_seeds_failed: " + str(n_rejected) + "/" + str(expected_n)
            )
        self._report_runtime_progress("batch_publication")
        publish_batch_decision(
            n_accepted=n_kept,
            n_rejected=n_rejected,
            accepted_batch=not bool(decision_reasons),
            reasons=decision_reasons,
        )
        self._report_runtime_progress(
            "batch_publication",
            completed=int(expected_n),
            total=int(expected_n),
            unit="seed results",
        )

        if n_kept == 0:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "ariadne_no_usable_seed_results: " + str(n_rejected)
                ),
            )

        # too many seeds lost (real failures + absentees) against the TRUE submitted count -> fail
        # rather than quietly commit a short batch as if the array had finished. only gated when we
        # actually know the submitted count (SELECTION.json present); mirrors the quantum phases.
        if any(reason.startswith("too_many_seeds_failed:") for reason in decision_reasons):
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "too_many_seeds_failed: " + str(n_rejected) + "/" + str(expected_n)
                ),
            )

        # per-iteration convergence scalar: a high quantile (p90) over the kept seeds, not the max.
        # tracks the worst-but-one rather than letting one intrinsically-hard seed pin it high
        # forever (A17), and being an order statistic it never reports a value no seed had (A44).
        # n_kept > 0 is guaranteed here, so the list is non-empty.
        last_alpha = _high_quantile(kept_alphas)
        self._journal_event(
            "phase_succeeded_live",
            phase=phase_name,
            iteration=int(state.iteration),
            n_kept=int(n_kept),
            n_rejected=int(n_rejected),
            last_acquisition_alpha0=float(last_alpha),
            acquisition_quantile=0.9,
            acquisition_quantile_policy="nearest_rank_observed",
        )
        return PhaseResult(
            is_complete=True,
            state_updates={
                "last_acquisition_alpha0": float(last_alpha),
                "last_n_anti_overlap_flagged": int(flagged_count),
            },
        )


    #-- Diversity parser body ----------------------------------

    def _parse_diversity_postprocess(
        self,
        state,
        phase,
        observations,
        *,
        emit_success_events: bool = True,
    ):
        """Parse the Phase A or Phase B diversity sample output.

        Phase A reads .DATA/BOOTSTRAP/selection/SELECTION.json. Phase B validates
        ACTIVE_LEARNING/iteration-NNNNNN/phase_b/SELECTION.json and its
        hash-bound selected geometry and finalised provenance records.

        Validation is presence + parseability via _count_xyz_frames. If the
        sample xyz is empty or unreadable, return a failure reason and
        let _handle_failure decide the next action.
        """
        from pathlib import Path as _Path
        from .phase_executor import PhaseResult

        phase_name = phase.value if hasattr(phase, "value") else str(phase)

        if phase_name == "PHASE_A_DIVERSITY":
            from ..layout import bootstrap_selection_dir

            outdir = bootstrap_selection_dir(self.campaign_dir)
            if not outdir.is_dir():
                return PhaseResult(
                    is_complete=True,
                    failure_reason="phase_a_outdir_missing: " + str(outdir),
                )
            from ..handoff_manifests import read_phase_a_sample_manifest
            try:
                phase_a_manifest = read_phase_a_sample_manifest(
                    outdir,
                    expected_campaign_uid=str(state.campaign_uid),
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_a_sample_manifest_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
            sample = _Path(str(phase_a_manifest["sample_xyz"]))
        else:
            iter_dir = self._iter_dir(state.iteration)
            from ..handoff_manifests import validate_phase_b_handoff
            try:
                phase_b_manifest = validate_phase_b_handoff(
                    iter_dir,
                    expected_iteration=int(state.iteration),
                    expected_campaign_uid=str(state.campaign_uid),
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_b_handoff_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
            sample = _Path(str(phase_b_manifest["selected_xyz"]["path"]))
            if not sample.is_file():
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_b_sample_missing: "
                        + str(sample)
                    ),
                )

        n_frames = self._count_xyz_frames(sample)
        if n_frames is None or n_frames <= 0:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "diversity_sample_unreadable_or_empty: " + str(sample)
                ),
            )

        if phase_name == "PHASE_A_DIVERSITY":
            expected = int(phase_a_manifest.get("n_select", 0))
            declared = phase_a_manifest.get("n_frames")
            if int(n_frames) != expected or (
                declared is not None and int(declared) != int(n_frames)
            ):
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_a_sample_count_mismatch: sample_frames="
                        + str(int(n_frames))
                        + " manifest_n_select="
                        + str(expected)
                    ),
                )

        if phase_name == "PHASE_B_DIVERSITY":
            final_records = list(phase_b_manifest.get("final", []))
            if int(n_frames) != len(final_records):
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_b_selection_count_mismatch: sample_frames="
                        + str(int(n_frames))
                        + " manifest_final="
                        + str(len(final_records))
                    ),
                )

        # if the Phase-B dedup ran, surface its counts in the journal.
        dedup_payload = {}
        deferred_journal_events = []
        if phase_name == "PHASE_B_DIVERSITY":
            d = phase_b_manifest.get("dedup", {}) if isinstance(phase_b_manifest, dict) else {}
            if isinstance(d, dict):
                relaxation = d.get("relaxation") if isinstance(d.get("relaxation"), dict) else {}
                dedup_payload = {
                    "n_kept": int(d.get("n_kept", 0)),
                    "n_dropped": int(d.get("n_dropped", 0)),
                    "min_separation": float(d.get("min_separation", 0.0)),
                    "threshold_mode": str(d.get("threshold_mode", "absolute")),
                }
                if d.get("effective_min_separation_angstrom") is not None:
                    dedup_payload["effective_min_separation_angstrom"] = float(
                        d.get("effective_min_separation_angstrom")
                    )
                if bool(relaxation.get("applied", False)):
                    relaxation_event = {
                        "event": "phase_b_novelty_threshold_relaxed",
                        "phase": phase_name,
                        "iteration": int(state.iteration),
                        "reason": str(relaxation.get("reason", "unknown")),
                        "n_admitted": int(relaxation.get("n_admitted", 0)),
                        "effective_min_separation_angstrom": relaxation.get(
                            "effective_min_separation_angstrom"
                        ),
                    }
                    if emit_success_events:
                        relaxation_name = str(relaxation_event.pop("event"))
                        self._journal_event(
                            relaxation_name,
                            **relaxation_event,
                        )
                    else:
                        deferred_journal_events.append(relaxation_event)
        success_event = {
            "event": "phase_succeeded_live",
            "phase": phase_name,
            "iteration": int(state.iteration),
            "sample_path": str(sample),
            "n_frames": int(n_frames),
            **dedup_payload,
        }
        if emit_success_events:
            success_name = str(success_event.pop("event"))
            self._journal_event(success_name, **success_event)
        else:
            deferred_journal_events.append(success_event)
        return PhaseResult(
            is_complete=True,
            state_updates={},
            journal_events=deferred_journal_events,
        )

    def _count_xyz_frames(self, sample_path):
        """Count complete positive-cardinality frames using the strict parser."""
        from ichor.core.files.xyz.strict_xyz import read_xyz_frames

        try:
            frames = read_xyz_frames(sample_path)
        except (OSError, ValueError):
            return None
        if any(len(frame) <= 0 for frame in frames):
            return None
        return len(frames)

    def _read_xyz_records(self, sample_path):
        from ichor.core.files.xyz.strict_xyz import read_xyz_frames

        parsed = read_xyz_frames(sample_path)
        if any(len(frame) <= 0 for frame in parsed):
            raise ValueError("XYZ frames must contain at least one atom")
        return [
            {
                "atom_types": [str(atom.type) for atom in frame],
                "coordinates": [
                    [float(atom.x), float(atom.y), float(atom.z)]
                    for atom in frame
                ],
            }
            for frame in parsed
        ]

    def _phase_b_sample_coordinate_mismatch(self, sample_path, final_records):
        from ..strict_json import strict_json as _json
        import math as _math

        try:
            frames = self._read_xyz_records(sample_path)
        except Exception as exc:
            return "sample_xyz_unreadable: " + type(exc).__name__ + ": " + str(exc)
        if len(frames) != len(final_records):
            return "sample frame count differs from final records"
        tolerance = 5.0e-6
        for index, (frame, rec) in enumerate(zip(frames, final_records)):
            result_path = Path(str(rec.get("result_json", "")))
            try:
                result = _json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return (
                    "result_json_unreadable for final_index "
                    + str(index)
                    + ": "
                    + type(exc).__name__
                )
            expected_atoms = [str(x) for x in (result.get("atom_types") or [])]
            expected_coords = result.get("final_coordinates") or []
            if frame["atom_types"] != expected_atoms:
                return "atom order mismatch at final_index " + str(index)
            if len(frame["coordinates"]) != len(expected_coords):
                return "coordinate row count mismatch at final_index " + str(index)
            for atom_i, (actual, expected) in enumerate(zip(frame["coordinates"], expected_coords)):
                if not isinstance(expected, list) or len(expected) != 3:
                    return "result coordinate shape mismatch at final_index " + str(index)
                for axis, (a, e) in enumerate(zip(actual, expected)):
                    if not _math.isclose(float(a), float(e), rel_tol=0.0, abs_tol=tolerance):
                        return (
                            "coordinate mismatch at final_index "
                            + str(index)
                            + " atom "
                            + str(atom_i)
                            + " axis "
                            + str(axis)
                        )
        return None



# ---module level: sbatch script builder -----------------


def _current_submission_job_name(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
    *,
    campaign_uid: Optional[str],
) -> str:
    """Return the durable attempt name when a submission intent is active."""
    try:
        from . import submission_intent as _submission_intent

        intent = _submission_intent.load_active_intent(
            campaign_dir,
            phase_name,
            int(iteration),
            expected_campaign_uid=(
                None if campaign_uid is None else str(campaign_uid)
            ),
        )
    except Exception as exc:
        raise BackendSubmissionError(
            "cannot render a scheduler job name from the active submission intent: "
            + type(exc).__name__
            + ": "
            + str(exc)[:160]
        ) from exc
    if isinstance(intent, dict):
        expected = intent.get("expected_job_name")
        if isinstance(expected, str) and expected:
            return expected
    return live_job_name(campaign_uid, phase_name, iteration)


def make_live_job_finder(
    sacct_runner=None,
    squeue_runner=None,
    *,
    campaign_dir: Optional[Path] = None,
    timeout_seconds: int = 60,
    scheduler_kind: Optional[str] = None,
):
    """the job_finder the daemon uses in live mode: given (state, phase) return the JobID of an
    already-running job for that exact phase+iteration, or None. lets the daemon adopt a job a crash
    orphaned rather than double-submit (A24/A25)."""
    from ..submit.sacct_poll import JobNameLookup

    backend = get_scheduler_backend(scheduler_kind or _configured_scheduler())

    def _finder(state, phase):
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        uid = getattr(state, "campaign_uid", None)
        iteration = getattr(state, "iteration", 0)
        intent = None
        if campaign_dir is not None:
            try:
                from . import submission_intent as _submission_intent

                intent = _submission_intent.load_active_intent(
                    campaign_dir,
                    phase_name,
                    int(iteration),
                    expected_campaign_uid=(None if uid is None else str(uid)),
                )
            except Exception:
                intent = None
        intent_name = (
            str(intent.get("expected_job_name") or "")
            if isinstance(intent, dict)
            else ""
        )
        names = (
            [intent_name]
            if intent_name
            else [live_job_name(uid, phase_name, iteration)]
        )
        is_identified_attempt = bool(
            isinstance(intent, dict) and intent.get("submission_identity")
        )
        if uid and not is_identified_attempt:
            legacy = str(uid)[:8] + "-" + str(phase_name) + "-" + str(int(iteration))
            if legacy not in names:
                names.append(legacy)
        names = [name for name in names if name]
        if backend.identity_kind == "sge":
            names = [sge_safe_job_name(name) for name in names]
        inconclusive: Optional[JobNameLookup] = None
        last_lookup = JobNameLookup(None, inconclusive=False)
        use_squeue_fallback = squeue_runner is not None or sacct_runner is None
        for name in names:
            found = backend.find_running_job_by_name(
                name,
                accounting_runner=(sacct_runner or subprocess.run),
                queue_runner=(squeue_runner or subprocess.run),
                timeout_seconds=int(timeout_seconds),
            )
            last_lookup = found
            if found.job_id:
                return found
            if found.inconclusive and inconclusive is None:
                inconclusive = found
        return inconclusive if inconclusive is not None else last_lookup

    return _finder


def make_live_job_accounting_finder(
    sacct_runner=None,
    squeue_runner=None,
    *,
    timeout_seconds: int = 60,
    scheduler_kind: Optional[str] = None,
):
    """Return a live-mode expected-job-name accounting lookup."""
    backend = get_scheduler_backend(scheduler_kind or _configured_scheduler())

    def _finder(state, phase, active_intent):
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        uid = getattr(state, "campaign_uid", None)
        iteration = getattr(state, "iteration", 0)
        expected_tasks = active_intent.get("expected_tasks")
        names = [str(active_intent.get("expected_job_name") or "")]
        is_identified_attempt = bool(active_intent.get("submission_identity"))
        if not is_identified_attempt:
            live_name = live_job_name(uid, phase_name, iteration)
            if live_name not in names:
                names.append(live_name)
        if uid and not is_identified_attempt:
            legacy = str(uid)[:8] + "-" + str(phase_name) + "-" + str(int(iteration))
            if legacy not in names:
                names.append(legacy)
        names = [name for name in names if name]
        if backend.identity_kind == "sge":
            names = [sge_safe_job_name(name) for name in names]
        inconclusive = None
        last_lookup = None
        use_squeue_fallback = squeue_runner is not None or sacct_runner is None
        for name in names:
            found = backend.find_accounted_job_by_name(
                name,
                expected_task_count=(
                    None if expected_tasks is None else int(expected_tasks)
                ),
                accounting_runner=(sacct_runner or subprocess.run),
                queue_runner=(squeue_runner or subprocess.run),
                submission_kind=str(active_intent.get("submission_kind")),
                timeout_seconds=int(timeout_seconds),
            )
            last_lookup = found
            if found.job_id:
                return found
            if found.inconclusive and inconclusive is None:
                inconclusive = found
        return inconclusive if inconclusive is not None else last_lookup

    return _finder


def make_live_job_liveness_checker(
    squeue_runner=None,
    *,
    timeout_seconds: int = 60,
    scheduler_kind: Optional[str] = None,
):
    """Return a live-mode checker for an existing scheduler JobID."""
    backend = get_scheduler_backend(scheduler_kind or _configured_scheduler())

    def _checker(
        job_id,
        *,
        expected_job_name=None,
        expected_owner=None,
    ):
        return backend.find_active_job_by_id(
            job_id,
            queue_runner=(squeue_runner or subprocess.run),
            timeout_seconds=int(timeout_seconds),
            expected_job_name=expected_job_name,
            expected_owner=expected_owner,
        )

    return _checker


def _reject_shell_control_chars(label: str, value: str) -> None:
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise BackendSubmissionError(label + " contains a control character")


def _shell_quote(value: Any) -> str:
    text = str(value)
    _reject_shell_control_chars("shell value", text)
    return shlex.quote(text)


def _shell_executable(value: Any) -> str:
    text = str(value)
    _reject_shell_control_chars("shell executable", text)
    if "$" in text and _SHELL_PATH_FRAGMENT_RE.fullmatch(text):
        return text
    return shlex.quote(text)


def _safe_shell_path_component(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = str(fallback)
    _reject_shell_control_chars("shell path component", text)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return safe or str(fallback)


def _python_executable_for_script() -> str:
    python_path = expanded_profile_value(
        "software", "python", "python_path", default=None
    )
    return _shell_executable(python_path or sys.executable)


def _normalise_module_list(raw: Any, *, label: str) -> List[str]:
    try:
        return normalise_module_list(raw, label=label)
    except ValueError as exc:
        raise BackendSubmissionError(str(exc)) from exc


def _configured_jobscript_shebang() -> str:
    raw = profile_value("hpc", "jobscript_shebang", default=None)
    if raw is None:
        return "#!/bin/bash --login"
    value = str(raw).strip()
    _reject_shell_control_chars("configured hpc.jobscript_shebang", value)
    if not _SHEBANG_RE.fullmatch(value):
        raise BackendSubmissionError(
            "configured hpc.jobscript_shebang is unsafe: " + repr(value)
        )
    return value


def _configured_max_array_task_id() -> Optional[int]:
    raw = profile_value("hpc", "max_array_task_id", default=None)
    if raw is None:
        return None
    if isinstance(raw, bool) or (
        isinstance(raw, float) and not raw.is_integer()
    ):
        raise BackendSubmissionError(
            "configured hpc.max_array_task_id must be an integer"
        )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(
            "configured hpc.max_array_task_id must be an integer"
        ) from exc
    if value < 0:
        raise BackendSubmissionError(
            "configured hpc.max_array_task_id must be >= 0"
        )
    return value


def _configured_max_array_tasks() -> Optional[int]:
    raw = profile_value("hpc", "max_array_tasks", default=None)
    if raw is None:
        legacy = _configured_max_array_task_id()
        return None if legacy is None else int(legacy) + 1
    if isinstance(raw, bool) or (
        isinstance(raw, float) and not raw.is_integer()
    ):
        raise BackendSubmissionError(
            "configured hpc.max_array_tasks must be an integer"
        )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(
            "configured hpc.max_array_tasks must be an integer"
        ) from exc
    if value <= 0:
        raise BackendSubmissionError(
            "configured hpc.max_array_tasks must be > 0"
        )
    return value


def _configured_max_job_log_files_per_directory() -> int:
    raw = profile_value(
        "hpc", "max_job_log_files_per_directory", default=5000
    )
    if isinstance(raw, bool) or (
        isinstance(raw, float) and not raw.is_integer()
    ):
        raise BackendSubmissionError(
            "configured hpc.max_job_log_files_per_directory must be an integer"
        )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(
            "configured hpc.max_job_log_files_per_directory must be an integer"
        ) from exc
    if value <= 0:
        raise BackendSubmissionError(
            "configured hpc.max_job_log_files_per_directory must be > 0"
        )
    return value


def _configured_scheduler() -> str:
    if active_machine() == "_default":
        raise BackendSubmissionError(
            "_default is a fallback configuration, not a live active-learning "
            "profile; set ICHOR_MACHINE to a live profile such as csf3, csf4, "
            "or ffluxlab"
        )
    raw = profile_value("hpc", "scheduler", default=None)
    if raw is None:
        return "slurm"
    value = str(raw).strip().lower()
    _reject_shell_control_chars("configured hpc.scheduler", value)
    if value not in {"slurm", "sge"}:
        raise BackendSubmissionError(
            "active-learning live mode supports hpc.scheduler='slurm' or "
            "'sge'; got "
            + repr(value)
        )
    return value


def _configured_ferebus_platform() -> str:
    raw = profile_value(
        "software", "ferebus", "pyferebus_platform", default=None
    )
    if raw:
        value = str(raw).strip()
    else:
        machine = active_machine()
        if machine:
            raise BackendSubmissionError(
                "live FEREBUS requires software.ferebus.pyferebus_platform "
                "in ichor_config.yaml"
            )
        value = "CSF4"
    _reject_shell_control_chars("configured ferebus pyferebus_platform", value)
    if not re.fullmatch(r"^[A-Za-z0-9_.-]+$", value):
        raise BackendSubmissionError(
            "configured ferebus pyferebus_platform is unsafe: " + repr(value)
        )
    return value


def _configured_daemon_runtime_modules() -> List[str]:
    """Modules loaded by daemon-owned scheduler scripts.

    Python and ARIADNE/MKL runtime modules are kept in ichor_config.yaml so a
    cluster-module update does not require a code edit. Missing config falls
    back to the current CSF4 stack for off-cluster tests and legacy configs.
    FEREBUS is not included: pyferebus generates the initial source script,
    which ICHOR replaces with a scheduler-native wrapper before submission.
    """
    try:
        return configured_daemon_runtime_modules()
    except Exception as exc:
        raise BackendSubmissionError(str(exc)) from exc


def _job_scratch_preamble(
    *,
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
    submission_intent: Dict[str, Any],
    resource_resolution_binding: Dict[str, Any],
    script_binding_path: Path,
) -> List[str]:
    python = _python_executable_for_script()
    camp = str(Path(campaign_dir).resolve())
    identity = str(submission_intent["submission_identity"])
    resolution_path = str(resource_resolution_binding["path"])
    resolution_sha = str(resource_resolution_binding["sha256"])
    binding_path = str(Path(script_binding_path).resolve())
    return [
        "# Verify immutable scientific, implementation, and script evidence.",
        "export ICHOR_RESOURCE_RESOLUTION=" + _shell_quote(resolution_path),
        "export ICHOR_RESOURCE_RESOLUTION_SHA256=" + _shell_quote(resolution_sha),
        "export ICHOR_SCRIPT_BINDING=" + _shell_quote(binding_path),
        ': "${ICHOR_SCRIPT_BINDING_SHA256:?missing submitted script-binding digest}"',
        python
        + " -c "
        + _shell_quote(
            "import sys; from ichor.hpc.active_learning.daemon.resource_records "
            "import verify_resolution; verify_resolution(sys.argv[1], sys.argv[2], "
            "campaign_dir=sys.argv[3])"
        )
        + ' "$ICHOR_RESOURCE_RESOLUTION" "$ICHOR_RESOURCE_RESOLUTION_SHA256" '
        + _shell_quote(camp),
        python
        + " -c "
        + _shell_quote(
            "import sys; from ichor.hpc.active_learning.daemon.script_bundles "
            "import verify_script_binding; verify_script_binding(sys.argv[1], sys.argv[2])"
        )
        + ' "$ICHOR_SCRIPT_BINDING" "$ICHOR_SCRIPT_BINDING_SHA256"',
        "ICHOR_JOB_SCRATCH=$("
        + python
        + " -m ichor.hpc.active_learning.daemon.scratch prepare"
        + " --campaign-dir "
        + _shell_quote(camp)
        + " --campaign-uid "
        + _shell_quote(str(submission_intent.get("campaign_uid") or ""))
        + " --phase "
        + _shell_quote(phase_name)
        + " --iteration "
        + str(int(iteration))
        + " --attempt-id "
        + _shell_quote(str(submission_intent.get("attempt_id") or ""))
        + " --submission-identity "
        + _shell_quote(identity)
        + ' --job-id "${ICHOR_SCHEDULER_JOB_ID}"'
        + ' --array-task-id "${ICHOR_SCHEDULER_ARRAY_TASK_ID:-0}"'
        + ' --resource-resolution "$ICHOR_RESOURCE_RESOLUTION"'
        + ' --resource-resolution-sha256 "$ICHOR_RESOURCE_RESOLUTION_SHA256"'
        + ' --script-binding "$ICHOR_SCRIPT_BINDING"'
        + ' --script-binding-sha256 "$ICHOR_SCRIPT_BINDING_SHA256"'
        + ")",
        "export ICHOR_JOB_SCRATCH",
        'export TMPDIR="$ICHOR_JOB_SCRATCH" TMP="$ICHOR_JOB_SCRATCH" TEMP="$ICHOR_JOB_SCRATCH"',
        "ichor_finish_scratch() {",
        "  status=$?",
        "  trap - EXIT",
        '  if [ "$status" -eq 0 ]; then',
        "    "
        + python
        + ' -m ichor.hpc.active_learning.daemon.scratch finish --path "$ICHOR_JOB_SCRATCH" --success || '
        + 'echo "WARNING: could not clean ICHOR task scratch" >&2',
        "  else",
        "    "
        + python
        + ' -m ichor.hpc.active_learning.daemon.scratch finish --path "$ICHOR_JOB_SCRATCH" || true',
        "  fi",
        '  exit "$status"',
        "}",
        "trap ichor_finish_scratch EXIT",
        "",
    ]


def build_scheduler_script(
    *,
    phase_name: str,
    iteration: int,
    campaign_dir: Path,
    config: CampaignConfig,
    array_size: Optional[int] = None,
    array_task_map: Optional[Path] = None,
    walltime_hours: Optional[float] = None,
    partition: Optional[str] = None,
    campaign_uid: Optional[str] = None,
    replacement_round: int = 0,
    resolved_resources: Optional[ResolvedPhaseResources] = None,
    attempt_bundle: Optional[AttemptBundle] = None,
    submission_intent: Optional[Dict[str, Any]] = None,
    resource_resolution_binding: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    scheduler_kind: Optional[str] = None,
) -> str:
    """Return a scheduler-native live script for the given phase.

    Resources come from config.resources (partition / walltime / mem /
    cpus-per-task / ntasks); walltime_hours and partition may still be passed
    to override them. Native task indexes are normalised to a zero-based
    ICHOR logical task ID before any scientific code is invoked.

    Paths are absolute (resolved campaign dir) so the script does not depend
    on submission being launched from any particular directory.
    """
    scheduler = str(scheduler_kind or _configured_scheduler()).strip().lower()
    if scheduler not in {"slurm", "sge"}:
        raise BackendSubmissionError("unsupported scheduler: " + repr(scheduler))
    res = config.resources
    part = str(partition if partition is not None else res.partition_for(phase_name))
    is_array = array_size is not None and int(array_size) > 0
    if is_array:
        if scheduler == "slurm":
            max_array_task_id = _configured_max_array_task_id()
            highest_task_id = int(array_size) - 1
            if max_array_task_id is not None and highest_task_id > max_array_task_id:
                raise BackendSubmissionError(
                    "array task id "
                    + str(highest_task_id)
                    + " exceeds configured hpc.max_array_task_id "
                    + str(max_array_task_id)
                )
        else:
            maximum_tasks = _configured_max_array_tasks()
            if maximum_tasks is not None and int(array_size) > maximum_tasks:
                raise BackendSubmissionError(
                    "array size "
                    + str(int(array_size))
                    + " exceeds configured hpc.max_array_tasks "
                    + str(maximum_tasks)
                )
    resolved = resolved_resources or resolve_phase_resources(
        phase_name=phase_name,
        config=config,
        partition=part,
        campaign_dir=(campaign_dir if submission_intent is not None else None),
        iteration=int(iteration),
        array_size=array_size,
        replacement_round=int(replacement_round),
        require_evidence=(submission_intent is not None),
    )
    wall = walltime_hours if walltime_hours is not None else res.walltime_for(phase_name)
    validate_partition_walltime(str(resolved.partition), wall)
    cpus = int(resolved.cpus_per_task)
    ntasks = int(resolved.ntasks)
    mem_per_cpu = str(resolved.mem_per_cpu)
    camp = str(Path(campaign_dir).resolve())
    try:
        job_name = _current_submission_job_name(
            Path(campaign_dir),
            phase_name,
            iteration,
            campaign_uid=campaign_uid,
        )
    except ValueError as exc:
        raise BackendSubmissionError(str(exc)) from exc
    if scheduler == "sge":
        job_name = sge_safe_job_name(job_name)
    _reject_shell_control_chars("scheduler job name", job_name)
    identity = str(
        (submission_intent or {}).get("submission_identity")
        or "unbound-preview"
    )
    bundle = attempt_bundle
    if bundle is None:
        root = bundle_root(
            campaign_dir,
            phase_name,
            int(iteration),
            identity,
        )
        bundle = AttemptBundle(
            root=root,
            script=root / "job.sh",
            outputs=root / "OUTPUTS",
            errors=root / "ERRORS",
            array_task_map=array_task_map,
        )
    log_paths = scheduler_log_paths(
        bundle,
        scheduler_kind=scheduler,
        is_array=is_array,
    )
    throttle = getattr(res, "array_concurrency_limit", None)
    throttle_i: Optional[int] = None
    if is_array and throttle is not None:
        try:
            throttle_i = int(throttle)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "resources.array_concurrency_limit must be a positive integer"
            ) from exc
        if throttle_i <= 0:
            raise BackendSubmissionError(
                "resources.array_concurrency_limit must be > 0"
            )
        throttle_i = min(throttle_i, int(array_size))
    lines: List[str] = [_configured_jobscript_shebang()]
    if scheduler == "slurm":
        lines += [
            "#SBATCH --job-name=" + job_name,
            "#SBATCH --partition=" + str(resolved.partition),
            "#SBATCH --time=" + _format_slurm_walltime_hours(wall),
            "#SBATCH --mem-per-cpu=" + str(mem_per_cpu),
            "#SBATCH --cpus-per-task=" + str(int(cpus)),
            "#SBATCH --ntasks=" + str(int(ntasks)),
        ]
        if is_array:
            array_spec = "0-" + str(int(array_size) - 1)
            if throttle_i is not None:
                array_spec += "%" + str(throttle_i)
            lines.append("#SBATCH --array=" + array_spec)
        lines += [
            "#SBATCH --output=" + log_paths["output"],
            "#SBATCH --error=" + log_paths["error"],
        ]
    else:
        if int(ntasks) != 1:
            raise BackendSubmissionError(
                "SGE live phases require resources.ntasks=1"
            )
        queue = str(resolved.extra.get("scheduler_queue") or "").strip()
        if not queue:
            raise BackendSubmissionError(
                "SGE resource resolution has no scheduler queue"
            )
        pe = str(resolved.extra.get("parallel_environment") or "").strip()
        total_memory_mib = int(
            math.ceil(slurm_memory_mib(mem_per_cpu) * float(max(1, cpus)))
        )
        lines += [
            "#$ -S /bin/bash",
            "#$ -V",
            "#$ -N " + job_name,
            "#$ -q " + queue,
            "#$ -l h_rt=" + _format_sge_walltime_hours(wall),
            "#$ -l h_vmem=" + str(total_memory_mib) + "M",
        ]
        if int(cpus) > 1 or pe:
            if not pe:
                raise BackendSubmissionError(
                    "parallel SGE work has no configured parallel environment"
                )
            lines.append("#$ -pe " + pe + " " + str(int(cpus)))
        if is_array:
            lines.append("#$ -t 1-" + str(int(array_size)))
            if throttle_i is not None:
                lines.append("#$ -tc " + str(throttle_i))
        lines += [
            "#$ -o " + log_paths["output"],
            "#$ -e " + log_paths["error"],
        ]
    lines += [
        "",
        "# Resolved ICHOR resources: backend="
        + str(resolved.backend)
        + " cpu_reason="
        + str(resolved.cpu_reason)
        + " memory_reason="
        + str(resolved.memory_reason),
        "export ICHOR_ACTIVE_WORKERS="
        + str(int(resolved.extra.get("active_workers", resolved.cpus_per_task))),
        "export ICHOR_MEMORY_ONLY_CPUS="
        + str(int(resolved.extra.get("memory_only_cpus", 0))),
        "set -euo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
    ]
    if scheduler == "slurm":
        lines += [
            'export ICHOR_SCHEDULER_JOB_ID="${SLURM_JOB_ID:?missing SLURM_JOB_ID}"',
            'export ICHOR_SCHEDULER_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"',
            'export ICHOR_SCHEDULER_CPUS="${SLURM_CPUS_PER_TASK:-1}"',
        ]
    else:
        lines += [
            'export ICHOR_SCHEDULER_JOB_ID="${JOB_ID:?missing JOB_ID}"',
            (
                ': "${SGE_TASK_ID:?missing SGE_TASK_ID}"; '
                'export ICHOR_SCHEDULER_ARRAY_TASK_ID="$((SGE_TASK_ID - 1))"'
                if is_array
                else "export ICHOR_SCHEDULER_ARRAY_TASK_ID=0"
            ),
            'export ICHOR_SCHEDULER_CPUS="${NSLOTS:-1}"',
        ]
    lines += [
        "",
        *module_initialisation_lines(),
        "module purge",
        *["module load " + m for m in _configured_daemon_runtime_modules()],
        *native_runtime_setup_lines(),
        *python_library_path_export_lines(configured_python_library_paths()),
        "",
    ]
    if str(resolved.backend) == "diversity":
        lines += [
            "export ICHOR_DIVERSITY_DISTANCE_STORE_MODE="
            + _shell_quote(str(resolved.extra.get("distance_store_mode", "memory"))),
            "export ICHOR_DIVERSITY_SCRATCH_REQUIRED_BYTES="
            + str(
                int(
                    resolved.extra.get("scratch_required_bytes", 0)
                )
            ),
            "",
        ]
    if resource_resolution_binding is not None and submission_intent is not None:
        lines += _job_scratch_preamble(
            campaign_dir=campaign_dir,
            phase_name=phase_name,
            iteration=int(iteration),
            submission_intent=submission_intent,
            resource_resolution_binding=resource_resolution_binding,
            script_binding_path=attempt_bundle.script_binding,
        )
    if dry_run:
        renderer_name = (
            "Slurm"
            if scheduler == "slurm"
            else "Sun Grid Engine"
        )
        lines += [
            "# DRY-RUN: production "
            + renderer_name
            + "/resource/environment renderer only.",
            "# No scientific backend is invoked.",
            "echo " + _shell_quote(
                "DRYRUN " + str(phase_name) + " iteration=" + str(int(iteration))
            ),
            "exit 0",
            "",
        ]
        return "\n".join(lines)
    if "REPLACEMENT" in phase_name:
        from ..replacement_sampling import replacement_round_dir

        context = "bootstrap" if phase_name.startswith("INITIAL_") else "active"
        points_path = replacement_round_dir(
            campaign_dir,
            context=context,
            iteration=0 if context == "bootstrap" else int(iteration),
            replacement_round=int(replacement_round),
        )
    else:
        from ..layout import staging_phase_dir

        points_path = staging_phase_dir(campaign_dir, phase_name, int(iteration))
    points_file = str((points_path / "POINTS.txt").resolve())

    if "GAUSSIAN" in phase_name:
        lines += _gaussian_invocation_block(
            phase_name,
            iteration,
            camp,
            config,
            points_file,
            array_task_map=array_task_map,
            campaign_uid=campaign_uid,
            resolved_resources=resolved,
        )
    elif "AIMALL" in phase_name:
        lines += _aimall_invocation_block(
            iteration,
            camp,
            config,
            points_file,
            array_task_map=array_task_map,
        )
    elif phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
        lines += _ferebus_invocation_block(iteration, camp, config)
    elif phase_name == "ARIADNE_ARRAY":
        lines += _ariadne_invocation_block(
            iteration,
            camp,
            config,
            array_task_map=array_task_map,
        )
    elif phase_name in ("PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"):
        lines += _diversity_invocation_block(phase_name, iteration, camp, config)
    else:
        lines.append("# No invocation block registered for phase " + phase_name)
        lines.append("exit 1")

    lines.append("")
    return "\n".join(lines)


def build_sbatch_script(**kwargs: Any) -> str:
    """Compatibility wrapper preserving the established Slurm renderer API."""
    if active_machine() == "_default":
        _configured_scheduler()
    supplied = dict(kwargs)
    supplied["scheduler_kind"] = "slurm"
    return build_scheduler_script(**supplied)


def _array_task_mapping_lines(array_task_map: Optional[Path]) -> List[str]:
    if array_task_map is None:
        return [
            'ICHOR_LOGICAL_ARRAY_TASK_ID="${ICHOR_SCHEDULER_ARRAY_TASK_ID}"',
        ]
    task_map = _shell_quote(str(Path(array_task_map).resolve()))
    python = _python_executable_for_script()
    from ..versioning.manifest import sha256_file

    expected_sha256 = sha256_file(Path(array_task_map))
    return [
        "if [ ! -f " + task_map + " ]; then echo "
        + _shell_quote("retry task map missing: " + str(Path(array_task_map).resolve()))
        + " >&2; exit 1; fi",
        "if ! "
        + python
        + " -c "
        + _shell_quote(
            "import hashlib,sys; observed=hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(); "
            "raise SystemExit(0 if observed == sys.argv[2] else 1)"
        )
        + " "
        + task_map
        + " "
        + _shell_quote(expected_sha256)
        + "; then echo "
        + _shell_quote("retry task map SHA-256 mismatch")
        + " >&2; exit 1; fi",
        "ICHOR_LOGICAL_ARRAY_TASK_ID=$("
        + python
        + " -c "
        + _shell_quote(
            "import sys; from ichor.hpc.active_learning.strict_json import load_path; "
            "data=load_path(sys.argv[1]); "
            "print(int(data['dense_to_logical'][int(sys.argv[2])]))"
        )
        + " "
        + task_map
        + ' "$ICHOR_SCHEDULER_ARRAY_TASK_ID")',
        'if [ -z "$ICHOR_LOGICAL_ARRAY_TASK_ID" ]; then echo "no logical task id for retry index $ICHOR_SCHEDULER_ARRAY_TASK_ID" >&2; exit 1; fi',
        'case "$ICHOR_LOGICAL_ARRAY_TASK_ID" in (*[!0-9]*|"") echo "unsafe logical task id: $ICHOR_LOGICAL_ARRAY_TASK_ID" >&2; exit 1 ;; esac',
    ]


def _pointdir_selection_lines(
    points_file: str,
    *,
    required_filename: str,
) -> List[str]:
    """Select one direct child of the exact daemon-owned staging directory."""
    points_path = Path(points_file).resolve(strict=False)
    points_file_q = _shell_quote(str(points_path))
    staging_root_q = _shell_quote(str(points_path.parent))
    return [
        "export ICHOR_STAGING_ROOT=" + staging_root_q,
        "if [ ! -f " + points_file_q + " ]; then echo "
        + _shell_quote("POINTS.txt missing: " + str(points_path))
        + " >&2; exit 1; fi",
        'POINT_DIR=$(sed -n "$((ICHOR_LOGICAL_ARRAY_TASK_ID + 1))p" '
        + points_file_q
        + ")",
        'if [ -z "$POINT_DIR" ]; then echo "no pointdir for logical index $ICHOR_LOGICAL_ARRAY_TASK_ID" >&2; exit 1; fi',
        'if [ -L "$POINT_DIR" ]; then echo "POINT_DIR is a symlink: $POINT_DIR" >&2; exit 1; fi',
        'if [ "$(dirname -- "$POINT_DIR")" != "$ICHOR_STAGING_ROOT" ]; then echo "POINT_DIR escapes exact campaign staging round: $POINT_DIR" >&2; exit 1; fi',
        'case "$(basename -- "$POINT_DIR")" in POINT_*.pointdir) ;; *) echo "unsafe staged pointdir name: $POINT_DIR" >&2; exit 1 ;; esac',
        'if [ ! -d "$POINT_DIR" ]; then echo "POINT_DIR is not a directory: $POINT_DIR" >&2; exit 1; fi',
        'if [ ! -f "$POINT_DIR/'
        + str(required_filename)
        + '" ]; then echo "'
        + str(required_filename)
        + ' missing in $POINT_DIR" >&2; exit 1; fi',
    ]


def _gaussian_invocation_block(
    phase_name,
    iteration,
    camp,
    config,
    points_file,
    *,
    array_task_map: Optional[Path] = None,
    campaign_uid: Optional[str] = None,
    resolved_resources: ResolvedPhaseResources,
) -> List[str]:
    gaussian_modules = _configured_backend_modules(
        "gaussian",
        ["gaussian/g16c01_em64t_detectcpu"],
    )
    gaussian_exe = _configured_backend_shell_executable("gaussian", "g16")
    mdef = gaussian_mdef(config, resolved_resources)
    python = _python_executable_for_script()
    camp_q = _shell_quote(camp)
    phase_q = _shell_quote(str(phase_name))
    uid = _safe_shell_path_component(campaign_uid)
    uid_q = _shell_quote(uid)
    return [
        *["module load " + m for m in gaussian_modules],
        "",
        "# per-point gaussian array: task N runs the Nth staged pointdir.",
        "export ICHOR_CAMPAIGN_DIR=" + camp_q,
        "export ICHOR_CAMPAIGN_UID=" + uid_q,
        "export ICHOR_GAUSSIAN_PHASE=" + phase_q,
        "export ICHOR_ITERATION=" + str(int(iteration)),
        'export GAUSS_SCRDIR="$ICHOR_JOB_SCRATCH/gaussian"',
        'export GAUSS_PDEF="${ICHOR_SCHEDULER_CPUS:-1}"',
        "export GAUSS_MDEF=" + mdef,
        'mkdir -p "$GAUSS_SCRDIR"',
        'echo "GAUSS_SCRDIR=$GAUSS_SCRDIR"',
        *_array_task_mapping_lines(array_task_map),
        *_pointdir_selection_lines(points_file, required_filename="input.gjf"),
        'cd "$POINT_DIR"',
        python
        + " -m ichor.hpc.active_learning.daemon.quantum_job_prepare"
        + ' --campaign-dir "$ICHOR_CAMPAIGN_DIR"'
        + ' --pointdir "$POINT_DIR" --backend gaussian',
        gaussian_exe + " < input.gjf > input.gau",
    ]


def _configured_backend_path(backend_name: str, fallback: str) -> str:
    """Look up the executable_path for a software backend from the user's
    ~/ichor_config.yaml. Falls back to a sensible default (usually the bare
    command) when no active profile/backend path is declared, so unit tests
    and non-cluster environments keep working.
    """
    try:
        raw = expanded_profile_value(
            "software", backend_name, "executable_path", default=None
        )
    except Exception:
        return fallback
    if not raw:
        return fallback
    value = os.path.expanduser(os.path.expandvars(str(raw)))
    _reject_shell_control_chars(
        "configured " + backend_name + " executable_path",
        value,
    )
    return value


def _configured_backend_shell_executable(backend_name: str, fallback: str) -> str:
    try:
        raw = profile_value(
            "software", backend_name, "executable_path", default=None
        )
    except Exception:
        raw = None
    value = os.path.expanduser(str(raw or fallback))
    _reject_shell_control_chars(
        "configured " + backend_name + " executable_path",
        value,
    )
    return _shell_executable(value)


def _configured_backend_modules(backend_name: str, fallback: List[str]) -> List[str]:
    try:
        raw = profile_value("software", backend_name, "modules", default=None)
    except Exception:
        return list(fallback)
    if raw is None:
        return list(fallback)
    return _normalise_module_list(raw, label=backend_name)


def _aimall_invocation_block(
    iteration,
    camp,
    config,
    points_file,
    *,
    array_task_map: Optional[Path] = None,
) -> List[str]:
    aimall_path = _configured_backend_path("aimall", "~/AIMAll/aimqb.ish")
    aimall_modules = _configured_backend_modules("aimall", [])
    aimall_cfg = getattr(config, "aimall", None)
    args: List[str] = []
    if bool(getattr(aimall_cfg, "nogui", True)):
        args.append("-nogui")
    args.append('-nproc="${ICHOR_SCHEDULER_CPUS:-1}"')
    args.append('-naat="$AIMALL_NAAT"')
    encomp = int(getattr(aimall_cfg, "encomp", 3))
    args.append("-encomp=" + str(encomp))
    boaq = str(getattr(aimall_cfg, "boaq", "auto")).strip().lower()
    if boaq not in VALID_AIMALL_BOAQ_VALUES:
        raise BackendSubmissionError("aimall.boaq is invalid: " + repr(boaq))
    args.append("-boaq=" + boaq)
    iasmesh = str(getattr(aimall_cfg, "iasmesh", "fine")).strip().lower()
    if iasmesh not in VALID_AIMALL_IASMESH_VALUES:
        raise BackendSubmissionError("aimall.iasmesh is invalid: " + repr(iasmesh))
    args.append("-iasmesh=" + iasmesh)
    camp_q = _shell_quote(camp)
    python = _python_executable_for_script()
    return [
        *["module load " + module for module in aimall_modules],
        "",
        "# per-point AIMAll array over the .wfn files gaussian produced.",
        "export ICHOR_CAMPAIGN_DIR=" + camp_q,
        "export ICHOR_ITERATION=" + str(int(iteration)),
        *_array_task_mapping_lines(array_task_map),
        *_pointdir_selection_lines(points_file, required_filename="input.wfn"),
        'cd "$POINT_DIR"',
        'if [ ! -f AIMALL_TASK.json ]; then echo "AIMALL_TASK.json missing in $POINT_DIR" >&2; exit 1; fi',
        python
        + " -c "
        + _shell_quote(
            "import hashlib,pathlib,sys; "
            "from ichor.hpc.active_learning.strict_json import load_path; "
            "m=load_path('AIMALL_TASK.json'); "
            "w=pathlib.Path('input.wfn'); "
            "g=pathlib.Path('input.gjf'); "
            "r=pathlib.Path(m['wfn_method_receipt']['path']); "
            "q=pathlib.Path(m['gaussian_task_receipt']['path']); "
            "sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(); "
            "ok=(m.get('schema_version')==2 and "
            "m.get('pointdir')==pathlib.Path.cwd().name and "
            "sha(w)==m['wfn_sha256'] and sha(g)==m['gjf_sha256'] and "
            "sha(r)==m['wfn_method_receipt']['sha256'] and "
            "sha(q)==m['gaussian_task_receipt']['sha256']); "
            "sys.exit(0 if ok else "
            "('AIMAll WFN method receipt binding mismatch'))"
        ),
        "AIMALL_NAAT=$("
        + python
        + " -c "
        + _shell_quote(
            "from ichor.hpc.active_learning.strict_json import load_path; "
            "print(int(load_path('AIMALL_TASK.json')['naat']))"
        )
        + ")",
        'if [ -z "$AIMALL_NAAT" ]; then echo "AIMALL_NAAT is empty in $POINT_DIR" >&2; exit 1; fi',
        python
        + " -m ichor.hpc.active_learning.daemon.quantum_job_prepare"
        + ' --campaign-dir "$ICHOR_CAMPAIGN_DIR"'
        + ' --pointdir "$POINT_DIR" --backend aimall',
        " ".join([_shell_quote(aimall_path)] + args + ["input.wfn"]),
        "if ! "
        + python
        + " -m ichor.hpc.active_learning.daemon.ferebus_row_cache"
        + ' produce --campaign-dir "$ICHOR_CAMPAIGN_DIR" --pointdir "$POINT_DIR"; then',
        '  echo "warning: FEREBUS row shard failed; REFERENCE_COMMIT will repair it serially" >&2',
        "fi",
    ]


def _ferebus_invocation_block(iteration, camp, config) -> List[str]:
    return [
        "# FEREBUS live phases are submitted through pyferebus_wrap.submit_ferebus().",
        "# This generic daemon sbatch renderer is intentionally not used for FEREBUS.",
        'echo "FEREBUS must be submitted via pyferebus wrapper, not build_sbatch_script" >&2',
        "exit 2",
    ]


def _ariadne_invocation_block(
    iteration,
    camp,
    config,
    *,
    array_task_map: Optional[Path] = None,
) -> List[str]:
    # single-threaded BLAS so the acquisition-gradient process-pool owns the
    # cores SLURM gave this task.
    python = _python_executable_for_script()
    camp_q = _shell_quote(camp)
    return [
        "# ARIADNE per-seed adversarial attack array.",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "cd " + camp_q,
        *_array_task_mapping_lines(array_task_map),
        ariadne_runtime_command_prefix()
        + python
        + " -m ichor.hpc.active_learning.acquisition.ariadne_runner \\",
        "    --array-task-id $ICHOR_LOGICAL_ARRAY_TASK_ID \\",
        "    --iteration " + str(iteration) + " \\",
        "    --campaign-dir " + camp_q,
    ]


def _diversity_invocation_block(phase_name, iteration, camp, config) -> List[str]:
    descriptor = (
        "rmsd_massweight"
        if phase_name == "PHASE_A_DIVERSITY"
        else config.phase_b.descriptor
    )
    wrapper_iteration = 0 if phase_name == "PHASE_A_DIVERSITY" else int(iteration)
    python = _python_executable_for_script()
    camp_q = _shell_quote(camp)
    return [
        "# ICHOR exact diversity selection (" + phase_name + ").",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "cd " + camp_q,
        python + " -m ichor.hpc.active_learning.sampling.diversity \\",
        "    --descriptor " + _shell_quote(descriptor) + " \\",
        "    --iteration " + str(wrapper_iteration) + " \\",
        "    --campaign-dir " + camp_q + " \\",
        '    --workers "$ICHOR_ACTIVE_WORKERS" \\',
        '    --distance-store "$ICHOR_JOB_SCRATCH/diversity-condensed.float64"',
    ]
