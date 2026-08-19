"""Operational recovery authority for interrupted FEREBUS staging.

This module deliberately remains outside ``ferebus_task_runner``'s import
closure.  Historical task receipts therefore keep the producer-code identity
under which they were written while reconcile and resume gain a shared view of
the staging tree that contains them.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..versioning.manifest import sha256_file
from ..versioning.trained_models import trained_models_commit_lock
from .filesystem import campaign_owned_path
from .input_staging import (
    FEREBUS_JOB_DETAILS,
    FEREBUS_TASK_MANIFEST,
    read_ferebus_manifest,
    resolve_ferebus_task_path,
)
from .reconcile_transaction import (
    inventory_reconcile_transactions,
    read_reconcile_transaction,
)
from .resource_records import read_resolution
from .scheduler_recovery import scheduler_terminal_recoveries
from .state import _fsync_parent_dir


FEREBUS_STAGING_DISPOSITIONS = frozenset(
    {
        "absent",
        "input_only",
        "prepared",
        "terminal_producer",
        "archived_terminal_producer",
        "partial_preparation",
        "contradictory",
    }
)

_FEREBUS_PHASES = frozenset({"INITIAL_FEREBUS", "FEREBUS"})
_TASK_MAP_FILENAME = "FEREBUS_TASK_MAP.json"
_TASK_RECEIPT_FILENAME = "FEREBUS_TASK_RECEIPT.json"


@dataclass(frozen=True)
class FerebusStagingRecoveryContext:
    disposition: str
    canonical_path: Path
    producer_path: Optional[Path] = None
    source_transaction_id: Optional[str] = None
    phase: str = ""
    iteration: int = 0
    replacement_round: int = 0
    scheduler_identity_kind: Optional[str] = None
    n_tasks: int = 0
    completed_logical_task_ids: Tuple[int, ...] = ()
    retry_logical_task_ids: Tuple[int, ...] = ()
    source_job_ids: Tuple[str, ...] = ()
    source_submission_identities: Tuple[str, ...] = ()
    terminal_receipt_paths: Tuple[str, ...] = ()
    task_map_sha256: Optional[str] = None
    manifest_sha256: Optional[str] = None
    manifest_identity_sha256: Optional[str] = None
    reason: str = ""

    @property
    def has_authenticated_task_map(self) -> bool:
        return self.disposition in {
            "prepared",
            "terminal_producer",
            "archived_terminal_producer",
        } and bool(self.task_map_sha256)

    @property
    def has_authenticated_producer_task_map(self) -> bool:
        return self.disposition in {
            "terminal_producer",
            "archived_terminal_producer",
        } and bool(self.task_map_sha256)

    @property
    def requires_restore(self) -> bool:
        return self.disposition == "archived_terminal_producer"

    def summary(self) -> Dict[str, Any]:
        return {
            "disposition": self.disposition,
            "canonical_path": str(self.canonical_path),
            "producer_path": (
                None if self.producer_path is None else str(self.producer_path)
            ),
            "source_transaction_id": self.source_transaction_id,
            "phase": self.phase,
            "iteration": int(self.iteration),
            "replacement_round": int(self.replacement_round),
            "scheduler_identity_kind": self.scheduler_identity_kind,
            "n_tasks": int(self.n_tasks),
            "scheduler_completed_candidates": len(
                self.completed_logical_task_ids
            ),
            "known_retry_candidates": len(self.retry_logical_task_ids),
            "completed_logical_task_ids": list(
                self.completed_logical_task_ids
            ),
            "retry_logical_task_ids": list(self.retry_logical_task_ids),
            "source_job_ids": list(self.source_job_ids),
            "source_submission_identities": list(
                self.source_submission_identities
            ),
            "terminal_receipt_paths": list(self.terminal_receipt_paths),
            "task_map_sha256": self.task_map_sha256,
            "manifest_sha256": self.manifest_sha256,
            "manifest_identity_sha256": self.manifest_identity_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _TreeInspection:
    disposition: str
    path: Path
    manifest: Optional[Mapping[str, Any]] = None
    task_map: Optional[Mapping[str, Any]] = None
    manifest_sha256: Optional[str] = None
    manifest_identity_sha256: Optional[str] = None
    producer_bound_task_ids: Tuple[int, ...] = ()
    reason: str = ""


def _canonical_json_sha256(payload: Any) -> str:
    from ..strict_json import strict_json as json

    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_scientific_identity(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(manifest)
    tasks = []
    for raw_task in list(payload.get("tasks") or []):
        if not isinstance(raw_task, Mapping):
            raise ValueError("FEREBUS task manifest contains a malformed task")
        task = dict(raw_task)
        task.pop("generated_config", None)
        tasks.append(task)
    payload["tasks"] = tasks
    return payload


def _validate_scientific_configuration(
    manifest: Mapping[str, Any],
    config: Optional[Any],
) -> None:
    if config is None:
        return
    from ..ferebus_prior import resolve_ferebus_prior_contract

    atoms = [str(value) for value in list(manifest.get("atoms") or [])]
    configured_properties = [
        str(value)
        for value in getattr(config.ferebus, "properties", ["iqa"])
    ]
    expected_prior = resolve_ferebus_prior_contract(
        config,
        atom_labels=atoms,
    ).to_dict()
    kernel = manifest.get("kernel_contract")
    if (
        str(manifest.get("system") or "") != str(config.system_name)
        or list(manifest.get("properties") or []) != configured_properties
        or not isinstance(kernel, Mapping)
        or str(kernel.get("family") or "") != str(config.ferebus.kernel)
        or manifest.get("prior_mean_contract") != expected_prior
    ):
        raise ValueError(
            "FEREBUS staging scientific configuration differs from campaign.yaml"
        )


def _read_task_map(path: Path) -> Dict[str, Any]:
    # Importing this private validator is intentional: it is the immutable
    # compute-side task-map grammar, and this module is not imported by it.
    from .ferebus_task_runner import _read_task_map as read_task_map

    return dict(read_task_map(path))


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " is missing, non-regular or symlinked")


def _validate_task_map_inputs(
    root: Path,
    manifest: Mapping[str, Any],
    task_map: Mapping[str, Any],
) -> None:
    manifest_path = root / FEREBUS_TASK_MANIFEST
    if str(task_map.get("task_manifest_path") or "") != manifest_path.name:
        raise ValueError("FEREBUS task map manifest path is invalid")
    if str(task_map.get("task_manifest_sha256") or "") != sha256_file(
        manifest_path
    ):
        raise ValueError("FEREBUS task map manifest digest mismatch")
    manifest_tasks = manifest.get("tasks")
    mapped_tasks = task_map.get("tasks")
    if (
        not isinstance(manifest_tasks, list)
        or not isinstance(mapped_tasks, list)
        or int(task_map.get("n_tasks", -1)) != len(mapped_tasks)
        or len(mapped_tasks) != len(manifest_tasks)
    ):
        raise ValueError("FEREBUS task map cardinality is invalid")
    for logical_task_id, (manifest_task, mapped_task) in enumerate(
        zip(manifest_tasks, mapped_tasks)
    ):
        if not isinstance(manifest_task, Mapping) or not isinstance(
            mapped_task, Mapping
        ):
            raise ValueError("FEREBUS task map contains a malformed task")
        if (
            mapped_task.get("task_index") != logical_task_id + 1
            or manifest_task.get("task_index") != logical_task_id + 1
            or mapped_task.get("property") != manifest_task.get("property")
            or mapped_task.get("atom") != manifest_task.get("atom")
            or mapped_task.get("expected_model_path")
            != manifest_task.get("expected_model_path")
        ):
            raise ValueError("FEREBUS task map ordering or identity is invalid")
        expected_performance = Path(
            str(manifest_task.get("expected_model_path") or "")
        ).with_suffix(".perf").as_posix()
        expected_receipt = (
            str(manifest_task.get("output_dir") or "")
            + "/"
            + _TASK_RECEIPT_FILENAME
        )
        if (
            mapped_task.get("expected_performance_path")
            != expected_performance
            or mapped_task.get("receipt_path") != expected_receipt
        ):
            raise ValueError("FEREBUS task output paths are invalid")
        generated_config = manifest_task.get("generated_config")
        if not isinstance(generated_config, Mapping):
            raise ValueError("prepared FEREBUS task lacks generated config evidence")
        config = mapped_task.get("config")
        if not isinstance(config, Mapping) or any(
            config.get(key) != generated_config.get(key)
            for key in ("path", "size", "sha256")
        ):
            raise ValueError("FEREBUS task-map config binding is invalid")
        datasets = mapped_task.get("datasets")
        manifest_datasets = manifest_task.get("datasets")
        if not isinstance(datasets, Mapping) or not isinstance(
            manifest_datasets, Mapping
        ):
            raise ValueError("FEREBUS task-map dataset binding is invalid")
        for split in ("train", "int_val", "ext_val"):
            bound = datasets.get(split)
            expected = manifest_datasets.get(split)
            if not isinstance(bound, Mapping) or not isinstance(
                expected, Mapping
            ) or any(
                bound.get(key) != expected.get(key)
                for key in ("path", "size", "sha256")
            ):
                raise ValueError(
                    "FEREBUS task-map " + split + " binding is invalid"
                )
        for label, binding in (
            ("config", config),
            *[("dataset " + split, datasets[split]) for split in (
                "train",
                "int_val",
                "ext_val",
            )],
        ):
            path = resolve_ferebus_task_path(
                root,
                binding.get("path"),
                "task-map " + label,
            )
            _regular_file(path, "FEREBUS " + label)
            if int(binding.get("size", -1)) != int(path.stat().st_size):
                raise ValueError("FEREBUS " + label + " size mismatch")
            if str(binding.get("sha256") or "") != sha256_file(path):
                raise ValueError("FEREBUS " + label + " digest mismatch")


def _expected_output_paths(
    root: Path,
    manifest: Mapping[str, Any],
) -> Tuple[Path, ...]:
    paths = []
    for task in list(manifest.get("tasks") or []):
        if not isinstance(task, Mapping):
            raise ValueError("FEREBUS task manifest contains a malformed task")
        model = resolve_ferebus_task_path(
            root,
            task.get("expected_model_path"),
            "expected_model_path",
        )
        output = resolve_ferebus_task_path(
            root,
            task.get("output_dir"),
            "output_dir",
        )
        paths.extend(
            (
                model,
                model.with_suffix(".perf"),
                output / _TASK_RECEIPT_FILENAME,
            )
        )
    return tuple(paths)


def _producer_bound_task_ids(
    root: Path,
    task_map: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Return tasks whose success receipt binds this exact producer map.

    This intentionally validates only control evidence.  Scientific model and
    performance payload hashes are checked later by resume, never by reconcile.
    """
    from ..strict_json import strict_json as json

    task_map_sha256 = str(task_map.get("task_map_sha256") or "")
    task_manifest_sha256 = str(task_map.get("task_manifest_sha256") or "")
    execution_kind = str(task_map.get("execution_kind") or "")
    executable = task_map.get("executable")
    tasks = task_map.get("tasks")
    if not isinstance(executable, Mapping) or not isinstance(tasks, list):
        raise ValueError("FEREBUS producer task-map controls are invalid")
    bound = []
    for logical_task_id, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError("FEREBUS producer task-map task is invalid")
        receipt_path = resolve_ferebus_task_path(
            root,
            task.get("receipt_path"),
            "receipt_path",
        )
        if not receipt_path.exists() and not receipt_path.is_symlink():
            continue
        _regular_file(receipt_path, "FEREBUS task receipt")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(receipt, Mapping):
            continue
        if (
            receipt.get("schema_version") != 1
            or receipt.get("task_map_sha256") != task_map_sha256
            or receipt.get("task_manifest_sha256") != task_manifest_sha256
            or receipt.get("task_index") != logical_task_id + 1
            or receipt.get("property") != task.get("property")
            or receipt.get("atom") != task.get("atom")
            or receipt.get("execution_kind") != execution_kind
            or receipt.get("argv") != task.get("argv")
            or receipt.get("executable") != executable
            or receipt.get("success") is not True
            or receipt.get("exit_code") != 0
        ):
            continue
        model = receipt.get("model")
        performance = receipt.get("performance")
        if (
            not isinstance(model, Mapping)
            or model.get("path") != task.get("expected_model_path")
        ):
            continue
        if task_map.get("performance_required") is True:
            if (
                not isinstance(performance, Mapping)
                or performance.get("path")
                != task.get("expected_performance_path")
            ):
                continue
        elif performance is not None:
            continue
        bound.append(logical_task_id)
    return tuple(bound)


def _inspect_tree(path: Path) -> _TreeInspection:
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        return _TreeInspection("absent", root)
    if root.is_symlink() or not root.is_dir():
        return _TreeInspection(
            "contradictory",
            root,
            reason="FEREBUS staging is non-directory or symlinked",
        )
    try:
        manifest_path = root / FEREBUS_TASK_MANIFEST
        if not manifest_path.exists() and not manifest_path.is_symlink():
            # A daemon can be interrupted after creating generated runtime but
            # before publishing the input manifest.  Such a tree has no
            # recoverable producer authority, but it is safe for the existing
            # preparation-cleanup path provided it contains no producer map or
            # obvious scheduler output.
            unsafe_names = {
                _TASK_MAP_FILENAME,
                _TASK_RECEIPT_FILENAME,
            }
            for entry in root.rglob("*"):
                if entry.is_symlink():
                    raise ValueError(
                        "manifestless FEREBUS staging contains a symlink"
                    )
                mode = entry.lstat().st_mode
                if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
                    raise ValueError(
                        "manifestless FEREBUS staging contains a special file"
                    )
                if entry.name in unsafe_names or (
                    stat.S_ISREG(mode)
                    and entry.suffix.lower() in {".model", ".perf"}
                ):
                    raise ValueError(
                        "manifestless FEREBUS staging contains producer evidence"
                    )
            return _TreeInspection(
                "partial_preparation",
                root,
                reason="generated runtime is missing its task manifest",
            )
        _regular_file(manifest_path, "FEREBUS task manifest")
        manifest = read_ferebus_manifest(root, verify_dataset_files=True)
        manifest_digest = sha256_file(manifest_path)
        identity_digest = _canonical_json_sha256(
            _manifest_scientific_identity(manifest)
        )
        declared_job_details = manifest.get("job_details")
        if declared_job_details is not None:
            if str(declared_job_details) != FEREBUS_JOB_DETAILS:
                raise ValueError(
                    "FEREBUS task manifest job-details path is invalid"
                )
            job_details = resolve_ferebus_task_path(
                root,
                declared_job_details,
                "job_details",
            )
            _regular_file(job_details, "FEREBUS job details")
        output_paths = _expected_output_paths(root, manifest)
        for output in output_paths:
            if output.is_symlink() or (output.exists() and not output.is_file()):
                raise ValueError(
                    "FEREBUS expected output is non-regular or symlinked: "
                    + str(output)
                )
        task_map_path = root / _TASK_MAP_FILENAME
        if not task_map_path.exists() and not task_map_path.is_symlink():
            has_outputs = any(output.is_file() for output in output_paths)
            generated = bool((root / "runFerebus.sh").exists()) or any(
                isinstance(task, Mapping)
                and isinstance(task.get("generated_config"), Mapping)
                for task in list(manifest.get("tasks") or [])
            )
            if has_outputs:
                raise ValueError(
                    "FEREBUS outputs exist without their producer task map"
                )
            return _TreeInspection(
                (
                    "partial_preparation"
                    if generated or declared_job_details is None
                    else "input_only"
                ),
                root,
                manifest=manifest,
                manifest_sha256=manifest_digest,
                manifest_identity_sha256=identity_digest,
                reason=(
                    "generated runtime is incomplete"
                    if generated or declared_job_details is None
                    else "authenticated input staging has no generated runtime"
                ),
            )
        task_map = _read_task_map(task_map_path)
        _validate_task_map_inputs(root, manifest, task_map)
        bound_task_ids = _producer_bound_task_ids(root, task_map)
        if declared_job_details is None and not bound_task_ids:
            return _TreeInspection(
                "partial_preparation",
                root,
                manifest=manifest,
                task_map=task_map,
                manifest_sha256=manifest_digest,
                manifest_identity_sha256=identity_digest,
                reason=(
                    "legacy generated runtime has no job-details binding or "
                    "successful producer receipt"
                ),
            )
        return _TreeInspection(
            "prepared",
            root,
            manifest=manifest,
            task_map=task_map,
            manifest_sha256=manifest_digest,
            manifest_identity_sha256=identity_digest,
            producer_bound_task_ids=bound_task_ids,
            reason="authenticated generated FEREBUS runtime is present",
        )
    except Exception as exc:
        return _TreeInspection(
            "contradictory",
            root,
            reason=type(exc).__name__ + ": " + str(exc),
        )


def _terminal_sources(
    campaign: Path,
    *,
    campaign_uid: str,
    phase: str,
    iteration: int,
    replacement_round: int,
    n_tasks: int,
) -> Dict[str, Any]:
    recoveries = scheduler_terminal_recoveries(
        campaign,
        campaign_uid=campaign_uid,
        phase=phase,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    latest: Dict[int, Tuple[int, bool]] = {}
    schedulers = set()
    jobs = []
    identities = []
    receipt_paths = []
    for recovery in recoveries:
        intent = recovery["intent"]
        receipt = recovery["receipt"]
        scheduler = str(intent.get("scheduler_identity_kind") or "")
        if scheduler not in {"slurm", "sge"}:
            raise ValueError("FEREBUS terminal producer scheduler is invalid")
        schedulers.add(scheduler)
        sequence = int(intent.get("attempt_sequence", 0))
        completed = {
            int(value)
            for value in receipt.get("completed_logical_task_ids", [])
        }
        task_ids = [int(value) for value in receipt.get("logical_task_ids", [])]
        for task_id in task_ids:
            if task_id < 0 or task_id >= int(n_tasks):
                raise ValueError(
                    "FEREBUS terminal receipt references an out-of-range task"
                )
            previous = latest.get(task_id)
            if previous is None or sequence > previous[0]:
                latest[task_id] = (sequence, task_id in completed)
        job_id = str(receipt.get("job_id") or "")
        if job_id and job_id not in jobs:
            jobs.append(job_id)
        identity = str(receipt.get("submission_identity") or "")
        if identity and identity not in identities:
            identities.append(identity)
        receipt_path = str(recovery["path"])
        if receipt_path not in receipt_paths:
            receipt_paths.append(receipt_path)
    if len(schedulers) > 1:
        raise ValueError("FEREBUS terminal receipts use conflicting schedulers")
    completed_ids = tuple(sorted(task_id for task_id, value in latest.items() if value[1]))
    retry_ids = tuple(
        task_id
        for task_id in range(int(n_tasks))
        if task_id not in set(completed_ids)
    )
    return {
        "recoveries": tuple(recoveries),
        "scheduler": next(iter(schedulers), None),
        "completed": completed_ids,
        "retry": retry_ids,
        "jobs": tuple(jobs),
        "identities": tuple(identities),
        "receipt_paths": tuple(receipt_paths),
    }


def _validate_historical_executable(
    campaign: Path,
    tree: _TreeInspection,
    recoveries: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(tree.task_map, Mapping):
        raise ValueError("FEREBUS producer task map is unavailable")
    executable = tree.task_map.get("executable")
    if not isinstance(executable, Mapping):
        raise ValueError("FEREBUS producer executable binding is invalid")
    map_path = str(executable.get("path") or "")
    map_digest = executable.get("sha256")
    for recovery in recoveries:
        intent = recovery.get("intent")
        if not isinstance(intent, Mapping):
            raise ValueError("FEREBUS terminal producer intent is invalid")
        resource_path = intent.get("resource_resolution_path")
        resource_digest = intent.get("resource_resolution_sha256")
        if not resource_path or not resource_digest:
            raise ValueError(
                "FEREBUS terminal producer lacks resource-resolution evidence"
            )
        source = campaign_owned_path(campaign, str(resource_path))
        _regular_file(source, "FEREBUS resource resolution")
        if sha256_file(source) != str(resource_digest):
            raise ValueError("FEREBUS resource-resolution digest mismatch")
        payload = read_resolution(source)
        if (
            str(payload.get("campaign_uid") or "")
            != str(intent.get("campaign_uid") or "")
            or str(payload.get("phase") or "")
            != str(intent.get("phase") or "")
            or int(payload.get("iteration", -1))
            != int(intent.get("iteration", -2))
            or str(payload.get("submission_identity") or "")
            != str(intent.get("submission_identity") or "")
        ):
            raise ValueError("FEREBUS resource-resolution ownership mismatch")
        implementation = payload.get("implementation_identity")
        backend = (
            implementation.get("backend_executable")
            if isinstance(implementation, Mapping)
            else None
        )
        if not isinstance(backend, Mapping):
            raise ValueError("FEREBUS executable provenance is unavailable")
        if (
            implementation.get("environment_generation_digest_sha256")
            != intent.get("environment_generation_digest_sha256")
        ):
            raise ValueError(
                "FEREBUS resource evidence belongs to another environment generation"
            )
        resolved = backend.get("resolved_file")
        configured = str(backend.get("configured_path") or "")
        if isinstance(resolved, Mapping):
            if (
                str(resolved.get("path") or "") != map_path
                or resolved.get("sha256") != map_digest
            ):
                raise ValueError(
                    "FEREBUS producer executable disagrees with resource evidence"
                )
        elif configured and configured != map_path:
            raise ValueError(
                "FEREBUS producer executable path disagrees with resource evidence"
            )


def _archive_candidates(
    campaign: Path,
    *,
    campaign_uid: str,
    iteration: int,
    models_dir_name: str = "TRAINED_MODELS",
) -> Tuple[Tuple[str, Path, _TreeInspection], ...]:
    expected_root = campaign_owned_path(campaign, campaign / models_dir_name)
    candidates = []
    for record in inventory_reconcile_transactions(campaign):
        if (
            record.get("status") != "COMMITTED"
            or int(record.get("schema_version", 0)) != 2
        ):
            continue
        transaction = read_reconcile_transaction(Path(str(record["path"])))
        if str(transaction.get("campaign_uid") or "") != str(campaign_uid):
            continue
        if int(transaction.get("proposed_iteration", -1)) != int(iteration):
            continue
        transaction_id = str(transaction["transaction_id"])
        expected = campaign_owned_path(
            campaign,
            expected_root
            / ("iteration-staging.before-reconcile-" + transaction_id),
        )
        for operation in list(transaction.get("completed_operations") or []):
            if (
                not isinstance(operation, Mapping)
                or operation.get("operation") != "archive_model_staging"
            ):
                continue
            paths = operation.get("paths")
            if not isinstance(paths, list):
                raise ValueError(
                    "reconcile model-staging archive record is malformed"
                )
            for raw_path in paths:
                archived = campaign_owned_path(campaign, str(raw_path))
                if archived != expected:
                    continue
                inspection = _inspect_tree(archived)
                candidates.append((transaction_id, archived, inspection))
    return tuple(candidates)


def classify_ferebus_staging_recovery(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    phase: str,
    iteration: int,
    replacement_round: int = 0,
    reference_data_version: Optional[int] = None,
    reference_head_manifest_sha256: Optional[str] = None,
    reference_view_sha256: Optional[str] = None,
    config: Optional[Any] = None,
    models_dir_name: str = "TRAINED_MODELS",
) -> FerebusStagingRecoveryContext:
    """Classify canonical and transaction-archived FEREBUS staging."""
    campaign = Path(campaign_dir).resolve()
    phase_name = str(phase)
    if phase_name not in _FEREBUS_PHASES:
        raise ValueError("FEREBUS staging recovery phase is unsupported")
    canonical = campaign_owned_path(
        campaign,
        campaign / models_dir_name / "iteration-staging",
    )
    canonical_tree = _inspect_tree(canonical)

    def validate_manifest(tree: _TreeInspection) -> None:
        manifest = tree.manifest
        if not isinstance(manifest, Mapping):
            raise ValueError("FEREBUS staging manifest is unavailable")
        expected_reference = (
            int(iteration)
            if reference_data_version is None
            else int(reference_data_version)
        )
        if (
            str(manifest.get("campaign_uid") or "") != str(campaign_uid)
            or int(manifest.get("reference_data_version", -1))
            != expected_reference
        ):
            raise ValueError("FEREBUS staging campaign/reference identity mismatch")
        if reference_head_manifest_sha256 is not None and str(
            manifest.get("reference_data_head_manifest_sha256") or ""
        ) != str(reference_head_manifest_sha256):
            raise ValueError("FEREBUS staging reference-head identity mismatch")
        if reference_view_sha256 is not None and str(
            manifest.get("reference_data_view_sha256") or ""
        ) != str(reference_view_sha256):
            raise ValueError("FEREBUS staging reference-view identity mismatch")
        _validate_scientific_configuration(manifest, config)

    if canonical_tree.disposition != "absent":
        if canonical_tree.disposition == "contradictory":
            return FerebusStagingRecoveryContext(
                "contradictory",
                canonical,
                phase=phase_name,
                iteration=int(iteration),
                replacement_round=int(replacement_round),
                reason=canonical_tree.reason,
            )
        if canonical_tree.manifest is not None:
            try:
                validate_manifest(canonical_tree)
            except Exception as exc:
                return FerebusStagingRecoveryContext(
                    "contradictory",
                    canonical,
                    phase=phase_name,
                    iteration=int(iteration),
                    replacement_round=int(replacement_round),
                    reason=type(exc).__name__ + ": " + str(exc),
                )

    n_tasks = int(
        (canonical_tree.manifest or {}).get("n_tasks", 0)
    )
    archive_records = _archive_candidates(
        campaign,
        campaign_uid=str(campaign_uid),
        iteration=int(iteration),
        models_dir_name=models_dir_name,
    )
    prepared_archives = []
    archive_errors = []
    for transaction_id, archived, inspection in archive_records:
        if inspection.disposition == "absent":
            continue
        if inspection.disposition in {
            "absent",
            "input_only",
            "partial_preparation",
        }:
            continue
        if inspection.disposition != "prepared":
            archive_errors.append(
                str(archived) + ": " + (inspection.reason or inspection.disposition)
            )
            continue
        try:
            validate_manifest(inspection)
            if (
                canonical_tree.manifest_identity_sha256
                and inspection.manifest_identity_sha256
                != canonical_tree.manifest_identity_sha256
            ):
                raise ValueError(
                    "archived producer scientific manifest differs from current staging"
                )
        except Exception as exc:
            archive_errors.append(
                str(archived) + ": " + type(exc).__name__ + ": " + str(exc)
            )
            continue
        prepared_archives.append((transaction_id, archived, inspection))
        if n_tasks == 0:
            n_tasks = int((inspection.manifest or {}).get("n_tasks", 0))

    if n_tasks <= 0:
        if archive_errors:
            return FerebusStagingRecoveryContext(
                "contradictory",
                canonical,
                phase=phase_name,
                iteration=int(iteration),
                replacement_round=int(replacement_round),
                reason="; ".join(archive_errors[:3]),
            )
        return FerebusStagingRecoveryContext(
            canonical_tree.disposition,
            canonical,
            phase=phase_name,
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            reason=canonical_tree.reason,
        )
    try:
        terminal = _terminal_sources(
            campaign,
            campaign_uid=str(campaign_uid),
            phase=phase_name,
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            n_tasks=n_tasks,
        )
    except Exception as exc:
        return FerebusStagingRecoveryContext(
            "contradictory",
            canonical,
            phase=phase_name,
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            n_tasks=n_tasks,
            reason=type(exc).__name__ + ": " + str(exc),
        )

    def context(
        disposition: str,
        tree: _TreeInspection,
        *,
        producer_path: Optional[Path] = None,
        source_transaction_id: Optional[str] = None,
        reason: str,
    ) -> FerebusStagingRecoveryContext:
        task_map_sha = (
            str((tree.task_map or {}).get("task_map_sha256") or "") or None
        )
        return FerebusStagingRecoveryContext(
            disposition,
            canonical,
            producer_path=producer_path,
            source_transaction_id=source_transaction_id,
            phase=phase_name,
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            scheduler_identity_kind=terminal["scheduler"],
            n_tasks=n_tasks,
            completed_logical_task_ids=tuple(terminal["completed"]),
            retry_logical_task_ids=tuple(terminal["retry"]),
            source_job_ids=tuple(terminal["jobs"]),
            source_submission_identities=tuple(terminal["identities"]),
            terminal_receipt_paths=tuple(terminal["receipt_paths"]),
            task_map_sha256=task_map_sha,
            manifest_sha256=tree.manifest_sha256,
            manifest_identity_sha256=tree.manifest_identity_sha256,
            reason=reason,
        )

    terminal_completed = set(terminal["completed"])

    def bound_to_terminal(tree: _TreeInspection) -> bool:
        return bool(
            terminal_completed.intersection(tree.producer_bound_task_ids)
        )

    if (
        canonical_tree.disposition == "prepared"
        and terminal["recoveries"]
        and bound_to_terminal(canonical_tree)
    ):
        try:
            _validate_historical_executable(
                campaign,
                canonical_tree,
                terminal["recoveries"],
            )
        except Exception as exc:
            return context(
                "contradictory",
                canonical_tree,
                producer_path=canonical,
                reason=type(exc).__name__ + ": " + str(exc),
            )
        return context(
            "terminal_producer",
            canonical_tree,
            producer_path=canonical,
            reason="canonical staging is bound to terminal scheduler evidence",
        )

    terminal_archives = [
        record for record in prepared_archives if bound_to_terminal(record[2])
    ]
    if archive_errors:
        return context(
            "contradictory",
            canonical_tree,
            reason="; ".join(archive_errors[:3]),
        )
    if len(terminal_archives) > 1:
        return context(
            "contradictory",
            terminal_archives[0][2],
            reason="multiple archived FEREBUS terminal producers match this phase",
        )
    if terminal_archives and terminal["recoveries"]:
        transaction_id, archived, tree = terminal_archives[0]
        try:
            _validate_historical_executable(
                campaign,
                tree,
                terminal["recoveries"],
            )
        except Exception as exc:
            return context(
                "contradictory",
                tree,
                producer_path=archived,
                source_transaction_id=transaction_id,
                reason=type(exc).__name__ + ": " + str(exc),
            )
        if canonical_tree.disposition not in {
            "absent",
            "input_only",
            "partial_preparation",
        }:
            return context(
                "contradictory",
                tree,
                producer_path=archived,
                source_transaction_id=transaction_id,
                reason=(
                    "archived terminal producer conflicts with canonical "
                    + canonical_tree.disposition
                    + " staging"
                ),
            )
        return context(
            "archived_terminal_producer",
            tree,
            producer_path=archived,
            source_transaction_id=transaction_id,
            reason="transaction archive contains the terminal FEREBUS producer",
        )
    reason = canonical_tree.reason
    if (
        canonical_tree.disposition == "prepared"
        and terminal["recoveries"]
        and not bound_to_terminal(canonical_tree)
    ):
        reason = (
            "prepared FEREBUS runtime has no successful receipt binding it "
            "to the historical terminal producer"
        )
    return context(
        canonical_tree.disposition,
        canonical_tree,
        producer_path=(
            canonical if canonical_tree.disposition == "prepared" else None
        ),
        reason=reason,
    )


def is_redundant_committed_parent_staging(
    campaign_dir: Path,
    *,
    campaign_uid: str,
    target_reference_data_version: int,
    parent_model_version: int,
    replacement_round: int = 0,
    artifact_snapshot: Optional[Any] = None,
    models_dir_name: str = "TRAINED_MODELS",
) -> bool:
    """Prove canonical staging is residue of the committed parent model.

    Successful publication intentionally leaves its source staging tree in
    place.  It is not evidence for the next FEREBUS attempt when its reference
    version is exactly that attempt's authenticated committed-model parent.
    """
    target_version = int(target_reference_data_version)
    parent_version = int(parent_model_version)
    if parent_version < 0 or target_version != parent_version + 1:
        return False
    campaign = Path(campaign_dir).resolve()
    try:
        if scheduler_terminal_recoveries(
            campaign,
            campaign_uid=str(campaign_uid),
            phase="FEREBUS",
            iteration=target_version,
            replacement_round=int(replacement_round),
        ):
            return False
    except Exception:
        return False
    staging = campaign_owned_path(
        campaign,
        campaign / models_dir_name / "iteration-staging",
    )
    try:
        if staging.is_symlink() or not staging.is_dir():
            return False
        manifest = read_ferebus_manifest(
            staging,
            verify_dataset_files=False,
        )
        if (
            str(manifest.get("campaign_uid") or "") != str(campaign_uid)
            or int(manifest.get("reference_data_version", -1))
            != parent_version
        ):
            return False
        if artifact_snapshot is None:
            from ..versioning.trained_models import TrainedModelVersioning

            parent = TrainedModelVersioning(
                campaign / models_dir_name
            ).resolve(parent_version, verification="metadata")
        else:
            parent = artifact_snapshot.model_set(parent_version)
        return bool(
            int(parent.version) == parent_version
            and str(parent.campaign_uid) == str(campaign_uid)
            and int(parent.reference_data_version) == parent_version
        )
    except Exception:
        return False


def restore_archived_ferebus_producer_staging(
    campaign_dir: Path,
    context: FerebusStagingRecoveryContext,
    *,
    transaction_id: str,
) -> Dict[str, Any]:
    """Atomically restore one authenticated transaction-archived producer."""
    campaign = Path(campaign_dir).resolve()
    identity = str(transaction_id)
    if len(identity) != 32 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError("reconcile transaction identity is invalid")
    canonical = campaign_owned_path(campaign, context.canonical_path)
    if context.disposition != "archived_terminal_producer":
        if context.disposition == "terminal_producer" and canonical.is_dir():
            return {
                "changed": False,
                "restored_path": str(canonical),
                "archived_input_path": None,
            }
        raise ValueError("FEREBUS staging context is not restorable")
    if context.producer_path is None:
        raise ValueError("archived FEREBUS producer path is missing")
    producer = campaign_owned_path(campaign, context.producer_path)
    input_archive = campaign_owned_path(
        campaign,
        canonical.with_name(
            canonical.name + ".before-reconcile-" + identity
        ),
    )
    with trained_models_commit_lock(campaign):
        canonical_inspection = _inspect_tree(canonical)
        if canonical_inspection.disposition == "prepared" and not producer.exists():
            if (
                canonical_inspection.task_map is None
                or str(
                    canonical_inspection.task_map.get("task_map_sha256") or ""
                )
                != str(context.task_map_sha256 or "")
                or canonical_inspection.manifest_identity_sha256
                != context.manifest_identity_sha256
            ):
                raise ValueError(
                    "replayed FEREBUS producer no longer matches inspected authority"
                )
            return {
                "changed": False,
                "restored_path": str(canonical),
                "archived_input_path": (
                    str(input_archive) if input_archive.exists() else None
                ),
            }
        if canonical_inspection.disposition in {
            "input_only",
            "partial_preparation",
        }:
            if (
                canonical_inspection.manifest_identity_sha256
                != context.manifest_identity_sha256
            ):
                raise ValueError(
                    "canonical FEREBUS inputs changed before restoration"
                )
            if input_archive.exists() or input_archive.is_symlink():
                raise ValueError(
                    "canonical and archived input-only FEREBUS staging both exist"
                )
            os.replace(canonical, input_archive)
            _fsync_parent_dir(canonical)
        elif canonical_inspection.disposition == "absent":
            if input_archive.exists():
                archived_input = _inspect_tree(input_archive)
                if archived_input.disposition not in {
                    "input_only",
                    "partial_preparation",
                } or (
                    archived_input.manifest_identity_sha256
                    != context.manifest_identity_sha256
                ):
                    raise ValueError(
                        "interrupted FEREBUS input archive is contradictory"
                    )
        else:
            raise ValueError(
                "canonical FEREBUS staging changed before restoration: "
                + canonical_inspection.disposition
            )
        producer_inspection = _inspect_tree(producer)
        if producer_inspection.disposition == "absent":
            if _inspect_tree(canonical).disposition == "prepared":
                return {
                    "changed": True,
                    "restored_path": str(canonical),
                    "archived_input_path": (
                        str(input_archive) if input_archive.exists() else None
                    ),
                }
            raise ValueError("archived FEREBUS producer disappeared before restoration")
        if (
            producer_inspection.disposition != "prepared"
            or producer_inspection.task_map is None
            or str(producer_inspection.task_map.get("task_map_sha256") or "")
            != str(context.task_map_sha256 or "")
            or producer_inspection.manifest_identity_sha256
            != context.manifest_identity_sha256
        ):
            raise ValueError(
                "archived FEREBUS producer changed before restoration"
            )
        if canonical.exists() or canonical.is_symlink():
            raise ValueError("canonical FEREBUS staging exists before producer restore")
        os.replace(producer, canonical)
        _fsync_parent_dir(canonical)
        restored = _inspect_tree(canonical)
        if (
            restored.disposition != "prepared"
            or restored.task_map is None
            or str(restored.task_map.get("task_map_sha256") or "")
            != str(context.task_map_sha256 or "")
            or restored.manifest_identity_sha256
            != context.manifest_identity_sha256
        ):
            raise ValueError(
                "restored FEREBUS producer no longer matches inspected authority"
            )
    return {
        "changed": True,
        "restored_path": str(canonical),
        "archived_input_path": (
            str(input_archive) if input_archive.exists() else None
        ),
    }


def archive_partial_ferebus_preparation(
    campaign_dir: Path,
    context: FerebusStagingRecoveryContext,
    *,
    attempt_id: str,
) -> Dict[str, Any]:
    """Retire generated-but-unsubmitted runtime before exact preparation."""
    campaign = Path(campaign_dir).resolve()
    identity = str(attempt_id)
    if len(identity) != 32 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError("FEREBUS preparation attempt identity is invalid")
    if context.disposition != "partial_preparation":
        raise ValueError("FEREBUS staging is not partial preparation")
    canonical = campaign_owned_path(campaign, context.canonical_path)
    archive = campaign_owned_path(
        campaign,
        canonical.with_name(
            canonical.name + ".partial-preparation-" + identity
        ),
    )
    with trained_models_commit_lock(campaign):
        current = _inspect_tree(canonical)
        if current.disposition == "absent":
            existing = _inspect_tree(archive)
            if (
                existing.disposition == "partial_preparation"
                and existing.manifest_identity_sha256
                == context.manifest_identity_sha256
            ):
                return {
                    "changed": False,
                    "archived_path": str(archive),
                }
            raise ValueError(
                "partial FEREBUS preparation disappeared before archival"
            )
        if (
            current.disposition != "partial_preparation"
            or current.manifest_identity_sha256
            != context.manifest_identity_sha256
        ):
            raise ValueError(
                "partial FEREBUS preparation changed before archival"
            )
        if archive.exists() or archive.is_symlink():
            raise ValueError(
                "partial FEREBUS preparation archive already exists"
            )
        os.replace(canonical, archive)
        _fsync_parent_dir(canonical)
    return {
        "changed": True,
        "archived_path": str(archive),
    }


__all__ = [
    "FEREBUS_STAGING_DISPOSITIONS",
    "FerebusStagingRecoveryContext",
    "archive_partial_ferebus_preparation",
    "classify_ferebus_staging_recovery",
    "is_redundant_committed_parent_staging",
    "restore_archived_ferebus_producer_staging",
]
