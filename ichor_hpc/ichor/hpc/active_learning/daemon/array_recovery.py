"""Partial recovery for daemon-owned Slurm array phases.

The daemon can safely reuse completed task outputs for phases whose array
tasks are independent.  This module keeps that logic outside the FSM: it
discovers the logical task set, validates existing task outputs, writes a
daemon-owned ledger, and emits dense retry task maps for Slurm.
"""
from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .state import atomic_write_json, atomic_write_text
from ..layout import staging_phase_dir


ARRAY_RECOVERY_SCHEMA_VERSION = 2
ARRAY_RECOVERY_DIR_NAME = "array_task_ledgers"
RETRY_TASKS_FILENAME_PREFIX = "RETRY_TASKS"

PARTIAL_RECOVERY_PHASES = frozenset(
    {
        "INITIAL_GAUSSIAN",
        "INITIAL_AIMALL",
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "INITIAL_REPLACEMENT_AIMALL",
        "GAUSSIAN",
        "AIMALL",
        "REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_AIMALL",
        "ARIADNE_ARRAY",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def supports_partial_array_recovery(phase_name: Any) -> bool:
    return str(getattr(phase_name, "value", phase_name)) in PARTIAL_RECOVERY_PHASES


def array_recovery_dir(campaign_dir: Union[str, Path]) -> Path:
    from .filesystem import operational_path

    return operational_path(campaign_dir, ARRAY_RECOVERY_DIR_NAME)


def array_ledger_path(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Path:
    phase = str(getattr(phase_name, "value", phase_name))
    safe_phase = phase.replace("/", "_").replace("\\", "_")
    round_suffix = ""
    if "REPLACEMENT" in phase:
        replacement_round, _round_dir = _replacement_identity(
            campaign_dir,
            phase,
            int(iteration),
        )
        round_suffix = "-r" + str(replacement_round).zfill(4)
    return array_recovery_dir(campaign_dir) / (
        safe_phase + "-" + str(int(iteration)).zfill(6) + round_suffix + ".json"
    )


def retry_task_file_path(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Path:
    phase = str(getattr(phase_name, "value", phase_name))
    safe_phase = phase.replace("/", "_").replace("\\", "_")
    round_suffix = ""
    if "REPLACEMENT" in phase:
        replacement_round, _round_dir = _replacement_identity(
            campaign_dir,
            phase,
            int(iteration),
        )
        round_suffix = ".r" + str(replacement_round).zfill(4)
    from .filesystem import operational_path

    return operational_path(
        campaign_dir,
        RETRY_TASKS_FILENAME_PREFIX
        + "."
        + safe_phase
        + "."
        + str(int(iteration)).zfill(6)
        + round_suffix
        + ".txt",
    )


def _sha_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _replacement_identity(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
) -> Tuple[int, Path]:
    from ..point_allocation import (
        pending_attempts,
        point_allocation_path,
        read_point_allocation,
    )
    from ..replacement_sampling import (
        read_replacement_sample_strict,
        replacement_round_dir,
    )
    from .filesystem import operational_path
    from .state import read_state

    context = "bootstrap" if str(phase_name).startswith("INITIAL_") else "active"
    allocation_iteration = 0 if context == "bootstrap" else int(iteration)
    state = read_state(operational_path(campaign_dir, "state.json"))
    allocation = read_point_allocation(
        point_allocation_path(
            campaign_dir,
            context=context,
            iteration=allocation_iteration,
        ),
        expected_campaign_uid=str(state.campaign_uid),
        expected_context=context,
        expected_iteration=allocation_iteration,
    )
    rounds = {
        int(attempt.get("round", -1)) for attempt in pending_attempts(allocation)
    }
    if len(rounds) != 1 or next(iter(rounds)) <= 0:
        raise ValueError("replacement recovery cannot resolve one pending round")
    replacement_round = next(iter(rounds))
    read_replacement_sample_strict(
        campaign_dir,
        context=context,
        iteration=allocation_iteration,
        replacement_round=replacement_round,
        expected_campaign_uid=str(state.campaign_uid),
    )
    return replacement_round, replacement_round_dir(
        campaign_dir,
        context=context,
        iteration=allocation_iteration,
        replacement_round=replacement_round,
    )


def _bucket_dir(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    if "REPLACEMENT" in str(phase_name):
        _replacement_round, round_dir = _replacement_identity(
            campaign_dir,
            phase_name,
            int(iteration),
        )
        return round_dir
    return staging_phase_dir(campaign_dir, phase_name, int(iteration))


def _points_file(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    return _bucket_dir(campaign_dir, phase_name, iteration) / "POINTS.txt"


def _read_point_paths(points_file: Path) -> List[Path]:
    if not Path(points_file).is_file():
        return []
    bucket = Path(points_file).parent
    if Path(points_file).is_symlink():
        raise ValueError("POINTS.txt must not be a symlink")
    out: List[Path] = []
    seen = set()
    for line_number, raw in enumerate(Path(points_file).read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if text:
            candidate = Path(text)
            expected = bucket / candidate.name
            if candidate.resolve(strict=False) != expected.resolve(strict=False):
                raise ValueError("POINTS.txt path escapes its bucket at line " + str(line_number))
            if expected.name in seen:
                raise ValueError("POINTS.txt contains a duplicate pointdir")
            seen.add(expected.name)
            out.append(expected)
    return out


def logical_task_ids(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> List[int]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        return []
    if phase == "ARIADNE_ARRAY":
        from ..layout import active_iteration_dir
        from ..seed_identity import read_ariadne_task_map

        task_map = read_ariadne_task_map(
            active_iteration_dir(campaign_dir, int(iteration)),
            expected_iteration=int(iteration),
        )
        return [int(task["array_task_id"]) for task in task_map["tasks"]]
    from .quantum_task_contracts import quantum_task_contract

    replacement_round = (
        _replacement_identity(campaign_dir, phase, int(iteration))[0]
        if "REPLACEMENT" in phase
        else 0
    )
    contract = quantum_task_contract(
        campaign_dir,
        phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=True,
    )
    return [task.logical_task_id for task in contract.tasks]


def _pointdir_for_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
) -> Optional[Path]:
    from .quantum_task_contracts import quantum_task_contract

    replacement_round = (
        _replacement_identity(campaign_dir, phase_name, int(iteration))[0]
        if "REPLACEMENT" in phase_name
        else 0
    )
    contract = quantum_task_contract(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=True,
    )
    index = int(task_id)
    if index < 0 or index >= len(contract.tasks):
        return None
    return contract.tasks[index].pointdir


def _validate_quantum_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
) -> Tuple[bool, str, str]:
    from .quantum_task_contracts import quantum_task_contract

    replacement_round = (
        _replacement_identity(campaign_dir, phase_name, int(iteration))[0]
        if "REPLACEMENT" in phase_name
        else 0
    )
    contract = quantum_task_contract(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=True,
    )
    index = int(task_id)
    if index < 0 or index >= len(contract.tasks):
        return False, "task_not_in_points_file", ""
    pointdir = contract.tasks[index].pointdir
    if "GAUSSIAN" in str(phase_name):
        from .input_staging import (
            POINTS_MEMBERSHIP_NONE,
            quantum_acceptance_manifest_path,
            read_quantum_acceptance_manifest,
        )
        manifest_path = quantum_acceptance_manifest_path(
            contract.staging_dir,
            phase_name=str(phase_name),
        )
        if manifest_path.exists() or manifest_path.is_symlink():
            try:
                if manifest_path.is_symlink():
                    raise ValueError(
                        "Gaussian acceptance manifest must not be a symlink"
                    )
                _accepted, manifest = read_quantum_acceptance_manifest(
                    contract.staging_dir,
                    expected_phase=str(phase_name),
                    expected_iteration=int(iteration),
                    require_nonempty=False,
                    points_membership=POINTS_MEMBERSHIP_NONE,
                    require_accepted_payloads=False,
                )
                dispositions = list(manifest["accepted_pointdirs"]) + [
                    str(record["pointdir"])
                    for record in list(manifest["rejected"])
                ]
                if (
                    int(manifest["n_total"]) != contract.logical_total
                    or set(dispositions) != set(contract.pointdir_names)
                    or len(dispositions) != len(set(dispositions))
                ):
                    raise ValueError(
                        "Gaussian acceptance does not cover its producer task set"
                    )
                rejected_names = {
                    str(record["pointdir"])
                    for record in list(manifest["rejected"])
                }
                if pointdir.name in rejected_names:
                    return (
                        True,
                        "terminal_gaussian_rejection",
                        str(pointdir.resolve(strict=False)),
                    )
            except Exception as exc:
                return (
                    False,
                    "gaussian_acceptance_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:140],
                    str(pointdir.resolve(strict=False)),
                )
    return _validate_quantum_task_path(
        campaign_dir,
        phase_name,
        int(iteration),
        int(task_id),
        pointdir,
        expected_campaign_uid=str(contract.campaign_uid),
    )


def _validate_quantum_task_path(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
    pointdir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Tuple[bool, str, str]:
    if pointdir.is_symlink():
        return False, "pointdir_symlinked", str(pointdir.resolve(strict=False))
    if not pointdir.exists():
        return False, "pointdir_missing", str(pointdir.resolve(strict=False))
    if not pointdir.is_dir():
        return False, "pointdir_wrong_type", str(pointdir.resolve(strict=False))
    try:
        from ichor.core.files.point_directory import PointDirectory
        from .live_executor import validate_aimall_completed, validate_gaussian_completed

        validator = validate_gaussian_completed if "GAUSSIAN" in phase_name else validate_aimall_completed
        ok, reason = validator(PointDirectory(pointdir))
        if ok:
            from .quantum_task_receipts import read_quantum_task_receipt
            from .submission_intent import (
                resolve_quantum_task_receipt_producer,
            )

            try:
                receipt = read_quantum_task_receipt(
                    pointdir,
                    phase_name=phase_name,
                    iteration=int(iteration),
                    logical_task_id=int(task_id),
                )
            except ValueError:
                if "GAUSSIAN" not in phase_name:
                    raise
                from .input_staging import (
                    read_gaussian_task_receipt_after_aimall,
                )
                from .quantum_task_contracts import (
                    aimall_phase_for_gaussian,
                )

                receipt = read_gaussian_task_receipt_after_aimall(
                    pointdir,
                    gaussian_phase=phase_name,
                    aimall_phase=aimall_phase_for_gaussian(phase_name),
                    iteration=int(iteration),
                    logical_task_id=int(task_id),
                )
            replacement_round = (
                _replacement_identity(
                    campaign_dir,
                    phase_name,
                    int(iteration),
                )[0]
                if "REPLACEMENT" in phase_name
                else 0
            )
            producer = resolve_quantum_task_receipt_producer(
                campaign_dir,
                receipt,
                expected_campaign_uid=(
                    str(expected_campaign_uid)
                    if expected_campaign_uid is not None
                    else str(receipt["campaign_uid"])
                ),
                phase_name=phase_name,
                iteration=int(iteration),
                logical_task_id=int(task_id),
                replacement_round=int(replacement_round),
            )
            observed_identity = (
                str(receipt["campaign_uid"]),
                str(receipt["attempt_id"]),
                str(receipt["submission_identity"]),
                str(receipt["job_id"]),
            )
            expected_identity = (
                str(producer["campaign_uid"]),
                str(producer["attempt_id"]),
                str(producer["submission_identity"]),
                str(producer["job_id"]),
            )
            if observed_identity != expected_identity:
                raise ValueError("quantum task receipt producer identity mismatch")
        return bool(ok), str(reason or ""), str(Path(pointdir).resolve(strict=False))
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:160], str(Path(pointdir).resolve(strict=False))


def scan_aimall_upstream_gaussian_recovery(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    """Rewind absent AIMAll pointdirs through their Gaussian producer tasks."""
    phase = str(getattr(phase_name, "value", phase_name))
    from .quantum_task_contracts import (
        AIMALL_PHASES,
        gaussian_phase_for_aimall,
        quantum_task_contract,
    )

    if phase not in AIMALL_PHASES:
        return None
    if "REPLACEMENT" in phase:
        replacement_round, staging = _replacement_identity(
            campaign_dir,
            phase,
            int(iteration),
        )
    else:
        replacement_round = 0
        staging = staging_phase_dir(campaign_dir, phase, int(iteration))
    from .input_staging import quantum_acceptance_manifest_path

    publication = quantum_acceptance_manifest_path(
        staging,
        phase_name=phase,
    )
    if publication.exists() or publication.is_symlink():
        return None
    gaussian_phase = gaussian_phase_for_aimall(phase)
    from .input_staging import (
        POINTS_MEMBERSHIP_NONE,
        _points_file_names,
        read_quantum_acceptance_manifest,
    )

    accepted_paths, gaussian_manifest = read_quantum_acceptance_manifest(
        staging,
        expected_phase=gaussian_phase,
        expected_iteration=int(iteration),
        require_nonempty=False,
        points_membership=POINTS_MEMBERSHIP_NONE,
        require_accepted_payloads=False,
    )
    missing_names = {
        Path(path).name
        for path in accepted_paths
        if not Path(path).exists() and not Path(path).is_symlink()
    }
    if not missing_names:
        return None
    aimall = quantum_task_contract(
        campaign_dir,
        phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    missing = [
        task for task in aimall.tasks if task.pointdir_name in missing_names
    ]
    if len(missing) != len(missing_names):
        raise ValueError(
            "missing AIMAll pointdirs do not match the accepted Gaussian task set"
        )
    gaussian = quantum_task_contract(
        campaign_dir,
        gaussian_phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    listed_names = tuple(_points_file_names(aimall.staging_dir))
    if listed_names not in {
        aimall.pointdir_names,
        gaussian.pointdir_names,
    }:
        raise ValueError(
            "AIMAll POINTS.txt matches neither its accepted task set nor the "
            "complete Gaussian producer task set"
        )
    prior_rejected = {
        str(record["pointdir"])
        for record in list(gaussian_manifest.get("rejected") or [])
    }
    missing_producer_ids = {
        int(task.producer_logical_task_id) for task in missing
    }
    tasks = []
    retry_ids = []
    n_complete = 0
    for task in gaussian.tasks:
        if int(task.logical_task_id) in missing_producer_ids:
            ok = False
            reason = "aimall_pointdir_missing_requires_gaussian"
            output_path = str(task.pointdir.resolve(strict=False))
        elif task.pointdir_name in prior_rejected:
            ok = True
            reason = "prior_gaussian_rejection_preserved"
            output_path = str(task.pointdir.resolve(strict=False))
        else:
            ok, reason, output_path = _validate_quantum_task_path(
                campaign_dir,
                gaussian_phase,
                int(iteration),
                int(task.logical_task_id),
                task.pointdir,
            )
        status = "complete" if ok else "pending"
        if ok:
            n_complete += 1
        else:
            retry_ids.append(int(task.logical_task_id))
        tasks.append(
            {
                "task_id": int(task.logical_task_id),
                "status": status,
                "complete": bool(ok),
                "reason": str(reason or ""),
                "output_path": str(output_path),
                "contract_hash": _sha_payload(
                    {
                        "phase": gaussian_phase,
                        "iteration": int(iteration),
                        "task_id": int(task.logical_task_id),
                        "output_path": str(output_path),
                    }
                ),
            }
        )
    return {
        "schema_version": ARRAY_RECOVERY_SCHEMA_VERSION,
        "phase": gaussian_phase,
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "updated_at_iso": _now_iso(),
        "force_resubmit": False,
        "logical_total": int(len(tasks)),
        "n_complete": int(n_complete),
        "n_reuse": int(n_complete),
        "n_retry": int(len(retry_ids)),
        "retry_task_ids": retry_ids,
        "all_complete": False,
        "tasks": tasks,
        "source_phase": phase,
        "upstream_rewind": "missing_aimall_pointdir",
        "missing_aimall_task_ids": [
            int(task.logical_task_id) for task in missing
        ],
    }


def prepare_aimall_upstream_gaussian_recovery(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Dict[str, Any]:
    """Restore the complete Gaussian task list for an AIMAll rewind."""
    phase = str(getattr(phase_name, "value", phase_name))
    from .input_staging import _points_file_names, write_points_file
    from .quantum_task_contracts import (
        AIMALL_PHASES,
        gaussian_phase_for_aimall,
        quantum_task_contract,
    )

    if phase not in AIMALL_PHASES:
        raise ValueError("phase is not an AIMAll recovery phase: " + phase)
    replacement_round = (
        _replacement_identity(campaign_dir, phase, int(iteration))[0]
        if "REPLACEMENT" in phase
        else 0
    )
    aimall = quantum_task_contract(
        campaign_dir,
        phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    missing = [
        task
        for task in aimall.tasks
        if not task.pointdir.exists() and not task.pointdir.is_symlink()
    ]
    if not missing:
        raise ValueError(
            "AIMAll recovery has no absent accepted point directory"
        )
    gaussian_phase = gaussian_phase_for_aimall(phase)
    gaussian = quantum_task_contract(
        campaign_dir,
        gaussian_phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    listed = tuple(_points_file_names(aimall.staging_dir))
    permitted = {aimall.pointdir_names, gaussian.pointdir_names}
    if listed not in permitted:
        raise ValueError(
            "AIMAll POINTS.txt is inconsistent with its Gaussian producer"
        )
    changed = listed != gaussian.pointdir_names
    if changed:
        write_points_file(
            gaussian.staging_dir,
            [task.pointdir for task in gaussian.tasks],
        )
    return {
        "phase": gaussian_phase,
        "source_phase": phase,
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "changed": bool(changed),
        "points_file": str(
            (gaussian.staging_dir / "POINTS.txt").resolve(strict=False)
        ),
        "logical_total": int(gaussian.logical_total),
        "retry_task_ids": [
            int(task.producer_logical_task_id) for task in missing
        ],
    }


def _validate_ariadne_task(
    campaign_dir: Union[str, Path],
    iteration: int,
    task_id: int,
) -> Tuple[bool, str, str]:
    from ..ariadne_outputs import validate_seed_output
    from ..layout import active_iteration_dir, ariadne_seed_dir
    from ..seed_identity import read_ariadne_task_map, task_for_array_task_id

    iter_dir = active_iteration_dir(campaign_dir, int(iteration))
    try:
        task_map = read_ariadne_task_map(
            iter_dir,
            expected_iteration=int(iteration),
        )
        task = task_for_array_task_id(task_map, int(task_id))
        seed_dir = ariadne_seed_dir(iter_dir, int(task["seed_id"]))
        output = validate_seed_output(
            seed_dir,
            expected_campaign_uid=str(task_map["campaign_uid"]),
            expected_iteration=int(iteration),
            expected_seed_id=int(task["seed_id"]),
            expected_seed_uid=str(task["seed_uid"]),
            expected_array_task_id=int(task_id),
        )
        if not bool(output.get("task_success", False)):
            return False, "task_success_false", str(seed_dir.resolve(strict=False))
        try:
            task_exit_code = int(output.get("task_exit_code", 1))
        except (TypeError, ValueError):
            return False, "task_exit_code_invalid", str(seed_dir.resolve(strict=False))
        if task_exit_code != 0:
            return (
                False,
                "task_exit_code_" + str(task_exit_code),
                str(seed_dir.resolve(strict=False)),
            )
        return True, "", str(seed_dir.resolve(strict=False))
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:160], str(iter_dir.resolve(strict=False))


def _validate_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
) -> Tuple[bool, str, str]:
    if phase_name == "ARIADNE_ARRAY":
        return _validate_ariadne_task(campaign_dir, int(iteration), int(task_id))
    return _validate_quantum_task(campaign_dir, phase_name, int(iteration), int(task_id))


def scan_array_tasks(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    force_resubmit: bool = False,
) -> Dict[str, Any]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        raise ValueError("phase does not support partial array recovery: " + phase)
    task_ids = logical_task_ids(campaign_dir, phase, int(iteration))
    tasks: List[Dict[str, Any]] = []
    n_complete = 0
    for task_id in task_ids:
        ok, reason, output_path = _validate_task(
            campaign_dir,
            phase,
            int(iteration),
            int(task_id),
        )
        if bool(force_resubmit):
            status = "pending"
            reason = "force_resubmit"
        elif ok:
            status = "complete"
            n_complete += 1
        else:
            status = "pending"
        tasks.append(
            {
                "task_id": int(task_id),
                "status": status,
                "complete": bool(status == "complete"),
                "reason": str(reason or ""),
                "output_path": str(output_path),
                "contract_hash": _sha_payload(
                    {
                        "phase": phase,
                        "iteration": int(iteration),
                        "task_id": int(task_id),
                        "output_path": str(output_path),
                    }
                ),
            }
        )
    if bool(force_resubmit):
        n_complete = 0
    retry_ids = [int(task["task_id"]) for task in tasks if task.get("status") != "complete"]
    payload: Dict[str, Any] = {
        "schema_version": ARRAY_RECOVERY_SCHEMA_VERSION,
        "phase": phase,
        "iteration": int(iteration),
        "replacement_round": (
            _replacement_identity(campaign_dir, phase, int(iteration))[0]
            if "REPLACEMENT" in phase
            else 0
        ),
        "updated_at_iso": _now_iso(),
        "force_resubmit": bool(force_resubmit),
        "logical_total": int(len(task_ids)),
        "n_complete": int(n_complete),
        "n_reuse": int(n_complete),
        "n_retry": int(len(retry_ids)),
        "retry_task_ids": retry_ids,
        "all_complete": bool(len(task_ids) > 0 and len(retry_ids) == 0),
        "tasks": tasks,
    }
    return payload


def write_array_ledger(
    campaign_dir: Union[str, Path],
    payload: Dict[str, Any],
) -> Path:
    phase = str(payload.get("phase"))
    iteration = int(payload.get("iteration"))
    path = array_ledger_path(campaign_dir, phase, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _validate_array_ledger(dict(payload), path=path, phase=phase, iteration=iteration))
    return path


def _ledger_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(label + " must be an exact integer >= " + str(minimum))
    return int(value)


def _validate_array_ledger(
    data: Any,
    *,
    path: Path,
    phase: str,
    iteration: int,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("array recovery ledger must be a JSON object: " + str(path))
    if _ledger_int(data.get("schema_version"), "array ledger schema_version") != ARRAY_RECOVERY_SCHEMA_VERSION:
        raise ValueError("unsupported array recovery ledger schema: " + str(path))
    if data.get("phase") != str(phase) or not supports_partial_array_recovery(phase):
        raise ValueError("array recovery ledger phase mismatch")
    if _ledger_int(data.get("iteration"), "array ledger iteration") != int(iteration):
        raise ValueError("array recovery ledger iteration mismatch")
    _ledger_int(data.get("replacement_round"), "array ledger replacement_round")
    for key in ("force_resubmit", "all_complete"):
        if not isinstance(data.get(key), bool):
            raise ValueError("array recovery ledger " + key + " must be a boolean")
    logical_total = _ledger_int(data.get("logical_total"), "array ledger logical_total")
    n_complete = _ledger_int(data.get("n_complete"), "array ledger n_complete")
    n_reuse = _ledger_int(data.get("n_reuse"), "array ledger n_reuse")
    n_retry = _ledger_int(data.get("n_retry"), "array ledger n_retry")
    tasks = data.get("tasks")
    retry_ids = data.get("retry_task_ids")
    if not isinstance(tasks, list) or len(tasks) != logical_total:
        raise ValueError("array recovery ledger task cardinality mismatch")
    if not isinstance(retry_ids, list):
        raise ValueError("array recovery retry_task_ids must be a list")
    seen = set()
    observed_retry = []
    observed_complete = 0
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("array recovery task must be an object")
        task_id = _ledger_int(task.get("task_id"), "array recovery task_id")
        if task_id in seen:
            raise ValueError("array recovery task IDs contain duplicates")
        seen.add(task_id)
        status = task.get("status")
        if status not in {"complete", "pending"}:
            raise ValueError("array recovery task status is invalid")
        if task.get("complete") is not (status == "complete"):
            raise ValueError("array recovery task status/complete mismatch")
        if not isinstance(task.get("reason"), str) or not isinstance(
            task.get("output_path"), str
        ):
            raise ValueError("array recovery task diagnostics are invalid")
        digest = task.get("contract_hash")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("array recovery task contract hash is invalid")
        if status == "complete":
            observed_complete += 1
        else:
            observed_retry.append(task_id)
    if seen != set(range(logical_total)):
        raise ValueError("array recovery logical task IDs must be contiguous from zero")
    parsed_retry = [
        _ledger_int(value, "array recovery retry task ID") for value in retry_ids
    ]
    if parsed_retry != observed_retry or len(parsed_retry) != len(set(parsed_retry)):
        raise ValueError("array recovery retry task set mismatch")
    if (
        n_complete != observed_complete
        or n_reuse != observed_complete
        or n_retry != len(observed_retry)
        or n_complete + n_retry != logical_total
    ):
        raise ValueError("array recovery ledger summary counts mismatch")
    if data["all_complete"] is not (logical_total > 0 and n_retry == 0):
        raise ValueError("array recovery all_complete is contradictory")
    return data


def read_array_ledger(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    path = array_ledger_path(campaign_dir, phase_name, int(iteration))
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ValueError("array recovery ledger must not be a symlink: " + str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data = _validate_array_ledger(
        data,
        path=path,
        phase=str(getattr(phase_name, "value", phase_name)),
        iteration=int(iteration),
    )
    data["path"] = str(path)
    return data


def refresh_array_ledger(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    force_resubmit: bool = False,
) -> Dict[str, Any]:
    payload = scan_array_tasks(
        campaign_dir,
        phase_name,
        int(iteration),
        force_resubmit=bool(force_resubmit),
    )
    path = write_array_ledger(campaign_dir, payload)
    payload["path"] = str(path)
    return payload


def write_retry_task_file(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    task_ids: Sequence[int],
) -> Path:
    path = retry_task_file_path(campaign_dir, phase_name, int(iteration))
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(int(task_id)) for task_id in task_ids)
    atomic_write_text(path, body + ("\n" if body else ""))
    return path


def clear_retry_task_file(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> None:
    path = retry_task_file_path(campaign_dir, phase_name, int(iteration))
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _ensure_inside_campaign(campaign_dir: Path, target: Path) -> None:
    campaign = campaign_dir.resolve(strict=False)
    resolved = target.resolve(strict=False)
    if resolved != campaign and campaign not in resolved.parents:
        raise ValueError("refusing to archive path outside campaign: " + str(target))


def _move_if_exists(source: Path, target_dir: Path, campaign_dir: Path) -> Optional[str]:
    from .filesystem import campaign_owned_path

    if not source.exists() and not source.is_symlink():
        return None
    if source.is_symlink():
        raise ValueError("refusing to archive symlinked array output: " + str(source))
    source = campaign_owned_path(campaign_dir, source)
    target_dir = campaign_owned_path(campaign_dir, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = campaign_owned_path(campaign_dir, target_dir)
    target = target_dir / source.name
    suffix = 1
    while target.exists():
        target = campaign_owned_path(
            campaign_dir, target_dir / (source.name + "." + str(suffix))
        )
        suffix += 1
    target = campaign_owned_path(campaign_dir, target)
    shutil.move(str(source), str(target))
    return str(target)


def archive_existing_array_task_outputs(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    task_ids: Optional[Sequence[int]] = None,
    archive_identity: Optional[str] = None,
) -> List[str]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        raise ValueError("phase does not support array output archive: " + phase)
    campaign = Path(campaign_dir)
    stamp = str(
        archive_identity
        or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    from .filesystem import campaign_owned_path, operational_path

    archive_root = campaign_owned_path(
        campaign,
        operational_path(
            campaign,
            "arr",
            phase,
            str(int(iteration)).zfill(6) + "-" + stamp,
        ),
    )
    archive_root.mkdir(parents=True, exist_ok=False)
    archive_receipt = archive_root / "ARCHIVE.json"
    ids = [int(x) for x in (task_ids if task_ids is not None else logical_task_ids(campaign, phase, int(iteration)))]
    archived: List[str] = []

    def record_archive(status: str) -> None:
        payload = {
            "schema_version": 1,
            "phase": phase,
            "iteration": int(iteration),
            "status": str(status),
            "moved": list(archived),
            "updated_at_iso": _now_iso(),
        }
        atomic_write_json(archive_receipt, payload)

    record_archive("moving")
    for task_id in ids:
        task_archive = campaign_owned_path(
            campaign, archive_root / str(int(task_id)).zfill(6)
        )
        if phase == "ARIADNE_ARRAY":
            from ..layout import active_iteration_dir, ariadne_seed_dir
            from ..seed_identity import read_ariadne_task_map, task_for_array_task_id

            iter_dir = active_iteration_dir(campaign, int(iteration))
            task_map = read_ariadne_task_map(
                iter_dir,
                expected_iteration=int(iteration),
            )
            task = task_for_array_task_id(task_map, int(task_id))
            seed_dir = ariadne_seed_dir(iter_dir, int(task["seed_id"]))
            moved = _move_if_exists(seed_dir, task_archive, campaign)
            if moved:
                archived.append(moved)
                record_archive("moving")
            partial_pattern = "." + seed_dir.name + ".partial-*"
            for candidate in sorted(seed_dir.parent.glob(partial_pattern)):
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved:
                    archived.append(moved)
                    record_archive("moving")
            continue
        pointdir = _pointdir_for_task(campaign, phase, int(iteration), int(task_id))
        if pointdir is None:
            continue
        pdir = Path(pointdir)
        if "GAUSSIAN" in phase:
            for candidate in [
                pdir / "input.gau",
                pdir / "input.log",
                pdir / "input.wfn",
                pdir / "GAUSSIAN_TASK_RECEIPT.json",
            ]:
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved:
                    archived.append(moved)
                    record_archive("moving")
        elif "AIMALL" in phase:
            patterns = [
                "*_atomicfiles",
                "*.int",
                "*.sum",
                "*.agpviz",
                "*.mgpviz",
                "AIMALL_COMPLETION_RECEIPT.json",
            ]
            for pattern in patterns:
                for candidate in sorted(pdir.glob(pattern)):
                    moved = _move_if_exists(candidate, task_archive, campaign)
                    if moved:
                        archived.append(moved)
                        record_archive("moving")
    record_archive("complete")
    return archived


def prepare_retry_submission(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    force_resubmit: bool = False,
) -> Dict[str, Any]:
    if not bool(force_resubmit):
        try:
            existing = read_array_ledger(campaign_dir, phase_name, int(iteration))
        except Exception:
            existing = None
        if isinstance(existing, dict) and bool(existing.get("force_resubmit", False)):
            force_resubmit = True
    payload = refresh_array_ledger(
        campaign_dir,
        phase_name,
        int(iteration),
        force_resubmit=bool(force_resubmit),
    )
    retry_ids = [int(x) for x in payload.get("retry_task_ids") or []]
    if retry_ids:
        retry_file = write_retry_task_file(campaign_dir, phase_name, int(iteration), retry_ids)
        payload["retry_task_file"] = str(retry_file)
    else:
        clear_retry_task_file(campaign_dir, phase_name, int(iteration))
        payload["retry_task_file"] = None
    return payload


def compact_array_recovery_summary(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "phase": payload.get("phase"),
        "iteration": payload.get("iteration"),
        "logical_total": int(payload.get("logical_total") or 0),
        "n_complete": int(payload.get("n_complete") or 0),
        "n_reuse": int(payload.get("n_reuse") or 0),
        "n_retry": int(payload.get("n_retry") or 0),
        "force_resubmit": bool(payload.get("force_resubmit", False)),
        "retry_task_ids_sample": list(payload.get("retry_task_ids") or [])[:12],
        "retry_task_ids_truncated": bool(len(list(payload.get("retry_task_ids") or [])) > 12),
        "ledger": payload.get("path"),
        "retry_task_file": payload.get("retry_task_file"),
    }


def discover_partial_array_recovery(
    campaign_dir: Union[str, Path],
    *,
    preferred_phase: Optional[Any] = None,
    preferred_iteration: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, int]] = []
    if preferred_phase is not None and supports_partial_array_recovery(preferred_phase):
        try:
            candidates.append((str(getattr(preferred_phase, "value", preferred_phase)), int(preferred_iteration or 0)))
        except Exception:
            pass
    if not candidates:
        return None
    for phase, iteration in candidates:
        try:
            task_ids = logical_task_ids(campaign_dir, phase, int(iteration))
        except Exception:
            continue
        if not task_ids:
            continue
        try:
            payload = refresh_array_ledger(campaign_dir, phase, int(iteration))
        except Exception:
            continue
        if int(payload.get("logical_total") or 0) > 0:
            return payload
    return None
