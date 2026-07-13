"""Partial recovery for daemon-owned Slurm array phases.

The daemon can safely reuse completed task outputs for phases whose array
tasks are independent.  This module keeps that logic outside the FSM: it
discovers the logical task set, validates existing task outputs, writes a
daemon-owned ledger, and emits dense retry task maps for Slurm.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .state import atomic_write_json
from ..layout import staging_phase_dir


ARRAY_RECOVERY_SCHEMA_VERSION = 1
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
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / ARRAY_RECOVERY_DIR_NAME


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
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / (RETRY_TASKS_FILENAME_PREFIX + "." + safe_phase + "." + str(int(iteration)).zfill(6) + round_suffix + ".txt")
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

    context = "bootstrap" if str(phase_name).startswith("INITIAL_") else "active"
    allocation_iteration = 0 if context == "bootstrap" else int(iteration)
    allocation = read_point_allocation(
        point_allocation_path(
            campaign_dir,
            context=context,
            iteration=allocation_iteration,
        )
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
    out: List[Path] = []
    for raw in Path(points_file).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text:
            out.append(Path(text))
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
    points = _read_point_paths(_points_file(campaign_dir, phase, int(iteration)))
    return list(range(len(points)))


def _pointdir_for_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
) -> Optional[Path]:
    points = _read_point_paths(_points_file(campaign_dir, phase_name, int(iteration)))
    index = int(task_id)
    if index < 0 or index >= len(points):
        return None
    return points[index]


def _validate_quantum_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
) -> Tuple[bool, str, str]:
    pointdir = _pointdir_for_task(campaign_dir, phase_name, iteration, task_id)
    if pointdir is None:
        return False, "task_not_in_points_file", ""
    try:
        from ichor.core.files.point_directory import PointDirectory
        from .live_executor import validate_aimall_completed, validate_gaussian_completed

        validator = validate_gaussian_completed if "GAUSSIAN" in phase_name else validate_aimall_completed
        ok, reason = validator(PointDirectory(pointdir))
        return bool(ok), str(reason or ""), str(Path(pointdir).resolve(strict=False))
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:160], str(Path(pointdir).resolve(strict=False))


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
    atomic_write_json(path, payload)
    return path


def read_array_ledger(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    path = array_ledger_path(campaign_dir, phase_name, int(iteration))
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("array recovery ledger must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != ARRAY_RECOVERY_SCHEMA_VERSION:
        raise ValueError("unsupported array recovery ledger schema: " + str(path))
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
    path.write_text(body + ("\n" if body else ""), encoding="utf-8", newline="\n")
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
    if not source.exists() and not source.is_symlink():
        return None
    if source.is_symlink():
        raise ValueError("refusing to archive symlinked array output: " + str(source))
    _ensure_inside_campaign(campaign_dir, source)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    suffix = 1
    while target.exists():
        target = target_dir / (source.name + "." + str(suffix))
        suffix += 1
    shutil.move(str(source), str(target))
    return str(target)


def archive_existing_array_task_outputs(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    task_ids: Optional[Sequence[int]] = None,
) -> List[str]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        raise ValueError("phase does not support array output archive: " + phase)
    campaign = Path(campaign_dir)
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    archive_root = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "arr"
        / phase
        / (str(int(iteration)).zfill(6) + "-" + stamp)
    )
    ids = [int(x) for x in (task_ids if task_ids is not None else logical_task_ids(campaign, phase, int(iteration)))]
    archived: List[str] = []
    for task_id in ids:
        task_archive = archive_root / str(int(task_id)).zfill(6)
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
            partial_pattern = "." + seed_dir.name + ".partial-*"
            for candidate in sorted(seed_dir.parent.glob(partial_pattern)):
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved:
                    archived.append(moved)
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
            ]:
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved:
                    archived.append(moved)
        elif "AIMALL" in phase:
            patterns = ["*_atomicfiles", "*.int", "*.sum", "*.agpviz", "*.mgpviz"]
            for pattern in patterns:
                for candidate in sorted(pdir.glob(pattern)):
                    moved = _move_if_exists(candidate, task_archive, campaign)
                    if moved:
                        archived.append(moved)
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
