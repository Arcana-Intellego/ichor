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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .state import atomic_write_json


ARRAY_RECOVERY_SCHEMA_VERSION = 1
ARRAY_RECOVERY_DIR_NAME = "array_task_ledgers"
RETRY_TASKS_FILENAME_PREFIX = "RETRY_TASKS"

PARTIAL_RECOVERY_PHASES = frozenset(
    {
        "INITIAL_GAUSSIAN",
        "INITIAL_AIMALL",
        "GAUSSIAN",
        "AIMALL",
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
    return array_recovery_dir(campaign_dir) / (
        safe_phase + "-" + str(int(iteration)).zfill(4) + ".json"
    )


def retry_task_file_path(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Path:
    phase = str(getattr(phase_name, "value", phase_name))
    safe_phase = phase.replace("/", "_").replace("\\", "_")
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / (RETRY_TASKS_FILENAME_PREFIX + "." + safe_phase + "." + str(int(iteration)).zfill(4) + ".txt")
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


def _bucket_dir(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    bucket = "initial" if str(phase_name).startswith("INITIAL_") else "iter_" + str(int(iteration))
    return Path(campaign_dir) / ".DATA" / "STAGING" / bucket


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


def _load_seed_records(campaign_dir: Union[str, Path], iteration: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from ..handoff_manifests import load_seeds_picked

    iter_dir = (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )
    picked = load_seeds_picked(iter_dir, expected_iteration=int(iteration))
    records = list(picked.get("seed_records") or [])
    if not records:
        frame_ids = list(picked.get("frame_ids") or [])
        records = [
            {"seed_index": int(i), "frame_id": frame_id}
            for i, frame_id in enumerate(frame_ids)
        ]
    return records, picked


def logical_task_ids(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> List[int]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        return []
    if phase == "ARIADNE_ARRAY":
        records, _picked = _load_seed_records(campaign_dir, int(iteration))
        ids: List[int] = []
        for record in records:
            try:
                ids.append(int(record.get("seed_index")))
            except (TypeError, ValueError):
                continue
        return sorted(set(ids))
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
    iter_dir = (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )
    seed_dir = iter_dir / "pool" / ("seed_" + str(int(task_id)).zfill(4))
    result_path = seed_dir / "result.json"
    if not result_path.is_file():
        return False, "missing_result_json", str(result_path.resolve(strict=False))
    try:
        from ..handoff_manifests import validate_ariadne_result

        records, picked = _load_seed_records(campaign_dir, int(iteration))
        by_seed = {
            int(record.get("seed_index")): dict(record)
            for record in records
            if record.get("seed_index") is not None
        }
        seed_record = by_seed.get(int(task_id))
        if seed_record is None:
            return False, "seed_record_missing", str(result_path.resolve(strict=False))
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_ariadne_result(
            payload,
            expected_iteration=int(iteration),
            seed_record=seed_record,
            expected_trajectory_sha256=str(picked.get("trajectory_sha256") or ""),
            accept_legacy_missing_landing_safety=True,
        )
        return True, "", str(result_path.resolve(strict=False))
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:160], str(result_path.resolve(strict=False))


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_root = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "array_task_archives"
        / (phase + "-" + str(int(iteration)).zfill(4) + "-" + stamp)
    )
    ids = [int(x) for x in (task_ids if task_ids is not None else logical_task_ids(campaign, phase, int(iteration)))]
    archived: List[str] = []
    for task_id in ids:
        task_archive = archive_root / ("task_" + str(int(task_id)).zfill(4))
        if phase == "ARIADNE_ARRAY":
            seed_dir = (
                campaign
                / "7_ACTIVE_LEARNING"
                / ("iteration-" + str(int(iteration)).zfill(4))
                / "pool"
                / ("seed_" + str(int(task_id)).zfill(4))
            )
            for candidate in [seed_dir / "result.json", seed_dir / "ARIADNE_TRACE.jsonl"]:
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved:
                    archived.append(moved)
            tmp_candidates = sorted(seed_dir.glob("result.json.*.tmp")) if seed_dir.is_dir() else []
            for candidate in tmp_candidates:
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
