"""Campaign-owned job scratch preparation, retention, and safe clean-up."""
from __future__ import annotations

import argparse
from ..strict_json import strict_json as json
import os
import stat
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .script_bundles import backend_name, campaign_owned_path
from .state import atomic_write_json


SCRATCH_TASK_SCHEMA_VERSION = 2
_SCRATCH_STATUSES = frozenset({"prepared", "failed_retained", "completed"})


def scratch_root(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / ".DATA" / "SCRATCH"


def _safe_token(value: Any, label: str) -> str:
    text = str(value)
    if not text or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for ch in text):
        raise ValueError(label + " is not a safe scratch token: " + repr(text))
    return text


def _check_campaign_relative_path(campaign_dir: Path, path: Path) -> Path:
    try:
        return campaign_owned_path(campaign_dir, path)
    except ValueError as exc:
        raise ValueError("unsafe campaign scratch path: " + str(path)) from exc


def _validate_resolution_ownership(
    campaign: Path,
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    attempt_id: str,
    submission_identity: str,
    resource_resolution_path: str,
    resource_resolution_sha256: str,
) -> Dict[str, Any]:
    from .resource_records import verify_resolution

    source = Path(resource_resolution_path)
    if not source.is_absolute():
        raise ValueError("scratch resource-resolution path must be absolute")
    source = campaign_owned_path(campaign, source)
    resolution_root = campaign_owned_path(
        campaign,
        campaign / ".DATA" / "ACTIVE_LEARNING" / "resource_resolutions",
    )
    try:
        source.relative_to(resolution_root)
    except ValueError as exc:
        raise ValueError(
            "scratch resource resolution is outside the resolution store"
        ) from exc
    payload = verify_resolution(source, str(resource_resolution_sha256))
    expected = {
        "campaign_uid": str(campaign_uid),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "attempt_id": str(attempt_id),
        "submission_identity": str(submission_identity),
    }
    observed = {
        "campaign_uid": str(payload.get("campaign_uid") or ""),
        "phase": str(payload.get("phase") or ""),
        "iteration": int(payload.get("iteration", -1)),
        "attempt_id": str(payload.get("attempt_id") or ""),
        "submission_identity": str(payload.get("submission_identity") or ""),
    }
    if observed != expected:
        raise ValueError(
            "scratch ownership does not match the immutable resource resolution"
        )
    return payload


def attempt_scratch_root(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    submission_identity: str,
) -> Path:
    campaign = Path(campaign_dir)
    backend = backend_name(phase_name)
    path = (
        scratch_root(campaign)
        / backend
        / _safe_token(phase_name, "phase")
        / ("iteration-" + str(int(iteration)).zfill(6))
        / _safe_token(submission_identity, "submission identity")
    )
    return _check_campaign_relative_path(campaign, path)


def scratch_path_template(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    submission_identity: str,
) -> str:
    return str(
        attempt_scratch_root(
            campaign_dir, phase_name, iteration, submission_identity
        )
        / "job-${SLURM_JOB_ID}"
        / "task-${SLURM_ARRAY_TASK_ID:-0}"
    )


def prepare_task_scratch(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    attempt_id: str,
    submission_identity: str,
    job_id: str,
    array_task_id: int,
    resource_resolution_path: str,
    resource_resolution_sha256: str,
) -> Path:
    campaign = Path(campaign_dir)
    _validate_resolution_ownership(
        campaign,
        campaign_uid=str(campaign_uid),
        phase_name=str(phase_name),
        iteration=int(iteration),
        attempt_id=str(attempt_id),
        submission_identity=str(submission_identity),
        resource_resolution_path=str(resource_resolution_path),
        resource_resolution_sha256=str(resource_resolution_sha256),
    )
    attempt = attempt_scratch_root(
        campaign, phase_name, int(iteration), submission_identity
    )
    leaf = attempt / (
        "job-" + _safe_token(job_id, "job ID")
    ) / ("task-" + str(int(array_task_id)))
    leaf = _check_campaign_relative_path(campaign, leaf)
    if leaf.is_symlink():
        raise ValueError("scratch task leaf must not be a symlink: " + str(leaf))
    leaf.mkdir(parents=True, exist_ok=True, mode=0o700)
    leaf.chmod(0o700)
    metadata = leaf.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or leaf.is_symlink():
        raise ValueError("scratch task leaf is not a regular directory: " + str(leaf))
    if os.name != "nt":
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise PermissionError(
                "scratch task leaf permissions are not private (0700): " + str(leaf)
            )
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise PermissionError(
                "scratch task leaf is not owned by the current user: " + str(leaf)
            )
    atomic_write_json(
        leaf / "TASK.json",
        {
            "schema_version": SCRATCH_TASK_SCHEMA_VERSION,
            "created_at_iso": datetime.now(timezone.utc).isoformat(),
            "campaign_dir": str(campaign.resolve()),
            "campaign_uid": str(campaign_uid),
            "phase": str(phase_name),
            "iteration": int(iteration),
            "attempt_id": str(attempt_id),
            "submission_identity": str(submission_identity),
            "job_id": str(job_id),
            "array_task_id": int(array_task_id),
            "resource_resolution_path": str(resource_resolution_path),
            "resource_resolution_sha256": str(resource_resolution_sha256),
            "status": "prepared",
        },
    )
    return leaf


def _prune_empty(path: Path, stop: Path) -> None:
    current = path
    stop_resolved = stop.resolve(strict=False)
    while current.resolve(strict=False) != stop_resolved:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def finish_task_scratch(path: Union[str, Path], *, success: bool) -> None:
    leaf = Path(path)
    task_path = leaf / "TASK.json"
    if task_path.is_symlink() or not task_path.is_file():
        raise ValueError("scratch TASK.json is missing: " + str(task_path))
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("scratch TASK.json is unreadable: " + str(task_path)) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCRATCH_TASK_SCHEMA_VERSION
        or isinstance(payload.get("schema_version"), bool)
    ):
        raise ValueError("scratch TASK.json has an unsupported schema")
    campaign_value = payload.get("campaign_dir")
    if not isinstance(campaign_value, str) or not campaign_value:
        raise ValueError("scratch TASK.json has no campaign directory ownership")
    campaign = Path(campaign_value)
    root = _check_campaign_relative_path(campaign, scratch_root(campaign))
    leaf = _check_campaign_relative_path(campaign, leaf)
    try:
        leaf.relative_to(root)
    except ValueError as exc:
        raise ValueError("scratch task leaf is outside .DATA/SCRATCH") from exc
    ownership = _task_record(campaign, root, leaf / "TASK.json")
    if ownership.get("status") == "invalid":
        raise ValueError(
            "scratch TASK.json ownership is invalid: "
            + str(ownership.get("reason"))
        )
    payload["finished_at_iso"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "completed" if success else "failed_retained"
    atomic_write_json(task_path, payload)
    if not success:
        return
    shutil.rmtree(leaf)
    _prune_empty(leaf.parent, root)


def _invalid(path: Path, reason: str) -> Dict[str, Any]:
    return {"status": "invalid", "path": str(path), "reason": str(reason)}


def _task_record(campaign: Path, root: Path, task_path: Path) -> Dict[str, Any]:
    leaf = task_path.parent
    try:
        relative = leaf.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return _invalid(leaf, "scratch task escapes the scratch root")
    parts = relative.parts
    if len(parts) != 6:
        return _invalid(leaf, "scratch task hierarchy is malformed")
    backend, phase, iteration_token, identity, job_token, task_token = parts
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _invalid(task_path, "TASK.json is unreadable")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCRATCH_TASK_SCHEMA_VERSION
        or isinstance(payload.get("schema_version"), bool)
    ):
        return _invalid(task_path, "TASK.json schema is invalid")
    required_text = (
        "campaign_dir",
        "campaign_uid",
        "phase",
        "attempt_id",
        "submission_identity",
        "job_id",
        "resource_resolution_path",
        "resource_resolution_sha256",
    )
    if any(not isinstance(payload.get(key), str) or not payload.get(key) for key in required_text):
        return _invalid(task_path, "TASK.json ownership metadata is incomplete")
    if any(
        isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int)
        for key in ("iteration", "array_task_id")
    ):
        return _invalid(task_path, "TASK.json numeric ownership metadata is malformed")
    parsed_iteration = int(payload["iteration"])
    parsed_task = int(payload["array_task_id"])
    if parsed_iteration < 0 or parsed_task < 0:
        return _invalid(task_path, "TASK.json numeric ownership metadata is negative")
    status = payload.get("status")
    if status not in _SCRATCH_STATUSES:
        return _invalid(task_path, "TASK.json status is invalid")
    for timestamp_key in ("created_at_iso", "finished_at_iso"):
        value = payload.get(timestamp_key)
        if timestamp_key == "finished_at_iso" and status == "prepared":
            if value is not None:
                return _invalid(task_path, "prepared TASK.json has a finished timestamp")
            continue
        if timestamp_key == "finished_at_iso" and status != "prepared" and value is None:
            return _invalid(task_path, "finished TASK.json lacks a finished timestamp")
        if value is None:
            return _invalid(task_path, "TASK.json lacks " + timestamp_key)
        try:
            parsed_timestamp = datetime.fromisoformat(str(value))
        except ValueError:
            return _invalid(task_path, "TASK.json timestamp is malformed")
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            return _invalid(task_path, "TASK.json timestamp lacks a timezone")
    try:
        _safe_token(payload["campaign_uid"], "TASK.json campaign UID")
    except ValueError:
        return _invalid(task_path, "TASK.json campaign UID is invalid")
    try:
        expected_backend = backend_name(str(payload["phase"]))
    except ValueError as exc:
        return _invalid(task_path, str(exc))
    expected = {
        "backend": expected_backend,
        "phase": str(payload["phase"]),
        "iteration": "iteration-" + str(parsed_iteration).zfill(6),
        "identity": str(payload["submission_identity"]),
        "job": "job-" + str(payload["job_id"]),
        "task": "task-" + str(parsed_task),
    }
    observed = {
        "backend": backend,
        "phase": phase,
        "iteration": iteration_token,
        "identity": identity,
        "job": job_token,
        "task": task_token,
    }
    if observed != expected:
        return _invalid(task_path, "TASK.json ownership metadata does not match its path")
    try:
        configured_campaign = Path(str(payload["campaign_dir"])).resolve(strict=False)
    except (OSError, RuntimeError):
        return _invalid(task_path, "TASK.json campaign directory is malformed")
    if configured_campaign != campaign.resolve(strict=False):
        return _invalid(
            task_path,
            "TASK.json campaign directory does not match the inventoried campaign",
        )
    digest = str(payload.get("resource_resolution_sha256"))
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return _invalid(task_path, "TASK.json resource-resolution digest is invalid")
    try:
        _validate_resolution_ownership(
            campaign,
            campaign_uid=str(payload["campaign_uid"]),
            phase_name=str(payload["phase"]),
            iteration=parsed_iteration,
            attempt_id=str(payload["attempt_id"]),
            submission_identity=str(payload["submission_identity"]),
            resource_resolution_path=str(payload["resource_resolution_path"]),
            resource_resolution_sha256=digest,
        )
    except Exception as exc:
        return _invalid(
            task_path,
            "TASK.json resource-resolution ownership is invalid: " + str(exc),
        )
    record = dict(payload)
    record["path"] = str(leaf)
    record["attempt_path"] = str(leaf.parent.parent)
    return record


def inventory(campaign_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    campaign = Path(campaign_dir)
    root = scratch_root(campaign)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        return [_invalid(root, "scratch root is not a regular directory")]
    records: List[Dict[str, Any]] = []
    task_directories: List[Path] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current)
        relative = directory.relative_to(root)
        depth = len(relative.parts)
        safe_directories: List[str] = []
        for name in directory_names:
            child = directory / name
            if child.is_symlink():
                records.append(_invalid(child, "scratch hierarchy contains a symlink"))
            else:
                safe_directories.append(name)
                child_depth = depth + 1
                valid_component = True
                if child_depth == 1:
                    valid_component = name in {"POLUS", "GAUSSIAN", "AIMALL", "ARIADNE", "FEREBUS"}
                elif child_depth == 2:
                    valid_component = bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", name))
                elif child_depth == 3:
                    valid_component = bool(re.fullmatch(r"iteration-[0-9]{6}", name))
                elif child_depth == 4:
                    valid_component = bool(re.fullmatch(r"[A-Za-z0-9_.-]+", name))
                elif child_depth == 5:
                    valid_component = bool(re.fullmatch(r"job-[A-Za-z0-9_.-]+", name))
                elif child_depth == 6:
                    valid_component = bool(re.fullmatch(r"task-[0-9]+", name))
                if child_depth <= 6 and not valid_component:
                    records.append(_invalid(child, "scratch hierarchy component is malformed"))
                    safe_directories.remove(name)
                    continue
                if child_depth == 6:
                    task_directories.append(child)
        directory_names[:] = safe_directories
        for name in file_names:
            child = directory / name
            if child.is_symlink():
                records.append(_invalid(child, "scratch hierarchy contains a symlink"))
            elif depth < 6:
                records.append(_invalid(child, "scratch file exists outside a task directory"))
    for leaf in sorted(task_directories):
        task = leaf / "TASK.json"
        if task.is_symlink() or not task.is_file():
            records.append(_invalid(leaf, "scratch task directory has no regular TASK.json"))
            continue
        records.append(_task_record(campaign, root, task))
    return records


def clean_inactive_attempts(
    campaign_dir: Union[str, Path],
    *,
    active_job_ids: Iterable[str],
    allowed_attempts: Optional[Iterable[str]] = None,
) -> List[str]:
    campaign = Path(campaign_dir)
    root = scratch_root(campaign)
    active = {str(value) for value in active_job_ids}
    selected = None if allowed_attempts is None else {str(value) for value in allowed_attempts}
    removed: List[str] = []
    attempts: Dict[str, Dict[str, Any]] = {}
    for record in inventory(campaign):
        if record.get("status") == "invalid":
            raise ValueError("invalid scratch evidence blocks clean-up: " + str(record.get("path")))
        identity = str(record.get("submission_identity") or "")
        attempt_id = str(record.get("attempt_id") or "")
        if selected is not None and identity not in selected and attempt_id not in selected:
            continue
        attempt_path = Path(str(record.get("attempt_path") or ""))
        key = str(attempt_path)
        item = attempts.setdefault(
            key,
            {"path": attempt_path, "job_ids": set(), "identity": identity},
        )
        item["job_ids"].add(str(record.get("job_id") or ""))
    for item in attempts.values():
        if item["job_ids"] & active:
            continue
        path = _check_campaign_relative_path(campaign, Path(item["path"]))
        try:
            path.relative_to(root.resolve(strict=False))
        except ValueError as exc:
            raise ValueError("scratch clean-up target escapes scratch root: " + str(path)) from exc
        if path.is_symlink() or not path.is_dir():
            raise ValueError("scratch clean-up target is not a regular directory: " + str(path))
        shutil.rmtree(path)
        removed.append(str(path))
        _prune_empty(path.parent, root)
    return removed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ichor-job-scratch")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--campaign-dir", required=True)
    prepare.add_argument("--campaign-uid", required=True)
    prepare.add_argument("--phase", required=True)
    prepare.add_argument("--iteration", type=int, required=True)
    prepare.add_argument("--attempt-id", required=True)
    prepare.add_argument("--submission-identity", required=True)
    prepare.add_argument("--job-id", required=True)
    prepare.add_argument("--array-task-id", type=int, required=True)
    prepare.add_argument("--resource-resolution", required=True)
    prepare.add_argument("--resource-resolution-sha256", required=True)
    finish = sub.add_parser("finish")
    finish.add_argument("--path", required=True)
    finish.add_argument("--success", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_task_scratch(
            args.campaign_dir,
            campaign_uid=args.campaign_uid,
            phase_name=args.phase,
            iteration=args.iteration,
            attempt_id=args.attempt_id,
            submission_identity=args.submission_identity,
            job_id=args.job_id,
            array_task_id=args.array_task_id,
            resource_resolution_path=args.resource_resolution,
            resource_resolution_sha256=args.resource_resolution_sha256,
        )
        print(str(path))
        return 0
    finish_task_scratch(args.path, success=bool(args.success))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
