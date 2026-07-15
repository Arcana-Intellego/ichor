"""One-based scientific seed identities and explicit scheduler task maps."""

from __future__ import annotations

import hashlib
from .strict_json import strict_json as json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .daemon.state import atomic_write_json
from .handoff_manifests import (
    ARIADNE_TASK_MAP_SCHEMA_VERSION,
    ariadne_task_map_path,
    seeds_picked_path,
)
from .layout import ariadne_seed_dir
from .versioning.manifest import sha256_file


class SeedIdentityError(ValueError):
    """Raised when a seed or scheduler mapping is ambiguous or unsafe."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise SeedIdentityError(label + " must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SeedIdentityError(label + " must be an integer") from exc
    try:
        if float(value) != float(parsed):
            raise SeedIdentityError(label + " must be an integer")
    except (TypeError, ValueError) as exc:
        raise SeedIdentityError(label + " must be an integer") from exc
    if parsed < minimum:
        raise SeedIdentityError(label + " must be >= " + str(minimum))
    return parsed


def selection_fingerprint_payload(selection: Mapping[str, Any]) -> Dict[str, Any]:
    records = selection.get("seed_records")
    if not isinstance(records, list):
        raise SeedIdentityError("seed selection records must be a list")
    return {
        "campaign_uid": str(selection.get("campaign_uid") or ""),
        "iteration": _integer(selection.get("iteration"), "iteration", minimum=1),
        "models_version": _integer(
            selection.get("models_version"),
            "models_version",
            minimum=0,
        ),
        "model_set_sha256": str(
            selection.get("model_set_sha256") or ""
        ),
        "trajectory_sha256": str(selection.get("trajectory_sha256") or ""),
        "records": [
            {
                "seed_id": _integer(record.get("seed_id"), "seed_id", minimum=1),
                "frame_id": record.get("frame_id"),
                "pool_row_index_zero_based": _integer(
                    record.get("pool_row_index_zero_based"),
                    "pool_row_index_zero_based",
                    minimum=0,
                ),
                "selection_origin": str(record.get("selection_origin") or ""),
            }
            for record in records
            if isinstance(record, Mapping)
        ],
    }


def selection_fingerprint_sha256(selection: Mapping[str, Any]) -> str:
    return _canonical_sha256(selection_fingerprint_payload(selection))


def deterministic_seed_uid(
    *,
    campaign_uid: str,
    iteration: int,
    seed_id: int,
    frame_id: Optional[int],
    models_version: int,
    model_set_sha256: str,
    selection_fingerprint_sha256_value: str,
) -> str:
    payload = {
        "campaign_uid": str(campaign_uid),
        "iteration": _integer(iteration, "iteration", minimum=1),
        "seed_id": _integer(seed_id, "seed_id", minimum=1),
        "frame_id": None if frame_id is None else _integer(
            frame_id,
            "frame_id",
            minimum=0,
        ),
        "models_version": _integer(models_version, "models_version", minimum=0),
        "model_set_sha256": str(model_set_sha256),
        "selection_fingerprint_sha256": str(
            selection_fingerprint_sha256_value
        ),
    }
    return _canonical_sha256(payload)


def write_ariadne_task_map(
    iter_dir: Path,
    selection: Mapping[str, Any],
) -> Path:
    selection_path = seeds_picked_path(iter_dir)
    if not selection_path.is_file() or selection_path.is_symlink():
        raise SeedIdentityError("seed selection manifest is not a regular file")
    campaign_uid = str(selection.get("campaign_uid") or "")
    if not campaign_uid:
        raise SeedIdentityError("seed selection campaign_uid is empty")
    iteration = _integer(selection.get("iteration"), "iteration", minimum=1)
    models_version = _integer(
        selection.get("models_version"),
        "models_version",
        minimum=0,
    )
    model_sha = str(selection.get("model_manifest_sha256") or "")
    model_set_sha = str(selection.get("model_set_sha256") or "")
    for label, value in (
        ("model manifest", model_sha),
        ("scientific model set", model_set_sha),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise SeedIdentityError(label + " SHA-256 is invalid")
    fingerprint = selection_fingerprint_sha256(selection)
    records = list(selection.get("seed_records") or [])
    tasks = []
    for array_task_id, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SeedIdentityError("seed selection record is invalid")
        seed_id = _integer(record.get("seed_id"), "seed_id", minimum=1)
        if seed_id != array_task_id + 1:
            raise SeedIdentityError("seed IDs must be contiguous from one")
        frame_id = record.get("frame_id")
        frame_id = None if frame_id is None else _integer(
            frame_id,
            "frame_id",
            minimum=0,
        )
        seed_uid = deterministic_seed_uid(
            campaign_uid=campaign_uid,
            iteration=iteration,
            seed_id=seed_id,
            frame_id=frame_id,
            models_version=models_version,
            model_set_sha256=model_set_sha,
            selection_fingerprint_sha256_value=fingerprint,
        )
        tasks.append(
            {
                "array_task_id": int(array_task_id),
                "seed_id": int(seed_id),
                "seed_uid": seed_uid,
                "frame_id": frame_id,
                "pool_row_index_zero_based": _integer(
                    record.get("pool_row_index_zero_based"),
                    "pool_row_index_zero_based",
                    minimum=0,
                ),
                "seed_directory": ariadne_seed_dir(
                    iter_dir,
                    seed_id,
                ).relative_to(iter_dir).as_posix(),
            }
        )
    payload = {
        "schema_version": ARIADNE_TASK_MAP_SCHEMA_VERSION,
        "campaign_uid": campaign_uid,
        "iteration": int(iteration),
        "models_version": int(models_version),
        "model_manifest_sha256": model_sha,
        "model_set_sha256": model_set_sha,
        "trajectory_sha256": str(selection.get("trajectory_sha256") or ""),
        "selection_fingerprint_sha256": fingerprint,
        "selection_manifest": {
            "path": selection_path.relative_to(iter_dir).as_posix(),
            "size": int(selection_path.stat().st_size),
            "sha256": sha256_file(selection_path),
        },
        "n_tasks": int(len(tasks)),
        "tasks": tasks,
    }
    path = ariadne_task_map_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def read_ariadne_task_map(
    iter_dir: Path,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = ariadne_task_map_path(iter_dir)
    if path.is_symlink() or not path.is_file():
        raise SeedIdentityError("ARIADNE task map is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SeedIdentityError("ARIADNE task map is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise SeedIdentityError("ARIADNE task map must be an object")
    if _integer(payload.get("schema_version"), "schema_version", minimum=1) != ARIADNE_TASK_MAP_SCHEMA_VERSION:
        raise SeedIdentityError("unsupported ARIADNE task-map schema")
    iteration = _integer(payload.get("iteration"), "iteration", minimum=1)
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise SeedIdentityError("ARIADNE task-map iteration mismatch")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise SeedIdentityError("ARIADNE task-map tasks must be a list")
    if _integer(payload.get("n_tasks"), "n_tasks", minimum=0) != len(tasks):
        raise SeedIdentityError("ARIADNE task-map task count mismatch")
    selection_path = seeds_picked_path(iter_dir)
    binding = payload.get("selection_manifest")
    if not isinstance(binding, Mapping):
        raise SeedIdentityError("ARIADNE task-map selection binding is missing")
    if str(binding.get("path") or "") != selection_path.relative_to(iter_dir).as_posix():
        raise SeedIdentityError("ARIADNE task-map selection path mismatch")
    if selection_path.is_symlink() or not selection_path.is_file():
        raise SeedIdentityError("ARIADNE task-map selection is not a regular file")
    if _integer(binding.get("size"), "selection size", minimum=0) != int(
        selection_path.stat().st_size
    ):
        raise SeedIdentityError("ARIADNE task-map selection size mismatch")
    if str(binding.get("sha256") or "") != sha256_file(selection_path):
        raise SeedIdentityError("ARIADNE task-map selection SHA-256 mismatch")
    from .handoff_manifests import load_seeds_picked

    selection = load_seeds_picked(
        iter_dir,
        expected_iteration=iteration,
    )
    fingerprint = selection_fingerprint_sha256(selection)
    if str(payload.get("selection_fingerprint_sha256") or "") != fingerprint:
        raise SeedIdentityError("ARIADNE task-map selection fingerprint mismatch")
    for field in (
        "campaign_uid",
        "models_version",
        "model_manifest_sha256",
        "model_set_sha256",
        "trajectory_sha256",
    ):
        if str(payload.get(field)) != str(selection.get(field)):
            raise SeedIdentityError("ARIADNE task-map " + field + " mismatch")
    selection_records = list(selection.get("seed_records") or [])
    if len(selection_records) != len(tasks):
        raise SeedIdentityError("ARIADNE task-map selection count mismatch")
    for array_task_id, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise SeedIdentityError("ARIADNE task-map record is invalid")
        if _integer(task.get("array_task_id"), "array_task_id", minimum=0) != array_task_id:
            raise SeedIdentityError("ARIADNE array task IDs are not contiguous")
        seed_id = _integer(task.get("seed_id"), "seed_id", minimum=1)
        if seed_id != array_task_id + 1:
            raise SeedIdentityError("ARIADNE seed IDs are not contiguous from one")
        expected_dir = ariadne_seed_dir(iter_dir, seed_id).relative_to(iter_dir).as_posix()
        if str(task.get("seed_directory") or "") != expected_dir:
            raise SeedIdentityError("ARIADNE seed directory mapping is invalid")
        selection_record = selection_records[array_task_id]
        if int(selection_record["seed_id"]) != seed_id:
            raise SeedIdentityError("ARIADNE task-map seed selection mismatch")
        if task.get("frame_id") != selection_record.get("frame_id"):
            raise SeedIdentityError("ARIADNE task-map frame mismatch")
        expected_uid = deterministic_seed_uid(
            campaign_uid=str(payload["campaign_uid"]),
            iteration=iteration,
            seed_id=seed_id,
            frame_id=task.get("frame_id"),
            models_version=int(payload["models_version"]),
            model_set_sha256=str(payload["model_set_sha256"]),
            selection_fingerprint_sha256_value=fingerprint,
        )
        if str(task.get("seed_uid") or "") != expected_uid:
            raise SeedIdentityError("ARIADNE task-map seed UID mismatch")
    return payload


def task_for_array_task_id(
    task_map: Mapping[str, Any],
    array_task_id: int,
) -> Dict[str, Any]:
    task_id = _integer(array_task_id, "array_task_id", minimum=0)
    tasks = list(task_map.get("tasks") or [])
    if task_id >= len(tasks):
        raise SeedIdentityError("array task ID is outside the ARIADNE task map")
    task = tasks[task_id]
    if not isinstance(task, dict):
        raise SeedIdentityError("ARIADNE task-map record is invalid")
    return dict(task)


__all__ = [
    "SeedIdentityError",
    "selection_fingerprint_payload",
    "selection_fingerprint_sha256",
    "deterministic_seed_uid",
    "write_ariadne_task_map",
    "read_ariadne_task_map",
    "task_for_array_task_id",
]
