"""Atomic, hash-bound artefacts produced by one ARIADNE seed task."""

from __future__ import annotations

from .strict_json import strict_json as json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .daemon.state import atomic_write_json, atomic_write_text
from .versioning.manifest import sha256_file


TRAJECTORY_DIRNAME = "trajectory"
TRAJECTORY_XYZ_FILENAME = "trajectory.xyz"
TRAJECTORY_METRICS_FILENAME = "metrics.jsonl"
TRAJECTORY_TRACE_FILENAME = "trace.jsonl"
TRAJECTORY_MANIFEST_FILENAME = "MANIFEST.json"
TRAJECTORY_SCHEMA_VERSION = 1
SEED_RESULT_FILENAME = "result.json"
SEED_OUTPUT_MANIFEST_FILENAME = "ARIADNE_OUTPUT_MANIFEST.json"
SEED_OUTPUT_SCHEMA_VERSION = 1


class AriadneOutputError(RuntimeError):
    """Raised when a per-seed output set is incomplete or inconsistent."""


def _validate_identity(
    *,
    campaign_uid: Any,
    iteration: Any,
    seed_id: Any,
    seed_uid: Any,
    array_task_id: Any,
) -> Dict[str, Any]:
    campaign = str(campaign_uid or "")
    uid = str(seed_uid or "")
    if any(isinstance(value, bool) for value in (iteration, seed_id, array_task_id)):
        raise AriadneOutputError("ARIADNE seed identity contains a boolean")
    try:
        iteration_value = int(iteration)
        seed_value = int(seed_id)
        task_value = int(array_task_id)
    except (TypeError, ValueError) as exc:
        raise AriadneOutputError("ARIADNE seed identity contains a non-integer") from exc
    if not campaign:
        raise AriadneOutputError("ARIADNE campaign_uid is empty")
    if iteration_value < 1:
        raise AriadneOutputError("ARIADNE iteration must be >= 1")
    if seed_value < 1:
        raise AriadneOutputError("ARIADNE seed_id must be >= 1")
    if task_value != seed_value - 1:
        raise AriadneOutputError("ARIADNE array task/seed identity mismatch")
    if len(uid) != 64 or any(character not in "0123456789abcdef" for character in uid):
        raise AriadneOutputError("ARIADNE seed_uid is invalid")
    return {
        "campaign_uid": campaign,
        "iteration": iteration_value,
        "seed_id": seed_value,
        "seed_uid": uid,
        "array_task_id": task_value,
    }


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _file_binding(root: Path, path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AriadneOutputError("ARIADNE output is not a regular file: " + str(path))
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise AriadneOutputError("ARIADNE output escapes its manifest root") from exc
    return {
        "path": relative,
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _xyz_text(
    atom_types: Sequence[str],
    coordinate_frames: Sequence[Sequence[Sequence[float]]],
) -> str:
    lines = []
    for frame_index_zero_based, coordinates in enumerate(coordinate_frames):
        if len(coordinates) != len(atom_types):
            raise AriadneOutputError("ARIADNE trajectory atom count changed")
        lines.append(str(len(atom_types)))
        lines.append(
            "ARIADNE optimisation frame " + str(frame_index_zero_based + 1)
        )
        for atom_type, row in zip(atom_types, coordinates):
            if len(row) != 3:
                raise AriadneOutputError("ARIADNE trajectory coordinates must be N x 3")
            xyz = [_finite_or_none(value) for value in row]
            if any(value is None for value in xyz):
                raise AriadneOutputError("ARIADNE trajectory contains non-finite coordinates")
            lines.append(
                str(atom_type)
                + " "
                + " ".join(format(float(value), ".16g") for value in xyz)
            )
    return "\n".join(lines) + "\n"


def write_optimisation_trajectory(
    seed_dir: Path,
    *,
    atom_types: Sequence[str],
    coordinate_frames: Sequence[Sequence[Sequence[float]]],
    alpha_values: Sequence[Any],
    gradient_norms: Sequence[Any],
    origins: Sequence[Any] = (),
) -> Path:
    """Write a compact XYZ trajectory and one metric record per frame."""
    seed_root = Path(seed_dir)
    trajectory_dir = seed_root / TRAJECTORY_DIRNAME
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    frames = list(coordinate_frames)
    if not frames:
        raise AriadneOutputError("ARIADNE optimisation trajectory is empty")

    xyz_path = trajectory_dir / TRAJECTORY_XYZ_FILENAME
    metrics_path = trajectory_dir / TRAJECTORY_METRICS_FILENAME
    atomic_write_text(xyz_path, _xyz_text(atom_types, frames))

    metric_lines = []
    for frame_index_zero_based in range(len(frames)):
        metric = {
            "frame_number": int(frame_index_zero_based + 1),
            "frame_index_zero_based": int(frame_index_zero_based),
            "alpha": _finite_or_none(
                alpha_values[frame_index_zero_based]
                if frame_index_zero_based < len(alpha_values)
                else None
            ),
            "gradient_norm": _finite_or_none(
                gradient_norms[frame_index_zero_based]
                if frame_index_zero_based < len(gradient_norms)
                else None
            ),
            "origin": str(
                origins[frame_index_zero_based]
                if frame_index_zero_based < len(origins)
                else "optimiser_iterate"
            ),
        }
        metric_lines.append(
            json.dumps(
                metric,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    atomic_write_text(metrics_path, "\n".join(metric_lines) + "\n")

    files = {
        "trajectory_xyz": _file_binding(trajectory_dir, xyz_path),
        "metrics_jsonl": _file_binding(trajectory_dir, metrics_path),
    }
    trace_path = trajectory_dir / TRAJECTORY_TRACE_FILENAME
    if trace_path.is_file() and not trace_path.is_symlink():
        files["trace_jsonl"] = _file_binding(trajectory_dir, trace_path)
    manifest = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "frame_count": int(len(frames)),
        "atom_count": int(len(atom_types)),
        "files": files,
    }
    path = trajectory_dir / TRAJECTORY_MANIFEST_FILENAME
    atomic_write_json(path, manifest)
    return path


def write_seed_output_manifest(
    seed_dir: Path,
    *,
    campaign_uid: str,
    iteration: int,
    seed_id: int,
    seed_uid: str,
    array_task_id: int,
    task_success: bool,
    task_exit_code: int,
) -> Path:
    """Publish the completeness marker after every required task file exists."""
    identity = _validate_identity(
        campaign_uid=campaign_uid,
        iteration=iteration,
        seed_id=seed_id,
        seed_uid=seed_uid,
        array_task_id=array_task_id,
    )
    success = bool(task_success)
    exit_code = int(task_exit_code)
    if success != (exit_code == 0):
        raise AriadneOutputError(
            "ARIADNE task_success must agree with a zero task_exit_code"
        )
    root = Path(seed_dir)
    result_path = root / SEED_RESULT_FILENAME
    result_payload = _read_json_object(result_path, "ARIADNE result")
    trajectory_manifest = root / TRAJECTORY_DIRNAME / TRAJECTORY_MANIFEST_FILENAME
    files = {
        "result_json": _file_binding(root, result_path),
        "trajectory_manifest": _file_binding(root, trajectory_manifest),
    }
    trajectory_payload = _read_json_object(
        trajectory_manifest,
        "ARIADNE trajectory manifest",
    )
    for label, binding in dict(trajectory_payload.get("files") or {}).items():
        if not isinstance(binding, Mapping):
            raise AriadneOutputError("invalid trajectory file binding: " + str(label))
        rel = str(binding.get("path") or "")
        bound_path = trajectory_manifest.parent / rel
        files["trajectory/" + str(label)] = _file_binding(root, bound_path)
    payload = {
        "schema_version": SEED_OUTPUT_SCHEMA_VERSION,
        **identity,
        "task_success": success,
        "task_exit_code": exit_code,
        "files": files,
    }
    if not success:
        failure_reason = result_payload.get("task_success_reason")
        if failure_reason is None:
            failure_reason = result_payload.get("failure_reason")
        if failure_reason is not None and str(failure_reason).strip():
            payload["task_failure_reason"] = str(failure_reason).strip()
    path = root / SEED_OUTPUT_MANIFEST_FILENAME
    atomic_write_json(path, payload)
    return path


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AriadneOutputError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise AriadneOutputError(label + " must be a JSON object")
    return payload


def validate_seed_output(
    seed_dir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
    expected_iteration: Optional[int] = None,
    expected_seed_id: Optional[int] = None,
    expected_seed_uid: Optional[str] = None,
    expected_array_task_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate identity and hashes for an atomically published seed output."""
    root = Path(seed_dir)
    manifest_path = root / SEED_OUTPUT_MANIFEST_FILENAME
    payload = _read_json_object(manifest_path, "ARIADNE seed output manifest")
    try:
        schema_version = int(payload.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise AriadneOutputError("invalid ARIADNE seed-output schema") from exc
    if schema_version != SEED_OUTPUT_SCHEMA_VERSION:
        raise AriadneOutputError("unsupported ARIADNE seed-output schema")
    identity = _validate_identity(
        campaign_uid=payload.get("campaign_uid"),
        iteration=payload.get("iteration"),
        seed_id=payload.get("seed_id"),
        seed_uid=payload.get("seed_uid"),
        array_task_id=payload.get("array_task_id"),
    )
    task_success = payload.get("task_success")
    if not isinstance(task_success, bool):
        raise AriadneOutputError("ARIADNE task_success must be a boolean")
    task_exit_code_raw = payload.get("task_exit_code")
    if isinstance(task_exit_code_raw, bool):
        raise AriadneOutputError("ARIADNE task_exit_code must be an integer")
    try:
        task_exit_code = int(task_exit_code_raw)
    except (TypeError, ValueError) as exc:
        raise AriadneOutputError("ARIADNE task_exit_code must be an integer") from exc
    if task_exit_code < 0:
        raise AriadneOutputError("ARIADNE task_exit_code must be non-negative")
    if task_success != (task_exit_code == 0):
        raise AriadneOutputError(
            "ARIADNE task_success does not agree with task_exit_code"
        )
    task_failure_reason = payload.get("task_failure_reason")
    if task_failure_reason is not None and (
        task_success
        or not isinstance(task_failure_reason, str)
        or not task_failure_reason.strip()
    ):
        raise AriadneOutputError(
            "ARIADNE task_failure_reason is only valid for a failed task"
        )
    expected = {
        "campaign_uid": expected_campaign_uid,
        "iteration": expected_iteration,
        "seed_id": expected_seed_id,
        "seed_uid": expected_seed_uid,
        "array_task_id": expected_array_task_id,
    }
    for key, value in expected.items():
        if value is not None and payload.get(key) != value:
            raise AriadneOutputError("ARIADNE seed-output " + key + " mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise AriadneOutputError("ARIADNE seed-output files are missing")
    required_paths = {
        "result_json": SEED_RESULT_FILENAME,
        "trajectory_manifest": (
            TRAJECTORY_DIRNAME + "/" + TRAJECTORY_MANIFEST_FILENAME
        ),
        "trajectory/trajectory_xyz": (
            TRAJECTORY_DIRNAME + "/" + TRAJECTORY_XYZ_FILENAME
        ),
        "trajectory/metrics_jsonl": (
            TRAJECTORY_DIRNAME + "/" + TRAJECTORY_METRICS_FILENAME
        ),
    }
    optional_paths = {
        "trajectory/trace_jsonl": (
            TRAJECTORY_DIRNAME + "/" + TRAJECTORY_TRACE_FILENAME
        )
    }
    labels = set(files)
    if not set(required_paths).issubset(labels):
        raise AriadneOutputError("ARIADNE seed-output required files are missing")
    if labels - set(required_paths) - set(optional_paths):
        raise AriadneOutputError("ARIADNE seed-output contains unknown file bindings")
    for label, binding in files.items():
        if not isinstance(binding, Mapping):
            raise AriadneOutputError("invalid ARIADNE output binding: " + str(label))
        rel = str(binding.get("path") or "")
        expected_rel = required_paths.get(label, optional_paths.get(label))
        if rel != expected_rel:
            raise AriadneOutputError(
                "ARIADNE output path is noncanonical: " + str(label)
            )
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise AriadneOutputError("ARIADNE output escapes its seed directory") from exc
        actual = _file_binding(root, candidate)
        if actual["size"] != int(binding.get("size", -1)):
            raise AriadneOutputError("ARIADNE output size mismatch: " + str(label))
        if actual["sha256"] != str(binding.get("sha256") or ""):
            raise AriadneOutputError("ARIADNE output hash mismatch: " + str(label))
    payload["task_success"] = task_success
    payload["task_exit_code"] = task_exit_code
    if task_failure_reason is not None:
        payload["task_failure_reason"] = task_failure_reason.strip()
    payload.update(identity)
    return payload


__all__ = [
    "AriadneOutputError",
    "TRAJECTORY_DIRNAME",
    "TRAJECTORY_XYZ_FILENAME",
    "TRAJECTORY_METRICS_FILENAME",
    "TRAJECTORY_TRACE_FILENAME",
    "TRAJECTORY_MANIFEST_FILENAME",
    "SEED_RESULT_FILENAME",
    "SEED_OUTPUT_MANIFEST_FILENAME",
    "write_optimisation_trajectory",
    "write_seed_output_manifest",
    "validate_seed_output",
]
