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
TRAJECTORY_SCHEMA_VERSION = 2
SEED_RESULT_FILENAME = "result.json"
SEED_OUTPUT_MANIFEST_FILENAME = "ARIADNE_OUTPUT_MANIFEST.json"
SEED_OUTPUT_SCHEMA_VERSION = 2


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
    if not isinstance(campaign_uid, str):
        raise AriadneOutputError("ARIADNE campaign_uid must be a string")
    if not isinstance(seed_uid, str):
        raise AriadneOutputError("ARIADNE seed_uid must be a string")
    campaign = campaign_uid
    uid = seed_uid
    iteration_value = _exact_int(iteration, "ARIADNE iteration", minimum=1)
    seed_value = _exact_int(seed_id, "ARIADNE seed_id", minimum=1)
    task_value = _exact_int(array_task_id, "ARIADNE array_task_id")
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


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AriadneOutputError(label + " must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise AriadneOutputError(label + " must be >= " + str(minimum))
    return parsed


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
    if not isinstance(task_success, bool):
        raise AriadneOutputError("ARIADNE task_success must be a boolean")
    success = task_success
    exit_code = _exact_int(task_exit_code, "ARIADNE task_exit_code")
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
    trajectory_payload = read_and_validate_optimisation_trajectory(root)
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


def _validate_binding(root: Path, binding: Any, *, label: str, expected: str) -> Path:
    if not isinstance(binding, Mapping):
        raise AriadneOutputError(label + " binding must be an object")
    if set(binding) != {"path", "size", "sha256"}:
        raise AriadneOutputError(label + " binding fields are invalid")
    relative = str(binding.get("path") or "")
    if relative != expected:
        raise AriadneOutputError(label + " path is noncanonical")
    candidate = root / relative
    actual = _file_binding(root, candidate)
    if _exact_int(binding.get("size"), label + " size") != actual["size"]:
        raise AriadneOutputError(label + " size mismatch")
    digest = binding.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or digest != actual["sha256"]
    ):
        raise AriadneOutputError(label + " SHA-256 mismatch")
    return candidate


def _read_xyz_frames(path: Path, *, frame_count: int, atom_count: int) -> Sequence[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AriadneOutputError("ARIADNE trajectory XYZ is unreadable") from exc
    frames = []
    position = 0
    expected_atoms: Optional[Sequence[str]] = None
    while position < len(lines):
        if not lines[position].strip():
            raise AriadneOutputError("ARIADNE trajectory XYZ contains a blank frame boundary")
        try:
            declared = int(lines[position])
        except ValueError as exc:
            raise AriadneOutputError("ARIADNE trajectory XYZ atom count is invalid") from exc
        if declared != atom_count or position + 2 + declared > len(lines):
            raise AriadneOutputError("ARIADNE trajectory XYZ frame cardinality mismatch")
        atoms = []
        coordinates = []
        for row in lines[position + 2 : position + 2 + declared]:
            parts = row.split()
            if len(parts) != 4 or not parts[0]:
                raise AriadneOutputError("ARIADNE trajectory XYZ atom row is invalid")
            xyz = []
            for token in parts[1:]:
                try:
                    value = float(token)
                except ValueError as exc:
                    raise AriadneOutputError("ARIADNE trajectory coordinate is invalid") from exc
                if not math.isfinite(value):
                    raise AriadneOutputError("ARIADNE trajectory coordinate is non-finite")
                xyz.append(value)
            atoms.append(parts[0])
            coordinates.append(xyz)
        if expected_atoms is None:
            expected_atoms = tuple(atoms)
        elif tuple(atoms) != tuple(expected_atoms):
            raise AriadneOutputError("ARIADNE trajectory atom identity/order changed")
        frames.append({"atom_types": atoms, "coordinates": coordinates})
        position += 2 + declared
    if len(frames) != frame_count:
        raise AriadneOutputError("ARIADNE trajectory XYZ frame count mismatch")
    return frames


def _read_json_lines(path: Path, *, label: str) -> Sequence[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AriadneOutputError(label + " is unreadable") from exc
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise AriadneOutputError(label + " contains a blank record")
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise AriadneOutputError(
                label + " record " + str(line_number) + " is invalid"
            ) from exc
        if not isinstance(record, dict):
            raise AriadneOutputError(label + " record must be an object")
        records.append(record)
    return records


def read_and_validate_optimisation_trajectory(seed_dir: Path) -> Dict[str, Any]:
    """Read and semantically validate one seed's optimisation trajectory."""
    seed_root = Path(seed_dir)
    root = seed_root / TRAJECTORY_DIRNAME
    manifest_path = root / TRAJECTORY_MANIFEST_FILENAME
    payload = _read_json_object(manifest_path, "ARIADNE trajectory manifest")
    if _exact_int(payload.get("schema_version"), "trajectory schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise AriadneOutputError("unsupported ARIADNE trajectory schema")
    frame_count = _exact_int(payload.get("frame_count"), "trajectory frame_count", minimum=1)
    atom_count = _exact_int(payload.get("atom_count"), "trajectory atom_count", minimum=1)
    files = payload.get("files")
    if not isinstance(files, dict):
        raise AriadneOutputError("ARIADNE trajectory files must be an object")
    required = {
        "trajectory_xyz": TRAJECTORY_XYZ_FILENAME,
        "metrics_jsonl": TRAJECTORY_METRICS_FILENAME,
    }
    optional = {"trace_jsonl": TRAJECTORY_TRACE_FILENAME}
    if not set(required).issubset(files) or set(files) - set(required) - set(optional):
        raise AriadneOutputError("ARIADNE trajectory file set is invalid")
    paths = {
        label: _validate_binding(root, files[label], label=label, expected=relative)
        for label, relative in {**required, **optional}.items()
        if label in files
    }
    frames = _read_xyz_frames(
        paths["trajectory_xyz"], frame_count=frame_count, atom_count=atom_count
    )
    metrics = _read_json_lines(paths["metrics_jsonl"], label="ARIADNE trajectory metrics")
    if len(metrics) != frame_count:
        raise AriadneOutputError("ARIADNE trajectory metric count mismatch")
    for index, record in enumerate(metrics):
        if _exact_int(record.get("frame_number"), "trajectory frame_number", minimum=1) != index + 1:
            raise AriadneOutputError("ARIADNE trajectory frame numbers are not contiguous")
        if _exact_int(record.get("frame_index_zero_based"), "trajectory frame index") != index:
            raise AriadneOutputError("ARIADNE trajectory frame indexes are not contiguous")
        for key in ("alpha", "gradient_norm"):
            value = record.get(key)
            if value is not None and _finite_or_none(value) is None:
                raise AriadneOutputError("ARIADNE trajectory " + key + " is non-finite")
        if not isinstance(record.get("origin"), str) or not record["origin"].strip():
            raise AriadneOutputError("ARIADNE trajectory origin is empty")
    trace = []
    if "trace_jsonl" in paths:
        trace = list(_read_json_lines(paths["trace_jsonl"], label="ARIADNE trajectory trace"))
    normalised = dict(payload)
    normalised["frame_count"] = frame_count
    normalised["atom_count"] = atom_count
    normalised["frames"] = list(frames)
    normalised["metrics"] = list(metrics)
    normalised["trace"] = trace
    return normalised


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
    schema_version = _exact_int(
        payload.get("schema_version"),
        "ARIADNE seed-output schema_version",
    )
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
    task_exit_code = _exact_int(
        payload.get("task_exit_code"),
        "ARIADNE task_exit_code",
    )
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
        if actual["size"] != _exact_int(
            binding.get("size"),
            "ARIADNE output size for " + str(label),
        ):
            raise AriadneOutputError("ARIADNE output size mismatch: " + str(label))
        if actual["sha256"] != str(binding.get("sha256") or ""):
            raise AriadneOutputError("ARIADNE output hash mismatch: " + str(label))
    read_and_validate_optimisation_trajectory(root)
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
    "read_and_validate_optimisation_trajectory",
    "write_seed_output_manifest",
    "validate_seed_output",
]
