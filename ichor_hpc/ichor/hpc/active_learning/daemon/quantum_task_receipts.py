"""Content-bound completion receipts for reusable quantum array tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..strict_json import load_path
from ..versioning.manifest import sha256_file
from .state import atomic_write_json
from .submission_intent import load_intent


QUANTUM_TASK_RECEIPT_SCHEMA_VERSION = 1
GAUSSIAN_TASK_RECEIPT = "GAUSSIAN_TASK_RECEIPT.json"
AIMALL_TASK_RECEIPT = "AIMALL_COMPLETION_RECEIPT.json"


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return int(value)


def _receipt_name(phase_name: str) -> str:
    if "GAUSSIAN" in str(phase_name):
        return GAUSSIAN_TASK_RECEIPT
    if "AIMALL" in str(phase_name):
        return AIMALL_TASK_RECEIPT
    raise ValueError("quantum task receipt phase is not Gaussian or AIMAll")


def _binding(root: Path, path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("quantum task artefact is missing or symlinked: " + str(path))
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError("quantum task artefact escapes its pointdir") from exc
    return {
        "path": relative,
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _regular_matches(root: Path, patterns: Iterable[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    unique = sorted({path.resolve(): path for path in paths}.values(), key=lambda p: p.relative_to(root).as_posix())
    if any(path.is_symlink() or not path.is_file() for path in unique):
        raise ValueError("quantum task artefact set contains a non-regular file")
    return unique


def _task_files(pointdir: Path, phase_name: str) -> tuple[List[Path], List[Path]]:
    if "GAUSSIAN" in phase_name:
        inputs = _regular_matches(pointdir, ("*.gjf",))
        logs = _regular_matches(pointdir, ("*.gau", "*.gaussianoutput"))
        wfns = _regular_matches(pointdir, ("*.wfn",))
        if len(inputs) != 1 or len(logs) != 1 or len(wfns) != 1:
            raise ValueError("Gaussian task requires one GJF, one output and one WFN")
        return inputs, logs + wfns
    inputs = _regular_matches(
        pointdir,
        ("*.wfn", "AIMALL_TASK.json", "WFN_METHOD_RECEIPT.json"),
    )
    ints = _regular_matches(pointdir, ("**/*.int",))
    input_names = {path.name for path in inputs}
    if (
        len([path for path in inputs if path.suffix.lower() == ".wfn"]) != 1
        or "AIMALL_TASK.json" not in input_names
        or "WFN_METHOD_RECEIPT.json" not in input_names
        or not ints
    ):
        raise ValueError("AIMAll task inputs or INT outputs are incomplete")
    return inputs, ints


def write_quantum_task_receipt(
    campaign_dir: Path,
    pointdir: Path,
    *,
    phase_name: str,
    iteration: int,
    logical_task_id: int,
) -> Path:
    """Publish a receipt only after a task's parser contract has succeeded."""
    iteration_value = _exact_int(iteration, "quantum task receipt iteration")
    logical_task_value = _exact_int(
        logical_task_id,
        "quantum task receipt logical_task_id",
    )
    root = Path(pointdir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("quantum task pointdir is missing or symlinked")
    intent = load_intent(campaign_dir, phase_name, iteration_value)
    if not isinstance(intent, dict) or str(intent.get("status")) not in {
        "SUBMITTED",
        "ADOPTED",
    }:
        raise ValueError("quantum task receipt requires an active submitted intent")
    inputs, outputs = _task_files(root, phase_name)
    payload = {
        "schema_version": QUANTUM_TASK_RECEIPT_SCHEMA_VERSION,
        "campaign_uid": str(intent["campaign_uid"]),
        "phase": str(phase_name),
        "iteration": iteration_value,
        "logical_task_id": logical_task_value,
        "attempt_id": str(intent["attempt_id"]),
        "submission_identity": str(intent["submission_identity"]),
        "job_id": str(intent["job_id"]),
        "pointdir": root.name,
        "inputs": [_binding(root, path) for path in inputs],
        "outputs": [_binding(root, path) for path in outputs],
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    target = root / _receipt_name(phase_name)
    atomic_write_json(target, payload)
    return target


def read_quantum_task_receipt(
    pointdir: Path,
    *,
    phase_name: str,
    iteration: int,
    logical_task_id: int,
) -> Dict[str, Any]:
    root = Path(pointdir)
    target = root / _receipt_name(phase_name)
    try:
        payload = load_path(target)
    except (OSError, ValueError) as exc:
        raise ValueError("quantum task receipt is unreadable: " + str(target)) from exc
    if not isinstance(payload, dict):
        raise ValueError("quantum task receipt must be a JSON object")
    if _exact_int(payload.get("schema_version"), "quantum task receipt schema") != QUANTUM_TASK_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported quantum task receipt schema")
    if payload.get("phase") != str(phase_name):
        raise ValueError("quantum task receipt phase mismatch")
    if _exact_int(payload.get("iteration"), "quantum task receipt iteration") != int(iteration):
        raise ValueError("quantum task receipt iteration mismatch")
    if _exact_int(payload.get("logical_task_id"), "quantum task logical_task_id") != int(logical_task_id):
        raise ValueError("quantum task receipt logical identity mismatch")
    if payload.get("pointdir") != root.name:
        raise ValueError("quantum task receipt pointdir mismatch")
    for key in ("campaign_uid", "attempt_id", "submission_identity", "job_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError("quantum task receipt " + key + " is empty")
    current_inputs, current_outputs = _task_files(root, str(phase_name))
    current_paths = {
        "inputs": {path.relative_to(root).as_posix() for path in current_inputs},
        "outputs": {path.relative_to(root).as_posix() for path in current_outputs},
    }
    for group in ("inputs", "outputs"):
        records = payload.get(group)
        if not isinstance(records, list) or not records:
            raise ValueError("quantum task receipt " + group + " are missing")
        seen = set()
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
                raise ValueError("quantum task receipt binding is invalid")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative or relative in seen:
                raise ValueError("quantum task receipt binding path is invalid")
            seen.add(relative)
            path = root / relative
            actual = _binding(root, path)
            if actual != dict(record):
                raise ValueError("quantum task receipt artefact binding mismatch")
        if seen != current_paths[group]:
            raise ValueError(
                "quantum task receipt does not bind the complete " + group + " set"
            )
    return payload


__all__ = [
    "AIMALL_TASK_RECEIPT",
    "GAUSSIAN_TASK_RECEIPT",
    "QUANTUM_TASK_RECEIPT_SCHEMA_VERSION",
    "read_quantum_task_receipt",
    "write_quantum_task_receipt",
]
