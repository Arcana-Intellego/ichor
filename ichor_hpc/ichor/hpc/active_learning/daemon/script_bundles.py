"""Immutable daemon submission-attempt script and log bundles."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from .state import atomic_write_json, atomic_write_text


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class AttemptBundle:
    root: Path
    script: Path
    outputs: Path
    errors: Path
    array_task_map: Optional[Path] = None


def safe_component(value: Any, label: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(label + " is not a safe path component: " + repr(text))
    return text


def campaign_owned_path(
    campaign_dir: Union[str, Path],
    path: Union[str, Path],
) -> Path:
    """Validate campaign containment without accepting symlinked components."""
    campaign_path = Path(
        os.path.abspath(os.fspath(Path(campaign_dir).expanduser()))
    )
    raw_candidate = Path(path).expanduser()
    if not raw_candidate.is_absolute():
        raw_candidate = campaign_path / raw_candidate
    candidate_path = Path(os.path.abspath(os.fspath(raw_candidate)))

    try:
        candidate_path.relative_to(campaign_path)
    except ValueError as exc:
        raise ValueError("daemon-owned path escapes campaign root: " + str(path)) from exc

    # Inspect the lexical path before resolve() follows anything. Checking only
    # the resolved path would miss a symlink that points back inside the
    # campaign, which is still unsafe for daemon-owned mutation.
    chain = []
    current = candidate_path
    while True:
        chain.append(current)
        if current == current.parent:
            break
        current = current.parent
    for component in reversed(chain):
        if component.is_symlink():
            raise ValueError(
                "daemon-owned path contains a symlink: " + str(component)
            )

    campaign = campaign_path.resolve(strict=False)
    candidate = candidate_path.resolve(strict=False)
    try:
        candidate.relative_to(campaign)
    except ValueError as exc:
        raise ValueError("daemon-owned path escapes campaign root: " + str(path)) from exc
    return candidate


def backend_name(phase_name: str) -> str:
    phase = str(phase_name)
    if phase in {"PHASE_A_POLUS", "PHASE_B_POLUS"}:
        return "POLUS"
    if "GAUSSIAN" in phase:
        return "GAUSSIAN"
    if "AIMALL" in phase:
        return "AIMALL"
    if phase == "ARIADNE_ARRAY":
        return "ARIADNE"
    if phase in {"INITIAL_FEREBUS", "FEREBUS"}:
        return "FEREBUS"
    raise ValueError("phase has no submitted backend: " + phase)


def bundle_root(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    submission_identity: str,
) -> Path:
    backend = backend_name(phase_name)
    phase = safe_component(phase_name, "phase")
    identity = safe_component(submission_identity, "submission identity")
    path = (
        Path(campaign_dir)
        / ".DATA"
        / "SCRIPTS"
        / "JOBS"
        / backend
        / phase
        / ("iteration-" + str(int(iteration)).zfill(6))
        / identity
    )
    return campaign_owned_path(campaign_dir, path)


def _logical_ids(source: Path) -> Sequence[int]:
    values = []
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                "array task map line " + str(line_number) + " is not an integer"
            ) from exc
        if value < 0:
            raise ValueError("array task map contains a negative logical task ID")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("array task map contains duplicate logical task IDs")
    return values


def read_source_array_task_ids(path: Union[str, Path]) -> Sequence[int]:
    """Read the producer-owned plain-text logical task map."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("array task map is not a regular file: " + str(source))
    return list(_logical_ids(source))


def prepare_attempt_bundle(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    submission_identity: str,
    *,
    array_size: Optional[int],
    max_log_files_per_directory: int,
    source_array_task_map: Optional[Union[str, Path]] = None,
) -> AttemptBundle:
    expected_logs = int(array_size) if array_size is not None else 1
    if expected_logs <= 0:
        raise ValueError("attempt bundle expected log count must be > 0")
    if isinstance(max_log_files_per_directory, bool):
        raise ValueError("hpc.max_job_log_files_per_directory must be an integer")
    limit = int(max_log_files_per_directory)
    if limit <= 0:
        raise ValueError("hpc.max_job_log_files_per_directory must be > 0")
    if expected_logs > limit:
        raise ValueError(
            "submission would create "
            + str(expected_logs)
            + " files in each attempt OUTPUTS/ and ERRORS/ directory, exceeding "
            + "hpc.max_job_log_files_per_directory="
            + str(limit)
        )
    root = bundle_root(
        campaign_dir, phase_name, int(iteration), submission_identity
    )
    if root.is_symlink():
        raise ValueError("attempt bundle root must not be a symlink: " + str(root))
    outputs = root / "OUTPUTS"
    errors = root / "ERRORS"
    outputs.mkdir(parents=True, exist_ok=True)
    errors.mkdir(parents=True, exist_ok=True)
    task_map_path: Optional[Path] = None
    if source_array_task_map is not None:
        source = Path(source_array_task_map)
        logical_ids = list(read_source_array_task_ids(source))
        if array_size is None or len(logical_ids) != int(array_size):
            raise ValueError("array task map length does not match submitted array size")
        task_map_path = root / "array_task_map.json"
        task_map_payload = {
            "schema_version": 1,
            "dense_to_logical": logical_ids,
        }
        if task_map_path.exists() or task_map_path.is_symlink():
            if task_map_path.is_symlink() or not task_map_path.is_file():
                raise ValueError(
                    "attempt array task map is not a regular file: "
                    + str(task_map_path)
                )
            try:
                existing_map = json.loads(
                    task_map_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "attempt array task map is unreadable: " + str(task_map_path)
                ) from exc
            if existing_map != task_map_payload:
                raise ValueError(
                    "attempt array task map already exists with different content"
                )
        else:
            atomic_write_json(task_map_path, task_map_payload)
    return AttemptBundle(
        root=root,
        script=root / "job.sh",
        outputs=outputs,
        errors=errors,
        array_task_map=task_map_path,
    )


def write_attempt_script(bundle: AttemptBundle, body: str) -> Path:
    if bundle.script.exists() or bundle.script.is_symlink():
        if bundle.script.is_symlink() or not bundle.script.is_file():
            raise ValueError(
                "attempt job script is not a regular file: " + str(bundle.script)
            )
        try:
            existing = bundle.script.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                "attempt job script is unreadable: " + str(bundle.script)
            ) from exc
        if existing != str(body):
            raise ValueError(
                "attempt job script already exists with different content"
            )
    else:
        atomic_write_text(bundle.script, body)
    try:
        bundle.script.chmod(0o700)
    except OSError:
        pass
    return bundle.script


def slurm_log_paths(bundle: AttemptBundle, *, is_array: bool) -> Dict[str, str]:
    stem = "%A_%a" if is_array else "%j_0"
    return {
        "output": str(bundle.outputs.resolve()) + "/" + stem + ".o",
        "error": str(bundle.errors.resolve()) + "/" + stem + ".e",
    }


def read_array_task_map(path: Union[str, Path]) -> Sequence[int]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("array task map is unreadable: " + str(source)) from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported array task map: " + str(source))
    values = payload.get("dense_to_logical")
    if not isinstance(values, list):
        raise ValueError("array task map dense_to_logical must be a list")
    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("array task map IDs must be non-negative integers")
        try:
            item = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("array task map IDs must be non-negative integers") from exc
        if item < 0:
            raise ValueError("array task map IDs must be non-negative integers")
        parsed.append(item)
    if len(set(parsed)) != len(parsed):
        raise ValueError("array task map contains duplicate logical task IDs")
    return parsed
