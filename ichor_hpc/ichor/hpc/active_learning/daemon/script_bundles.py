"""Immutable daemon submission-attempt script and log bundles."""
from __future__ import annotations

from ..strict_json import strict_json as json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from .state import atomic_write_json, atomic_write_text
from .filesystem import campaign_owned_path
from ..versioning.manifest import sha256_file


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SCRIPT_BINDING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttemptBundle:
    root: Path
    script: Path
    outputs: Path
    errors: Path
    array_task_map: Optional[Path] = None

    @property
    def script_binding(self) -> Path:
        return self.root / "SCRIPT.json"


def safe_component(value: Any, label: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(label + " is not a safe path component: " + repr(text))
    return text


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
        if not re.fullmatch(r"0|[1-9][0-9]*", text):
            raise ValueError(
                "array task map line " + str(line_number) + " is not an integer"
            )
        value = int(text)
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
    logical_task_ids: Optional[Sequence[int]] = None,
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
    outputs = campaign_owned_path(campaign_dir, root / "OUTPUTS")
    errors = campaign_owned_path(campaign_dir, root / "ERRORS")
    outputs.mkdir(parents=True, exist_ok=True)
    errors.mkdir(parents=True, exist_ok=True)
    # Re-check after creation so a concurrent path substitution cannot redirect
    # Slurm's stdout or stderr outside the campaign.
    outputs = campaign_owned_path(campaign_dir, outputs)
    errors = campaign_owned_path(campaign_dir, errors)
    task_map_path: Optional[Path] = None
    if source_array_task_map is not None and logical_task_ids is not None:
        raise ValueError(
            "attempt bundle accepts either a source task map or logical IDs, not both"
        )
    if source_array_task_map is not None or logical_task_ids is not None:
        if source_array_task_map is not None:
            source = Path(source_array_task_map)
            logical_ids = list(read_source_array_task_ids(source))
        else:
            logical_ids = []
            for value in list(logical_task_ids or []):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "attempt logical task IDs must be non-negative integers"
                    )
                logical_ids.append(value)
            if len(set(logical_ids)) != len(logical_ids):
                raise ValueError("attempt logical task IDs contain duplicates")
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
    write_script_binding(bundle)
    return bundle.script


def write_script_binding(bundle: AttemptBundle) -> Dict[str, Any]:
    """Bind the final submitted script bytes to an immutable sidecar."""
    script = bundle.script
    if script.is_symlink() or not script.is_file():
        raise ValueError("attempt job script is not a regular file: " + str(script))
    payload = {
        "schema_version": SCRIPT_BINDING_SCHEMA_VERSION,
        "script_path": str(script.resolve()),
        "script_size": int(script.stat().st_size),
        "script_sha256": sha256_file(script),
    }
    path = bundle.script_binding
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("attempt script binding is not a regular file: " + str(path))
        existing = _read_script_binding_payload(path)
        if existing != payload:
            raise ValueError("attempt script binding already exists with different content")
    else:
        atomic_write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        **payload,
    }


def _read_script_binding_payload(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt script binding is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("attempt script binding is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("attempt script binding must contain a JSON object")
    if payload.get("schema_version") != SCRIPT_BINDING_SCHEMA_VERSION:
        raise ValueError("attempt script binding has an unsupported schema")
    if not isinstance(payload.get("script_path"), str) or not payload["script_path"]:
        raise ValueError("attempt script binding has no script path")
    if (
        isinstance(payload.get("script_size"), bool)
        or not isinstance(payload.get("script_size"), int)
        or payload["script_size"] < 0
    ):
        raise ValueError("attempt script binding has an invalid script size")
    digest = payload.get("script_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise ValueError("attempt script binding has an invalid script SHA-256")
    return payload


def verify_script_binding(
    path: Union[str, Path], expected_sha256: str
) -> Dict[str, Any]:
    source = Path(path)
    observed_binding_sha = sha256_file(source)
    if observed_binding_sha != str(expected_sha256):
        raise ValueError(
            "attempt script-binding SHA-256 mismatch: expected "
            + str(expected_sha256)
            + " got "
            + observed_binding_sha
        )
    payload = _read_script_binding_payload(source)
    script = Path(str(payload["script_path"]))
    if script.is_symlink() or not script.is_file():
        raise ValueError("bound attempt script is not a regular file: " + str(script))
    if int(script.stat().st_size) != int(payload["script_size"]):
        raise ValueError("bound attempt script size has drifted")
    if sha256_file(script) != str(payload["script_sha256"]):
        raise ValueError("bound attempt script SHA-256 has drifted")
    return payload


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
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("array task map IDs must be non-negative integers")
        item = value
        if item < 0:
            raise ValueError("array task map IDs must be non-negative integers")
        parsed.append(item)
    if len(set(parsed)) != len(parsed):
        raise ValueError("array task map contains duplicate logical task IDs")
    return parsed


__all__ = [
    "AttemptBundle",
    "SCRIPT_BINDING_SCHEMA_VERSION",
    "backend_name",
    "bundle_root",
    "prepare_attempt_bundle",
    "read_array_task_map",
    "read_source_array_task_ids",
    "safe_component",
    "slurm_log_paths",
    "verify_script_binding",
    "write_attempt_script",
    "write_script_binding",
]
