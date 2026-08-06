"""Execute one authenticated FEREBUS array task without shell word splitting."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from .state import atomic_write_json


FEREBUS_TASK_MAP_FILENAME = "FEREBUS_TASK_MAP.json"
FEREBUS_TASK_MAP_SCHEMA_VERSION = 1
FEREBUS_TASK_RECEIPT_FILENAME = "FEREBUS_TASK_RECEIPT.json"
FEREBUS_TASK_RECEIPT_SCHEMA_VERSION = 1


class FerebusTaskRunnerError(RuntimeError):
    """Raised when an array member cannot authenticate its exact task."""


def _sha256(value: Any, label: str) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FerebusTaskRunnerError(label + " must be a lowercase SHA-256")
    return text


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FerebusTaskRunnerError(label + " must be an exact integer")
    parsed = int(value)
    if parsed < minimum:
        raise FerebusTaskRunnerError(label + " must be >= " + str(minimum))
    return parsed


def _contained_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise FerebusTaskRunnerError(label + " must be a relative POSIX path")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise FerebusTaskRunnerError(label + " must be a canonical relative path")
    path = root.joinpath(*parts)
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise FerebusTaskRunnerError(label + " escapes FEREBUS staging") from exc
    current = root
    for part in Path(*parts).parts:
        current = current / part
        if current.is_symlink():
            raise FerebusTaskRunnerError(label + " contains a symlink")
    return path


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_task_map(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FerebusTaskRunnerError("FEREBUS task map is missing or symlinked")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusTaskRunnerError("FEREBUS task map is unreadable") from exc
    if not isinstance(payload, dict):
        raise FerebusTaskRunnerError("FEREBUS task map must be an object")
    if _exact_int(payload.get("schema_version"), "task-map schema") != FEREBUS_TASK_MAP_SCHEMA_VERSION:
        raise FerebusTaskRunnerError("unsupported FEREBUS task-map schema")
    material = dict(payload)
    declared = _sha256(material.pop("task_map_sha256", None), "task-map digest")
    if declared != _canonical_sha256(material):
        raise FerebusTaskRunnerError("FEREBUS task-map digest mismatch")
    return payload


def _validate_input(root: Path, record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping):
        raise FerebusTaskRunnerError(label + " binding must be an object")
    path = _contained_file(root, record.get("path"), label + ".path")
    if not path.is_file() or path.is_symlink():
        raise FerebusTaskRunnerError(label + " is missing")
    if _exact_int(record.get("size"), label + ".size") != int(path.stat().st_size):
        raise FerebusTaskRunnerError(label + " size mismatch")
    if _sha256(record.get("sha256"), label + ".sha256") != sha256_file(path):
        raise FerebusTaskRunnerError(label + " SHA-256 mismatch")
    return path


def execute_task(task_map_path: Path, task_index: int) -> int:
    task_map_path = Path(task_map_path).absolute()
    if task_map_path.is_symlink():
        raise FerebusTaskRunnerError("FEREBUS task map must not be a symlink")
    root = task_map_path.parent.resolve()
    payload = _read_task_map(task_map_path)
    if payload.get("execution_kind") != "native_ferebus":
        raise FerebusTaskRunnerError(
            "only native_ferebus task maps may be executed"
        )
    if payload.get("performance_required") is not True:
        raise FerebusTaskRunnerError(
            "native FEREBUS execution requires a performance receipt"
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise FerebusTaskRunnerError("FEREBUS task map has no tasks")
    if _exact_int(payload.get("n_tasks"), "task-map n_tasks", minimum=1) != len(
        tasks
    ):
        raise FerebusTaskRunnerError("FEREBUS task-map cardinality mismatch")
    index = _exact_int(task_index, "array task index")
    if index >= len(tasks):
        raise FerebusTaskRunnerError(
            "FEREBUS array task index " + str(index) + " is out of range"
        )
    task = tasks[index]
    if not isinstance(task, dict) or _exact_int(
        task.get("task_index"), "logical task index", minimum=1
    ) != index + 1:
        raise FerebusTaskRunnerError("FEREBUS logical task identity is invalid")
    manifest_path = _contained_file(
        root, payload.get("task_manifest_path"), "task_manifest_path"
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FerebusTaskRunnerError("FEREBUS task manifest is missing")
    if _sha256(
        payload.get("task_manifest_sha256"), "task manifest SHA-256"
    ) != sha256_file(manifest_path):
        raise FerebusTaskRunnerError("FEREBUS task manifest changed after submission")
    config_path = _validate_input(root, task.get("config"), "FEREBUS config")
    datasets = task.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {
        "train",
        "int_val",
        "ext_val",
    }:
        raise FerebusTaskRunnerError("FEREBUS dataset bindings are invalid")
    for split in ("train", "int_val", "ext_val"):
        _validate_input(root, datasets.get(split), "FEREBUS " + split)
    executable = payload.get("executable")
    if not isinstance(executable, dict) or not isinstance(executable.get("path"), str):
        raise FerebusTaskRunnerError("FEREBUS executable binding is invalid")
    executable_path = executable["path"]
    if executable.get("sha256") is not None:
        executable_file = Path(executable_path)
        if executable_file.is_symlink() or not executable_file.is_file():
            raise FerebusTaskRunnerError("bound FEREBUS executable is missing")
        if _sha256(executable.get("sha256"), "FEREBUS executable SHA-256") != sha256_file(executable_file):
            raise FerebusTaskRunnerError("bound FEREBUS executable changed")
    argv = task.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or argv[0] != executable_path
        or any(not isinstance(value, str) or "\x00" in value for value in argv)
    ):
        raise FerebusTaskRunnerError("FEREBUS argv contract is invalid")
    if str(config_path.relative_to(root).as_posix()) not in argv:
        raise FerebusTaskRunnerError("FEREBUS argv is not bound to its config")
    receipt_path = _contained_file(root, task.get("receipt_path"), "receipt_path")
    model_path = _contained_file(root, task.get("expected_model_path"), "expected_model_path")
    performance_path = _contained_file(
        root,
        task.get("expected_performance_path"),
        "expected_performance_path",
    )
    receipt: Dict[str, Any] = {
        "schema_version": FEREBUS_TASK_RECEIPT_SCHEMA_VERSION,
        "task_map_sha256": str(payload["task_map_sha256"]),
        "task_manifest_sha256": str(payload["task_manifest_sha256"]),
        "task_index": index + 1,
        "property": task.get("property"),
        "atom": task.get("atom"),
        "argv": list(argv),
        "executable": dict(executable),
        "execution_kind": "native_ferebus",
        "success": False,
        "exit_code": None,
        "model": None,
        "performance": None,
    }
    try:
        completed = subprocess.run(argv, cwd=str(root), check=False)
        exit_code = int(completed.returncode)
        receipt["exit_code"] = exit_code
        if exit_code == 0:
            if (
                model_path.is_symlink()
                or not model_path.is_file()
                or model_path.stat().st_size <= 0
                or performance_path.is_symlink()
                or not performance_path.is_file()
                or performance_path.stat().st_size <= 0
            ):
                receipt["failure_reason"] = "expected_model_or_performance_receipt_missing"
                exit_code = 66
                receipt["exit_code"] = exit_code
            else:
                receipt["model"] = {
                    "path": model_path.relative_to(root).as_posix(),
                    "size": int(model_path.stat().st_size),
                    "sha256": sha256_file(model_path),
                }
                receipt["performance"] = {
                    "path": performance_path.relative_to(root).as_posix(),
                    "size": int(performance_path.stat().st_size),
                    "sha256": sha256_file(performance_path),
                }
                receipt["success"] = True
                # Training success is authoritative independently of optional
                # quality measurement.  Publish it first so an interruption in
                # the metric tail can never cause scientifically valid training
                # to be submitted again.
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(receipt_path, receipt)
                for variable in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ):
                    os.environ[variable] = "1"
                optional_model = None
                try:
                    from ichor.core.models import Model

                    optional_model = Model(model_path)
                except Exception as exc:
                    print(
                        "WARNING: FEREBUS training succeeded but its optional "
                        "postprocessing model could not be loaded: "
                        + type(exc).__name__
                        + ": "
                        + str(exc),
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    from .ferebus_quality import enrich_task_receipt_with_quality

                    enrich_task_receipt_with_quality(
                        root,
                        index,
                        model=optional_model,
                    )
                except Exception as exc:
                    print(
                        "WARNING: FEREBUS training succeeded but optional quality "
                        "measurement was not published: "
                        + type(exc).__name__
                        + ": "
                        + str(exc),
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    from .ferebus_model_admission import (
                        enrich_task_receipt_with_model_admission,
                    )

                    enrich_task_receipt_with_model_admission(
                        root,
                        index,
                        model=optional_model,
                    )
                except Exception as exc:
                    print(
                        "WARNING: FEREBUS training succeeded but optional model "
                        "admission evidence was not published: "
                        + type(exc).__name__
                        + ": "
                        + str(exc),
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    from .ferebus_model_factors import publish_task_factor

                    publish_task_factor(
                        root,
                        index,
                        model=optional_model,
                    )
                except Exception as exc:
                    print(
                        "WARNING: FEREBUS training succeeded but optional model "
                        "factor evidence was not published: "
                        + type(exc).__name__
                        + ": "
                        + str(exc),
                        file=sys.stderr,
                        flush=True,
                    )
        else:
            receipt["failure_reason"] = "ferebus_nonzero_exit"
    except OSError as exc:
        exit_code = 127
        receipt["exit_code"] = exit_code
        receipt["failure_reason"] = type(exc).__name__ + ": " + str(exc)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if not receipt.get("success") or not receipt_path.is_file():
        atomic_write_json(receipt_path, receipt)
    return int(exit_code)


def write_preexisting_model_receipts(
    staging_dir: Path,
    *,
    execution_kind: str,
) -> None:
    """Authenticate imported or synthetic outputs without running FEREBUS."""
    root = Path(staging_dir).resolve()
    task_map_path = root / FEREBUS_TASK_MAP_FILENAME
    payload = _read_task_map(task_map_path)
    if execution_kind not in {"imported_model_bootstrap", "synthetic_dry_run"}:
        raise FerebusTaskRunnerError(
            "pre-existing FEREBUS outputs require an imported or dry execution kind"
        )
    if payload.get("execution_kind") != execution_kind:
        raise FerebusTaskRunnerError(
            "pre-existing FEREBUS output kind does not match its task map"
        )
    performance_required = payload.get("performance_required") is True
    if execution_kind == "imported_model_bootstrap" and performance_required:
        raise FerebusTaskRunnerError("imported model bootstrap cannot claim native performance")
    if execution_kind == "synthetic_dry_run" and not performance_required:
        raise FerebusTaskRunnerError("dry-run FEREBUS evidence requires synthetic performance")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise FerebusTaskRunnerError("FEREBUS task map has no imported tasks")
    for expected_index, task in enumerate(tasks, start=1):
        if not isinstance(task, Mapping) or _exact_int(
            task.get("task_index"), "logical task index", minimum=1
        ) != expected_index:
            raise FerebusTaskRunnerError("imported FEREBUS task identity is invalid")
        model_path = _contained_file(
            root,
            task.get("expected_model_path"),
            "expected_model_path",
        )
        if model_path.is_symlink() or not model_path.is_file() or model_path.stat().st_size <= 0:
            raise FerebusTaskRunnerError("imported FEREBUS model is missing")
        performance = None
        if performance_required:
            performance_path = _contained_file(
                root,
                task.get("expected_performance_path"),
                "expected_performance_path",
            )
            if (
                performance_path.is_symlink()
                or not performance_path.is_file()
                or performance_path.stat().st_size <= 0
            ):
                raise FerebusTaskRunnerError(
                    "pre-existing FEREBUS performance evidence is missing"
                )
            performance = {
                "path": performance_path.relative_to(root).as_posix(),
                "size": int(performance_path.stat().st_size),
                "sha256": sha256_file(performance_path),
            }
        receipt_path = _contained_file(
            root,
            task.get("receipt_path"),
            "receipt_path",
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            receipt_path,
            {
                "schema_version": FEREBUS_TASK_RECEIPT_SCHEMA_VERSION,
                "task_map_sha256": str(payload["task_map_sha256"]),
                "task_manifest_sha256": str(payload["task_manifest_sha256"]),
                "task_index": expected_index,
                "property": task.get("property"),
                "atom": task.get("atom"),
                "argv": list(task.get("argv") or []),
                "executable": dict(payload.get("executable") or {}),
                "execution_kind": execution_kind,
                "success": True,
                "exit_code": 0,
                "model": {
                    "path": model_path.relative_to(root).as_posix(),
                    "size": int(model_path.stat().st_size),
                    "sha256": sha256_file(model_path),
                },
                "performance": performance,
            },
        )


def write_imported_model_receipts(staging_dir: Path) -> None:
    """Authenticate imported models without pretending FEREBUS was executed."""
    write_preexisting_model_receipts(
        staging_dir,
        execution_kind="imported_model_bootstrap",
    )


def _validate_task_receipt_with_map(
    root: Path,
    payload: Mapping[str, Any],
    logical_task_id: int,
    *,
    verify_payload_hashes: bool,
) -> tuple[Dict[str, Any], Dict[str, Any], Mapping[str, Any]]:
    """Validate one receipt against an already authenticated task map."""
    execution_kind = payload.get("execution_kind")
    if execution_kind not in {
        "native_ferebus",
        "imported_model_bootstrap",
        "synthetic_dry_run",
    }:
        raise FerebusTaskRunnerError("FEREBUS task-map execution kind is invalid")
    performance_required = payload.get("performance_required")
    if not isinstance(performance_required, bool):
        raise FerebusTaskRunnerError(
            "FEREBUS task-map performance requirement is invalid"
        )
    tasks = payload.get("tasks")
    if (
        not isinstance(tasks, list)
        or _exact_int(payload.get("n_tasks"), "task-map n_tasks", minimum=1)
        != len(tasks)
    ):
        raise FerebusTaskRunnerError("FEREBUS task-map cardinality mismatch")
    task_id = _exact_int(logical_task_id, "logical task ID")
    if task_id >= len(tasks):
        raise FerebusTaskRunnerError("FEREBUS logical task ID is out of range")
    task = tasks[task_id]
    if (
        not isinstance(task, Mapping)
        or _exact_int(
            task.get("task_index"),
            "logical task index",
            minimum=1,
        )
        != task_id + 1
    ):
        raise FerebusTaskRunnerError("FEREBUS task-map task identity is invalid")
    receipt_path = _contained_file(root, task.get("receipt_path"), "receipt_path")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FerebusTaskRunnerError(
            "FEREBUS task receipt is missing for task " + str(task.get("task_index"))
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusTaskRunnerError("FEREBUS task receipt is unreadable") from exc
    if not isinstance(receipt, dict):
        raise FerebusTaskRunnerError("FEREBUS task receipt must be an object")
    if (
        _exact_int(receipt.get("schema_version"), "task receipt schema")
        != FEREBUS_TASK_RECEIPT_SCHEMA_VERSION
    ):
        raise FerebusTaskRunnerError("unsupported FEREBUS task receipt schema")
    if (
        receipt.get("task_map_sha256") != payload["task_map_sha256"]
        or receipt.get("task_manifest_sha256") != payload["task_manifest_sha256"]
        or receipt.get("task_index") != task.get("task_index")
        or receipt.get("property") != task.get("property")
        or receipt.get("atom") != task.get("atom")
        or receipt.get("execution_kind") != execution_kind
        or receipt.get("argv") != task.get("argv")
        or receipt.get("executable") != payload.get("executable")
        or receipt.get("success") is not True
        or receipt.get("exit_code") != 0
    ):
        raise FerebusTaskRunnerError("FEREBUS task receipt identity/status mismatch")
    model = receipt.get("model")
    if (
        not isinstance(model, dict)
        or model.get("path") != task.get("expected_model_path")
    ):
        raise FerebusTaskRunnerError("FEREBUS task receipt model binding is invalid")
    model_path = _contained_file(root, model.get("path"), "FEREBUS model.path")
    if model_path.is_symlink() or not model_path.is_file():
        raise FerebusTaskRunnerError("FEREBUS model is missing")
    if _exact_int(model.get("size"), "FEREBUS model.size") != int(
        model_path.stat().st_size
    ):
        raise FerebusTaskRunnerError("FEREBUS model size mismatch")
    _sha256(model.get("sha256"), "FEREBUS model.sha256")
    if verify_payload_hashes and str(model["sha256"]) != sha256_file(model_path):
        raise FerebusTaskRunnerError("FEREBUS model SHA-256 mismatch")
    performance = receipt.get("performance")
    performance_path = None
    if performance_required:
        if (
            not isinstance(performance, dict)
            or performance.get("path") != task.get("expected_performance_path")
        ):
            raise FerebusTaskRunnerError(
                "FEREBUS task receipt performance binding is invalid"
            )
        performance_path = _contained_file(
            root,
            performance.get("path"),
            "FEREBUS performance receipt.path",
        )
        if performance_path.is_symlink() or not performance_path.is_file():
            raise FerebusTaskRunnerError("FEREBUS performance receipt is missing")
        if _exact_int(
            performance.get("size"),
            "FEREBUS performance receipt.size",
        ) != int(performance_path.stat().st_size):
            raise FerebusTaskRunnerError(
                "FEREBUS performance receipt size mismatch"
            )
        _sha256(
            performance.get("sha256"),
            "FEREBUS performance receipt.sha256",
        )
        if verify_payload_hashes and str(performance["sha256"]) != sha256_file(
            performance_path
        ):
            raise FerebusTaskRunnerError(
                "FEREBUS performance receipt SHA-256 mismatch"
            )
    elif performance is not None:
        raise FerebusTaskRunnerError(
            "non-executed FEREBUS task must not claim native performance evidence"
        )
    normalised = {
        "task_index": task["task_index"],
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": sha256_file(receipt_path),
        "model_path": model_path.relative_to(root).as_posix(),
        "model_sha256": model["sha256"],
        "performance_path": (
            None
            if performance_path is None
            else performance_path.relative_to(root).as_posix()
        ),
        "performance_sha256": (
            None if performance is None else performance["sha256"]
        ),
    }
    return normalised, receipt, task


def validate_task_receipt(
    staging_dir: Path,
    logical_task_id: int,
) -> Dict[str, Any]:
    """Authenticate one zero-based logical FEREBUS task and its outputs."""
    root = Path(staging_dir).resolve()
    payload = _read_task_map(root / FEREBUS_TASK_MAP_FILENAME)
    normalised, unused_receipt, unused_task = _validate_task_receipt_with_map(
        root,
        payload,
        int(logical_task_id),
        verify_payload_hashes=True,
    )
    del unused_receipt, unused_task
    return normalised


def quarantine_task_outputs(
    staging_dir: Path,
    logical_task_ids: Sequence[int],
    quarantine_dir: Path,
) -> Sequence[Dict[str, Any]]:
    """Move only retry-task receipts, models and performance files aside."""
    root = Path(staging_dir).resolve()
    task_map = _read_task_map(root / FEREBUS_TASK_MAP_FILENAME)
    tasks = task_map.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise FerebusTaskRunnerError("FEREBUS task map has no tasks")
    quarantine = Path(quarantine_dir)
    if quarantine.is_symlink():
        raise FerebusTaskRunnerError("FEREBUS retry quarantine is a symlink")
    quarantine.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for logical_task_id in logical_task_ids:
        task_id = _exact_int(logical_task_id, "logical task ID")
        if task_id in seen:
            raise FerebusTaskRunnerError(
                "FEREBUS retry task list contains duplicate IDs"
            )
        seen.add(task_id)
        if task_id >= len(tasks):
            raise FerebusTaskRunnerError("FEREBUS retry task ID is out of range")
        task = tasks[task_id]
        if (
            not isinstance(task, Mapping)
            or task.get("task_index") != task_id + 1
        ):
            raise FerebusTaskRunnerError("FEREBUS retry task identity is invalid")
        task_quarantine = quarantine / ("task-" + f"{task_id + 1:06d}")
        for kind, raw_path in (
            ("receipt", task.get("receipt_path")),
            ("model", task.get("expected_model_path")),
            ("performance", task.get("expected_performance_path")),
        ):
            source = _contained_file(root, raw_path, "FEREBUS " + kind)
            destination = task_quarantine / kind / source.name
            if destination.is_symlink():
                raise FerebusTaskRunnerError(
                    "FEREBUS retry quarantine destination is a symlink"
                )
            if source.exists() or source.is_symlink():
                if source.is_symlink() or not source.is_file():
                    raise FerebusTaskRunnerError(
                        "FEREBUS retry output is not a regular file: "
                        + str(source)
                    )
                if destination.exists() or destination.is_symlink():
                    raise FerebusTaskRunnerError(
                        "FEREBUS retry source and quarantine destination both exist"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                records.append(
                    {
                        "logical_task_id": task_id,
                        "kind": kind,
                        "source": source.relative_to(root).as_posix(),
                        "destination": str(destination),
                    }
                )
            elif destination.exists():
                if not destination.is_file() or destination.is_symlink():
                    raise FerebusTaskRunnerError(
                        "FEREBUS retry quarantine entry is invalid"
                    )
                records.append(
                    {
                        "logical_task_id": task_id,
                        "kind": kind,
                        "source": source.relative_to(root).as_posix(),
                        "destination": str(destination),
                        "already_quarantined": True,
                    }
                )
    return tuple(records)


def validate_task_receipts(
    staging_dir: Path,
    *,
    verify_payload_hashes: bool = True,
    include_receipt_payloads: bool = False,
) -> Dict[str, Any]:
    """Authenticate complete successful task coverage before postprocessing."""
    root = Path(staging_dir).resolve()
    task_map_path = root / FEREBUS_TASK_MAP_FILENAME
    payload = _read_task_map(task_map_path)
    execution_kind = payload.get("execution_kind")
    if execution_kind not in {
        "native_ferebus",
        "imported_model_bootstrap",
        "synthetic_dry_run",
    }:
        raise FerebusTaskRunnerError("FEREBUS task-map execution kind is invalid")
    performance_required = payload.get("performance_required")
    if not isinstance(performance_required, bool):
        raise FerebusTaskRunnerError(
            "FEREBUS task-map performance requirement is invalid"
        )
    if _exact_int(payload.get("n_tasks"), "task-map n_tasks", minimum=1) != len(
        payload.get("tasks") or []
    ):
        raise FerebusTaskRunnerError("FEREBUS task-map cardinality mismatch")
    validated = [
        _validate_task_receipt_with_map(
            root,
            payload,
            logical_task_id,
            verify_payload_hashes=bool(verify_payload_hashes),
        )
        for logical_task_id in range(len(payload["tasks"]))
    ]
    records = [record for record, unused_receipt, unused_task in validated]
    result = {
        "task_map_path": FEREBUS_TASK_MAP_FILENAME,
        "task_map_sha256": payload["task_map_sha256"],
        "task_map_file_sha256": sha256_file(task_map_path),
        "execution_kind": execution_kind,
        "performance_required": performance_required,
        "n_tasks": len(records),
        "receipts": records,
    }
    if include_receipt_payloads:
        result["task_map"] = dict(payload)
        result["receipt_payloads"] = [
            receipt for unused_record, receipt, unused_task in validated
        ]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-map", required=True)
    parser.add_argument("--scheduler-task-map", default=None)
    parser.add_argument("--task-index", type=int, default=None)
    args = parser.parse_args(argv)
    task_index = args.task_index
    if task_index is None:
        raw = (
            os.environ.get("ICHOR_SCHEDULER_ARRAY_TASK_ID")
            or os.environ.get("SLURM_ARRAY_TASK_ID")
        )
        if raw is None or not raw.isdigit():
            raise FerebusTaskRunnerError(
                "ICHOR scheduler array task ID is missing or invalid"
            )
        task_index = int(raw)
    if args.scheduler_task_map is not None:
        from .script_bundles import read_array_task_map

        dense_mapping = list(read_array_task_map(args.scheduler_task_map))
        if not 0 <= int(task_index) < len(dense_mapping):
            raise FerebusTaskRunnerError(
                "FEREBUS dense scheduler task ID is out of range"
            )
        task_index = int(dense_mapping[int(task_index)])
    return execute_task(Path(args.task_map), task_index)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FEREBUS_TASK_MAP_FILENAME",
    "FEREBUS_TASK_MAP_SCHEMA_VERSION",
    "FEREBUS_TASK_RECEIPT_FILENAME",
    "FEREBUS_TASK_RECEIPT_SCHEMA_VERSION",
    "FerebusTaskRunnerError",
    "execute_task",
    "main",
    "quarantine_task_outputs",
    "validate_task_receipt",
    "validate_task_receipts",
    "write_imported_model_receipts",
    "write_preexisting_model_receipts",
]
