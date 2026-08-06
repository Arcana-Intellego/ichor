"""Task-produced FEREBUS model admission evidence and legacy fallback."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from .filesystem import campaign_owned_path
from .state import atomic_write_json


FEREBUS_TASK_MODEL_ADMISSION_KIND = "ferebus_task_model_admission_v1"
FEREBUS_MODEL_ADMISSION_CACHE_SCHEMA_VERSION = 1


class FerebusModelAdmissionError(ValueError):
    """Raised when candidate model admission evidence is not trustworthy."""


@dataclass(frozen=True)
class FerebusModelAdmissionContext:
    """Candidate authority authenticated once for one postprocessing attempt."""

    staging: Path
    task_manifest: Mapping[str, Any]
    task_execution: Mapping[str, Any]
    admissions: Tuple[Mapping[str, Any], ...]
    sources: Tuple[str, ...]
    statistics: Mapping[str, int]


AdmissionProgressCallback = Optional[
    Callable[[str, Mapping[str, Any]], None]
]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalised_symbol_source(symbol: Any) -> str:
    source = inspect.getsource(symbol)
    tree = ast.parse(source)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


@lru_cache(maxsize=1)
def ferebus_model_admission_evaluator_identity() -> Dict[str, Any]:
    from .ferebus_dataset import iter_feature_target_chunks
    from .model_contract import (
        _check_section_rows,
        _validate_kernel_family,
        _validate_model_object,
        _validate_task_bootstrap_prefix,
        validate_ferebus_task_model_semantics,
    )
    from ..ferebus_prior import (
        validate_ferebus_config_contract,
        validate_model_prior_mean,
    )

    try:
        import scipy

        scipy_version = str(scipy.__version__)
    except Exception:
        scipy_version = None

    material = {
        "kind": FEREBUS_TASK_MODEL_ADMISSION_KIND,
        "numpy_version": str(np.__version__),
        "scipy_version": scipy_version,
        "symbols": [
            _normalised_symbol_source(_validate_model_object),
            _normalised_symbol_source(_validate_kernel_family),
            _normalised_symbol_source(_check_section_rows),
            _normalised_symbol_source(_validate_task_bootstrap_prefix),
            _normalised_symbol_source(validate_ferebus_task_model_semantics),
            _normalised_symbol_source(iter_feature_target_chunks),
            _normalised_symbol_source(validate_ferebus_config_contract),
            _normalised_symbol_source(validate_model_prior_mean),
        ],
    }
    return {
        "kind": FEREBUS_TASK_MODEL_ADMISSION_KIND,
        "fingerprint_sha256": _canonical_sha256(material),
        "numpy_version": material["numpy_version"],
        "scipy_version": material["scipy_version"],
    }


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FerebusModelAdmissionError(label + " is missing or symlinked")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusModelAdmissionError(label + " is unreadable") from exc
    if not isinstance(payload, dict):
        raise FerebusModelAdmissionError(label + " must contain an object")
    return payload


def _file_stat_identity(path: Path) -> Dict[str, int]:
    if path.is_symlink() or not path.is_file():
        raise FerebusModelAdmissionError(
            "FEREBUS admission input is missing or symlinked: " + str(path)
        )
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _capture_file_binding(
    staging: Path,
    record: Mapping[str, Any],
    label: str,
) -> Dict[str, Any]:
    from .ferebus_task_runner import _validate_input

    path = _validate_input(staging, record, label)
    return {
        "path": str(record["path"]),
        "size": int(record["size"]),
        "sha256": str(record["sha256"]),
        "stat": _file_stat_identity(path),
    }


def _validate_file_binding(
    staging: Path,
    evidence: Any,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    from .ferebus_task_runner import _contained_file

    if not isinstance(evidence, Mapping):
        raise FerebusModelAdmissionError(label + " file evidence is invalid")
    if {
        "path": evidence.get("path"),
        "size": evidence.get("size"),
        "sha256": evidence.get("sha256"),
    } != {
        "path": expected.get("path"),
        "size": expected.get("size"),
        "sha256": expected.get("sha256"),
    }:
        raise FerebusModelAdmissionError(label + " file binding mismatch")
    path = _contained_file(staging, expected.get("path"), label + ".path")
    if dict(evidence.get("stat") or {}) != _file_stat_identity(path):
        raise FerebusModelAdmissionError(label + " changed after task admission")


def _task_material(
    staging: Path,
    logical_task_id: int,
    *,
    task_manifest: Optional[Mapping[str, Any]] = None,
    task_map: Optional[Mapping[str, Any]] = None,
    receipt: Optional[Mapping[str, Any]] = None,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    from . import input_staging as _stg
    from .ferebus_task_runner import (
        FEREBUS_TASK_MAP_FILENAME,
        _read_task_map,
        validate_task_receipt,
    )

    manifest = (
        _stg.read_ferebus_manifest(staging, verify_dataset_files=False)
        if task_manifest is None
        else task_manifest
    )
    selected_map = (
        _read_task_map(staging / FEREBUS_TASK_MAP_FILENAME)
        if task_map is None
        else task_map
    )
    tasks = selected_map.get("tasks")
    manifest_tasks = manifest.get("tasks")
    if (
        not isinstance(tasks, list)
        or not isinstance(manifest_tasks, list)
        or not 0 <= int(logical_task_id) < len(tasks)
        or len(tasks) != len(manifest_tasks)
    ):
        raise FerebusModelAdmissionError("FEREBUS admission task coverage is invalid")
    map_task = tasks[int(logical_task_id)]
    manifest_task = manifest_tasks[int(logical_task_id)]
    if (
        not isinstance(map_task, Mapping)
        or not isinstance(manifest_task, Mapping)
        or map_task.get("task_index") != manifest_task.get("task_index")
        or map_task.get("property") != manifest_task.get("property")
        or map_task.get("atom") != manifest_task.get("atom")
    ):
        raise FerebusModelAdmissionError("FEREBUS admission task identity mismatch")
    selected_receipt = receipt
    if selected_receipt is None:
        validate_task_receipt(staging, int(logical_task_id))
        receipt_path = staging.joinpath(*str(map_task["receipt_path"]).split("/"))
        selected_receipt = _read_json_object(
            receipt_path,
            "FEREBUS task receipt",
        )
    return manifest, selected_map, map_task, selected_receipt


def measure_ferebus_task_model_admission(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Any = None,
) -> Dict[str, Any]:
    """Perform exact per-task model admission and return portable evidence."""
    from .ferebus_quality import validate_ferebus_task_measurement
    from .model_contract import validate_ferebus_task_model_semantics

    staging = Path(staging_dir).resolve()
    manifest, task_map, map_task, receipt = _task_material(
        staging,
        int(logical_task_id),
    )
    manifest_task = manifest["tasks"][int(logical_task_id)]
    quality = receipt.get("quality_measurement")
    validated_quality = None
    if quality is not None:
        validated_quality = validate_ferebus_task_measurement(
            staging,
            int(logical_task_id),
            quality,
        )
    semantic = validate_ferebus_task_model_semantics(
        staging,
        manifest,
        manifest_task,
        model=model,
        training_binding_proven=validated_quality is not None,
    )
    datasets = map_task.get("datasets")
    if not isinstance(datasets, Mapping):
        raise FerebusModelAdmissionError("FEREBUS admission datasets are invalid")
    files = {
        "config": _capture_file_binding(
            staging,
            map_task["config"],
            "FEREBUS config",
        ),
        "model": _capture_file_binding(
            staging,
            receipt["model"],
            "FEREBUS model",
        ),
        "datasets": {
            split: _capture_file_binding(
                staging,
                datasets[split],
                "FEREBUS " + split,
            )
            for split in ("train", "int_val", "ext_val")
        },
    }
    if receipt.get("performance") is not None:
        files["performance"] = _capture_file_binding(
            staging,
            receipt["performance"],
            "FEREBUS performance receipt",
        )
    else:
        files["performance"] = None
    return {
        "kind": FEREBUS_TASK_MODEL_ADMISSION_KIND,
        "evaluator_identity": ferebus_model_admission_evaluator_identity(),
        "task_map_sha256": str(task_map["task_map_sha256"]),
        "task_manifest_sha256": str(task_map["task_manifest_sha256"]),
        "task_index": int(map_task["task_index"]),
        "property": str(map_task["property"]),
        "atom": str(map_task["atom"]),
        "quality_measurement_sha256": (
            None if validated_quality is None else _canonical_sha256(validated_quality)
        ),
        "files": files,
        "semantic": semantic,
        "numerical_threads": 1,
    }


def validate_ferebus_task_model_admission(
    staging_dir: Path,
    logical_task_id: int,
    evidence: Mapping[str, Any],
    *,
    task_manifest: Optional[Mapping[str, Any]] = None,
    task_map: Optional[Mapping[str, Any]] = None,
    receipt: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate task evidence through bindings and immutable stat identities."""
    staging = Path(staging_dir).resolve()
    if not isinstance(evidence, Mapping):
        raise FerebusModelAdmissionError("FEREBUS task model admission is invalid")
    manifest, selected_map, map_task, selected_receipt = _task_material(
        staging,
        int(logical_task_id),
        task_manifest=task_manifest,
        task_map=task_map,
        receipt=receipt,
    )
    manifest_task = manifest["tasks"][int(logical_task_id)]
    if (
        evidence.get("kind") != FEREBUS_TASK_MODEL_ADMISSION_KIND
        or evidence.get("evaluator_identity")
        != ferebus_model_admission_evaluator_identity()
        or evidence.get("task_map_sha256") != selected_map.get("task_map_sha256")
        or evidence.get("task_manifest_sha256")
        != selected_map.get("task_manifest_sha256")
        or evidence.get("task_index") != map_task.get("task_index")
        or evidence.get("property") != map_task.get("property")
        or evidence.get("atom") != map_task.get("atom")
        or evidence.get("numerical_threads") != 1
    ):
        raise FerebusModelAdmissionError(
            "FEREBUS task model admission identity mismatch"
        )
    files = evidence.get("files")
    if not isinstance(files, Mapping):
        raise FerebusModelAdmissionError("FEREBUS admission file bindings are invalid")
    _validate_file_binding(staging, files.get("config"), map_task["config"], "config")
    _validate_file_binding(
        staging,
        files.get("model"),
        selected_receipt["model"],
        "model",
    )
    map_datasets = map_task.get("datasets")
    evidence_datasets = files.get("datasets")
    if not isinstance(map_datasets, Mapping) or not isinstance(
        evidence_datasets, Mapping
    ):
        raise FerebusModelAdmissionError("FEREBUS admission datasets are invalid")
    for split in ("train", "int_val", "ext_val"):
        _validate_file_binding(
            staging,
            evidence_datasets.get(split),
            map_datasets[split],
            split,
        )
    if selected_receipt.get("performance") is None:
        if files.get("performance") is not None:
            raise FerebusModelAdmissionError(
                "FEREBUS admission claims unexpected performance evidence"
            )
    else:
        _validate_file_binding(
            staging,
            files.get("performance"),
            selected_receipt["performance"],
            "performance",
        )
    quality = selected_receipt.get("quality_measurement")
    expected_quality_sha = None if quality is None else _canonical_sha256(quality)
    if evidence.get("quality_measurement_sha256") != expected_quality_sha:
        raise FerebusModelAdmissionError(
            "FEREBUS admission quality binding mismatch"
        )
    semantic = evidence.get("semantic")
    task_alf = [int(value) - 1 for value in manifest_task.get("alf_1_indexed", [])]
    expected_rows = int((manifest_task.get("row_counts") or {}).get("train", -1))
    kernel = manifest.get("kernel_contract") or {}
    semantic_prior = (
        semantic.get("prior_mean") if isinstance(semantic, Mapping) else None
    )
    task_prior = manifest_task.get("prior_mean")
    if quality is not None:
        expected_prior = quality.get("prior_mean")
    else:
        expected_prior = None
    prior_matches = (
        semantic_prior == expected_prior
        if expected_prior is not None
        else (
            isinstance(semantic_prior, Mapping)
            and isinstance(task_prior, Mapping)
            and semantic_prior.get("contract_sha256")
            == task_prior.get("contract_sha256")
            and semantic_prior.get("expected_mean_ha")
            == task_prior.get("expected_mean_ha")
            and semantic_prior.get("observed_mean_ha")
            == task_prior.get("expected_mean_ha")
            and semantic_prior.get("units") == "ha"
        )
    )
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("system") != manifest.get("system")
        or semantic.get("property") != manifest_task.get("property")
        or semantic.get("atom") != manifest_task.get("atom")
        or semantic.get("alf_zero_indexed") != task_alf
        or semantic.get("ntrain") != expected_rows
        or not isinstance(semantic.get("nfeats"), int)
        or int(semantic["nfeats"]) <= 0
        or semantic.get("kernel_family") != kernel.get("family")
        or not isinstance(semantic.get("numeric_model_identity"), str)
        or not semantic.get("numeric_model_identity")
        or semantic.get("training_data_binding_complete") is not True
        or semantic.get("bootstrap_validation_complete") is not True
        or semantic.get("config_validation_complete") is not True
        or not prior_matches
    ):
        raise FerebusModelAdmissionError(
            "FEREBUS task model semantic admission is invalid"
        )
    return dict(evidence)


def enrich_task_receipt_with_model_admission(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Any = None,
) -> Dict[str, Any]:
    """Atomically enrich a successful receipt with optional admission proof."""
    from .ferebus_task_runner import FEREBUS_TASK_MAP_FILENAME, _read_task_map

    staging = Path(staging_dir).resolve()
    task_map = _read_task_map(staging / FEREBUS_TASK_MAP_FILENAME)
    task = task_map["tasks"][int(logical_task_id)]
    receipt_path = staging.joinpath(*str(task["receipt_path"]).split("/"))
    before = _read_json_object(receipt_path, "FEREBUS task receipt")
    existing = before.get("model_admission")
    if existing is not None:
        return validate_ferebus_task_model_admission(
            staging,
            int(logical_task_id),
            existing,
            task_map=task_map,
            receipt=before,
        )
    evidence = measure_ferebus_task_model_admission(
        staging,
        int(logical_task_id),
        model=model,
    )
    current = _read_json_object(receipt_path, "FEREBUS task receipt")
    if current != before:
        raise FerebusModelAdmissionError(
            "FEREBUS task receipt changed during model admission"
        )
    enriched = dict(current)
    enriched["model_admission"] = evidence
    atomic_write_json(receipt_path, enriched)
    return validate_ferebus_task_model_admission(
        staging,
        int(logical_task_id),
        evidence,
        task_map=task_map,
        receipt=enriched,
    )


def _cache_identity(
    manifest: Mapping[str, Any],
    task_map: Mapping[str, Any],
    map_task: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": FEREBUS_TASK_MODEL_ADMISSION_KIND,
        "campaign_uid": str(manifest.get("campaign_uid") or ""),
        "reference_data_version": int(manifest.get("reference_data_version", -1)),
        "task_map_sha256": str(task_map.get("task_map_sha256") or ""),
        "task_manifest_sha256": str(task_map.get("task_manifest_sha256") or ""),
        "task_index": int(map_task.get("task_index", -1)),
        "property": str(map_task.get("property") or ""),
        "atom": str(map_task.get("atom") or ""),
        "model": dict(receipt.get("model") or {}),
        "performance": (
            None
            if receipt.get("performance") is None
            else dict(receipt["performance"])
        ),
        "config": dict(map_task.get("config") or {}),
        "datasets": {
            split: dict((map_task.get("datasets") or {}).get(split) or {})
            for split in ("train", "int_val", "ext_val")
        },
        "evaluator_identity": ferebus_model_admission_evaluator_identity(),
    }


def _cache_path(staging: Path, identity: Mapping[str, Any]) -> Path:
    campaign = staging.parent.parent.resolve()
    return campaign_owned_path(
        campaign,
        campaign
        / ".DATA"
        / "CACHE"
        / "FEREBUS_MODEL_VALIDATION"
        / (_canonical_sha256(identity) + ".json"),
    )


def _read_cache(
    staging: Path,
    logical_task_id: int,
    *,
    manifest: Mapping[str, Any],
    task_map: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    map_task = task_map["tasks"][int(logical_task_id)]
    identity = _cache_identity(manifest, task_map, map_task, receipt)
    path = _cache_path(staging, identity)
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise FerebusModelAdmissionError("FEREBUS model admission cache is unsafe")
    try:
        payload = _read_json_object(path, "FEREBUS model admission cache")
        if (
            payload.get("schema_version")
            != FEREBUS_MODEL_ADMISSION_CACHE_SCHEMA_VERSION
            or payload.get("identity") != identity
            or payload.get("identity_sha256") != _canonical_sha256(identity)
        ):
            raise FerebusModelAdmissionError(
                "FEREBUS model admission cache identity mismatch"
            )
        return validate_ferebus_task_model_admission(
            staging,
            int(logical_task_id),
            payload.get("model_admission"),
            task_manifest=manifest,
            task_map=task_map,
            receipt=receipt,
        )
    except Exception:
        path.unlink()
        return None


def _write_cache(staging: Path, logical_task_id: int, evidence: Mapping[str, Any]) -> Path:
    manifest, task_map, map_task, receipt = _task_material(
        staging,
        int(logical_task_id),
    )
    identity = _cache_identity(manifest, task_map, map_task, receipt)
    path = _cache_path(staging, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise FerebusModelAdmissionError("FEREBUS model admission cache root is unsafe")
    payload = {
        "schema_version": FEREBUS_MODEL_ADMISSION_CACHE_SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "model_admission": dict(evidence),
    }
    if path.exists() or path.is_symlink():
        existing = _read_cache(
            staging,
            int(logical_task_id),
            manifest=manifest,
            task_map=task_map,
            receipt=receipt,
        )
        if existing is None and not path.exists() and not path.is_symlink():
            atomic_write_json(path, payload)
            return path
        if existing != dict(evidence):
            raise FerebusModelAdmissionError(
                "FEREBUS model admission cache conflict"
            )
        return path
    atomic_write_json(path, payload)
    return path


def _cache_worker(staging_dir: Path, logical_task_id: int) -> Path:
    evidence = measure_ferebus_task_model_admission(
        Path(staging_dir),
        int(logical_task_id),
    )
    return _write_cache(Path(staging_dir).resolve(), int(logical_task_id), evidence)


def _run_cache_subprocess(staging: Path, logical_task_id: int) -> None:
    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ichor.hpc.active_learning.daemon.ferebus_model_admission",
            "--validate-cache",
            "--staging-dir",
            str(staging),
            "--logical-task-id",
            str(int(logical_task_id)),
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        raise FerebusModelAdmissionError(
            "isolated FEREBUS model admission failed: " + diagnostic[-1200:]
        )


def _emit_progress(
    callback: AdmissionProgressCallback,
    stage: str,
    **fields: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), dict(fields))
    except Exception:
        pass


def build_ferebus_model_admission_context(
    staging_dir: Path,
    *,
    progress_callback: AdmissionProgressCallback = None,
) -> FerebusModelAdmissionContext:
    """Authenticate completion and exact model admission once per candidate."""
    from . import input_staging as _stg
    from .ferebus_task_runner import validate_task_receipts

    staging = Path(staging_dir).resolve()
    _emit_progress(progress_callback, "ferebus_task_control_validation")
    manifest = _stg.read_ferebus_manifest(staging, verify_dataset_files=False)
    execution = validate_task_receipts(
        staging,
        verify_payload_hashes=False,
        include_receipt_payloads=True,
    )
    task_map = execution["task_map"]
    receipts = execution["receipt_payloads"]
    total = int(execution["n_tasks"])
    _emit_progress(
        progress_callback,
        "ferebus_task_control_validation",
        completed=total,
        total=total,
        unit="tasks",
    )
    admissions: Dict[int, Dict[str, Any]] = {}
    sources: Dict[int, str] = {}
    missing = []
    _emit_progress(
        progress_callback,
        "ferebus_model_admission",
        completed=0,
        total=total,
        unit="models",
    )
    for logical_task_id, receipt in enumerate(receipts):
        inline = receipt.get("model_admission")
        if inline is not None:
            try:
                admissions[logical_task_id] = validate_ferebus_task_model_admission(
                    staging,
                    logical_task_id,
                    inline,
                    task_manifest=manifest,
                    task_map=task_map,
                    receipt=receipt,
                )
                sources[logical_task_id] = "task"
            except Exception:
                inline = None
        if logical_task_id not in admissions:
            cached = _read_cache(
                staging,
                logical_task_id,
                manifest=manifest,
                task_map=task_map,
                receipt=receipt,
            )
            if cached is None:
                missing.append(logical_task_id)
            else:
                admissions[logical_task_id] = cached
                sources[logical_task_id] = "cache"
        _emit_progress(
            progress_callback,
            "ferebus_model_admission",
            completed=logical_task_id + 1,
            total=total,
            unit="models",
        )
    if missing:
        _emit_progress(
            progress_callback,
            "ferebus_local_admission",
            completed=0,
            total=len(missing),
            unit="models",
        )
        failures: Dict[int, Exception] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(missing)))) as pool:
            futures = {
                pool.submit(_run_cache_subprocess, staging, task_id): task_id
                for task_id in missing
            }
            for completed_count, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                task_id = futures[future]
                try:
                    future.result()
                    cached = _read_cache(
                        staging,
                        task_id,
                        manifest=manifest,
                        task_map=task_map,
                        receipt=receipts[task_id],
                    )
                    if cached is None:
                        raise FerebusModelAdmissionError(
                            "model admission worker published no cache"
                        )
                    admissions[task_id] = cached
                    sources[task_id] = "local"
                except Exception as exc:
                    failures[task_id] = exc
                _emit_progress(
                    progress_callback,
                    "ferebus_local_admission",
                    completed=completed_count,
                    total=len(missing),
                    unit="models",
                )
        if failures:
            task_id = sorted(failures)[0]
            exc = failures[task_id]
            raise FerebusModelAdmissionError(
                "FEREBUS model admission failed for task "
                + str(task_id + 1)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
    if sorted(admissions) != list(range(total)):
        raise FerebusModelAdmissionError(
            "FEREBUS model admission coverage is incomplete"
        )
    expected_models = {
        str(task.get("expected_model_path"))
        for task in manifest.get("tasks", [])
        if isinstance(task, Mapping)
    }
    actual_models = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*.model")
        if path.is_file() and not path.is_symlink()
    }
    if actual_models != expected_models:
        raise FerebusModelAdmissionError(
            "FEREBUS candidate model inventory is contradictory"
        )
    source_values = tuple(sources[index] for index in range(total))
    statistics = {
        "task": sum(value == "task" for value in source_values),
        "cache": sum(value == "cache" for value in source_values),
        "local": sum(value == "local" for value in source_values),
        "total": total,
    }
    return FerebusModelAdmissionContext(
        staging=staging,
        task_manifest=manifest,
        task_execution=execution,
        admissions=tuple(admissions[index] for index in range(total)),
        sources=source_values,
        statistics=statistics,
    )


def assert_ferebus_model_admission_context_unchanged(
    context: FerebusModelAdmissionContext,
) -> None:
    """Recheck exact candidate controls and stat-bound payloads before publish."""
    from . import input_staging as _stg
    from .ferebus_task_runner import FEREBUS_TASK_MAP_FILENAME

    staging = Path(context.staging).resolve()
    task_map = context.task_execution.get("task_map")
    receipts = context.task_execution.get("receipt_payloads")
    records = context.task_execution.get("receipts")
    if (
        not isinstance(task_map, Mapping)
        or not isinstance(receipts, list)
        or not isinstance(records, list)
        or len(receipts) != len(context.admissions)
        or len(records) != len(context.admissions)
    ):
        raise FerebusModelAdmissionError(
            "FEREBUS model admission context is incomplete"
        )
    manifest_path = _stg.ferebus_manifest_path(staging)
    task_map_path = staging / FEREBUS_TASK_MAP_FILENAME
    if sha256_file(manifest_path) != str(task_map.get("task_manifest_sha256") or ""):
        raise FerebusModelAdmissionError(
            "FEREBUS task manifest changed after model admission"
        )
    if sha256_file(task_map_path) != str(
        context.task_execution.get("task_map_file_sha256") or ""
    ):
        raise FerebusModelAdmissionError(
            "FEREBUS task map changed after model admission"
        )
    for logical_task_id, (evidence, receipt, record) in enumerate(
        zip(context.admissions, receipts, records)
    ):
        map_task = task_map["tasks"][logical_task_id]
        receipt_path = staging.joinpath(*str(map_task["receipt_path"]).split("/"))
        if sha256_file(receipt_path) != str(record.get("receipt_sha256") or ""):
            raise FerebusModelAdmissionError(
                "FEREBUS task receipt changed after model admission"
            )
        validate_ferebus_task_model_admission(
            staging,
            logical_task_id,
            evidence,
            task_manifest=context.task_manifest,
            task_map=task_map,
            receipt=receipt,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-cache", action="store_true")
    parser.add_argument("--staging-dir")
    parser.add_argument("--logical-task-id", type=int)
    args = parser.parse_args(argv)
    if not args.validate_cache:
        parser.error("an internal model-admission action is required")
    if args.staging_dir is None or args.logical_task_id is None:
        parser.error("--staging-dir and --logical-task-id are required")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    _cache_worker(Path(args.staging_dir), int(args.logical_task_id))
    return 0


__all__ = [
    "FEREBUS_MODEL_ADMISSION_CACHE_SCHEMA_VERSION",
    "FEREBUS_TASK_MODEL_ADMISSION_KIND",
    "FerebusModelAdmissionContext",
    "FerebusModelAdmissionError",
    "build_ferebus_model_admission_context",
    "assert_ferebus_model_admission_context_unchanged",
    "enrich_task_receipt_with_model_admission",
    "ferebus_model_admission_evaluator_identity",
    "measure_ferebus_task_model_admission",
    "validate_ferebus_task_model_admission",
]


if __name__ == "__main__":
    raise SystemExit(main())
