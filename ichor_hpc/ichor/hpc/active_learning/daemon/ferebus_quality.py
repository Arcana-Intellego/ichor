"""FEREBUS held-out metric sidecar generation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..strict_json import strict_json as json
from .state import atomic_write_json


FEREBUS_QUALITY_MANIFEST = "FEREBUS_QUALITY.json"
FEREBUS_QUALITY_SCHEMA_VERSION = 4
FEREBUS_QUALITY_DECISION_MANIFEST = "FEREBUS_QUALITY_DECISION.json"
FEREBUS_QUALITY_DECISION_SCHEMA_VERSION = 2
FEREBUS_QUALITY_DECISION_POLICY = "absolute_hard_relative_advisory_v1"
FEREBUS_RELATIVE_REGRESSION_WARNINGS = frozenset(
    {
        "ferebus_aggregate_ext_rmse_regressed",
        "ferebus_task_ext_rmse_regressed",
    }
)

_PERFORMANCE_METRIC_ALIASES = {
    "weights_l2_nor": "weights_l2_norm",
    "covariance_con": "covariance_condition_number",
}
_PREDICTION_CHUNK_SIZE = 512
FEREBUS_TASK_QUALITY_KIND = "exact_streaming_ferebus_quality_v1"
FEREBUS_QUALITY_CACHE_SCHEMA_VERSION = 1
_AUTO_INCUMBENT = object()


@dataclass(frozen=True)
class FerebusQualityContext:
    """Authenticated inputs reused throughout one postprocessing operation."""

    staging: Path
    task_manifest: Mapping[str, Any]
    task_execution: Mapping[str, Any]
    incumbent_set: Any


QualityProgressCallback = Optional[
    Callable[[str, Mapping[str, Any]], None]
]


class FerebusQualityDecisionError(ValueError):
    """Raised when FEREBUS quality evidence or its decision is untrustworthy."""


class FerebusQualityMeasurementIncomplete(FerebusQualityDecisionError):
    """Raised when raw FEREBUS outputs cannot yet support a quality decision."""


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FerebusQualityDecisionError(label + " must be an exact integer")
    if value < minimum:
        raise FerebusQualityDecisionError(label + " must be >= " + str(minimum))
    return int(value)


def _threshold(
    gates: Any,
    name: str,
    *,
    allow_negative: bool = False,
) -> Optional[float]:
    value = getattr(gates, name, None)
    if value is None:
        return None
    if isinstance(value, bool):
        raise FerebusQualityDecisionError(name + " must be numeric or null")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FerebusQualityDecisionError(name + " must be numeric or null") from exc
    if not math.isfinite(out) or (out < 0.0 and not allow_negative):
        qualifier = "finite" if allow_negative else "finite and non-negative"
        raise FerebusQualityDecisionError(name + " must be " + qualifier)
    return out


def _new_metric_state() -> Dict[str, float]:
    return {
        "n": 0.0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "target_mean": 0.0,
        "target_m2": 0.0,
    }


def _update_metric_state(
    state: Dict[str, float],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    y_true = np.asarray(targets, dtype=float).reshape(-1)
    y_pred = np.asarray(predictions, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("prediction/target row count mismatch")
    residual = y_pred - y_true
    if not np.all(np.isfinite(residual)):
        raise ValueError("FEREBUS predictions contain non-finite values")
    batch_n = float(y_true.size)
    batch_mean = float(np.mean(y_true))
    batch_m2 = float(np.sum((y_true - batch_mean) ** 2))
    previous_n = float(state["n"])
    combined_n = previous_n + batch_n
    delta = batch_mean - float(state["target_mean"])
    state["target_mean"] = float(state["target_mean"]) + delta * (
        batch_n / combined_n
    )
    state["target_m2"] = (
        float(state["target_m2"])
        + batch_m2
        + delta * delta * previous_n * batch_n / combined_n
    )
    state["n"] = combined_n
    state["absolute_error_sum"] += float(np.sum(np.abs(residual)))
    state["squared_error_sum"] += float(np.sum(residual * residual))


def _finish_metric_state(state: Mapping[str, float]) -> Dict[str, float]:
    count = int(state["n"])
    if count <= 0:
        raise ValueError("FEREBUS metric stream has no rows")
    squared_error_sum = float(state["squared_error_sum"])
    rmse = math.sqrt(squared_error_sum / count)
    mae = float(state["absolute_error_sum"]) / count
    denominator = float(state["target_m2"])
    r2 = 1.0 if denominator == 0.0 and rmse == 0.0 else (
        0.0 if denominator == 0.0 else 1.0 - squared_error_sum / denominator
    )
    if not all(math.isfinite(value) for value in (rmse, mae, r2)):
        raise ValueError("FEREBUS metric is non-finite")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _stream_model_metrics(
    csv_path: Path,
    prop: str,
    candidate_model: Any,
    *,
    incumbent_model: Any = None,
    bind_training_data: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]], int]:
    from .ferebus_dataset import iter_feature_target_chunks

    candidate_state = _new_metric_state()
    incumbent_state = _new_metric_state() if incumbent_model is not None else None
    expected_x = np.asarray(candidate_model.x, dtype=float)
    expected_y = np.asarray(candidate_model.y, dtype=float).reshape(-1)
    offset = 0
    for features, targets in iter_feature_target_chunks(
        csv_path,
        prop,
        chunk_size=_PREDICTION_CHUNK_SIZE,
    ):
        candidate_predictions = np.asarray(
            candidate_model.predict(features),
            dtype=float,
        ).reshape(-1)
        _update_metric_state(candidate_state, targets, candidate_predictions)
        if bind_training_data:
            stop = offset + int(targets.shape[0])
            if (
                stop > expected_x.shape[0]
                or features.shape != expected_x[offset:stop].shape
                or not np.allclose(
                    features,
                    expected_x[offset:stop],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.allclose(
                    targets,
                    expected_y[offset:stop],
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise ValueError(
                    "FEREBUS model training X/Y do not match the bound CSV"
                )
            offset = stop
        if incumbent_model is not None and incumbent_state is not None:
            incumbent_predictions = np.asarray(
                incumbent_model.predict(features),
                dtype=float,
            ).reshape(-1)
            _update_metric_state(
                incumbent_state,
                targets,
                incumbent_predictions,
            )
    if bind_training_data and (
        offset != expected_x.shape[0] or offset != expected_y.shape[0]
    ):
        raise ValueError("FEREBUS model training row count disagrees with the CSV")
    return (
        _finish_metric_state(candidate_state),
        (
            None
            if incumbent_state is None
            else _finish_metric_state(incumbent_state)
        ),
        int(candidate_state["n"]),
    )


def _parse_perf(path: Path) -> Dict[str, float]:
    perf_path = Path(path)
    if perf_path.is_symlink() or not perf_path.is_file():
        raise ValueError("FEREBUS performance receipt is missing: " + str(perf_path))
    values: Dict[str, float] = {}
    for line_number, raw in enumerate(
        perf_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(
                "FEREBUS performance receipt row has invalid cardinality at line "
                + str(line_number)
            )
        raw_key = parts[0]
        key = _PERFORMANCE_METRIC_ALIASES.get(raw_key, raw_key)
        if key in values:
            raise ValueError(
                "duplicate or ambiguous FEREBUS performance metric "
                + repr(key)
            )
        try:
            value = float(parts[1])
        except ValueError as exc:
            raise ValueError("non-numeric FEREBUS performance metric " + repr(key)) from exc
        if not math.isfinite(value):
            raise ValueError("non-finite FEREBUS performance metric " + repr(key))
        values[key] = value
    required = {"RMSE", "MAE", "covariance_condition_number"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError("FEREBUS performance receipt is missing " + repr(missing))
    return values


def _aggregate_iqa_rmse(records: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    squared_error_sum = 0.0
    rows = 0
    for record in records:
        if str(record.get("property") or "") != "iqa":
            continue
        if field == "candidate":
            metric = record.get("metrics", {}).get("ext_val", {})
        else:
            metric = record.get("incumbent_ext_metrics") or {}
        if not isinstance(metric, Mapping):
            continue
        row_counts = record.get("row_counts")
        if not isinstance(row_counts, Mapping):
            continue
        count = _exact_int(
            row_counts.get("ext_val"),
            "FEREBUS external-validation row count",
        )
        try:
            rmse = float(metric.get("rmse"))
        except (TypeError, ValueError):
            continue
        if count <= 0 or not math.isfinite(rmse):
            continue
        squared_error_sum += rmse * rmse * count
        rows += count
    return math.sqrt(squared_error_sum / rows) if rows else None


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalised_symbol_source(symbol: Any) -> str:
    tree = ast.parse(inspect.getsource(symbol))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


@lru_cache(maxsize=1)
def ferebus_quality_evaluator_identity() -> Dict[str, Any]:
    """Return a path-independent identity for the exact metric producer."""
    from .ferebus_dataset import iter_feature_target_chunks
    from ..ferebus_prior import validate_model_prior_mean

    try:
        import scipy

        scipy_version = str(scipy.__version__)
    except Exception:
        scipy_version = None
    material = {
        "kind": FEREBUS_TASK_QUALITY_KIND,
        "prediction_chunk_size": int(_PREDICTION_CHUNK_SIZE),
        "numpy_version": str(np.__version__),
        "scipy_version": scipy_version,
        "symbols": [
            _normalised_symbol_source(_new_metric_state),
            _normalised_symbol_source(_update_metric_state),
            _normalised_symbol_source(_finish_metric_state),
            _normalised_symbol_source(_stream_model_metrics),
            _normalised_symbol_source(_parse_perf),
            _normalised_symbol_source(iter_feature_target_chunks),
            _normalised_symbol_source(validate_model_prior_mean),
        ],
    }
    return {
        "kind": FEREBUS_TASK_QUALITY_KIND,
        "fingerprint_sha256": _canonical_sha256(material),
        "numpy_version": material["numpy_version"],
        "scipy_version": material["scipy_version"],
        "prediction_chunk_size": int(_PREDICTION_CHUNK_SIZE),
    }


def _emit_quality_progress(
    callback: QualityProgressCallback,
    stage: str,
    **fields: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), dict(fields))
    except Exception:
        pass


def build_ferebus_quality_context(
    staging_dir: Path,
    *,
    incumbent_set: Any = _AUTO_INCUMBENT,
) -> FerebusQualityContext:
    """Authenticate staging once and bind the incumbent without chain replay."""
    from . import input_staging as _stg
    from .ferebus_task_runner import validate_task_receipts

    staging = Path(staging_dir).resolve()
    manifest = _stg.read_ferebus_manifest(staging)
    try:
        task_execution = validate_task_receipts(staging)
    except Exception as exc:
        raise FerebusQualityDecisionError(
            "FEREBUS task execution evidence is invalid: " + str(exc)
        ) from exc
    reference_version = int(manifest.get("reference_data_version", -1))
    selected_incumbent = incumbent_set
    if selected_incumbent is _AUTO_INCUMBENT:
        if reference_version > 0:
            from ..versioning.trained_models import resolve_trained_model_set

            selected_incumbent = resolve_trained_model_set(
                staging.parent.parent,
                reference_version - 1,
                verification="metadata",
            )
        else:
            selected_incumbent = None
    if reference_version == 0 and selected_incumbent is not None:
        raise FerebusQualityDecisionError(
            "bootstrap FEREBUS quality must not bind an incumbent model set"
        )
    if reference_version > 0:
        if selected_incumbent is None:
            raise FerebusQualityDecisionError(
                "active FEREBUS quality requires an incumbent model set"
            )
        if (
            int(selected_incumbent.version) != reference_version - 1
            or str(selected_incumbent.campaign_uid)
            != str(manifest.get("campaign_uid") or "")
        ):
            raise FerebusQualityDecisionError(
                "FEREBUS incumbent model-set authority mismatch"
            )
    return FerebusQualityContext(
        staging=staging,
        task_manifest=manifest,
        task_execution=task_execution,
        incumbent_set=selected_incumbent,
    )


def _raw_task_receipt(
    staging: Path,
    logical_task_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    from .ferebus_task_runner import _read_task_map, validate_task_receipt

    root = Path(staging).resolve()
    normalised = validate_task_receipt(root, int(logical_task_id))
    task_map = _read_task_map(root / "FEREBUS_TASK_MAP.json")
    task = task_map["tasks"][int(logical_task_id)]
    receipt_path = root.joinpath(*str(task["receipt_path"]).split("/"))
    receipt = _read_json_object(receipt_path, "FEREBUS task receipt")
    return normalised, receipt, task_map, task


def _measurement_dataset_bindings(
    manifest_task: Mapping[str, Any],
    map_task: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    manifest_datasets = manifest_task.get("datasets")
    map_datasets = map_task.get("datasets")
    if not isinstance(manifest_datasets, Mapping) or not isinstance(
        map_datasets, Mapping
    ):
        raise FerebusQualityDecisionError("FEREBUS measurement datasets are invalid")
    bindings: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "int_val", "ext_val"):
        manifest_record = manifest_datasets.get(split)
        map_record = map_datasets.get(split)
        if not isinstance(manifest_record, Mapping) or not isinstance(
            map_record, Mapping
        ):
            raise FerebusQualityDecisionError(
                "FEREBUS measurement dataset binding is missing for " + split
            )
        for field in ("path", "size", "sha256"):
            if map_record.get(field) != manifest_record.get(field):
                raise FerebusQualityDecisionError(
                    "FEREBUS measurement dataset binding mismatch for " + split
                )
        bindings[split] = {
            "path": str(manifest_record.get("path") or ""),
            "size": _exact_int(
                manifest_record.get("size"),
                "FEREBUS " + split + " dataset size",
            ),
            "sha256": str(manifest_record.get("sha256") or ""),
            "row_identity_sha256": str(
                manifest_record.get("row_identity_sha256") or ""
            ),
            "row_identity_count": _exact_int(
                manifest_record.get("row_identity_count"),
                "FEREBUS " + split + " row-identity count",
            ),
        }
    return bindings


def measure_ferebus_task(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute candidate metrics for one authenticated successful task."""
    from ichor.core.models import Model
    from . import input_staging as _stg
    from .ferebus_task_runner import _validate_input
    from ..ferebus_prior import contract_from_payload, validate_model_prior_mean

    staging = Path(staging_dir).resolve()
    normalised, receipt, task_map, map_task = _raw_task_receipt(
        staging, int(logical_task_id)
    )
    manifest = _stg.read_ferebus_manifest(
        staging,
        verify_dataset_files=False,
    )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or int(logical_task_id) >= len(tasks):
        raise FerebusQualityDecisionError("FEREBUS measurement task is out of range")
    task = tasks[int(logical_task_id)]
    if (
        not isinstance(task, Mapping)
        or task.get("task_index") != map_task.get("task_index")
        or task.get("property") != map_task.get("property")
        or task.get("atom") != map_task.get("atom")
    ):
        raise FerebusQualityDecisionError("FEREBUS measurement task identity mismatch")
    datasets = _measurement_dataset_bindings(task, map_task)
    for split in ("train", "int_val", "ext_val"):
        _validate_input(
            staging,
            map_task["datasets"][split],
            "FEREBUS " + split,
        )
    model_path = _stg.resolve_ferebus_task_path(
        staging, normalised["model_path"], "measurement model"
    )
    if model is None:
        model = Model(model_path)
    elif (
        str(getattr(model, "atom", getattr(model, "atom_name", "")))
        != str(task.get("atom") or "")
        or str(getattr(model, "prop", "")) != str(task.get("property") or "")
    ):
        raise FerebusQualityDecisionError(
            "FEREBUS supplied measurement model identity mismatch"
        )
    performance = None
    performance_path = None
    if task_map.get("performance_required") is True:
        performance_path = _stg.resolve_ferebus_task_path(
            staging,
            normalised["performance_path"],
            "measurement performance receipt",
        )
        performance = _parse_perf(performance_path)
    section_metrics: Dict[str, Dict[str, float]] = {}
    row_counts: Dict[str, int] = {}
    for section, key in (
        ("train", "training_csv"),
        ("int_val", "int_validation_csv"),
        ("ext_val", "ext_validation_csv"),
    ):
        csv_path = _stg.resolve_ferebus_task_path(staging, task[key], key)
        metrics, unused_incumbent, n_rows = _stream_model_metrics(
            csv_path,
            str(task.get("property") or ""),
            model,
            bind_training_data=(section == "train"),
        )
        del unused_incumbent
        section_metrics[section] = metrics
        row_counts[section] = int(n_rows)
    prior_contract = contract_from_payload(manifest.get("prior_mean_contract"))
    prior_evidence = validate_model_prior_mean(
        model,
        contract=prior_contract,
        property_name=str(task.get("property") or ""),
        atom=str(task.get("atom") or ""),
        training_values=np.asarray(model.y, dtype=float).reshape(-1),
    )
    return {
        "kind": FEREBUS_TASK_QUALITY_KIND,
        "evaluator_identity": ferebus_quality_evaluator_identity(),
        "task_map_sha256": str(task_map["task_map_sha256"]),
        "task_manifest_sha256": str(task_map["task_manifest_sha256"]),
        "task_index": int(task["task_index"]),
        "property": str(task["property"]),
        "atom": str(task["atom"]),
        "model": dict(receipt["model"]),
        "performance": (
            None if receipt.get("performance") is None else dict(receipt["performance"])
        ),
        "datasets": datasets,
        "prior_mean": prior_evidence,
        "row_counts": row_counts,
        "condition_number": (
            None
            if performance is None
            else float(performance["covariance_condition_number"])
        ),
        "native_performance": performance,
        "metrics": section_metrics,
        "numerical_threads": 1,
    }


def validate_ferebus_task_measurement(
    staging_dir: Path,
    logical_task_id: int,
    measurement: Mapping[str, Any],
    *,
    context: Optional[FerebusQualityContext] = None,
) -> Dict[str, Any]:
    """Validate optional task evidence without invalidating its base receipt."""
    from . import input_staging as _stg

    if not isinstance(measurement, Mapping):
        raise FerebusQualityDecisionError("FEREBUS task quality measurement is invalid")
    staging = Path(staging_dir).resolve()
    if context is not None and context.staging != staging:
        raise FerebusQualityDecisionError(
            "FEREBUS task quality context path mismatch"
        )
    normalised, receipt, task_map, map_task = _raw_task_receipt(
        staging, int(logical_task_id)
    )
    manifest = (
        context.task_manifest
        if context is not None
        else _stg.read_ferebus_manifest(staging)
    )
    task = manifest["tasks"][int(logical_task_id)]
    expected_datasets = _measurement_dataset_bindings(task, map_task)
    if (
        measurement.get("kind") != FEREBUS_TASK_QUALITY_KIND
        or measurement.get("evaluator_identity")
        != ferebus_quality_evaluator_identity()
        or measurement.get("task_map_sha256") != task_map.get("task_map_sha256")
        or measurement.get("task_manifest_sha256")
        != task_map.get("task_manifest_sha256")
        or measurement.get("task_index") != task.get("task_index")
        or measurement.get("property") != task.get("property")
        or measurement.get("atom") != task.get("atom")
        or measurement.get("model") != receipt.get("model")
        or measurement.get("performance") != receipt.get("performance")
        or measurement.get("datasets") != expected_datasets
        or measurement.get("numerical_threads") != 1
    ):
        raise FerebusQualityDecisionError(
            "FEREBUS task quality measurement identity mismatch"
        )
    expected_counts = task.get("row_counts")
    counts = measurement.get("row_counts")
    if not isinstance(expected_counts, Mapping) or not isinstance(counts, Mapping):
        raise FerebusQualityDecisionError("FEREBUS task quality row counts are invalid")
    if {
        split: _exact_int(counts.get(split), "FEREBUS measurement row count")
        for split in ("train", "int_val", "ext_val")
    } != {
        split: _exact_int(expected_counts.get(split), "FEREBUS task row count")
        for split in ("train", "int_val", "ext_val")
    }:
        raise FerebusQualityDecisionError("FEREBUS task quality row counts mismatch")
    metrics = measurement.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "train",
        "int_val",
        "ext_val",
    }:
        raise FerebusQualityDecisionError("FEREBUS task quality split coverage is invalid")
    for split in ("train", "int_val", "ext_val"):
        values = metrics.get(split)
        if not isinstance(values, Mapping) or set(values) != {"rmse", "mae", "r2"}:
            raise FerebusQualityDecisionError("FEREBUS task quality metrics are invalid")
        for name in ("rmse", "mae", "r2"):
            value = float(values[name])
            if not math.isfinite(value) or (name in {"rmse", "mae"} and value < 0.0):
                raise FerebusQualityDecisionError(
                    "FEREBUS task quality metric is invalid"
                )
    task_prior = task.get("prior_mean")
    prior = measurement.get("prior_mean")
    if (
        not isinstance(task_prior, Mapping)
        or not isinstance(prior, Mapping)
        or str(prior.get("contract_sha256") or "")
        != str(task_prior.get("contract_sha256") or "")
        or not _same_optional_metric(
            prior.get("expected_mean_ha"),
            _finite_metric(task_prior, "expected_mean_ha"),
        )
        or not _same_optional_metric(
            prior.get("observed_mean_ha"),
            _finite_metric(task_prior, "expected_mean_ha"),
        )
        or prior.get("units") != "ha"
    ):
        raise FerebusQualityDecisionError(
            "FEREBUS task quality prior-mean binding mismatch"
        )
    if task_map.get("performance_required") is True:
        performance_path = _stg.resolve_ferebus_task_path(
            staging,
            normalised["performance_path"],
            "measurement performance receipt",
        )
        parsed = _parse_perf(performance_path)
        if measurement.get("native_performance") != parsed or not math.isclose(
            float(measurement.get("condition_number")),
            float(parsed["covariance_condition_number"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise FerebusQualityDecisionError(
                "FEREBUS task quality performance binding mismatch"
            )
    elif measurement.get("condition_number") is not None or measurement.get(
        "native_performance"
    ) is not None:
        raise FerebusQualityDecisionError(
            "imported FEREBUS task quality claims native performance"
        )
    return dict(measurement)


def enrich_task_receipt_with_quality(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Atomically add optional quality evidence to a successful base receipt."""
    staging = Path(staging_dir).resolve()
    unused_normalised, before, task_map, task = _raw_task_receipt(
        staging, int(logical_task_id)
    )
    del unused_normalised
    existing = before.get("quality_measurement")
    if existing is not None:
        return validate_ferebus_task_measurement(
            staging, int(logical_task_id), existing
        )
    measurement = measure_ferebus_task(
        staging,
        int(logical_task_id),
        model=model,
    )
    receipt_path = staging.joinpath(*str(task["receipt_path"]).split("/"))
    current = _read_json_object(receipt_path, "FEREBUS task receipt")
    if current != before:
        raise FerebusQualityDecisionError(
            "FEREBUS task receipt changed during quality measurement"
        )
    enriched = dict(current)
    enriched["quality_measurement"] = measurement
    atomic_write_json(receipt_path, enriched)
    validate_ferebus_task_measurement(staging, int(logical_task_id), measurement)
    return measurement


def _quality_cache_identity(
    context: FerebusQualityContext,
    logical_task_id: int,
) -> Dict[str, Any]:
    normalised, unused_receipt, task_map, map_task = _raw_task_receipt(
        context.staging, int(logical_task_id)
    )
    del unused_receipt
    task = context.task_manifest["tasks"][int(logical_task_id)]
    return {
        "kind": FEREBUS_TASK_QUALITY_KIND,
        "campaign_uid": str(context.task_manifest.get("campaign_uid") or ""),
        "reference_data_version": int(
            context.task_manifest.get("reference_data_version", -1)
        ),
        "task_map_sha256": str(task_map["task_map_sha256"]),
        "task_manifest_sha256": str(task_map["task_manifest_sha256"]),
        "task_index": int(task["task_index"]),
        "property": str(task["property"]),
        "atom": str(task["atom"]),
        "receipt_sha256": str(normalised["receipt_sha256"]),
        "model_sha256": str(normalised["model_sha256"]),
        "performance_sha256": normalised["performance_sha256"],
        "datasets": _measurement_dataset_bindings(task, map_task),
        "prior_mean_contract": dict(
            context.task_manifest.get("prior_mean_contract") or {}
        ),
        "evaluator_identity": ferebus_quality_evaluator_identity(),
    }


def _quality_cache_path(
    context: FerebusQualityContext,
    identity: Mapping[str, Any],
) -> Path:
    from .filesystem import campaign_owned_path

    campaign = context.staging.parent.parent.resolve()
    digest = _canonical_sha256(identity)
    return campaign_owned_path(
        campaign,
        campaign / ".DATA" / "CACHE" / "FEREBUS_QUALITY" / (digest + ".json"),
    )


def _read_quality_cache(
    context: FerebusQualityContext,
    logical_task_id: int,
) -> Optional[Dict[str, Any]]:
    identity = _quality_cache_identity(context, int(logical_task_id))
    path = _quality_cache_path(context, identity)
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise FerebusQualityDecisionError("FEREBUS quality cache path is unsafe")
    try:
        payload = _read_json_object(path, "FEREBUS quality cache")
        if (
            payload.get("schema_version") != FEREBUS_QUALITY_CACHE_SCHEMA_VERSION
            or payload.get("identity") != identity
            or payload.get("identity_sha256") != _canonical_sha256(identity)
        ):
            raise FerebusQualityDecisionError("FEREBUS quality cache identity mismatch")
        measurement = validate_ferebus_task_measurement(
            context.staging,
            int(logical_task_id),
            payload.get("quality_measurement"),
            context=context,
        )
        return measurement
    except Exception:
        path.unlink()
        return None


def _write_quality_cache(
    context: FerebusQualityContext,
    logical_task_id: int,
    measurement: Mapping[str, Any],
) -> Path:
    identity = _quality_cache_identity(context, int(logical_task_id))
    path = _quality_cache_path(context, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise FerebusQualityDecisionError("FEREBUS quality cache root is symlinked")
    payload = {
        "schema_version": FEREBUS_QUALITY_CACHE_SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "quality_measurement": dict(measurement),
    }
    if path.exists() or path.is_symlink():
        existing = _read_quality_cache(context, int(logical_task_id))
        if existing != dict(measurement):
            raise FerebusQualityDecisionError("FEREBUS quality cache conflict")
        return path
    atomic_write_json(path, payload)
    return path


def _measure_task_cache_worker(staging_dir: Path, logical_task_id: int) -> Path:
    from . import input_staging as _stg
    from .ferebus_model_factors import publish_task_factor
    from ichor.core.models import Model

    staging = Path(staging_dir).resolve()
    context = FerebusQualityContext(
        staging=staging,
        task_manifest=_stg.read_ferebus_manifest(
            staging,
            verify_dataset_files=False,
        ),
        task_execution={},
        incumbent_set=None,
    )
    normalised, unused_receipt, unused_map, unused_task = _raw_task_receipt(
        context.staging,
        int(logical_task_id),
    )
    del unused_receipt, unused_map, unused_task
    model_path = _stg.resolve_ferebus_task_path(
        context.staging,
        normalised["model_path"],
        "measurement model",
    )
    model = Model(model_path)
    measurement = measure_ferebus_task(
        context.staging,
        int(logical_task_id),
        model=model,
    )
    path = _write_quality_cache(context, int(logical_task_id), measurement)
    try:
        publish_task_factor(
            context.staging,
            int(logical_task_id),
            model=model,
        )
    except Exception:
        # Optional factor evidence cannot invalidate exact quality evidence.
        pass
    return path


def _run_measurement_subprocess(staging: Path, logical_task_id: int) -> None:
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
            "ichor.hpc.active_learning.daemon.ferebus_quality",
            "--measure-cache",
            "--staging-dir",
            str(Path(staging).resolve()),
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
        raise FerebusQualityDecisionError(
            "isolated FEREBUS quality measurement failed: " + diagnostic[-1000:]
        )


def _bound_model_root_json(model_set: Any, filename: str) -> Dict[str, Any]:
    from ..versioning.manifest import sha256_file

    records = [
        record
        for record in model_set.root_files
        if str(record.relative_path) == str(filename)
    ]
    if len(records) != 1:
        raise FerebusQualityDecisionError(
            "incumbent model set has no unique " + str(filename)
        )
    record = records[0]
    path = Path(record.path)
    if (
        path.is_symlink()
        or not path.is_file()
        or int(path.stat().st_size) != int(record.size)
        or sha256_file(path) != str(record.sha256)
    ):
        raise FerebusQualityDecisionError(
            "incumbent model-set evidence changed: " + str(filename)
        )
    return _read_json_object(path, "incumbent " + str(filename))


def _incumbent_metric_reuse_index(
    context: FerebusQualityContext,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    incumbent_set = context.incumbent_set
    if incumbent_set is None:
        return {}
    try:
        task_manifest = _bound_model_root_json(incumbent_set, "FEREBUS_TASKS.json")
        quality = _bound_model_root_json(
            incumbent_set, FEREBUS_QUALITY_MANIFEST
        )
        if (
            quality.get("measurement_complete") is not True
            or int(quality.get("reference_data_version", -1))
            != int(incumbent_set.reference_data_version)
        ):
            return {}
        manifest_tasks = task_manifest.get("tasks")
        quality_records = quality.get("records")
        if not isinstance(manifest_tasks, list) or not isinstance(
            quality_records, list
        ):
            return {}
        manifest_by_key = {
            (str(task.get("property") or ""), str(task.get("atom") or "")): task
            for task in manifest_tasks
            if isinstance(task, Mapping)
        }
        quality_by_key = {
            (str(record.get("property") or ""), str(record.get("atom") or "")): record
            for record in quality_records
            if isinstance(record, Mapping)
        }
        incumbent_tasks = {task.key: task for task in incumbent_set.tasks}
        result: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for current_task in context.task_manifest.get("tasks", []):
            key = (
                str(current_task.get("property") or ""),
                str(current_task.get("atom") or ""),
            )
            previous_task = manifest_by_key.get(key)
            previous_quality = quality_by_key.get(key)
            incumbent_task = incumbent_tasks.get(key)
            if (
                not isinstance(previous_task, Mapping)
                or not isinstance(previous_quality, Mapping)
                or incumbent_task is None
            ):
                continue
            current_ext = (current_task.get("datasets") or {}).get("ext_val")
            previous_ext = (previous_task.get("datasets") or {}).get("ext_val")
            metrics = (previous_quality.get("metrics") or {}).get("ext_val")
            if (
                not isinstance(current_ext, Mapping)
                or not isinstance(previous_ext, Mapping)
                or not isinstance(metrics, Mapping)
                or previous_quality.get("model_sha256")
                != incumbent_task.model.sha256
                or previous_ext.get("sha256") != current_ext.get("sha256")
                or previous_ext.get("row_identity_sha256")
                != current_ext.get("row_identity_sha256")
                or previous_ext.get("row_identity_count")
                != current_ext.get("row_identity_count")
                or (previous_quality.get("row_counts") or {}).get("ext_val")
                != (current_task.get("row_counts") or {}).get("ext_val")
            ):
                continue
            validated_metrics = {
                name: _finite_metric(previous_quality, "metrics", "ext_val", name)
                for name in ("rmse", "mae", "r2")
            }
            if validated_metrics["rmse"] < 0.0 or validated_metrics["mae"] < 0.0:
                continue
            result[key] = validated_metrics
        return result
    except Exception:
        return {}


def _local_incumbent_metric(
    staging: Path,
    task: Mapping[str, Any],
    incumbent_task: Any,
) -> Dict[str, float]:
    from ichor.core.models import Model
    from . import input_staging as _stg

    incumbent = Model(incumbent_task.model.path)
    ext_path = _stg.resolve_ferebus_task_path(
        staging, task["ext_validation_csv"], "ext_validation_csv"
    )
    metrics, unused, unused_count = _stream_model_metrics(
        ext_path,
        str(task.get("property") or ""),
        incumbent,
    )
    del unused, unused_count
    return metrics


def evaluate_ferebus_quality(
    staging_dir: Path,
    gates: Any = None,
    *,
    context: Optional[FerebusQualityContext] = None,
    progress_callback: QualityProgressCallback = None,
    measurement_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    from . import input_staging as _stg
    from ..ferebus_prior import contract_from_payload
    from ..versioning.manifest import sha256_file

    del gates
    quality_context = context or build_ferebus_quality_context(staging_dir)
    staging = quality_context.staging
    manifest = quality_context.task_manifest
    task_receipts = quality_context.task_execution
    prior_contract = contract_from_payload(manifest.get("prior_mean_contract"))
    tasks = list(manifest.get("tasks", []))
    stats = {
        "inline": 0,
        "cached": 0,
        "local": 0,
        "incumbent_reused": 0,
        "incumbent_local": 0,
    }
    measurements: Dict[int, Dict[str, Any]] = {}
    missing: List[int] = []
    _emit_quality_progress(
        progress_callback,
        "ferebus_task_quality",
        completed=0,
        total=len(tasks),
        unit="models",
    )
    for logical_task_id in range(len(tasks)):
        unused_normalised, receipt, unused_map, unused_task = _raw_task_receipt(
            staging, logical_task_id
        )
        del unused_normalised, unused_map, unused_task
        raw_measurement = receipt.get("quality_measurement")
        if raw_measurement is not None:
            try:
                measurements[logical_task_id] = validate_ferebus_task_measurement(
                    staging,
                    logical_task_id,
                    raw_measurement,
                    context=quality_context,
                )
                stats["inline"] += 1
            except Exception:
                raw_measurement = None
        if logical_task_id not in measurements:
            cached = _read_quality_cache(quality_context, logical_task_id)
            if cached is None:
                missing.append(logical_task_id)
            else:
                measurements[logical_task_id] = cached
                stats["cached"] += 1
        _emit_quality_progress(
            progress_callback,
            "ferebus_task_quality",
            completed=logical_task_id + 1,
            total=len(tasks),
            unit="models",
        )
    if missing:
        _emit_quality_progress(
            progress_callback,
            "ferebus_local_quality",
            completed=0,
            total=len(missing),
            unit="models",
        )
        failures: Dict[int, Exception] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(missing)))) as pool:
            futures = {
                pool.submit(_run_measurement_subprocess, staging, task_id): task_id
                for task_id in missing
            }
            completed_count = 0
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    future.result()
                    cached = _read_quality_cache(quality_context, task_id)
                    if cached is None:
                        raise FerebusQualityDecisionError(
                            "isolated FEREBUS quality worker published no cache"
                        )
                    measurements[task_id] = cached
                    stats["local"] += 1
                except Exception as exc:
                    failures[task_id] = exc
                completed_count += 1
                _emit_quality_progress(
                    progress_callback,
                    "ferebus_local_quality",
                    completed=completed_count,
                    total=len(missing),
                    unit="models",
                )
        for task_id, exc in failures.items():
            measurements[task_id] = {
                "measurement_error": (
                    "ferebus_quality_metric_failed:"
                    + type(exc).__name__
                    + ":"
                    + str(exc)
                )
            }

    incumbent_set = quality_context.incumbent_set
    incumbent_tasks = (
        {} if incumbent_set is None else {task.key: task for task in incumbent_set.tasks}
    )
    reusable_incumbent = _incumbent_metric_reuse_index(quality_context)
    _emit_quality_progress(
        progress_callback,
        "ferebus_incumbent_quality",
        completed=0,
        total=len(tasks),
        unit="models",
    )
    records: List[Dict[str, Any]] = []
    measurement_errors: List[str] = []
    for logical_task_id, task in enumerate(tasks):
        prop = str(task.get("property"))
        atom = str(task.get("atom"))
        model_path = _stg.resolve_ferebus_task_path(
            staging, task.get("expected_model_path"), "expected_model_path"
        )
        measurement = measurements.get(logical_task_id) or {}
        if "measurement_error" in measurement:
            reason = str(measurement["measurement_error"])
            measurement_errors.append(reason)
            records.append(
                {
                    "property": prop,
                    "atom": atom,
                    "model_path": _stg.ferebus_relative_path(staging, model_path),
                    "measurement_error": reason,
                }
            )
            continue
        try:
            validated = validate_ferebus_task_measurement(
                staging,
                logical_task_id,
                measurement,
                context=quality_context,
            )
            incumbent_metrics = None
            incumbent_binding = None
            if incumbent_set is not None:
                incumbent_task = incumbent_tasks.get((prop, atom))
                if incumbent_task is None:
                    raise ValueError("incumbent FEREBUS task coverage changed")
                incumbent_metrics = reusable_incumbent.get((prop, atom))
                if incumbent_metrics is None:
                    incumbent_metrics = _local_incumbent_metric(
                        staging, task, incumbent_task
                    )
                    stats["incumbent_local"] += 1
                else:
                    stats["incumbent_reused"] += 1
                incumbent_binding = {
                    "models_version": int(incumbent_set.version),
                    "model_set_sha256": str(incumbent_set.model_set_sha256),
                    "model_path": incumbent_task.model.relative_path,
                    "model_sha256": incumbent_task.model.sha256,
                    "holdout_row_identity_sha256": str(
                        task["datasets"]["ext_val"]["row_identity_sha256"]
                    ),
                }
            performance_binding = validated.get("performance")
            records.append(
                {
                    "property": prop,
                    "atom": atom,
                    "model_path": _stg.ferebus_relative_path(staging, model_path),
                    "model_sha256": str(validated["model"]["sha256"]),
                    "prior_mean": dict(validated["prior_mean"]),
                    "row_counts": dict(validated["row_counts"]),
                    "condition_number": validated.get("condition_number"),
                    "performance_path": (
                        None
                        if performance_binding is None
                        else str(performance_binding["path"])
                    ),
                    "performance_sha256": (
                        None
                        if performance_binding is None
                        else str(performance_binding["sha256"])
                    ),
                    "native_performance": validated.get("native_performance"),
                    "metrics": dict(validated["metrics"]),
                    "incumbent_ext_metrics": incumbent_metrics,
                    "incumbent_binding": incumbent_binding,
                }
            )
        except Exception as exc:
            reason = (
                "ferebus_quality_metric_failed:"
                + type(exc).__name__
                + ":"
                + str(exc)
            )
            measurement_errors.append(reason)
            records.append(
                {
                    "property": prop,
                    "atom": atom,
                    "model_path": _stg.ferebus_relative_path(staging, model_path),
                    "measurement_error": reason,
                }
            )
        _emit_quality_progress(
            progress_callback,
            "ferebus_incumbent_quality",
            completed=logical_task_id + 1,
            total=len(tasks),
            unit="models",
        )
    if measurement_stats is not None:
        measurement_stats.clear()
        measurement_stats.update(stats)
    summary = _quality_summary(records)
    return {
        "schema_version": FEREBUS_QUALITY_SCHEMA_VERSION,
        "campaign_uid": str(manifest.get("campaign_uid") or ""),
        "system": str(manifest.get("system")),
        "reference_data_version": int(manifest.get("reference_data_version", -1)),
        "reference_data_head_manifest_sha256": str(
            manifest.get("reference_data_head_manifest_sha256") or ""
        ),
        "reference_data_view_sha256": str(
            manifest.get("reference_data_view_sha256") or ""
        ),
        "source_task_manifest_sha256": sha256_file(
            _stg.ferebus_manifest_path(staging)
        ),
        "prior_mean_contract": prior_contract.to_dict(),
        "task_execution": task_receipts,
        "summary": summary,
        "records": records,
        "measurement_complete": not measurement_errors,
        "measurement_errors": sorted(set(measurement_errors)),
    }


def write_ferebus_quality_manifest(staging_dir: Path, payload: Mapping[str, Any]) -> Path:
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / FEREBUS_QUALITY_MANIFEST
    material = dict(payload)
    if material.get("measurement_complete") is not True:
        raise FerebusQualityMeasurementIncomplete(
            "incomplete FEREBUS measurement cannot become canonical quality evidence"
        )
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FerebusQualityDecisionError(
                "existing FEREBUS quality evidence is not a regular file"
            )
        existing = _read_json_object(path, "FEREBUS quality manifest")
        if existing != material:
            raise FerebusQualityDecisionError(
                "immutable FEREBUS quality evidence changed on re-evaluation"
            )
        return path
    atomic_write_json(path, material)
    return path


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FerebusQualityDecisionError(label + " is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusQualityDecisionError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise FerebusQualityDecisionError(label + " must be a JSON object")
    return payload


def _finite_metric(record: Mapping[str, Any], *path: str) -> float:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise FerebusQualityDecisionError(
                "FEREBUS quality metric is missing: " + ".".join(path)
            )
        value = value[key]
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FerebusQualityDecisionError(
            "FEREBUS quality metric is not numeric: " + ".".join(path)
        ) from exc
    if not math.isfinite(out):
        raise FerebusQualityDecisionError(
            "FEREBUS quality metric is non-finite: " + ".".join(path)
        )
    return out


def _same_optional_metric(observed: Any, expected: Optional[float]) -> bool:
    if expected is None:
        return observed is None
    if isinstance(observed, bool):
        return False
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value,
        float(expected),
        rel_tol=1.0e-14,
        abs_tol=1.0e-15,
    )


def _quality_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    measured = [record for record in records if isinstance(record.get("metrics"), Mapping)]
    ext_rmses = [
        _finite_metric(record, "metrics", "ext_val", "rmse")
        for record in measured
    ]
    ext_r2s = [
        _finite_metric(record, "metrics", "ext_val", "r2")
        for record in measured
    ]
    conditions = [
        _finite_metric(record, "condition_number")
        for record in measured
        if record.get("condition_number") is not None
    ]
    return {
        "n_tasks": len(records),
        "n_measured": len(measured),
        "n_measurement_failures": len(records) - len(measured),
        "mean_ext_rmse": float(np.mean(ext_rmses)) if ext_rmses else None,
        "min_ext_r2": float(np.min(ext_r2s)) if ext_r2s else None,
        "max_condition_number": (
            float(np.max(conditions)) if conditions else None
        ),
        "aggregate_iqa_ext_rmse": _aggregate_iqa_rmse(records, "candidate"),
        "incumbent_aggregate_iqa_ext_rmse": _aggregate_iqa_rmse(
            records,
            "incumbent",
        ),
    }


def validate_ferebus_quality_evidence(
    staging_dir: Path,
    *,
    context: Optional[FerebusQualityContext] = None,
) -> Dict[str, Any]:
    """Validate raw metric evidence and all model/task hashes it binds."""
    from . import input_staging as _stg
    from ..versioning.manifest import sha256_file

    quality_context = context or build_ferebus_quality_context(staging_dir)
    staging = quality_context.staging
    quality_path = staging / FEREBUS_QUALITY_MANIFEST
    quality = _read_json_object(quality_path, "FEREBUS quality manifest")
    if _exact_int(
        quality.get("schema_version"),
        "FEREBUS quality schema",
    ) != FEREBUS_QUALITY_SCHEMA_VERSION:
        raise FerebusQualityDecisionError("unsupported FEREBUS quality schema")
    task_path = _stg.ferebus_manifest_path(staging)
    if str(quality.get("source_task_manifest_sha256") or "") != sha256_file(task_path):
        raise FerebusQualityDecisionError("FEREBUS quality task-manifest hash mismatch")
    task_manifest = quality_context.task_manifest
    expected_execution = quality_context.task_execution
    if quality.get("task_execution") != expected_execution:
        raise FerebusQualityDecisionError("FEREBUS task-execution binding mismatch")
    for field in (
        "campaign_uid",
        "system",
        "reference_data_head_manifest_sha256",
        "reference_data_view_sha256",
    ):
        if str(quality.get(field) or "") != str(task_manifest.get(field) or ""):
            raise FerebusQualityDecisionError(
                "FEREBUS quality " + field + " binding mismatch"
            )
    reference_version = _exact_int(
        quality.get("reference_data_version"),
        "FEREBUS quality reference_data_version",
    )
    if reference_version != _exact_int(
        task_manifest.get("reference_data_version"),
        "FEREBUS task reference_data_version",
    ):
        raise FerebusQualityDecisionError(
            "FEREBUS quality reference-data version mismatch"
        )
    if quality.get("prior_mean_contract") != task_manifest.get(
        "prior_mean_contract"
    ):
        raise FerebusQualityDecisionError(
            "FEREBUS quality prior-mean contract mismatch"
        )
    records = quality.get("records")
    if not isinstance(records, list) or not records:
        raise FerebusQualityDecisionError("FEREBUS quality records are empty")
    tasks = task_manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(records):
        raise FerebusQualityDecisionError("FEREBUS quality/task cardinality mismatch")
    task_keys = [
        (str(task.get("property") or ""), str(task.get("atom") or ""))
        for task in tasks
        if isinstance(task, Mapping)
    ]
    record_keys = [
        (str(record.get("property") or ""), str(record.get("atom") or ""))
        for record in records
        if isinstance(record, Mapping)
    ]
    if len(task_keys) != len(tasks) or record_keys != task_keys:
        raise FerebusQualityDecisionError(
            "FEREBUS quality/task record order or coverage mismatch"
        )
    execution_records = expected_execution.get("receipts")
    if not isinstance(execution_records, list) or len(execution_records) != len(tasks):
        raise FerebusQualityDecisionError(
            "FEREBUS quality task-execution coverage mismatch"
        )
    incumbent_set = quality_context.incumbent_set
    incumbent_tasks: Dict[Tuple[str, str], Any] = {}
    if incumbent_set is not None:
        incumbent_tasks = {task.key: task for task in incumbent_set.tasks}

    failed_reasons: List[str] = []
    for task, receipt, record in zip(tasks, execution_records, records):
        if not isinstance(record, dict):
            raise FerebusQualityDecisionError("FEREBUS quality record must be an object")
        key = (str(record.get("property") or ""), str(record.get("atom") or ""))
        if not isinstance(task, Mapping) or not isinstance(receipt, Mapping):
            raise FerebusQualityDecisionError("FEREBUS quality task binding is invalid")
        expected_model_relative = str(task.get("expected_model_path") or "")
        if str(record.get("model_path") or "") != expected_model_relative:
            raise FerebusQualityDecisionError("FEREBUS quality model path mismatch")
        model_path = _stg.resolve_ferebus_task_path(
            staging,
            record.get("model_path"),
            "quality model_path",
        )
        if (
            _exact_int(receipt.get("task_index"), "FEREBUS receipt task index", minimum=1)
            != _exact_int(task.get("task_index"), "FEREBUS task index", minimum=1)
            or receipt.get("model_path") != expected_model_relative
            or str(receipt.get("model_sha256") or "") != sha256_file(model_path)
        ):
            raise FerebusQualityDecisionError(
                "FEREBUS quality execution/model binding mismatch"
            )
        expected_counts = task.get("row_counts")
        row_counts = record.get("row_counts")
        declared_model_hash = str(record.get("model_sha256") or "")
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            if (
                not isinstance(expected_counts, Mapping)
                or not isinstance(row_counts, Mapping)
                or set(row_counts) != {"train", "int_val", "ext_val"}
                or {
                    split: _exact_int(
                        row_counts.get(split),
                        "FEREBUS quality " + split + " row count",
                    )
                    for split in ("train", "int_val", "ext_val")
                }
                != {
                    split: _exact_int(
                        expected_counts.get(split),
                        "FEREBUS task " + split + " row count",
                    )
                    for split in ("train", "int_val", "ext_val")
                }
            ):
                raise FerebusQualityDecisionError(
                    "FEREBUS quality row-count binding mismatch"
                )
            if declared_model_hash != str(receipt.get("model_sha256") or ""):
                raise FerebusQualityDecisionError("FEREBUS quality model hash mismatch")
            prior = record.get("prior_mean")
            task_prior = task.get("prior_mean")
            if not isinstance(prior, Mapping) or not isinstance(task_prior, Mapping):
                raise FerebusQualityDecisionError(
                    "FEREBUS quality prior-mean evidence is invalid"
                )
            if (
                str(prior.get("contract_sha256") or "")
                != str(task_prior.get("contract_sha256") or "")
                or not _same_optional_metric(
                    prior.get("expected_mean_ha"),
                    _finite_metric(task_prior, "expected_mean_ha"),
                )
                or not _same_optional_metric(
                    prior.get("observed_mean_ha"),
                    _finite_metric(task_prior, "expected_mean_ha"),
                )
                or prior.get("units") != "ha"
            ):
                raise FerebusQualityDecisionError(
                    "FEREBUS quality prior-mean evidence mismatch"
                )
            if expected_execution.get("performance_required") is True:
                condition = _finite_metric(record, "condition_number")
                if condition < 0.0:
                    raise FerebusQualityDecisionError(
                        "FEREBUS condition number is negative"
                    )
                performance_path = _stg.resolve_ferebus_task_path(
                    staging,
                    record.get("performance_path"),
                    "quality performance_path",
                )
                if (
                    record.get("performance_path") != receipt.get("performance_path")
                    or performance_path.is_symlink()
                    or not performance_path.is_file()
                    or str(record.get("performance_sha256") or "")
                    != str(receipt.get("performance_sha256") or "")
                    or str(record.get("performance_sha256") or "")
                    != sha256_file(performance_path)
                ):
                    raise FerebusQualityDecisionError(
                        "FEREBUS quality performance receipt mismatch"
                    )
                parsed_performance = _parse_perf(performance_path)
                if record.get("native_performance") != parsed_performance:
                    raise FerebusQualityDecisionError(
                        "FEREBUS native performance metrics changed"
                    )
                if not math.isclose(
                    condition,
                    float(parsed_performance["covariance_condition_number"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise FerebusQualityDecisionError(
                        "FEREBUS condition number disagrees with native evidence"
                    )
            elif any(
                record.get(key) is not None
                for key in (
                    "condition_number",
                    "performance_path",
                    "performance_sha256",
                    "native_performance",
                )
            ):
                raise FerebusQualityDecisionError(
                    "imported FEREBUS quality claims native performance evidence"
                )
            if set(metrics) != {"train", "int_val", "ext_val"}:
                raise FerebusQualityDecisionError(
                    "FEREBUS quality metric split coverage is invalid"
                )
            for split in ("train", "int_val", "ext_val"):
                split_metrics = metrics.get(split)
                if not isinstance(split_metrics, Mapping) or set(split_metrics) != {
                    "rmse",
                    "mae",
                    "r2",
                }:
                    raise FerebusQualityDecisionError(
                        "FEREBUS quality metric set is invalid for " + split
                    )
                for metric in ("rmse", "mae", "r2"):
                    value = _finite_metric(record, "metrics", split, metric)
                    if metric in {"rmse", "mae"} and value < 0.0:
                        raise FerebusQualityDecisionError(
                            "FEREBUS " + metric + " is negative"
                        )
            incumbent_metrics = record.get("incumbent_ext_metrics")
            incumbent_binding = record.get("incumbent_binding")
            if incumbent_set is None:
                if incumbent_metrics is not None or incumbent_binding is not None:
                    raise FerebusQualityDecisionError(
                        "bootstrap FEREBUS quality claims incumbent evidence"
                    )
            else:
                incumbent_task = incumbent_tasks.get(key)
                if (
                    incumbent_task is None
                    or not isinstance(incumbent_metrics, Mapping)
                    or set(incumbent_metrics) != {"rmse", "mae", "r2"}
                    or not isinstance(incumbent_binding, Mapping)
                ):
                    raise FerebusQualityDecisionError(
                        "FEREBUS incumbent comparison evidence is incomplete"
                    )
                for metric in ("rmse", "mae", "r2"):
                    value = _finite_metric(record, "incumbent_ext_metrics", metric)
                    if metric in {"rmse", "mae"} and value < 0.0:
                        raise FerebusQualityDecisionError(
                            "FEREBUS incumbent " + metric + " is negative"
                        )
                ext_identity = (task.get("datasets") or {}).get("ext_val") or {}
                if (
                    _exact_int(
                        incumbent_binding.get("models_version"),
                        "FEREBUS incumbent models_version",
                    )
                    != incumbent_set.version
                    or incumbent_binding.get("model_set_sha256")
                    != incumbent_set.model_set_sha256
                    or incumbent_binding.get("model_path")
                    != incumbent_task.model.relative_path
                    or incumbent_binding.get("model_sha256")
                    != incumbent_task.model.sha256
                    or incumbent_binding.get("holdout_row_identity_sha256")
                    != ext_identity.get("row_identity_sha256")
                ):
                    raise FerebusQualityDecisionError(
                        "FEREBUS incumbent comparison binding mismatch"
                    )
        else:
            reason = record.get("measurement_error")
            if not isinstance(reason, str) or not reason:
                raise FerebusQualityDecisionError(
                    "failed FEREBUS quality record has no diagnostic reason"
                )
            failed_reasons.append(reason)
            if any(
                record.get(field) is not None
                for field in (
                    "row_counts",
                    "condition_number",
                    "performance_path",
                    "performance_sha256",
                    "native_performance",
                    "incumbent_ext_metrics",
                    "incumbent_binding",
                )
            ):
                raise FerebusQualityDecisionError(
                    "failed FEREBUS quality record claims measured evidence"
                )
    errors = quality.get("measurement_errors")
    if (
        not isinstance(errors, list)
        or any(not isinstance(reason, str) or not reason for reason in errors)
        or errors != sorted(set(failed_reasons))
        or not isinstance(quality.get("measurement_complete"), bool)
        or quality.get("measurement_complete") != (len(failed_reasons) == 0)
    ):
        raise FerebusQualityDecisionError(
            "FEREBUS quality measurement-completion summary is invalid"
        )
    summary = quality.get("summary")
    expected_summary = _quality_summary(records)
    if not isinstance(summary, Mapping) or set(summary) != set(expected_summary):
        raise FerebusQualityDecisionError("FEREBUS quality summary is invalid")
    for field in ("n_tasks", "n_measured", "n_measurement_failures"):
        if _exact_int(summary.get(field), "FEREBUS summary " + field) != int(
            expected_summary[field]
        ):
            raise FerebusQualityDecisionError(
                "FEREBUS quality summary " + field + " mismatch"
            )
    for field in (
        "mean_ext_rmse",
        "min_ext_r2",
        "max_condition_number",
        "aggregate_iqa_ext_rmse",
        "incumbent_aggregate_iqa_ext_rmse",
    ):
        if not _same_optional_metric(summary.get(field), expected_summary[field]):
            raise FerebusQualityDecisionError(
                "FEREBUS quality summary " + field + " mismatch"
            )
    return quality


def evaluate_ferebus_quality_decision(
    quality: Mapping[str, Any],
    gates: Any,
) -> Dict[str, Any]:
    """Apply current policy thresholds to immutable raw FEREBUS metrics."""
    if quality.get("measurement_complete") is not True:
        errors = quality.get("measurement_errors")
        detail = (
            ";".join(str(value) for value in errors[:3])
            if isinstance(errors, list) and errors
            else "measurement evidence is incomplete"
        )
        raise FerebusQualityMeasurementIncomplete(
            "FEREBUS quality measurement is incomplete: " + detail[:300]
        )
    min_ext_r2 = _threshold(
        gates,
        "ferebus_min_ext_r2",
        allow_negative=True,
    )
    max_ext_rmse = _threshold(gates, "ferebus_max_ext_rmse_ha")
    max_cond = _threshold(gates, "ferebus_max_condition_number")
    aggregate_increase = _threshold(
        gates, "ferebus_max_aggregate_ext_rmse_increase_fraction"
    )
    task_increase = _threshold(
        gates, "ferebus_max_task_ext_rmse_increase_fraction"
    )
    regression_abs = _threshold(gates, "ferebus_regression_abs_tolerance_ha")
    if aggregate_increase is None or task_increase is None or regression_abs is None:
        raise FerebusQualityDecisionError(
            "FEREBUS incumbent-promotion thresholds must be finite"
        )
    task_decisions: List[Dict[str, Any]] = []
    all_reasons: List[str] = []
    all_warnings: List[str] = []
    for record in list(quality.get("records") or []):
        reasons: List[str] = []
        warnings: List[str] = []
        metrics = record.get("metrics") if isinstance(record, Mapping) else None
        if not isinstance(metrics, Mapping):
            reason = record.get("measurement_error")
            reasons.append(
                str(reason) if isinstance(reason, str) and reason else "ferebus_quality_metric_missing"
            )
        else:
            ext_r2 = _finite_metric(record, "metrics", "ext_val", "r2")
            ext_rmse = _finite_metric(record, "metrics", "ext_val", "rmse")
            condition = None
            if record.get("condition_number") is not None:
                condition = _finite_metric(record, "condition_number")
            if min_ext_r2 is not None and ext_r2 < min_ext_r2:
                reasons.append("ferebus_ext_r2_below_threshold")
            if (
                str(record.get("property") or "") == "iqa"
                and max_ext_rmse is not None
                and ext_rmse > max_ext_rmse
            ):
                reasons.append("ferebus_ext_rmse_threshold_exceeded")
            if max_cond is not None:
                if condition is None:
                    reasons.append("ferebus_condition_number_evidence_missing")
                elif condition > max_cond:
                    reasons.append("ferebus_condition_number_threshold_exceeded")
            incumbent_metrics = record.get("incumbent_ext_metrics")
            incumbent_binding = record.get("incumbent_binding")
            relative_limit = None
            if incumbent_metrics is not None:
                if not isinstance(incumbent_metrics, Mapping) or not isinstance(
                    incumbent_binding, Mapping
                ):
                    reasons.append("ferebus_incumbent_evidence_invalid")
                elif str(record.get("property") or "") == "iqa":
                    incumbent_rmse = _finite_metric(
                        record, "incumbent_ext_metrics", "rmse"
                    )
                    relative_limit = incumbent_rmse + max(
                        incumbent_rmse * task_increase,
                        regression_abs,
                    )
                    if ext_rmse > relative_limit:
                        warnings.append("ferebus_task_ext_rmse_regressed")
        all_reasons.extend(reasons)
        all_warnings.extend(warnings)
        task_decisions.append(
            {
                "property": str(record.get("property") or ""),
                "atom": str(record.get("atom") or ""),
                "accepted": not reasons,
                "reasons": reasons,
                "warnings": warnings,
                "candidate_ext_rmse": (
                    None if not isinstance(metrics, Mapping) else ext_rmse
                ),
                "incumbent_ext_rmse": (
                    None
                    if not isinstance(record.get("incumbent_ext_metrics"), Mapping)
                    else _finite_metric(record, "incumbent_ext_metrics", "rmse")
                ),
                "relative_rmse_limit": (
                    relative_limit if isinstance(metrics, Mapping) else None
                ),
            }
        )
    candidate_aggregate_raw = (quality.get("summary") or {}).get(
        "aggregate_iqa_ext_rmse"
    )
    if candidate_aggregate_raw is None:
        candidate_aggregate = None
        all_reasons.append("ferebus_aggregate_metric_missing")
    else:
        candidate_aggregate = _finite_metric(
            quality, "summary", "aggregate_iqa_ext_rmse"
        )
    incumbent_aggregate_raw = (quality.get("summary") or {}).get(
        "incumbent_aggregate_iqa_ext_rmse"
    )
    aggregate_limit = None
    if incumbent_aggregate_raw is not None:
        incumbent_aggregate = _finite_metric(
            quality, "summary", "incumbent_aggregate_iqa_ext_rmse"
        )
        aggregate_limit = incumbent_aggregate + max(
            incumbent_aggregate * aggregate_increase,
            regression_abs,
        )
        if candidate_aggregate is not None and candidate_aggregate > aggregate_limit:
            all_warnings.append("ferebus_aggregate_ext_rmse_regressed")
    else:
        incumbent_aggregate = None
    return {
        "decision_policy": FEREBUS_QUALITY_DECISION_POLICY,
        "thresholds": {
            "ferebus_min_ext_r2": min_ext_r2,
            "ferebus_max_ext_rmse_ha": max_ext_rmse,
            "ferebus_max_condition_number": max_cond,
            "ferebus_max_aggregate_ext_rmse_increase_fraction": aggregate_increase,
            "ferebus_max_task_ext_rmse_increase_fraction": task_increase,
            "ferebus_regression_abs_tolerance_ha": regression_abs,
        },
        "promotion": {
            "candidate_aggregate_iqa_ext_rmse": candidate_aggregate,
            "incumbent_aggregate_iqa_ext_rmse": incumbent_aggregate,
            "aggregate_rmse_limit": aggregate_limit,
            "bootstrap_without_incumbent": incumbent_aggregate is None,
        },
        "n_tasks": len(task_decisions),
        "n_accepted": sum(1 for item in task_decisions if item["accepted"]),
        "n_rejected": sum(1 for item in task_decisions if not item["accepted"]),
        "n_warned": sum(1 for item in task_decisions if item["warnings"]),
        "n_warnings": len(all_warnings),
        "accepted": not all_reasons,
        "reasons": sorted(set(all_reasons)),
        "warnings": sorted(set(all_warnings)),
        "tasks": task_decisions,
    }


def write_ferebus_quality_decision(
    staging_dir: Path,
    *,
    config_sha256: str,
    gates: Any,
    context: Optional[FerebusQualityContext] = None,
    validated_quality: Optional[Mapping[str, Any]] = None,
) -> Path:
    from .completion_receipts import canonical_sha256
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir)
    quality = (
        dict(validated_quality)
        if validated_quality is not None
        else validate_ferebus_quality_evidence(staging, context=context)
    )
    quality_path = staging / FEREBUS_QUALITY_MANIFEST
    evaluation = evaluate_ferebus_quality_decision(quality, gates)
    evaluation["config_sha256"] = str(config_sha256)
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    path = staging / FEREBUS_QUALITY_DECISION_MANIFEST
    evaluations: List[Dict[str, Any]] = []
    if path.is_file() and not path.is_symlink():
        existing = read_ferebus_quality_decision(
            staging,
            require_accepted=False,
            verify_current_config=False,
            context=context,
            validated_quality=quality,
        )
        if str(existing.get("campaign_uid") or "") != str(
            quality.get("campaign_uid") or ""
        ):
            raise FerebusQualityDecisionError("FEREBUS decision campaign UID changed")
        if str((existing.get("quality") or {}).get("sha256") or "") != sha256_file(
            quality_path
        ):
            raise FerebusQualityDecisionError("FEREBUS quality changed after decision")
        evaluations = [dict(item) for item in existing.get("evaluations", [])]
    digest = str(evaluation["evaluation_sha256"])
    if not any(str(item.get("evaluation_sha256") or "") == digest for item in evaluations):
        evaluation["evaluated_at_iso"] = datetime.now(timezone.utc).isoformat()
        evaluations.append(evaluation)
    payload = {
        "schema_version": FEREBUS_QUALITY_DECISION_SCHEMA_VERSION,
        "campaign_uid": str(quality.get("campaign_uid") or ""),
        "reference_data_version": int(quality.get("reference_data_version", -1)),
        "quality": {
            "path": FEREBUS_QUALITY_MANIFEST,
            "size": int(quality_path.stat().st_size),
            "sha256": sha256_file(quality_path),
        },
        "evaluations": evaluations,
        "current_evaluation_sha256": digest,
    }
    atomic_write_json(path, payload)
    return path


def read_ferebus_quality_decision(
    staging_dir: Path,
    *,
    expected_config_sha256: Optional[str] = None,
    require_accepted: bool = True,
    verify_current_config: bool = True,
    context: Optional[FerebusQualityContext] = None,
    validated_quality: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    from .completion_receipts import canonical_sha256
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir)
    path = staging / FEREBUS_QUALITY_DECISION_MANIFEST
    payload = _read_json_object(path, "FEREBUS quality decision")
    if _exact_int(payload.get("schema_version"), "FEREBUS decision schema") != FEREBUS_QUALITY_DECISION_SCHEMA_VERSION:
        raise FerebusQualityDecisionError("unsupported FEREBUS quality decision schema")
    quality = (
        dict(validated_quality)
        if validated_quality is not None
        else validate_ferebus_quality_evidence(staging, context=context)
    )
    binding = payload.get("quality")
    if not isinstance(binding, dict) or str(binding.get("path") or "") != FEREBUS_QUALITY_MANIFEST:
        raise FerebusQualityDecisionError("FEREBUS decision quality binding is invalid")
    quality_path = staging / FEREBUS_QUALITY_MANIFEST
    if _exact_int(
        binding.get("size"),
        "FEREBUS decision quality size",
    ) != int(quality_path.stat().st_size):
        raise FerebusQualityDecisionError("FEREBUS decision quality size mismatch")
    if str(binding.get("sha256") or "") != sha256_file(quality_path):
        raise FerebusQualityDecisionError("FEREBUS decision quality hash mismatch")
    if str(payload.get("campaign_uid") or "") != str(quality.get("campaign_uid") or ""):
        raise FerebusQualityDecisionError("FEREBUS decision campaign UID mismatch")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise FerebusQualityDecisionError("FEREBUS decision evaluations are empty")
    by_digest: Dict[str, Dict[str, Any]] = {}
    for raw in evaluations:
        if not isinstance(raw, dict):
            raise FerebusQualityDecisionError("FEREBUS evaluation must be an object")
        material = dict(raw)
        declared = str(material.pop("evaluation_sha256", ""))
        material.pop("evaluated_at_iso", None)
        if declared != canonical_sha256(material) or declared in by_digest:
            raise FerebusQualityDecisionError("FEREBUS evaluation digest is invalid")
        by_digest[declared] = dict(raw)
    current_digest = str(payload.get("current_evaluation_sha256") or "")
    if current_digest not in by_digest:
        raise FerebusQualityDecisionError("FEREBUS current evaluation is missing")
    current = by_digest[current_digest]
    if verify_current_config and expected_config_sha256 is not None and str(
        current.get("config_sha256") or ""
    ) != str(expected_config_sha256):
        raise FerebusQualityDecisionError("FEREBUS decision config digest mismatch")
    if require_accepted and not bool(current.get("accepted", False)):
        raise FerebusQualityDecisionError(
            "FEREBUS quality decision rejected: "
            + ";".join(str(reason) for reason in current.get("reasons", []))
        )
    out = dict(payload)
    out["current_evaluation"] = current
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-cache", action="store_true")
    parser.add_argument("--staging-dir")
    parser.add_argument("--logical-task-id", type=int)
    args = parser.parse_args(argv)
    if not args.measure_cache:
        parser.error("an internal measurement action is required")
    if args.staging_dir is None or args.logical_task_id is None:
        parser.error("--staging-dir and --logical-task-id are required")
    _measure_task_cache_worker(
        Path(args.staging_dir),
        int(args.logical_task_id),
    )
    return 0


__all__ = [
    "FEREBUS_QUALITY_DECISION_MANIFEST",
    "FEREBUS_QUALITY_DECISION_POLICY",
    "FEREBUS_QUALITY_DECISION_SCHEMA_VERSION",
    "FEREBUS_QUALITY_MANIFEST",
    "FEREBUS_RELATIVE_REGRESSION_WARNINGS",
    "FEREBUS_QUALITY_SCHEMA_VERSION",
    "FEREBUS_TASK_QUALITY_KIND",
    "FerebusQualityContext",
    "FerebusQualityMeasurementIncomplete",
    "FerebusQualityDecisionError",
    "build_ferebus_quality_context",
    "enrich_task_receipt_with_quality",
    "evaluate_ferebus_quality",
    "evaluate_ferebus_quality_decision",
    "ferebus_quality_evaluator_identity",
    "main",
    "measure_ferebus_task",
    "read_ferebus_quality_decision",
    "validate_ferebus_task_measurement",
    "validate_ferebus_quality_evidence",
    "write_ferebus_quality_decision",
    "write_ferebus_quality_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
