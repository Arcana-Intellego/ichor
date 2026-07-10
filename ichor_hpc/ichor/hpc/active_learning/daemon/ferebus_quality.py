"""FEREBUS held-out metric sidecar generation."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .state import atomic_write_json


FEREBUS_QUALITY_MANIFEST = "FEREBUS_QUALITY.json"
FEREBUS_QUALITY_SCHEMA_VERSION = 3


def _threshold(gates: Any, name: str) -> Optional[float]:
    value = getattr(gates, name, None)
    if value is None or isinstance(value, bool):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _read_features_and_target(csv_path: Path, prop: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("FEREBUS metric CSV is empty: " + str(path))
    header = [c.strip() for c in rows[0]]
    feature_cols = [i for i, name in enumerate(header) if name.startswith("f") and name[1:].isdigit()]
    if not feature_cols:
        raise ValueError("FEREBUS metric CSV has no feature columns: " + str(path))
    if prop not in header:
        raise ValueError("FEREBUS metric CSV missing property " + repr(prop) + ": " + str(path))
    prop_idx = header.index(prop)
    features: List[List[float]] = []
    targets: List[float] = []
    for row_no, row in enumerate(rows[1:], start=2):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise ValueError("FEREBUS metric CSV row length mismatch at row " + str(row_no))
        try:
            features.append([float(row[i]) for i in feature_cols])
            targets.append(float(row[prop_idx]))
        except ValueError as exc:
            raise ValueError("FEREBUS metric CSV non-numeric value at row " + str(row_no)) from exc
    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("FEREBUS metric CSV shape mismatch: " + str(path))
    if X.shape[0] == 0:
        raise ValueError("FEREBUS metric CSV has no rows: " + str(path))
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("FEREBUS metric CSV contains non-finite values: " + str(path))
    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(y_pred, dtype=float).reshape(-1) - np.asarray(y_true, dtype=float).reshape(-1)
    if residual.shape != y_true.reshape(-1).shape:
        raise ValueError("prediction/target row count mismatch")
    if not np.all(np.isfinite(residual)):
        raise ValueError("FEREBUS predictions contain non-finite values")
    mse = float(np.mean(residual * residual))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        r2 = 1.0 if rmse == 0.0 else 0.0
    else:
        r2 = 1.0 - float(np.sum(residual * residual)) / denom
    if not all(math.isfinite(v) for v in (rmse, mae, r2)):
        raise ValueError("FEREBUS metric is non-finite")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _condition_number(model: Any) -> float:
    value = float(np.linalg.cond(np.asarray(model.R, dtype=float)))
    if not math.isfinite(value):
        raise ValueError("FEREBUS model condition number is non-finite")
    return value


def evaluate_ferebus_quality(staging_dir: Path, gates: Any = None) -> Dict[str, Any]:
    from ichor.core.models import Model
    from . import input_staging as _stg
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir)
    manifest = _stg.read_ferebus_manifest(staging)
    tasks = list(manifest.get("tasks", []))
    records: List[Dict[str, Any]] = []
    reasons: List[str] = []
    min_ext_r2 = _threshold(gates, "ferebus_min_ext_r2")
    max_ext_rmse = _threshold(gates, "ferebus_max_ext_rmse_ha")
    max_cond = _threshold(gates, "ferebus_max_condition_number")

    for task in tasks:
        prop = str(task.get("property"))
        atom = str(task.get("atom"))
        model_path = _stg.resolve_ferebus_task_path(
            staging,
            task.get("expected_model_path"),
            "expected_model_path",
        )
        try:
            model = Model(model_path)
            model_sha256 = sha256_file(model_path)
            cond = _condition_number(model)
            section_metrics: Dict[str, Dict[str, float]] = {}
            row_counts: Dict[str, int] = {}
            for section, key in (
                ("train", "training_csv"),
                ("int_val", "int_validation_csv"),
                ("ext_val", "ext_validation_csv"),
            ):
                csv_path = _stg.resolve_ferebus_task_path(staging, task[key], key)
                X, y = _read_features_and_target(csv_path, prop)
                pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
                section_metrics[section] = _metrics(y, pred)
                row_counts[section] = int(y.shape[0])
            task_reasons: List[str] = []
            ext = section_metrics["ext_val"]
            if min_ext_r2 is not None and float(ext["r2"]) < min_ext_r2:
                task_reasons.append("ferebus_ext_r2_below_threshold")
            if max_ext_rmse is not None and float(ext["rmse"]) > max_ext_rmse:
                task_reasons.append("ferebus_ext_rmse_threshold_exceeded")
            if max_cond is not None and cond > max_cond:
                task_reasons.append("ferebus_condition_number_threshold_exceeded")
            reasons.extend(task_reasons)
            records.append(
                {
                    "property": prop,
                    "atom": atom,
                    "model_path": _stg.ferebus_relative_path(staging, model_path),
                    "model_sha256": model_sha256,
                    "row_counts": row_counts,
                    "condition_number": cond,
                    "metrics": section_metrics,
                    "accepted": not task_reasons,
                    "reasons": task_reasons,
                }
            )
        except Exception as exc:
            reason = "ferebus_quality_metric_failed:" + type(exc).__name__ + ":" + str(exc)
            reasons.append(reason)
            records.append(
                {
                    "property": prop,
                    "atom": atom,
                    "model_path": _stg.ferebus_relative_path(staging, model_path),
                    "accepted": False,
                    "reasons": [reason],
                }
            )

    ext_rmses = [
        float(r["metrics"]["ext_val"]["rmse"])
        for r in records
        if r.get("metrics") and "ext_val" in r["metrics"]
    ]
    ext_r2s = [
        float(r["metrics"]["ext_val"]["r2"])
        for r in records
        if r.get("metrics") and "ext_val" in r["metrics"]
    ]
    conds = [
        float(r["condition_number"])
        for r in records
        if r.get("condition_number") is not None
    ]
    summary = {
        "n_tasks": int(len(records)),
        "n_accepted": int(sum(1 for r in records if bool(r.get("accepted")))),
        "n_rejected": int(sum(1 for r in records if not bool(r.get("accepted")))),
        "mean_ext_rmse": float(np.mean(ext_rmses)) if ext_rmses else None,
        "min_ext_r2": float(np.min(ext_r2s)) if ext_r2s else None,
        "max_condition_number": float(np.max(conds)) if conds else None,
    }
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
        "summary": summary,
        "records": records,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def write_ferebus_quality_manifest(staging_dir: Path, payload: Mapping[str, Any]) -> Path:
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / FEREBUS_QUALITY_MANIFEST
    atomic_write_json(path, dict(payload))
    return path
