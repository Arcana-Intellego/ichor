"""FEREBUS held-out metric sidecar generation."""
from __future__ import annotations

import csv
from ..strict_json import strict_json as json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .state import atomic_write_json


FEREBUS_QUALITY_MANIFEST = "FEREBUS_QUALITY.json"
FEREBUS_QUALITY_SCHEMA_VERSION = 3
FEREBUS_QUALITY_DECISION_MANIFEST = "FEREBUS_QUALITY_DECISION.json"
FEREBUS_QUALITY_DECISION_SCHEMA_VERSION = 1


class FerebusQualityDecisionError(ValueError):
    """Raised when FEREBUS quality evidence or its decision is untrustworthy."""


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
    from ..ferebus_prior import (
        contract_from_payload,
        validate_model_prior_mean,
    )

    prior_contract = contract_from_payload(manifest.get("prior_mean_contract"))
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
            prior_evidence = validate_model_prior_mean(
                model,
                contract=prior_contract,
                property_name=prop,
                atom=atom,
            )
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
            if (
                prop == "iqa"
                and max_ext_rmse is not None
                and float(ext["rmse"]) > max_ext_rmse
            ):
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
                    "prior_mean": prior_evidence,
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
        "prior_mean_contract": prior_contract.to_dict(),
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


def validate_ferebus_quality_evidence(staging_dir: Path) -> Dict[str, Any]:
    """Validate raw metric evidence and all model/task hashes it binds."""
    from . import input_staging as _stg
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir)
    quality_path = staging / FEREBUS_QUALITY_MANIFEST
    quality = _read_json_object(quality_path, "FEREBUS quality manifest")
    if int(quality.get("schema_version", -1)) != FEREBUS_QUALITY_SCHEMA_VERSION:
        raise FerebusQualityDecisionError("unsupported FEREBUS quality schema")
    task_path = _stg.ferebus_manifest_path(staging)
    if str(quality.get("source_task_manifest_sha256") or "") != sha256_file(task_path):
        raise FerebusQualityDecisionError("FEREBUS quality task-manifest hash mismatch")
    task_manifest = _stg.read_ferebus_manifest(staging)
    if str(quality.get("campaign_uid") or "") != str(
        task_manifest.get("campaign_uid") or ""
    ):
        raise FerebusQualityDecisionError("FEREBUS quality campaign UID mismatch")
    records = quality.get("records")
    if not isinstance(records, list) or not records:
        raise FerebusQualityDecisionError("FEREBUS quality records are empty")
    task_keys = {
        (str(task.get("property") or ""), str(task.get("atom") or ""))
        for task in task_manifest.get("tasks", [])
    }
    record_keys = set()
    for record in records:
        if not isinstance(record, dict):
            raise FerebusQualityDecisionError("FEREBUS quality record must be an object")
        key = (str(record.get("property") or ""), str(record.get("atom") or ""))
        if key in record_keys:
            raise FerebusQualityDecisionError("duplicate FEREBUS quality task record")
        record_keys.add(key)
        model_path = _stg.resolve_ferebus_task_path(
            staging,
            record.get("model_path"),
            "quality model_path",
        )
        declared_model_hash = str(record.get("model_sha256") or "")
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            if not declared_model_hash or declared_model_hash != sha256_file(model_path):
                raise FerebusQualityDecisionError("FEREBUS quality model hash mismatch")
            _finite_metric(record, "condition_number")
            for split in ("train", "int_val", "ext_val"):
                for metric in ("rmse", "mae", "r2"):
                    _finite_metric(record, "metrics", split, metric)
        else:
            reasons = record.get("reasons")
            if not isinstance(reasons, list) or not reasons:
                raise FerebusQualityDecisionError(
                    "failed FEREBUS quality record has no diagnostic reason"
                )
    if record_keys != task_keys:
        raise FerebusQualityDecisionError("FEREBUS quality/task record set mismatch")
    return quality


def evaluate_ferebus_quality_decision(
    quality: Mapping[str, Any],
    gates: Any,
) -> Dict[str, Any]:
    """Apply current policy thresholds to immutable raw FEREBUS metrics."""
    min_ext_r2 = _threshold(gates, "ferebus_min_ext_r2")
    max_ext_rmse = _threshold(gates, "ferebus_max_ext_rmse_ha")
    max_cond = _threshold(gates, "ferebus_max_condition_number")
    task_decisions: List[Dict[str, Any]] = []
    all_reasons: List[str] = []
    for record in list(quality.get("records") or []):
        reasons: List[str] = []
        metrics = record.get("metrics") if isinstance(record, Mapping) else None
        if not isinstance(metrics, Mapping):
            reasons.extend(str(reason) for reason in (record.get("reasons") or []))
            if not reasons:
                reasons.append("ferebus_quality_metric_missing")
        else:
            ext_r2 = _finite_metric(record, "metrics", "ext_val", "r2")
            ext_rmse = _finite_metric(record, "metrics", "ext_val", "rmse")
            condition = _finite_metric(record, "condition_number")
            if min_ext_r2 is not None and ext_r2 < min_ext_r2:
                reasons.append("ferebus_ext_r2_below_threshold")
            if (
                str(record.get("property") or "") == "iqa"
                and max_ext_rmse is not None
                and ext_rmse > max_ext_rmse
            ):
                reasons.append("ferebus_ext_rmse_threshold_exceeded")
            if max_cond is not None and condition > max_cond:
                reasons.append("ferebus_condition_number_threshold_exceeded")
        all_reasons.extend(reasons)
        task_decisions.append(
            {
                "property": str(record.get("property") or ""),
                "atom": str(record.get("atom") or ""),
                "accepted": not reasons,
                "reasons": reasons,
            }
        )
    return {
        "thresholds": {
            "ferebus_min_ext_r2": min_ext_r2,
            "ferebus_max_ext_rmse_ha": max_ext_rmse,
            "ferebus_max_condition_number": max_cond,
        },
        "n_tasks": len(task_decisions),
        "n_accepted": sum(1 for item in task_decisions if item["accepted"]),
        "n_rejected": sum(1 for item in task_decisions if not item["accepted"]),
        "accepted": not all_reasons,
        "reasons": sorted(set(all_reasons)),
        "tasks": task_decisions,
    }


def write_ferebus_quality_decision(
    staging_dir: Path,
    *,
    config_sha256: str,
    gates: Any,
) -> Path:
    from .completion_receipts import canonical_sha256
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir)
    quality = validate_ferebus_quality_evidence(staging)
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
) -> Dict[str, Any]:
    from .completion_receipts import canonical_sha256
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir)
    path = staging / FEREBUS_QUALITY_DECISION_MANIFEST
    payload = _read_json_object(path, "FEREBUS quality decision")
    if int(payload.get("schema_version", -1)) != FEREBUS_QUALITY_DECISION_SCHEMA_VERSION:
        raise FerebusQualityDecisionError("unsupported FEREBUS quality decision schema")
    quality = validate_ferebus_quality_evidence(staging)
    binding = payload.get("quality")
    if not isinstance(binding, dict) or str(binding.get("path") or "") != FEREBUS_QUALITY_MANIFEST:
        raise FerebusQualityDecisionError("FEREBUS decision quality binding is invalid")
    quality_path = staging / FEREBUS_QUALITY_MANIFEST
    if int(binding.get("size", -1)) != int(quality_path.stat().st_size):
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


__all__ = [
    "FEREBUS_QUALITY_DECISION_MANIFEST",
    "FEREBUS_QUALITY_DECISION_SCHEMA_VERSION",
    "FEREBUS_QUALITY_MANIFEST",
    "FEREBUS_QUALITY_SCHEMA_VERSION",
    "FerebusQualityDecisionError",
    "evaluate_ferebus_quality",
    "evaluate_ferebus_quality_decision",
    "read_ferebus_quality_decision",
    "validate_ferebus_quality_evidence",
    "write_ferebus_quality_decision",
    "write_ferebus_quality_manifest",
]
