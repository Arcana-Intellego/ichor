"""Empirical uncertainty-to-error calibration for active learning.

The daemon already pays for the expensive pieces: model predictions before QM
and AIMAll IQA truth after QM. This module only joins those existing records and
fits a tiny monotone binned lookup table, so it is safe to run on every accepted
AIMAll batch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .state import atomic_write_json


ERROR_CALIBRATION_RECORDS_FILENAME = "error_calibration_records.json"
ERROR_CALIBRATION_MODEL_FILENAME = "error_calibration_model.json"
ERROR_CALIBRATION_AUDIT_FILENAME = "ERROR_CALIBRATION_AUDIT.json"
ERROR_CALIBRATION_SCHEMA_VERSION = 1


class ErrorCalibrationError(RuntimeError):
    """Raised when a calibration manifest is malformed."""


def _active_learning_dir(campaign_dir: Any) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING"


def records_path(campaign_dir: Any) -> Path:
    return _active_learning_dir(campaign_dir) / ERROR_CALIBRATION_RECORDS_FILENAME


def model_path(campaign_dir: Any) -> Path:
    return _active_learning_dir(campaign_dir) / ERROR_CALIBRATION_MODEL_FILENAME


def audit_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / ERROR_CALIBRATION_AUDIT_FILENAME


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _block(config: Any) -> Any:
    return getattr(config, "error_calibration", config)


def _enabled(config: Any) -> bool:
    return bool(_cfg_value(_block(config), "enabled", True))


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ErrorCalibrationError("calibration JSON unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ErrorCalibrationError("calibration JSON must be an object: " + str(path))
    return data


def load_records(campaign_dir: Any) -> List[Dict[str, Any]]:
    path = records_path(campaign_dir)
    if not path.is_file():
        return []
    data = _read_json_object(path)
    if int(data.get("schema_version", -1)) != ERROR_CALIBRATION_SCHEMA_VERSION:
        raise ErrorCalibrationError("unsupported calibration records schema")
    records = data.get("records")
    if not isinstance(records, list):
        raise ErrorCalibrationError("calibration records must be a list")
    return [dict(r) for r in records if isinstance(r, dict)]


def write_records(campaign_dir: Any, records: Sequence[Mapping[str, Any]]) -> Path:
    path = records_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
        "n_records": int(len(records)),
        "records": [dict(r) for r in records],
    }
    atomic_write_json(path, payload)
    return path


def _record_key(record: Mapping[str, Any]) -> str:
    return "|".join(
        str(record.get(k, ""))
        for k in ("iteration", "model_version", "pointdir", "atom", "property")
    )


def append_records(
    campaign_dir: Any,
    new_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    existing = load_records(campaign_dir)
    by_key = {_record_key(r): dict(r) for r in existing}
    added = 0
    skipped = 0
    for record in new_records:
        key = _record_key(record)
        if key in by_key:
            skipped += 1
            continue
        by_key[key] = dict(record)
        added += 1
    merged = list(by_key.values())
    merged.sort(
        key=lambda r: (
            int(r.get("iteration", -1)),
            str(r.get("pointdir", "")),
            str(r.get("atom", "")),
            str(r.get("property", "")),
        )
    )
    write_records(campaign_dir, merged)
    return merged, added, skipped


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float((1.0 - frac) * sorted_values[lo] + frac * sorted_values[hi])


def _make_table(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bins: int,
    min_bin_records: int,
) -> Dict[str, Any]:
    pairs = []
    for record in records:
        raw = _finite_float(record.get("raw_uncertainty"))
        err = _finite_float(record.get("abs_error_ha"))
        if raw is None or err is None:
            continue
        pairs.append((float(raw), float(err)))
    pairs.sort(key=lambda item: item[0])
    usable = len(pairs) >= int(min_bin_records)
    if not pairs:
        return {"n_records": 0, "usable": False, "bins": []}
    bin_count = max(1, min(int(n_bins), max(1, len(pairs) // max(1, int(min_bin_records)))))
    if not usable:
        bin_count = 1
    bins = []
    running = 0.0
    for idx in range(bin_count):
        start = int(round(idx * len(pairs) / bin_count))
        end = int(round((idx + 1) * len(pairs) / bin_count))
        chunk = pairs[start:end] or pairs[start:start + 1]
        raws = [p[0] for p in chunk]
        errs = sorted(p[1] for p in chunk)
        med = float(median(errs))
        running = max(running, med)
        bins.append(
            {
                "raw_uncertainty_min": float(min(raws)),
                "raw_uncertainty_max": float(max(raws)),
                "n": int(len(chunk)),
                "mean_abs_error_ha": float(mean(errs)),
                "median_abs_error_ha": med,
                "q90_abs_error_ha": _percentile(errs, 0.90),
                "calibrated_abs_error_ha": float(running),
            }
        )
    return {
        "n_records": int(len(pairs)),
        "usable": bool(usable),
        "bins": bins,
    }


def _current_model_version(records: Sequence[Mapping[str, Any]]) -> Optional[int]:
    versions = []
    for record in records:
        try:
            versions.append(int(record.get("model_version")))
        except (TypeError, ValueError):
            continue
    return max(versions) if versions else None


def _filter_records_by_model_version(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    current_model_version: Optional[int],
) -> List[Dict[str, Any]]:
    if str(policy) == "all":
        return [dict(r) for r in records]
    version = current_model_version
    if version is None:
        version = _current_model_version(records)
    if version is None:
        return []
    out = []
    for record in records:
        try:
            if int(record.get("model_version")) == int(version):
                out.append(dict(record))
        except (TypeError, ValueError):
            continue
    return out


def _total_error_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            record.get("iteration"),
            record.get("model_version"),
            record.get("pointdir"),
            record.get("seed_index"),
            record.get("property", "iqa"),
        )
        grouped.setdefault(key, []).append(record)

    totals: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        raw_total = _finite_float(group[0].get("raw_total_energy_variance"))
        if raw_total is None:
            continue
        pred_sum = 0.0
        truth_sum = 0.0
        n_atoms = 0
        for record in group:
            pred = _finite_float(record.get("predicted_iqa_ha"))
            truth = _finite_float(record.get("true_iqa_ha"))
            if pred is None or truth is None:
                continue
            pred_sum += float(pred)
            truth_sum += float(truth)
            n_atoms += 1
        if n_atoms <= 0:
            continue
        iteration, model_version, pointdir, seed_index, prop = key
        totals.append(
            {
                "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
                "iteration": iteration,
                "model_version": model_version,
                "pointdir": pointdir,
                "seed_index": seed_index,
                "property": prop,
                "n_atoms": int(n_atoms),
                "predicted_total_iqa_ha": float(pred_sum),
                "true_total_iqa_ha": float(truth_sum),
                "abs_error_ha": float(abs(pred_sum - truth_sum)),
                "raw_uncertainty": float(raw_total),
                "raw_total_energy_variance": float(raw_total),
                "raw_total_score": _finite_float(group[0].get("raw_total_score")),
                "landing_policy": str(group[0].get("landing_policy", "unknown")),
            }
        )
    totals.sort(
        key=lambda r: (
            int(r.get("iteration", -1)),
            str(r.get("pointdir", "")),
            str(r.get("property", "")),
        )
    )
    return totals


def build_calibration_model(
    records: Sequence[Mapping[str, Any]],
    config: Any,
    *,
    iteration: int,
    current_model_version: Optional[int] = None,
) -> Dict[str, Any]:
    block = _block(config)
    n_bins = int(_cfg_value(block, "n_bins", 10))
    min_bin_records = int(_cfg_value(block, "min_bin_records", 8))
    min_records_to_apply = int(_cfg_value(block, "min_records_to_apply", 100))
    group_by_atom_type = bool(_cfg_value(block, "group_by_atom_type", True))
    group_by_landing_policy = bool(_cfg_value(block, "group_by_landing_policy", False))
    model_version_policy = str(_cfg_value(block, "model_version_policy", "current"))

    candidate_records = _filter_records_by_model_version(
        records,
        policy=model_version_policy,
        current_model_version=current_model_version,
    )
    finite_records = [
        dict(r)
        for r in candidate_records
        if _finite_float(r.get("raw_uncertainty")) is not None
        and _finite_float(r.get("abs_error_ha")) is not None
    ]
    total_records = _total_error_records(finite_records)
    tables: Dict[str, Any] = {
        "global": _make_table(
            finite_records,
            n_bins=n_bins,
            min_bin_records=min_bin_records,
        ),
        "global_total": _make_table(
            total_records,
            n_bins=n_bins,
            min_bin_records=min_bin_records,
        )
    }
    if group_by_atom_type:
        atom_types = sorted({str(r.get("atom_type", "")) for r in finite_records if r.get("atom_type")})
        for atom_type in atom_types:
            grouped = [r for r in finite_records if str(r.get("atom_type")) == atom_type]
            table = _make_table(
                grouped,
                n_bins=n_bins,
                min_bin_records=min_bin_records,
            )
            if table.get("usable"):
                tables["atom_type:" + atom_type] = table
    if group_by_landing_policy:
        policies = sorted({str(r.get("landing_policy", "")) for r in finite_records if r.get("landing_policy")})
        for policy in policies:
            grouped = [r for r in finite_records if str(r.get("landing_policy")) == policy]
            table = _make_table(
                grouped,
                n_bins=n_bins,
                min_bin_records=min_bin_records,
            )
            if table.get("usable"):
                tables["landing_policy:" + policy] = table

    errors = sorted(float(r["abs_error_ha"]) for r in (total_records or finite_records))
    reference_error = max(_percentile(errors, 0.50), 1.0e-12) if errors else 1.0
    global_usable = bool(tables["global_total"].get("usable"))
    usable_for_acquisition = (
        bool(_enabled(config))
        and str(_cfg_value(block, "mode", "record_only")) == "apply_to_acquisition"
        and float(_cfg_value(block, "apply_strength", 0.0)) > 0.0
        and len(total_records) >= min_records_to_apply
        and global_usable
    )
    return {
        "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
        "iteration": int(iteration),
        "n_records": int(len(finite_records)),
        "n_total_error_records": int(len(total_records)),
        "n_total_records": int(len(records)),
        "model_version_policy": str(model_version_policy),
        "current_model_version": (
            None
            if (current_model_version if current_model_version is not None else _current_model_version(records)) is None
            else int(current_model_version if current_model_version is not None else _current_model_version(records))
        ),
        "n_bins": int(n_bins),
        "min_bin_records": int(min_bin_records),
        "min_records_to_apply": int(min_records_to_apply),
        "reference_error_ha": float(reference_error),
        "group_by_atom_type": bool(group_by_atom_type),
        "group_by_landing_policy": bool(group_by_landing_policy),
        "output_units": "ha",
        "usable_for_acquisition": bool(usable_for_acquisition),
        "tables": tables,
    }


def write_calibration_model(
    campaign_dir: Any,
    model: Mapping[str, Any],
) -> Path:
    path = model_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(model))
    return path


def load_calibration_model_for_acquisition(
    campaign_dir: Any,
    config: Any,
) -> Tuple[Optional[Dict[str, Any]], str]:
    block = _block(config)
    if not _enabled(config):
        return None, "disabled"
    if str(_cfg_value(block, "mode", "record_only")) != "apply_to_acquisition":
        return None, "record_only"
    if float(_cfg_value(block, "apply_strength", 0.0)) <= 0.0:
        return None, "zero_apply_strength"
    path = model_path(campaign_dir)
    if not path.is_file():
        return None, "missing_model"
    try:
        data = _read_json_object(path)
        if int(data.get("schema_version", -1)) != ERROR_CALIBRATION_SCHEMA_VERSION:
            return None, "unsupported_schema"
        if not bool(data.get("usable_for_acquisition", False)):
            return None, "not_enough_records"
        if not isinstance(data.get("tables"), dict):
            return None, "missing_tables"
    except ErrorCalibrationError:
        return None, "malformed_model"
    return data, "loaded"


def lookup_calibrated_abs_error(
    model: Mapping[str, Any],
    raw_uncertainty: Any,
    *,
    table_key: str = "global_total",
) -> Optional[float]:
    """Return calibrated absolute IQA error for a raw uncertainty value.

    Seed selection only has a cheap total posterior variance for the whole
    pool scan, so it uses the total calibration table by default. ARIADNE can
    still fall back to per-atom diagnostics when a total table is unavailable.
    """
    raw = _finite_float(raw_uncertainty)
    if raw is None:
        return None
    tables = model.get("tables") if isinstance(model, Mapping) else None
    if not isinstance(tables, Mapping):
        return None
    table = tables.get(str(table_key))
    if not isinstance(table, Mapping):
        table = tables.get("global_total") or tables.get("global")
    if not isinstance(table, Mapping):
        return None
    bins = table.get("bins")
    if not isinstance(bins, list) or not bins:
        return None
    last_value = None
    for entry in bins:
        if not isinstance(entry, Mapping):
            continue
        value = _finite_float(entry.get("calibrated_abs_error_ha"))
        if value is None:
            value = _finite_float(entry.get("median_abs_error_ha"))
        if value is None:
            continue
        last_value = float(value)
        high = _finite_float(entry.get("raw_uncertainty_max"))
        if high is not None and float(raw) <= float(high):
            return float(value)
    return last_value


def write_iteration_audit(iter_dir: Any, payload: Mapping[str, Any]) -> Path:
    path = audit_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = ERROR_CALIBRATION_SCHEMA_VERSION
    atomic_write_json(path, data)
    return path


def _quality_by_pointdir(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record.get("pointdir")): record
        for record in records
        if isinstance(record, Mapping) and bool(record.get("accepted", False))
    }


def _build_records_for_pointdir(
    pointdir: Path,
    quality_record: Mapping[str, Any],
    *,
    iteration: int,
    fallback_model_version: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    from ..versioning.provenance import PROVENANCE_FILENAME, read_provenance

    skipped: List[str] = []
    try:
        provenance = read_provenance(pointdir)
    except Exception as exc:
        return [], ["missing_or_invalid_provenance:" + type(exc).__name__]
    source = provenance.get("error_calibration_input")
    if not isinstance(source, Mapping):
        return [], ["missing_error_calibration_input"]
    per_atom_raw = source.get("per_atom")
    if not isinstance(per_atom_raw, list):
        return [], ["missing_per_atom_prediction_diagnostics"]
    predicted_by_atom = {
        str(row.get("atom")): row
        for row in per_atom_raw
        if isinstance(row, Mapping) and row.get("atom") is not None
    }
    quality_atoms = quality_record.get("per_atom")
    if not isinstance(quality_atoms, list):
        return [], ["missing_quality_per_atom"]

    model_version_raw = source.get("model_version", fallback_model_version)
    try:
        model_version = int(model_version_raw)
    except (TypeError, ValueError):
        model_version = int(fallback_model_version)
    seed_index = source.get("seed_index")
    try:
        seed_index_value = None if seed_index is None else int(seed_index)
    except (TypeError, ValueError):
        seed_index_value = None

    records: List[Dict[str, Any]] = []
    for q_atom in quality_atoms:
        if not isinstance(q_atom, Mapping):
            skipped.append("malformed_quality_atom")
            continue
        atom = str(q_atom.get("atom"))
        pred = predicted_by_atom.get(atom)
        if not isinstance(pred, Mapping):
            skipped.append("missing_prediction_for_atom:" + atom)
            continue
        true_iqa = _finite_float(q_atom.get("iqa_ha"))
        predicted_iqa = _finite_float(pred.get("predicted_iqa_ha"))
        raw_uncertainty = _finite_float(pred.get("raw_variance"))
        total_variance = _finite_float(source.get("total_energy_variance"))
        raw_total_score = _finite_float(source.get("raw_total_score"))
        if true_iqa is None or predicted_iqa is None or raw_uncertainty is None:
            skipped.append("nonfinite_prediction_truth_or_uncertainty:" + atom)
            continue
        record = {
            "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
            "iteration": int(iteration),
            "model_version": int(model_version),
            "pointdir": pointdir.name,
            "seed_index": seed_index_value,
            "atom": atom,
            "atom_type": str(pred.get("atom_type", "")),
            "property": str(pred.get("property", source.get("property", "iqa"))),
            "predicted_iqa_ha": float(predicted_iqa),
            "true_iqa_ha": float(true_iqa),
            "abs_error_ha": float(abs(predicted_iqa - true_iqa)),
            "raw_uncertainty": float(raw_uncertainty),
            "raw_atom_variance": float(raw_uncertainty),
            "raw_total_energy_variance": total_variance,
            "raw_total_score": raw_total_score,
            "landing_policy": str(source.get("landing_policy", "unknown")),
            "safety_metrics": dict(source.get("safety_metrics") or {}),
            "provenance": {
                "pointdir": str(pointdir.resolve()),
                "provenance_json": str((pointdir / PROVENANCE_FILENAME).resolve()),
                "result_json": str(source.get("result_json", "")),
            },
        }
        record["record_id"] = _record_key(record)
        records.append(record)
    return records, skipped


def update_from_aimall_acceptance(
    *,
    campaign_dir: Any,
    iter_dir: Any,
    config: Any,
    iteration: int,
    models_version: int,
    accepted_pointdirs: Sequence[Any],
    quality_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not _enabled(config):
        audit = {
            "iteration": int(iteration),
            "enabled": False,
            "n_new_records": 0,
            "n_added_records": 0,
            "n_duplicate_records": 0,
            "n_total_records": int(len(load_records(campaign_dir))),
            "skipped": {"disabled": int(len(accepted_pointdirs))},
        }
        write_iteration_audit(iter_dir, audit)
        return audit

    q_by_pointdir = _quality_by_pointdir(quality_records)
    new_records: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for raw_pointdir in accepted_pointdirs:
        pointdir = Path(getattr(raw_pointdir, "path", raw_pointdir))
        q_record = q_by_pointdir.get(pointdir.name)
        if q_record is None:
            skipped["missing_quality_record"] = skipped.get("missing_quality_record", 0) + 1
            continue
        built, reasons = _build_records_for_pointdir(
            pointdir,
            q_record,
            iteration=int(iteration),
            fallback_model_version=int(models_version),
        )
        new_records.extend(built)
        for reason in reasons:
            skipped[reason] = skipped.get(reason, 0) + 1

    all_records, added, duplicate = append_records(campaign_dir, new_records)
    model = build_calibration_model(
        all_records,
        config,
        iteration=int(iteration),
        current_model_version=int(models_version),
    )
    model_file = write_calibration_model(campaign_dir, model)
    audit = {
        "iteration": int(iteration),
        "enabled": True,
        "mode": str(_cfg_value(_block(config), "mode", "record_only")),
        "n_new_records": int(len(new_records)),
        "n_added_records": int(added),
        "n_duplicate_records": int(duplicate),
        "n_total_records": int(len(all_records)),
        "n_usable_records": int(model.get("n_records", 0)),
        "n_usable_total_records": int(model.get("n_total_error_records", 0)),
        "usable_for_acquisition": bool(model.get("usable_for_acquisition", False)),
        "model": str(model_file.resolve()),
        "skipped": skipped,
    }
    write_iteration_audit(iter_dir, audit)
    return audit


def synthetic_dry_records(
    *,
    iteration: int,
    models_version: int,
    n_points: int,
) -> List[Dict[str, Any]]:
    records = []
    for i in range(int(n_points)):
        predicted = -1.0 - 1.0e-4 * i
        true = -1.0 - 2.0e-4 * i
        record = {
            "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
            "iteration": int(iteration),
            "model_version": int(models_version),
            "pointdir": "POINT_" + str(i).zfill(4) + ".pointdir",
            "seed_index": int(i),
            "atom": "X1",
            "atom_type": "X",
            "property": "iqa",
            "predicted_iqa_ha": float(predicted),
            "true_iqa_ha": float(true),
            "abs_error_ha": float(abs(predicted - true)),
            "raw_uncertainty": float(1.0e-3 + 1.0e-4 * i),
            "raw_atom_variance": float(1.0e-3 + 1.0e-4 * i),
            "raw_total_energy_variance": float(1.0e-3 + 1.0e-4 * i),
            "raw_total_score": float(1.0 + 1.0e-2 * i),
            "landing_policy": "dry_run",
            "safety_metrics": {"synthetic": True},
            "provenance": {"pointdir": "", "provenance_json": "", "result_json": ""},
        }
        record["record_id"] = _record_key(record)
        records.append(record)
    return records
