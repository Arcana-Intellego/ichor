"""Empirical uncertainty-to-error calibration for active learning.

The daemon already pays for the expensive pieces: model predictions before QM
and AIMAll IQA truth after QM. This module only joins those existing records and
fits a tiny monotone binned lookup table, so it is safe to run on every accepted
AIMAll batch.
"""
from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
import math
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .state import atomic_write_json
from ..layout import staging_pointdir_name


ERROR_CALIBRATION_RECORDS_FILENAME = "error_calibration_records.json"
ERROR_CALIBRATION_MODEL_FILENAME = "error_calibration_model.json"
ERROR_CALIBRATION_AUDIT_FILENAME = "ERROR_CALIBRATION_AUDIT.json"
ERROR_CALIBRATION_SCHEMA_VERSION = 1


class ErrorCalibrationError(RuntimeError):
    """Raised when a calibration manifest is malformed."""


def _active_learning_dir(campaign_dir: Any) -> Path:
    from .filesystem import operational_data_dir

    return operational_data_dir(campaign_dir)


def records_path(campaign_dir: Any) -> Path:
    return _active_learning_dir(campaign_dir) / ERROR_CALIBRATION_RECORDS_FILENAME


def model_path(campaign_dir: Any) -> Path:
    return _active_learning_dir(campaign_dir) / ERROR_CALIBRATION_MODEL_FILENAME


def audit_path(iter_dir: Any) -> Path:
    from ..layout import active_calibration_dir

    return active_calibration_dir(iter_dir) / ERROR_CALIBRATION_AUDIT_FILENAME


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


def _quarantine_file(path: Path, reason: str, label: str) -> Optional[Path]:
    if not path.exists():
        return None
    suffix = ".corrupt." + str(time.time_ns())
    target = path.with_name(path.name + suffix)
    try:
        path.rename(target)
    except OSError as exc:
        raise ErrorCalibrationError(
            "failed to quarantine malformed "
            + label
            + " "
            + str(path)
            + ": "
            + str(exc)
        ) from exc
    marker = target.with_suffix(target.suffix + ".reason.txt")
    try:
        marker.write_text(str(reason) + "\n", encoding="utf-8", newline="\n")
    except OSError:
        pass
    return target


def _quarantine_records_file(campaign_dir: Any, reason: str) -> Optional[Path]:
    return _quarantine_file(records_path(campaign_dir), reason, "calibration records")


def _quarantine_model_file(campaign_dir: Any, reason: str) -> Optional[Path]:
    return _quarantine_file(model_path(campaign_dir), reason, "calibration model")


def _record_key(record: Mapping[str, Any]) -> str:
    return "|".join(
        str(record.get(k, ""))
        for k in (
            "iteration",
            "model_version",
            "prior_mean_contract_sha256",
            "pointdir",
            "atom",
            "property",
        )
    )


def _frame_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        record.get("iteration"),
        record.get("model_version"),
        record.get("pointdir"),
        record.get("seed_id"),
        record.get("seed_uid"),
        record.get("prior_mean_contract_sha256"),
        record.get("property", "iqa"),
    )


def append_records(
    campaign_dir: Any,
    new_records: Sequence[Mapping[str, Any]],
    *,
    max_records: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    try:
        existing = load_records(campaign_dir)
    except ErrorCalibrationError as exc:
        _quarantine_records_file(campaign_dir, str(exc))
        existing = []
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
    if max_records is not None and int(max_records) > 0 and len(merged) > int(max_records):
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for record in merged:
            grouped.setdefault(_frame_key(record), []).append(record)
        retained_groups: List[List[Dict[str, Any]]] = []
        retained_count = 0
        ordered_groups = sorted(
            grouped.values(),
            key=lambda group: (
                int(group[0].get("iteration", -1)),
                str(group[0].get("pointdir", "")),
                str(group[0].get("property", "")),
            ),
        )
        for group in reversed(ordered_groups):
            if retained_groups and retained_count + len(group) > int(max_records):
                continue
            retained_groups.append(group)
            retained_count += len(group)
        merged = [record for group in reversed(retained_groups) for record in group]
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


def _pava_non_decreasing(values: Sequence[float], weights: Sequence[int]) -> List[float]:
    """Weighted pool-adjacent-violators fit for monotone calibration bins."""
    if len(values) != len(weights):
        raise ValueError("PAVA values and weights length mismatch")
    blocks: List[Dict[str, float]] = []
    for idx, (value, weight) in enumerate(zip(values, weights)):
        w = max(float(weight), 1.0)
        blocks.append({
            "start": float(idx),
            "end": float(idx),
            "weight": w,
            "value": float(value),
        })
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left["weight"] + right["weight"]
            pooled = (
                left["value"] * left["weight"] + right["value"] * right["weight"]
            ) / total_weight
            blocks.append({
                "start": left["start"],
                "end": right["end"],
                "weight": total_weight,
                "value": float(pooled),
            })
    fitted = [0.0] * len(values)
    for block in blocks:
        for idx in range(int(block["start"]), int(block["end"]) + 1):
            fitted[idx] = float(block["value"])
    return fitted


def _make_table(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bins: int,
    min_bin_records: int,
    monotone: bool,
    quantile: float,
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
    calibrated_values: List[float] = []
    bin_weights: List[int] = []
    q = min(1.0, max(1.0e-12, float(quantile)))
    for idx in range(bin_count):
        start = int(round(idx * len(pairs) / bin_count))
        end = int(round((idx + 1) * len(pairs) / bin_count))
        chunk = pairs[start:end] or pairs[start:start + 1]
        raws = [p[0] for p in chunk]
        errs = sorted(p[1] for p in chunk)
        med = float(median(errs))
        calibrated = float(_percentile(errs, q))
        calibrated_values.append(calibrated)
        bin_weights.append(int(len(chunk)))
        bins.append(
            {
                "raw_uncertainty_min": float(min(raws)),
                "raw_uncertainty_max": float(max(raws)),
                "n": int(len(chunk)),
                "mean_abs_error_ha": float(mean(errs)),
                "median_abs_error_ha": med,
                "q90_abs_error_ha": _percentile(errs, 0.90),
                "calibration_quantile": float(q),
                "calibrated_abs_error_ha": float(calibrated),
            }
        )
    if bool(monotone) and bins:
        fitted = _pava_non_decreasing(calibrated_values, bin_weights)
        for entry, value in zip(bins, fitted):
            entry["raw_calibrated_abs_error_ha"] = float(
                entry["calibrated_abs_error_ha"]
            )
            entry["calibrated_abs_error_ha"] = float(value)
    return {
        "n_records": int(len(pairs)),
        "usable": bool(usable),
        "monotone": bool(monotone),
        "estimator": "pava_quantile_bins" if bool(monotone) else "raw_quantile_bins",
        "pava_applied": bool(monotone and bins),
        "quantile": float(q),
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
    if str(policy) in {"all", "rolling_normalised"}:
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


def _normalise_total_uncertainty_by_model(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for record in records:
        try:
            version = int(record.get("model_version"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(version, []).append(record)
    normalised: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for version in sorted(grouped):
        group = grouped[version]
        raw_values = sorted(
            float(value)
            for value in (
                _finite_float(record.get("raw_uncertainty")) for record in group
            )
            if value is not None and float(value) >= 0.0
        )
        positive = [value for value in raw_values if value > 0.0]
        scale = float(median(positive)) if positive else 1.0
        scale = max(scale, 1.0e-18)
        errors = sorted(
            float(value)
            for value in (
                _finite_float(record.get("abs_error_ha")) for record in group
            )
            if value is not None
        )
        diagnostics[str(version)] = {
            "n_total_frames": int(len(group)),
            "raw_uncertainty_median": float(scale),
            "raw_uncertainty_min": float(min(raw_values)) if raw_values else None,
            "raw_uncertainty_max": float(max(raw_values)) if raw_values else None,
            "realised_error_median_ha": (
                float(median(errors)) if errors else None
            ),
        }
        for record in group:
            raw = _finite_float(record.get("raw_uncertainty"))
            if raw is None:
                continue
            item = dict(record)
            item["raw_uncertainty_unnormalised"] = float(raw)
            item["raw_uncertainty"] = float(raw) / scale
            item["model_uncertainty_scale"] = float(scale)
            normalised.append(item)
    return normalised, diagnostics


def _filter_records_by_age(
    records: Sequence[Mapping[str, Any]],
    *,
    iteration: int,
    max_age_iterations: int,
) -> List[Dict[str, Any]]:
    cutoff = int(iteration) - int(max_age_iterations)
    out: List[Dict[str, Any]] = []
    for record in records:
        try:
            rec_iteration = int(record.get("iteration"))
        except (TypeError, ValueError):
            continue
        if rec_iteration >= cutoff:
            out.append(dict(record))
    return out


def _total_error_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_frame_key(record), []).append(record)

    totals: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        try:
            expected_atoms = int(group[0].get("frame_atom_count"))
        except (TypeError, ValueError):
            continue
        atom_names = [str(record.get("atom") or "") for record in group]
        frame_counts: List[int] = []
        for record in group:
            try:
                frame_counts.append(int(record.get("frame_atom_count", -1)))
            except (TypeError, ValueError):
                frame_counts.append(-1)
        identity_hashes = {
            str(record.get("frame_atom_identity_sha256") or "")
            for record in group
        }
        if (
            expected_atoms <= 0
            or len(group) != expected_atoms
            or len(set(atom_names)) != expected_atoms
            or any(not atom for atom in atom_names)
            or any(value != expected_atoms for value in frame_counts)
            or len(identity_hashes) != 1
            or "" in identity_hashes
        ):
            continue
        raw_totals = [
            _finite_float(record.get("raw_total_energy_variance"))
            for record in group
        ]
        if any(value is None for value in raw_totals):
            continue
        raw_total = float(raw_totals[0])
        if any(
            not math.isclose(float(value), raw_total, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for value in raw_totals[1:]
        ):
            continue
        pred_sum = 0.0
        truth_sum = 0.0
        complete = True
        for record in group:
            pred = _finite_float(record.get("predicted_iqa_ha"))
            truth = _finite_float(record.get("true_iqa_ha"))
            if pred is None or truth is None:
                complete = False
                break
            pred_sum += float(pred)
            truth_sum += float(truth)
        if not complete:
            continue
        (
            iteration,
            model_version,
            pointdir,
            seed_id,
            seed_uid,
            prior_contract_hash,
            prop,
        ) = key
        totals.append(
            {
                "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
                "iteration": iteration,
                "model_version": model_version,
                "prior_mean_contract_sha256": prior_contract_hash,
                "pointdir": pointdir,
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "property": prop,
                "n_atoms": int(expected_atoms),
                "predicted_total_iqa_ha": float(pred_sum),
                "true_total_iqa_ha": float(truth_sum),
                "abs_error_ha": float(abs(pred_sum - truth_sum)),
                "raw_uncertainty": float(raw_total),
                "raw_total_energy_variance": float(raw_total),
                "raw_total_score": _finite_float(group[0].get("raw_total_score")),
                "landing_policy": str(group[0].get("landing_policy", "unknown")),
                "sampling_aggressiveness": group[0].get("sampling_aggressiveness"),
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
    min_model_versions_to_apply = int(
        _cfg_value(block, "min_model_versions_to_apply", 2)
    )
    max_model_age_iterations = int(_cfg_value(block, "max_model_age_iterations", 10))
    monotone_estimator = bool(_cfg_value(block, "monotone_estimator", True))
    quantile = float(_cfg_value(block, "quantile", 0.75))
    group_by_atom_type = bool(_cfg_value(block, "group_by_atom_type", True))
    group_by_landing_policy = bool(_cfg_value(block, "group_by_landing_policy", False))
    model_version_policy = str(
        _cfg_value(block, "model_version_policy", "rolling_normalised")
    )
    aggressiveness_match_required = bool(
        _cfg_value(block, "aggressiveness_match_required", True)
    )
    sampling_aggressiveness = int(
        getattr(getattr(config, "campaign", object()), "sampling_aggressiveness", 5)
    )
    try:
        from ..ferebus_prior import resolve_ferebus_prior_contract

        prior_contract_hash = resolve_ferebus_prior_contract(config).contract_sha256
    except Exception as exc:
        raise ErrorCalibrationError(
            "cannot resolve FEREBUS prior contract for calibration: " + str(exc)
        ) from exc

    current_version = (
        current_model_version
        if current_model_version is not None
        else _current_model_version(records)
    )
    n_records_by_model_version: Dict[str, int] = {}
    for record in records:
        try:
            key = str(int(record.get("model_version")))
        except (TypeError, ValueError):
            key = "unknown"
        n_records_by_model_version[key] = n_records_by_model_version.get(key, 0) + 1

    candidate_records = _filter_records_by_model_version(
        records,
        policy=model_version_policy,
        current_model_version=current_model_version,
    )
    candidate_records = _filter_records_by_age(
        candidate_records,
        iteration=int(iteration),
        max_age_iterations=max_model_age_iterations,
    )
    prior_matched = [
        dict(record)
        for record in candidate_records
        if str(record.get("prior_mean_contract_sha256") or "")
        == prior_contract_hash
    ]
    n_prior_contract_mismatch = len(candidate_records) - len(prior_matched)
    candidate_records = prior_matched
    n_aggressiveness_mismatch = 0
    if aggressiveness_match_required:
        matched: List[Dict[str, Any]] = []
        for record in candidate_records:
            try:
                matches = int(record.get("sampling_aggressiveness")) == sampling_aggressiveness
            except (TypeError, ValueError):
                matches = False
            if matches:
                matched.append(record)
            else:
                n_aggressiveness_mismatch += 1
        candidate_records = matched
    n_records_by_iteration: Dict[str, int] = {}
    for record in candidate_records:
        try:
            key = str(int(record.get("iteration")))
        except (TypeError, ValueError):
            key = "unknown"
        n_records_by_iteration[key] = n_records_by_iteration.get(key, 0) + 1

    finite_records = [
        dict(r)
        for r in candidate_records
        if _finite_float(r.get("raw_uncertainty")) is not None
        and _finite_float(r.get("abs_error_ha")) is not None
    ]
    total_records = _total_error_records(finite_records)
    uncertainty_axis = "raw"
    model_normalisation: Dict[str, Dict[str, Any]] = {}
    table_total_records = total_records
    if model_version_policy == "rolling_normalised":
        table_total_records, model_normalisation = (
            _normalise_total_uncertainty_by_model(total_records)
        )
        uncertainty_axis = "model_normalised"
    contributing_versions_set: set[int] = set()
    for record in table_total_records:
        try:
            contributing_versions_set.add(int(record.get("model_version")))
        except (TypeError, ValueError):
            continue
    contributing_model_versions = sorted(contributing_versions_set)
    tables: Dict[str, Any] = {
        "global": _make_table(
            finite_records,
            n_bins=n_bins,
            min_bin_records=min_bin_records,
            monotone=monotone_estimator,
            quantile=quantile,
        ),
        "global_total": _make_table(
            table_total_records,
            n_bins=n_bins,
            min_bin_records=min_bin_records,
            monotone=monotone_estimator,
            quantile=quantile,
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
                monotone=monotone_estimator,
                quantile=quantile,
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
                monotone=monotone_estimator,
                quantile=quantile,
            )
            if table.get("usable"):
                tables["landing_policy:" + policy] = table

    errors = sorted(float(r["abs_error_ha"]) for r in (total_records or finite_records))
    reference_error = max(_percentile(errors, 0.50), 1.0e-12) if errors else 1.0
    global_usable = bool(tables["global_total"].get("usable"))
    activation_blockers: List[str] = []
    mode = str(_cfg_value(block, "mode", "record_only"))
    apply_strength = float(_cfg_value(block, "apply_strength", 0.0))
    if not bool(_enabled(config)):
        activation_blockers.append("disabled")
    if mode != "apply_to_acquisition":
        activation_blockers.append("record_only")
    if apply_strength <= 0.0:
        activation_blockers.append("zero_apply_strength")
    if len(total_records) < min_records_to_apply:
        activation_blockers.append("not_enough_total_records")
    if (
        model_version_policy == "rolling_normalised"
        and len(contributing_model_versions) < min_model_versions_to_apply
    ):
        activation_blockers.append("not_enough_model_versions")
    if not global_usable:
        activation_blockers.append("global_total_unusable")
    usable_for_acquisition = not activation_blockers
    activation_reason = "usable" if usable_for_acquisition else activation_blockers[0]
    return {
        "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
        "iteration": int(iteration),
        "n_records": int(len(finite_records)),
        "n_total_error_records": int(len(total_records)),
        "n_total_records": int(len(records)),
        "n_records_by_model_version": n_records_by_model_version,
        "n_records_by_iteration_window": n_records_by_iteration,
        "model_version_policy": str(model_version_policy),
        "uncertainty_axis": uncertainty_axis,
        "model_uncertainty_normalisation": model_normalisation,
        "contributing_model_versions": contributing_model_versions,
        "n_contributing_model_versions": int(len(contributing_model_versions)),
        "min_model_versions_to_apply": int(min_model_versions_to_apply),
        "sampling_aggressiveness": int(sampling_aggressiveness),
        "prior_mean_contract_sha256": prior_contract_hash,
        "n_prior_contract_mismatch_excluded": int(n_prior_contract_mismatch),
        "aggressiveness_match_required": bool(aggressiveness_match_required),
        "n_aggressiveness_mismatch_excluded": int(n_aggressiveness_mismatch),
        "current_model_version": (
            None
            if current_version is None
            else int(current_version)
        ),
        "n_bins": int(n_bins),
        "min_bin_records": int(min_bin_records),
        "min_records_to_apply": int(min_records_to_apply),
        "max_model_age_iterations": int(max_model_age_iterations),
        "monotone_estimator": bool(monotone_estimator),
        "estimator": "pava_quantile_bins" if monotone_estimator else "raw_quantile_bins",
        "quantile": float(quantile),
        "reference_error_ha": float(reference_error),
        "group_by_atom_type": bool(group_by_atom_type),
        "group_by_landing_policy": bool(group_by_landing_policy),
        "output_units": "ha",
        "usable_for_acquisition": bool(usable_for_acquisition),
        "activation_reason": str(activation_reason),
        "activation_blockers": activation_blockers,
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


def mark_calibration_model_stale(
    campaign_dir: Any,
    *,
    reason: str,
    iteration: Optional[int] = None,
) -> Path:
    payload = {
        "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
        "usable_for_acquisition": False,
        "stale": True,
        "stale_reason": str(reason),
        "iteration": None if iteration is None else int(iteration),
        "tables": {},
    }
    return write_calibration_model(campaign_dir, payload)


def load_calibration_model_for_acquisition(
    campaign_dir: Any,
    config: Any,
    *,
    current_model_version: Optional[int] = None,
    current_iteration: Optional[int] = None,
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
            _quarantine_model_file(campaign_dir, "unsupported calibration model schema")
            return None, "unsupported_schema"
        if bool(data.get("stale", False)):
            return None, "stale_model"
        if not bool(data.get("usable_for_acquisition", False)):
            return None, "not_enough_records"
        if not isinstance(data.get("tables"), dict):
            _quarantine_model_file(campaign_dir, "calibration model missing tables")
            return None, "missing_tables"
    except ErrorCalibrationError as exc:
        _quarantine_model_file(campaign_dir, str(exc))
        return None, "malformed_model"
    configured_policy = str(
        _cfg_value(block, "model_version_policy", "rolling_normalised")
    )
    if str(data.get("model_version_policy")) != configured_policy:
        return None, "model_version_policy_changed"
    try:
        from ..ferebus_prior import resolve_ferebus_prior_contract

        current_prior_hash = resolve_ferebus_prior_contract(config).contract_sha256
    except Exception:
        return None, "prior_mean_contract_unavailable"
    if str(data.get("prior_mean_contract_sha256") or "") != current_prior_hash:
        return None, "prior_mean_contract_mismatch"
    if current_model_version is None:
        try:
            from ..versioning.trained_models import TrainedModelVersioning

            current_model_version = TrainedModelVersioning(
                Path(campaign_dir) / "6_TRAINED_MODELS"
            ).current_version()
        except Exception:
            current_model_version = None
    if configured_policy == "current":
        try:
            calibrated_version = int(data.get("current_model_version"))
        except (TypeError, ValueError):
            return None, "calibrated_model_version_missing"
        if current_model_version is None or calibrated_version != int(current_model_version):
            return None, "calibrated_model_version_mismatch"
    elif configured_policy == "rolling_normalised":
        versions = data.get("contributing_model_versions")
        if not isinstance(versions, list):
            return None, "normalised_model_versions_missing"
        try:
            version_values = sorted({int(value) for value in versions})
        except (TypeError, ValueError):
            return None, "normalised_model_versions_malformed"
        required_versions = int(
            _cfg_value(block, "min_model_versions_to_apply", 2)
        )
        if len(version_values) < required_versions:
            return None, "not_enough_model_versions"
        if current_model_version is None:
            return None, "current_model_version_unavailable"
        if version_values and int(current_model_version) < max(version_values):
            return None, "calibration_is_ahead_of_current_model"
    if bool(_cfg_value(block, "aggressiveness_match_required", True)):
        try:
            model_aggressiveness = int(data.get("sampling_aggressiveness"))
            current_aggressiveness = int(config.campaign.sampling_aggressiveness)
        except (AttributeError, TypeError, ValueError):
            return None, "sampling_aggressiveness_unavailable"
        if model_aggressiveness != current_aggressiveness:
            return None, "sampling_aggressiveness_mismatch"
    if current_iteration is not None:
        try:
            age = int(current_iteration) - int(data.get("iteration"))
        except (TypeError, ValueError):
            return None, "calibration_iteration_missing"
        max_age = int(_cfg_value(block, "max_model_age_iterations", 10))
        if age < 0 or age > max_age:
            return None, "calibration_model_too_old"
    return data, "loaded"


def lookup_calibrated_abs_error(
    model: Mapping[str, Any],
    raw_uncertainty: Any,
    *,
    table_key: str = "global_total",
    application_uncertainty_scale: Optional[float] = None,
) -> Optional[float]:
    """Return calibrated absolute IQA error for a raw uncertainty value.

    Acquisition-facing callers use the total calibration table. Per-atom
    calibration tables remain diagnostic-only and must not be combined into a
    total-energy acquisition signal.
    """
    raw = _finite_float(raw_uncertainty)
    if raw is None:
        return None
    if str(model.get("uncertainty_axis", "raw")) == "model_normalised":
        scale = _finite_float(application_uncertainty_scale)
        if scale is None or scale <= 0.0:
            return None
        raw = float(raw) / float(scale)
    tables = model.get("tables") if isinstance(model, Mapping) else None
    if not isinstance(tables, Mapping):
        return None
    table = tables.get(str(table_key))
    if not isinstance(table, Mapping):
        table = tables.get("global_total")
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
    sampling_aggressiveness: int,
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
    quality_atoms = quality_record.get("per_atom")
    if not isinstance(quality_atoms, list):
        return [], ["missing_quality_per_atom"]
    predicted_rows = [row for row in per_atom_raw if isinstance(row, Mapping)]
    quality_rows = [row for row in quality_atoms if isinstance(row, Mapping)]
    predicted_names = [str(row.get("atom") or "") for row in predicted_rows]
    quality_names = [str(row.get("atom") or "") for row in quality_rows]
    if (
        len(predicted_rows) != len(per_atom_raw)
        or len(quality_rows) != len(quality_atoms)
        or not predicted_names
        or any(not name for name in predicted_names + quality_names)
        or len(set(predicted_names)) != len(predicted_names)
        or len(set(quality_names)) != len(quality_names)
        or set(predicted_names) != set(quality_names)
    ):
        return [], ["incomplete_or_mismatched_atom_identity"]
    predicted_by_atom = {
        name: row for name, row in zip(predicted_names, predicted_rows)
    }

    model_version_raw = source.get("model_version", fallback_model_version)
    try:
        model_version = int(model_version_raw)
    except (TypeError, ValueError):
        model_version = int(fallback_model_version)
    seed_id = source.get("seed_id")
    try:
        seed_id_value = None if seed_id is None else int(seed_id)
    except (TypeError, ValueError):
        seed_id_value = None
    seed_uid = str(source.get("seed_uid") or "") or None
    total_variance = _finite_float(source.get("total_energy_variance"))
    raw_total_score = _finite_float(source.get("raw_total_score"))
    prior_contract_hash = str(source.get("prior_mean_contract_sha256") or "")
    if len(prior_contract_hash) != 64:
        return [], ["missing_or_invalid_prior_mean_contract_sha256"]
    atom_identity_sha = hashlib.sha256(
        json.dumps(quality_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    records: List[Dict[str, Any]] = []
    for q_atom in quality_rows:
        atom = str(q_atom.get("atom"))
        pred = predicted_by_atom.get(atom)
        true_iqa = _finite_float(q_atom.get("iqa_ha"))
        predicted_iqa = _finite_float(pred.get("predicted_iqa_ha"))
        raw_uncertainty = _finite_float(pred.get("raw_variance"))
        if true_iqa is None or predicted_iqa is None or raw_uncertainty is None:
            return [], ["nonfinite_prediction_truth_or_uncertainty:" + atom]
        record = {
            "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
            "iteration": int(iteration),
            "model_version": int(model_version),
            "prior_mean_contract_sha256": prior_contract_hash,
            "pointdir": pointdir.name,
            "seed_id": seed_id_value,
            "seed_uid": seed_uid,
            "sampling_aggressiveness": int(sampling_aggressiveness),
            "frame_atom_count": int(len(quality_names)),
            "frame_atom_identity_sha256": atom_identity_sha,
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
            sampling_aggressiveness=int(
                config.campaign.sampling_aggressiveness
            ),
        )
        new_records.extend(built)
        for reason in reasons:
            skipped[reason] = skipped.get(reason, 0) + 1

    all_records, added, duplicate = append_records(
        campaign_dir,
        new_records,
        max_records=int(_cfg_value(_block(config), "max_records", 5000)),
    )
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
    sampling_aggressiveness: int = 5,
    prior_mean_contract_sha256: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if prior_mean_contract_sha256 is None:
        from ..ferebus_prior import FerebusPriorContract

        prior_mean_contract_sha256 = FerebusPriorContract(
            strategy="physical_atomic_iqa",
            mean_type=21,
            level_of_theory="b3lyp/aug-cc-pvtz",
            physical_prior_scale=1.0,
        ).contract_sha256
    records = []
    for i in range(int(n_points)):
        predicted = -1.0 - 1.0e-4 * i
        true = -1.0 - 2.0e-4 * i
        record = {
            "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
            "iteration": int(iteration),
            "model_version": int(models_version),
            "prior_mean_contract_sha256": str(prior_mean_contract_sha256),
            "pointdir": staging_pointdir_name(i),
            "seed_id": int(i) + 1,
            "seed_uid": hashlib.sha256(
                (str(iteration) + ":" + str(i + 1)).encode("ascii")
            ).hexdigest(),
            "sampling_aggressiveness": int(sampling_aggressiveness),
            "frame_atom_count": 1,
            "frame_atom_identity_sha256": hashlib.sha256(b'["X1"]').hexdigest(),
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
