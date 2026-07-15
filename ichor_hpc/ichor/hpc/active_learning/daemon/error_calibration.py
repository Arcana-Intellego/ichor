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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .state import atomic_write_json
from ..layout import staging_pointdir_name
from .error_calibration_contract import (
    CALIBRATION_AUDIT_SCHEMA_VERSION,
    CALIBRATION_MODEL_SCHEMA_VERSION,
    CALIBRATION_RECORDS_SCHEMA_VERSION,
    CalibrationContractError,
    active_environment_binding,
    calibration_context_sha256,
    canonical_json_bytes,
    canonical_sha256,
    estimator_settings,
    estimator_settings_sha256,
    record_key as contract_record_key,
    validate_model,
    validate_record,
)
from ichor.core.adversarial.error_calibration import CALIBRATION_OUTPUT_UNITS, lookup_calibrated_error


ERROR_CALIBRATION_RECORDS_FILENAME = "error_calibration_records.json"
ERROR_CALIBRATION_MODEL_FILENAME = "error_calibration_model.json"
ERROR_CALIBRATION_AUDIT_FILENAME = "ERROR_CALIBRATION_AUDIT.json"
ERROR_CALIBRATION_SCHEMA_VERSION = CALIBRATION_RECORDS_SCHEMA_VERSION


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
    if data.get("schema_version") != CALIBRATION_RECORDS_SCHEMA_VERSION:
        raise ErrorCalibrationError("unsupported calibration records schema")
    declared_count = data.get("n_records")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise ErrorCalibrationError("calibration n_records must be an exact integer")
    records = data.get("records")
    if not isinstance(records, list):
        raise ErrorCalibrationError("calibration records must be a list")
    if declared_count != len(records):
        raise ErrorCalibrationError("calibration n_records does not match records")
    validated: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        try:
            record = validate_record(raw)
        except CalibrationContractError as exc:
            raise ErrorCalibrationError(
                "invalid calibration record " + str(index) + ": " + str(exc)
            ) from exc
        record_id = str(record["record_id"])
        if record_id in seen:
            raise ErrorCalibrationError("duplicate calibration record identity")
        seen.add(record_id)
        validated.append(record)
    return validated


def write_records(campaign_dir: Any, records: Sequence[Mapping[str, Any]]) -> Path:
    validated: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        try:
            record = validate_record(raw)
        except CalibrationContractError as exc:
            raise ErrorCalibrationError(
                "invalid calibration record " + str(index) + ": " + str(exc)
            ) from exc
        if record["record_id"] in seen:
            raise ErrorCalibrationError("duplicate calibration record identity")
        seen.add(record["record_id"])
        validated.append(record)
    path = records_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CALIBRATION_RECORDS_SCHEMA_VERSION,
        "n_records": int(len(validated)),
        "records": validated,
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
    return contract_record_key(record)


def _frame_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        record.get("iteration"),
        record.get("model_version"),
        record.get("model_set_sha256"),
        record.get("pointdir"),
        record.get("seed_id"),
        record.get("seed_uid"),
        record.get("prior_mean_contract_sha256"),
        record.get("environment_generation"),
        record.get("environment_generation_digest_sha256"),
        record.get("calibration_context_sha256"),
        dict(record.get("source_digests") or {}).get("sampling_protocol_sha256"),
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
    for index, raw in enumerate(new_records):
        try:
            record = validate_record(raw)
        except CalibrationContractError as exc:
            raise ErrorCalibrationError(
                "invalid new calibration record " + str(index) + ": " + str(exc)
            ) from exc
        key = _record_key(record)
        if key in by_key:
            if canonical_json_bytes(by_key[key]) != canonical_json_bytes(record):
                raise ErrorCalibrationError(
                    "calibration replay conflict for record " + key
                )
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


def build_calibration_model(
    records: Sequence[Mapping[str, Any]],
    config: Any,
    *,
    iteration: int,
    current_model_version: Optional[int] = None,
    environment_binding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one strict, environment-bound calibration model."""
    from .calibration_model import build_calibration_model_v2

    try:
        return build_calibration_model_v2(
            records,
            config,
            iteration=int(iteration),
            current_model_version=current_model_version,
            environment_binding=environment_binding,
        )
    except CalibrationContractError as exc:
        raise ErrorCalibrationError(str(exc)) from exc


def write_calibration_model(
    campaign_dir: Any,
    model: Mapping[str, Any],
) -> Path:
    try:
        validated = validate_model(model)
    except CalibrationContractError as exc:
        raise ErrorCalibrationError("invalid calibration model: " + str(exc)) from exc
    path = model_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, validated)
    return path


def mark_calibration_model_stale(
    campaign_dir: Any,
    *,
    reason: str,
    iteration: Optional[int] = None,
) -> Path:
    payload = {
        "schema_version": CALIBRATION_MODEL_SCHEMA_VERSION,
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
        data = validate_model(_read_json_object(path))
    except (ErrorCalibrationError, CalibrationContractError) as exc:
        _quarantine_model_file(campaign_dir, str(exc))
        return None, "malformed_model"
    if bool(data.get("stale", False)):
        return None, "stale_model"
    if data.get("estimator_settings_sha256") != estimator_settings_sha256(config):
        return None, "estimator_settings_changed"
    try:
        from ..ferebus_prior import resolve_ferebus_prior_contract

        current_prior_hash = resolve_ferebus_prior_contract(config).contract_sha256
    except Exception:
        return None, "prior_mean_contract_unavailable"
    if str(data.get("prior_mean_contract_sha256") or "") != current_prior_hash:
        return None, "prior_mean_contract_mismatch"
    try:
        environment = active_environment_binding(campaign_dir)
    except CalibrationContractError:
        return None, "environment_generation_unavailable"
    if int(data.get("environment_generation")) != int(environment["generation"]):
        return None, "environment_generation_mismatch"
    if data.get("environment_generation_digest_sha256") != environment[
        "generation_digest_sha256"
    ]:
        return None, "environment_generation_mismatch"
    expected_context = calibration_context_sha256(
        config,
        prior_mean_contract_sha256=current_prior_hash,
        environment_generation_digest_sha256=environment[
            "generation_digest_sha256"
        ],
    )
    if data.get("calibration_context_sha256") != expected_context:
        return None, "calibration_context_mismatch"
    if current_model_version is None:
        try:
            from ..versioning.trained_models import TrainedModelVersioning

            current_model_version = TrainedModelVersioning(
                Path(campaign_dir) / "6_TRAINED_MODELS"
            ).current_version()
        except Exception:
            current_model_version = None
    if current_model_version is None and not bool(environment.get("bound", False)):
        current_model_version = data.get("current_model_version")
    if current_model_version is None:
        return None, "current_model_version_unavailable"
    versions = list(data.get("contributing_model_versions") or [])
    if versions and int(current_model_version) < max(int(value) for value in versions):
        return None, "calibration_is_ahead_of_current_model"
    if current_iteration is not None:
        try:
            age = int(current_iteration) - int(data.get("iteration"))
        except (TypeError, ValueError):
            return None, "calibration_iteration_missing"
        max_age = int(_cfg_value(block, "max_model_age_iterations", 10))
        if age < 0 or age > max_age:
            return None, "calibration_model_too_old"
    if not bool(data.get("usable_for_acquisition", False)):
        return None, str(data.get("activation_reason") or "not_enough_records")
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
    if str(table_key) != "global_total":
        return None
    try:
        validated = validate_model(model)
    except CalibrationContractError:
        return None
    if not bool(validated.get("usable_for_acquisition", False)):
        return None
    return lookup_calibrated_error(
        validated,
        raw_uncertainty,
        application_uncertainty_scale=application_uncertainty_scale,
    )


def write_iteration_audit(iter_dir: Any, payload: Mapping[str, Any]) -> Path:
    path = audit_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = CALIBRATION_AUDIT_SCHEMA_VERSION
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
    campaign_dir: Any,
    iteration: int,
    fallback_model_version: int,
    sampling_aggressiveness: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    from ..versioning.provenance import PROVENANCE_FILENAME, read_provenance
    from ..versioning.manifest import sha256_file

    skipped: List[str] = []
    campaign_root = Path(campaign_dir).resolve()
    try:
        resolved_pointdir = pointdir.resolve(strict=True)
        pointdir_relative = resolved_pointdir.relative_to(campaign_root)
    except (OSError, ValueError):
        return [], ["pointdir_path_escapes_campaign"]
    if pointdir.is_symlink() or not resolved_pointdir.is_dir():
        return [], ["pointdir_missing_or_symlinked"]
    try:
        provenance = read_provenance(resolved_pointdir)
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

    model_version_raw = source.get("model_version")
    if (
        isinstance(model_version_raw, bool)
        or not isinstance(model_version_raw, int)
        or int(model_version_raw) != int(fallback_model_version)
    ):
        return [], ["missing_or_mismatched_model_version"]
    model_version = int(model_version_raw)
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
    model_set_hash = str(source.get("model_set_sha256") or "")
    environment_digest = str(
        source.get("environment_generation_digest_sha256") or ""
    )
    context_digest = str(source.get("calibration_context_sha256") or "")
    sampling_protocol_digest = str(source.get("sampling_protocol_sha256") or "")
    digests = (
        model_set_hash,
        environment_digest,
        context_digest,
        sampling_protocol_digest,
    )
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digests
    ):
        return [], ["missing_or_invalid_calibration_binding_digest"]
    environment_generation = source.get("environment_generation")
    if isinstance(environment_generation, bool) or not isinstance(
        environment_generation, int
    ):
        return [], ["missing_or_invalid_environment_generation"]
    if total_variance is None or total_variance < 0.0:
        return [], ["missing_or_invalid_total_energy_variance"]
    provenance_path = resolved_pointdir / PROVENANCE_FILENAME
    result_relative = Path(str(source.get("result_json") or ""))
    if result_relative.is_absolute() or not result_relative.parts:
        return [], ["missing_or_invalid_result_json_path"]
    try:
        result_path = (campaign_root / result_relative).resolve()
        result_path.relative_to(campaign_root)
    except (OSError, ValueError):
        return [], ["result_json_path_escapes_campaign"]
    if (
        provenance_path.is_symlink()
        or not provenance_path.is_file()
        or result_path.is_symlink()
        or not result_path.is_file()
    ):
        return [], ["calibration_source_file_missing_or_symlinked"]
    source_digests = {
        "provenance_json_sha256": sha256_file(provenance_path),
        "result_json_sha256": sha256_file(result_path),
        "sampling_protocol_sha256": sampling_protocol_digest,
    }
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
            "model_set_sha256": model_set_hash,
            "prior_mean_contract_sha256": prior_contract_hash,
            "environment_generation": int(environment_generation),
            "environment_generation_digest_sha256": environment_digest,
            "calibration_context_sha256": context_digest,
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
            "abs_error_ha_per_sqrt_atom": float(
                abs(predicted_iqa - true_iqa)
            ),
            "raw_uncertainty": float(raw_uncertainty),
            "raw_atom_variance": float(raw_uncertainty),
            "raw_total_energy_variance": total_variance,
            "raw_total_score": raw_total_score,
            "landing_policy": str(source.get("landing_policy", "unknown")),
            "safety_metrics": dict(source.get("safety_metrics") or {}),
            "source_digests": dict(source_digests),
            "provenance": {
                "pointdir": pointdir_relative.as_posix(),
                "provenance_json": provenance_path.resolve().relative_to(
                    campaign_root
                ).as_posix(),
                "result_json": result_relative.as_posix(),
            },
        }
        record["record_id"] = _record_key(record)
        try:
            records.append(validate_record(record))
        except CalibrationContractError as exc:
            return [], ["calibration_record_contract:" + str(exc)]
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
            campaign_dir=campaign_dir,
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
    environment = active_environment_binding(campaign_dir)
    model_error = None
    try:
        model = build_calibration_model(
            all_records,
            config,
            iteration=int(iteration),
            current_model_version=int(models_version),
            environment_binding=environment,
        )
        model_file = write_calibration_model(campaign_dir, model)
    except Exception as exc:
        model_error = type(exc).__name__ + ": " + str(exc)
        model_file = mark_calibration_model_stale(
            campaign_dir,
            reason="calibration_model_build_failed: " + model_error,
            iteration=int(iteration),
        )
        model = {
            "n_records": 0,
            "n_total_error_records": 0,
            "usable_for_acquisition": False,
        }
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
        "model_error": model_error,
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
    config: Optional[Any] = None,
    environment_binding: Optional[Mapping[str, Any]] = None,
    model_set_sha256: Optional[str] = None,
    sampling_protocol_sha256: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if config is None:
        from ..config import CampaignConfig

        config = CampaignConfig()
        config.campaign.sampling_aggressiveness = int(sampling_aggressiveness)
    elif int(config.campaign.sampling_aggressiveness) != int(
        sampling_aggressiveness
    ):
        raise ErrorCalibrationError(
            "synthetic calibration aggressiveness does not match config"
        )
    if prior_mean_contract_sha256 is None:
        from ..ferebus_prior import resolve_ferebus_prior_contract

        prior_mean_contract_sha256 = resolve_ferebus_prior_contract(
            config
        ).contract_sha256
    environment = dict(
        environment_binding
        or {
            "generation": 0,
            "generation_digest_sha256": "0" * 64,
        }
    )
    if model_set_sha256 is None:
        model_set_sha256 = canonical_sha256(
            {"dry_run_model_version": int(models_version)}
        )
    if sampling_protocol_sha256 is None:
        sampling_protocol_sha256 = canonical_sha256(
            {
                "dry_run_iteration": int(iteration),
                "sampling_aggressiveness": int(sampling_aggressiveness),
            }
        )
    context_sha = calibration_context_sha256(
        config,
        prior_mean_contract_sha256=str(prior_mean_contract_sha256),
        environment_generation_digest_sha256=str(
            environment["generation_digest_sha256"]
        ),
    )
    records = []
    for i in range(int(n_points)):
        predicted = -1.0 - 1.0e-4 * i
        true = -1.0 - 2.0e-4 * i
        record = {
            "schema_version": ERROR_CALIBRATION_SCHEMA_VERSION,
            "iteration": int(iteration),
            "model_version": int(models_version),
            "model_set_sha256": str(model_set_sha256),
            "prior_mean_contract_sha256": str(prior_mean_contract_sha256),
            "environment_generation": int(environment["generation"]),
            "environment_generation_digest_sha256": str(
                environment["generation_digest_sha256"]
            ),
            "calibration_context_sha256": context_sha,
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
            "abs_error_ha_per_sqrt_atom": float(abs(predicted - true)),
            "raw_uncertainty": float(1.0e-3 + 1.0e-4 * i),
            "raw_atom_variance": float(1.0e-3 + 1.0e-4 * i),
            "raw_total_energy_variance": float(1.0e-3 + 1.0e-4 * i),
            "raw_total_score": float(1.0 + 1.0e-2 * i),
            "landing_policy": "dry_run",
            "safety_metrics": {"synthetic": True},
            "source_digests": {
                "provenance_json_sha256": canonical_sha256(
                    {"dry_provenance": int(i)}
                ),
                "result_json_sha256": canonical_sha256(
                    {"dry_result": int(i)}
                ),
                "sampling_protocol_sha256": str(sampling_protocol_sha256),
            },
            "provenance": {
                "pointdir": staging_pointdir_name(i),
                "provenance_json": staging_pointdir_name(i) + "/.provenance.json",
                "result_json": "dry-result-" + str(i) + ".json",
            },
        }
        record["record_id"] = _record_key(record)
        records.append(validate_record(record))
    return records
