"""Strict schema-v2 contracts for empirical IQA error calibration."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ichor.core.adversarial.error_calibration import CALIBRATION_OUTPUT_UNITS

from ..execution_identity import environment_current_path
from ..strict_json import strict_json as json


CALIBRATION_RECORDS_SCHEMA_VERSION = 2
CALIBRATION_MODEL_SCHEMA_VERSION = 2
CALIBRATION_AUDIT_SCHEMA_VERSION = 2
CALIBRATION_ESTIMATOR = "grouped_quantile_isotonic_v1"
CALIBRATION_MODEL_POLICY = "rolling_normalised"
UNBOUND_ENVIRONMENT_DIGEST = "0" * 64


class CalibrationContractError(ValueError):
    """Raised when calibration evidence violates schema v2."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError(
            "calibration content is not canonical finite JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationContractError(label + " must be an exact integer")
    parsed = int(value)
    if parsed < int(minimum):
        raise CalibrationContractError(
            label + " must be >= " + str(int(minimum))
        )
    return parsed


def finite_number(
    value: Any,
    label: str,
    *,
    minimum: Optional[float] = None,
    allow_none: bool = False,
) -> Optional[float]:
    if value is None and allow_none:
        return None
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationContractError(label + " must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CalibrationContractError(label + " must be finite")
    if minimum is not None and parsed < float(minimum):
        raise CalibrationContractError(label + " must be >= " + str(float(minimum)))
    return parsed


def digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CalibrationContractError(label + " must be a lowercase SHA-256")
    return value


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CalibrationContractError(label + " must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise CalibrationContractError(label + " contains a control character")
    return value


def estimator_settings(config: Any) -> Dict[str, Any]:
    block = getattr(config, "error_calibration", config)
    return {
        "schema_version": 1,
        "estimator": CALIBRATION_ESTIMATOR,
        "model_policy": CALIBRATION_MODEL_POLICY,
        "output_units": CALIBRATION_OUTPUT_UNITS,
        "min_records_to_apply": int(getattr(block, "min_records_to_apply", 100)),
        "min_model_versions_to_apply": int(
            getattr(block, "min_model_versions_to_apply", 2)
        ),
        "n_bins": int(getattr(block, "n_bins", 10)),
        "min_bin_records": int(getattr(block, "min_bin_records", 8)),
        "max_records": int(getattr(block, "max_records", 5000)),
        "max_model_age_iterations": int(
            getattr(block, "max_model_age_iterations", 10)
        ),
        "quantile": float(getattr(block, "quantile", 0.75)),
        "group_by_atom_type": bool(getattr(block, "group_by_atom_type", True)),
        "group_by_landing_policy": bool(
            getattr(block, "group_by_landing_policy", False)
        ),
        "aggressiveness_match_required": True,
        "environment_match_required": True,
    }


def estimator_settings_sha256(config: Any) -> str:
    return canonical_sha256(estimator_settings(config))


def active_environment_binding(campaign_dir: Any) -> Dict[str, Any]:
    """Read the active generation, or return an explicit unit-test sentinel."""
    path = environment_current_path(campaign_dir)
    if not path.is_file() or path.is_symlink():
        return {
            "generation": 0,
            "generation_digest_sha256": UNBOUND_ENVIRONMENT_DIGEST,
            "bound": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationContractError(
            "active environment-generation record is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise CalibrationContractError(
            "active environment-generation record must be an object"
        )
    if payload.get("schema_version") != 1:
        raise CalibrationContractError("unsupported active environment schema")
    return {
        "generation": exact_int(
            payload.get("generation"),
            "environment generation",
        ),
        "generation_digest_sha256": digest(
            payload.get("generation_digest_sha256"),
            "environment generation digest",
        ),
        "bound": True,
    }


def calibration_context_payload(
    config: Any,
    *,
    prior_mean_contract_sha256: str,
    environment_generation_digest_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "estimator_settings_sha256": estimator_settings_sha256(config),
        "prior_mean_contract_sha256": digest(
            prior_mean_contract_sha256,
            "calibration prior-mean digest",
        ),
        "sampling_aggressiveness": exact_int(
            int(config.campaign.sampling_aggressiveness),
            "sampling aggressiveness",
            minimum=1,
        ),
        "environment_generation_digest_sha256": digest(
            environment_generation_digest_sha256,
            "calibration environment digest",
        ),
    }


def calibration_context_sha256(
    config: Any,
    *,
    prior_mean_contract_sha256: str,
    environment_generation_digest_sha256: str,
) -> str:
    return canonical_sha256(
        calibration_context_payload(
            config,
            prior_mean_contract_sha256=prior_mean_contract_sha256,
            environment_generation_digest_sha256=(
                environment_generation_digest_sha256
            ),
        )
    )


def record_identity_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "iteration",
            "model_version",
            "model_set_sha256",
            "prior_mean_contract_sha256",
            "environment_generation_digest_sha256",
            "pointdir",
            "atom",
            "property",
        )
    }


def record_key(record: Mapping[str, Any]) -> str:
    return canonical_sha256(record_identity_payload(record))


def _validate_source_digests(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise CalibrationContractError("calibration source_digests must be an object")
    required = (
        "provenance_json_sha256",
        "result_json_sha256",
        "sampling_protocol_sha256",
    )
    return {key: digest(value.get(key), "calibration " + key) for key in required}


def validate_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CalibrationContractError("calibration record must be an object")
    record = dict(raw)
    if record.get("schema_version") != CALIBRATION_RECORDS_SCHEMA_VERSION:
        raise CalibrationContractError("unsupported calibration record schema")
    record["iteration"] = exact_int(record.get("iteration"), "record iteration")
    record["model_version"] = exact_int(
        record.get("model_version"), "record model_version"
    )
    record["model_set_sha256"] = digest(
        record.get("model_set_sha256"), "record model-set digest"
    )
    record["prior_mean_contract_sha256"] = digest(
        record.get("prior_mean_contract_sha256"), "record prior-mean digest"
    )
    record["environment_generation"] = exact_int(
        record.get("environment_generation"), "record environment generation"
    )
    record["environment_generation_digest_sha256"] = digest(
        record.get("environment_generation_digest_sha256"),
        "record environment digest",
    )
    record["calibration_context_sha256"] = digest(
        record.get("calibration_context_sha256"), "record calibration context"
    )
    record["pointdir"] = nonempty_string(record.get("pointdir"), "record pointdir")
    if Path(record["pointdir"]).name != record["pointdir"]:
        raise CalibrationContractError("record pointdir must be a basename")
    seed_id = record.get("seed_id")
    record["seed_id"] = (
        None if seed_id is None else exact_int(seed_id, "record seed_id", minimum=1)
    )
    seed_uid = record.get("seed_uid")
    record["seed_uid"] = None if seed_uid is None else digest(seed_uid, "record seed_uid")
    record["sampling_aggressiveness"] = exact_int(
        record.get("sampling_aggressiveness"),
        "record sampling aggressiveness",
        minimum=1,
    )
    if record["sampling_aggressiveness"] > 10:
        raise CalibrationContractError("record sampling aggressiveness must be <= 10")
    record["frame_atom_count"] = exact_int(
        record.get("frame_atom_count"), "record frame_atom_count", minimum=1
    )
    record["frame_atom_identity_sha256"] = digest(
        record.get("frame_atom_identity_sha256"), "record atom-identity digest"
    )
    record["atom"] = nonempty_string(record.get("atom"), "record atom")
    record["atom_type"] = nonempty_string(record.get("atom_type"), "record atom_type")
    if nonempty_string(record.get("property"), "record property") != "iqa":
        raise CalibrationContractError("calibration record property must be iqa")
    record["property"] = "iqa"
    predicted = finite_number(record.get("predicted_iqa_ha"), "predicted IQA")
    truth = finite_number(record.get("true_iqa_ha"), "true IQA")
    observed_error = finite_number(
        record.get("abs_error_ha"), "absolute IQA error", minimum=0.0
    )
    expected_error = abs(float(predicted) - float(truth))
    if not math.isclose(
        float(observed_error), expected_error, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise CalibrationContractError("record absolute IQA error is inconsistent")
    record["predicted_iqa_ha"] = float(predicted)
    record["true_iqa_ha"] = float(truth)
    record["abs_error_ha"] = float(observed_error)
    normalised_error = finite_number(
        record.get("abs_error_ha_per_sqrt_atom"),
        "normalised absolute IQA error",
        minimum=0.0,
    )
    if not math.isclose(
        float(normalised_error),
        float(observed_error),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise CalibrationContractError(
            "per-atom normalised IQA error must equal the per-atom IQA error"
        )
    record["abs_error_ha_per_sqrt_atom"] = float(normalised_error)
    for field_name in (
        "raw_uncertainty",
        "raw_atom_variance",
        "raw_total_energy_variance",
    ):
        record[field_name] = float(
            finite_number(record.get(field_name), field_name, minimum=0.0)
        )
    record["raw_total_score"] = finite_number(
        record.get("raw_total_score"),
        "raw total score",
        allow_none=True,
    )
    record["landing_policy"] = nonempty_string(
        record.get("landing_policy"), "record landing policy"
    )
    if not isinstance(record.get("safety_metrics"), Mapping):
        raise CalibrationContractError("record safety_metrics must be an object")
    record["safety_metrics"] = dict(record["safety_metrics"])
    canonical_json_bytes(record["safety_metrics"])
    record["source_digests"] = _validate_source_digests(
        record.get("source_digests")
    )
    expected_id = record_key(record)
    if record.get("record_id") != expected_id:
        raise CalibrationContractError("calibration record_id mismatch")
    record["record_id"] = expected_id
    canonical_json_bytes(record)
    return record


def validate_table(raw: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CalibrationContractError(label + " table must be an object")
    table = dict(raw)
    n_records = exact_int(table.get("n_records"), label + " n_records")
    if not isinstance(table.get("usable"), bool):
        raise CalibrationContractError(label + " usable must be a boolean")
    if table.get("estimator") != CALIBRATION_ESTIMATOR:
        raise CalibrationContractError(label + " estimator mismatch")
    quantile = finite_number(
        table.get("quantile"), label + " quantile", minimum=0.0
    )
    if float(quantile) <= 0.0 or float(quantile) > 1.0:
        raise CalibrationContractError(label + " quantile must be in (0, 1]")
    bins = table.get("bins")
    if not isinstance(bins, list):
        raise CalibrationContractError(label + " bins must be a list")
    parsed_bins = []
    previous_max = None
    total_bin_records = 0
    for index, raw_bin in enumerate(bins):
        if not isinstance(raw_bin, Mapping):
            raise CalibrationContractError(label + " bin must be an object")
        entry = dict(raw_bin)
        count = exact_int(entry.get("n"), label + " bin n", minimum=1)
        low = finite_number(
            entry.get("raw_uncertainty_min"), label + " bin minimum", minimum=0.0
        )
        high = finite_number(
            entry.get("raw_uncertainty_max"), label + " bin maximum", minimum=0.0
        )
        if float(high) < float(low):
            raise CalibrationContractError(label + " bin bounds are reversed")
        if previous_max is not None and float(low) <= float(previous_max):
            raise CalibrationContractError(label + " bins overlap or split tied scores")
        previous_max = float(high)
        for field_name in (
            "mean_abs_error_ha_per_sqrt_atom",
            "median_abs_error_ha_per_sqrt_atom",
            "q90_abs_error_ha_per_sqrt_atom",
            "raw_calibrated_abs_error_ha_per_sqrt_atom",
            "calibrated_abs_error_ha_per_sqrt_atom",
        ):
            entry[field_name] = float(
                finite_number(
                    entry.get(field_name),
                    label + " " + field_name,
                    minimum=0.0,
                )
            )
        entry["calibration_quantile"] = float(
            finite_number(
                entry.get("calibration_quantile"),
                label + " calibration quantile",
                minimum=0.0,
            )
        )
        if not math.isclose(
            entry["calibration_quantile"], float(quantile), rel_tol=0.0, abs_tol=0.0
        ):
            raise CalibrationContractError(label + " bin quantile mismatch")
        entry["n"] = count
        entry["raw_uncertainty_min"] = float(low)
        entry["raw_uncertainty_max"] = float(high)
        parsed_bins.append(entry)
        total_bin_records += count
    if total_bin_records != n_records:
        raise CalibrationContractError(label + " bin counts do not match n_records")
    if bool(table["usable"]) and not parsed_bins:
        raise CalibrationContractError(label + " usable table has no bins")
    table.update(
        {
            "n_records": n_records,
            "quantile": float(quantile),
            "bins": parsed_bins,
        }
    )
    return table


def validate_model(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CalibrationContractError("calibration model must be an object")
    model = dict(raw)
    if model.get("schema_version") != CALIBRATION_MODEL_SCHEMA_VERSION:
        raise CalibrationContractError("unsupported calibration model schema")
    if bool(model.get("stale", False)):
        if bool(model.get("usable_for_acquisition", False)):
            raise CalibrationContractError("stale calibration model cannot be usable")
        nonempty_string(model.get("stale_reason"), "stale calibration reason")
        stale_iteration = model.get("iteration")
        if stale_iteration is not None:
            model["iteration"] = exact_int(
                stale_iteration, "stale calibration iteration"
            )
        if model.get("tables") != {}:
            raise CalibrationContractError("stale calibration model tables must be empty")
        canonical_json_bytes(model)
        return model
    model["iteration"] = exact_int(model.get("iteration"), "model iteration")
    for key in ("n_records", "n_total_error_records", "n_total_records"):
        model[key] = exact_int(model.get(key), "model " + key)
    if model["n_records"] > model["n_total_records"]:
        raise CalibrationContractError("model usable-record count exceeds total records")
    if model["n_total_error_records"] > model["n_records"]:
        raise CalibrationContractError("model total-frame count exceeds atom records")
    if model.get("output_units") != CALIBRATION_OUTPUT_UNITS:
        raise CalibrationContractError("calibration output units mismatch")
    if model.get("estimator") != CALIBRATION_ESTIMATOR:
        raise CalibrationContractError("calibration estimator mismatch")
    if model.get("model_policy") != CALIBRATION_MODEL_POLICY:
        raise CalibrationContractError("calibration model policy mismatch")
    if model.get("uncertainty_axis") != "model_normalised":
        raise CalibrationContractError("calibration uncertainty axis mismatch")
    for key in (
        "estimator_settings_sha256",
        "calibration_context_sha256",
        "prior_mean_contract_sha256",
        "environment_generation_digest_sha256",
        "source_records_sha256",
    ):
        model[key] = digest(model.get(key), "calibration model " + key)
    model["environment_generation"] = exact_int(
        model.get("environment_generation"), "model environment generation"
    )
    model["sampling_aggressiveness"] = exact_int(
        model.get("sampling_aggressiveness"),
        "model sampling aggressiveness",
        minimum=1,
    )
    if model["sampling_aggressiveness"] > 10:
        raise CalibrationContractError("model sampling aggressiveness must be <= 10")
    settings = model.get("estimator_settings")
    if not isinstance(settings, Mapping):
        raise CalibrationContractError("calibration estimator settings are missing")
    if canonical_sha256(dict(settings)) != model["estimator_settings_sha256"]:
        raise CalibrationContractError("calibration estimator-settings digest mismatch")
    model["estimator_settings"] = dict(settings)
    source_ids = model.get("source_record_ids")
    if not isinstance(source_ids, list) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in source_ids
    ):
        raise CalibrationContractError("calibration source record IDs are invalid")
    if source_ids != sorted(set(source_ids)):
        raise CalibrationContractError("calibration source record IDs are not canonical")
    if canonical_sha256(source_ids) != model["source_records_sha256"]:
        raise CalibrationContractError("calibration source-record digest mismatch")
    if len(source_ids) != model["n_records"]:
        raise CalibrationContractError(
            "calibration source-record count does not match n_records"
        )
    contributing = model.get("contributing_model_versions")
    if not isinstance(contributing, list):
        raise CalibrationContractError("contributing model versions must be a list")
    parsed_versions = [
        exact_int(value, "contributing model version") for value in contributing
    ]
    if parsed_versions != sorted(set(parsed_versions)):
        raise CalibrationContractError(
            "contributing model versions are not sorted and unique"
        )
    model["contributing_model_versions"] = parsed_versions
    model["n_contributing_model_versions"] = exact_int(
        model.get("n_contributing_model_versions"),
        "number of contributing model versions",
    )
    if model["n_contributing_model_versions"] != len(parsed_versions):
        raise CalibrationContractError(
            "contributing model-version count is inconsistent"
        )
    current_version = model.get("current_model_version")
    model["current_model_version"] = (
        None
        if current_version is None
        else exact_int(current_version, "current model version")
    )
    normalisation = model.get("model_uncertainty_normalisation")
    if not isinstance(normalisation, Mapping):
        raise CalibrationContractError("model uncertainty normalisation is missing")
    for version in parsed_versions:
        entry = normalisation.get(str(version))
        if not isinstance(entry, Mapping):
            raise CalibrationContractError(
                "model uncertainty normalisation is missing version " + str(version)
            )
        finite_number(
            entry.get("raw_uncertainty_median"),
            "model uncertainty normalisation scale",
            minimum=0.0,
        )
        if float(entry["raw_uncertainty_median"]) <= 0.0:
            raise CalibrationContractError(
                "model uncertainty normalisation scale must be positive"
            )
    model["model_uncertainty_normalisation"] = {
        str(key): dict(value) for key, value in normalisation.items()
    }
    model["reference_error_ha_per_sqrt_atom"] = float(
        finite_number(
            model.get("reference_error_ha_per_sqrt_atom"),
            "calibration reference error",
            minimum=0.0,
        )
    )
    tables = model.get("tables")
    if not isinstance(tables, Mapping):
        raise CalibrationContractError("calibration model tables are missing")
    parsed_tables = {
        str(key): validate_table(value, "calibration " + str(key))
        for key, value in tables.items()
    }
    if "global_total" not in parsed_tables or "global" not in parsed_tables:
        raise CalibrationContractError("calibration global tables are missing")
    model["tables"] = parsed_tables
    if not isinstance(model.get("usable_for_acquisition"), bool):
        raise CalibrationContractError("calibration usability must be a boolean")
    blockers = model.get("activation_blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(value, str) or not value for value in blockers
    ):
        raise CalibrationContractError("calibration activation blockers are invalid")
    reason = nonempty_string(model.get("activation_reason"), "activation reason")
    if bool(model["usable_for_acquisition"]):
        if blockers:
            raise CalibrationContractError("usable calibration model has blockers")
        if reason != "usable":
            raise CalibrationContractError("usable calibration reason must be usable")
        if not bool(parsed_tables["global_total"].get("usable")):
            raise CalibrationContractError("usable model has unusable global-total table")
    elif not blockers:
        raise CalibrationContractError("unusable calibration model has no blocker")
    canonical_json_bytes(model)
    return model


def records_digest(records: Sequence[Mapping[str, Any]]) -> Tuple[list[str], str]:
    record_ids = sorted(str(record["record_id"]) for record in records)
    return record_ids, canonical_sha256(record_ids)


__all__ = [
    "CALIBRATION_AUDIT_SCHEMA_VERSION",
    "CALIBRATION_ESTIMATOR",
    "CALIBRATION_MODEL_POLICY",
    "CALIBRATION_MODEL_SCHEMA_VERSION",
    "CALIBRATION_RECORDS_SCHEMA_VERSION",
    "CalibrationContractError",
    "UNBOUND_ENVIRONMENT_DIGEST",
    "active_environment_binding",
    "calibration_context_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "estimator_settings",
    "estimator_settings_sha256",
    "record_key",
    "records_digest",
    "validate_model",
    "validate_record",
]
