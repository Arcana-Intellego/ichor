"""Strict empirical IQA calibration model construction and validation."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ichor.core.adversarial.error_calibration import CALIBRATION_OUTPUT_UNITS

from ..ferebus_prior import resolve_ferebus_prior_contract
from .error_calibration_contract import (
    CALIBRATION_ESTIMATOR,
    CALIBRATION_MODEL_POLICY,
    CALIBRATION_MODEL_SCHEMA_VERSION,
    UNBOUND_ENVIRONMENT_DIGEST,
    CalibrationContractError,
    calibration_context_sha256,
    digest,
    estimator_settings,
    estimator_settings_sha256,
    exact_int,
    records_digest,
    validate_model,
    validate_record,
)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        (1.0 - fraction) * sorted_values[lower]
        + fraction * sorted_values[upper]
    )


def _group_equal_predictors(
    pairs: Sequence[Tuple[float, float]],
) -> List[Tuple[float, List[float]]]:
    grouped: List[Tuple[float, List[float]]] = []
    for predictor, error in sorted(pairs, key=lambda item: (item[0], item[1])):
        if grouped and predictor == grouped[-1][0]:
            grouped[-1][1].append(float(error))
        else:
            grouped.append((float(predictor), [float(error)]))
    return grouped


def _partition_predictor_groups(
    groups: Sequence[Tuple[float, List[float]]],
    *,
    n_bins: int,
    min_bin_records: int,
) -> List[List[Tuple[float, List[float]]]]:
    if not groups:
        return []
    total = sum(len(errors) for _predictor, errors in groups)
    requested = min(
        int(n_bins),
        len(groups),
        max(1, total // max(1, int(min_bin_records))),
    )
    bins: List[List[Tuple[float, List[float]]]] = []
    cursor = 0
    remaining_records = total
    for bin_index in range(requested):
        bins_left = requested - bin_index
        target = float(remaining_records) / float(bins_left)
        chunk: List[Tuple[float, List[float]]] = []
        chunk_count = 0
        while cursor < len(groups):
            groups_after = len(groups) - (cursor + 1)
            records_after = remaining_records - len(groups[cursor][1])
            leave_groups = groups_after >= bins_left - 1
            leave_records = records_after >= (
                max(1, int(min_bin_records)) * (bins_left - 1)
            )
            if chunk and chunk_count >= target and leave_groups and leave_records:
                break
            group = groups[cursor]
            chunk.append(group)
            cursor += 1
            chunk_count += len(group[1])
            remaining_records -= len(group[1])
            if bins_left > 1 and cursor >= len(groups) - (bins_left - 1):
                break
        bins.append(chunk)
    if cursor < len(groups):
        bins[-1].extend(groups[cursor:])
    return [chunk for chunk in bins if chunk]


def _isotonic_quantile_values(
    bin_errors: Sequence[Sequence[float]],
    quantile: float,
) -> List[float]:
    blocks: List[Dict[str, Any]] = []
    for index, errors in enumerate(bin_errors):
        observations = sorted(float(value) for value in errors)
        blocks.append(
            {
                "start": int(index),
                "end": int(index),
                "observations": observations,
                "value": float(_percentile(observations, quantile)),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            observations = sorted(
                list(left["observations"]) + list(right["observations"])
            )
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "observations": observations,
                    "value": float(_percentile(observations, quantile)),
                }
            )
    fitted = [0.0] * len(bin_errors)
    for block in blocks:
        for index in range(int(block["start"]), int(block["end"]) + 1):
            fitted[index] = float(block["value"])
    return fitted


def make_calibration_table(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bins: int,
    min_bin_records: int,
    quantile: float,
) -> Dict[str, Any]:
    pairs = [
        (
            float(record["raw_uncertainty"]),
            float(record["abs_error_ha_per_sqrt_atom"]),
        )
        for record in records
    ]
    value_quantile = min(1.0, max(1.0e-12, float(quantile)))
    groups = _group_equal_predictors(pairs)
    chunks = _partition_predictor_groups(
        groups,
        n_bins=int(n_bins),
        min_bin_records=int(min_bin_records),
    )
    errors_by_bin = [
        [error for _predictor, errors in chunk for error in errors]
        for chunk in chunks
    ]
    raw_values = [
        _percentile(sorted(errors), value_quantile) for errors in errors_by_bin
    ]
    fitted_values = _isotonic_quantile_values(errors_by_bin, value_quantile)
    bins: List[Dict[str, Any]] = []
    for chunk, errors, raw_value, fitted_value in zip(
        chunks, errors_by_bin, raw_values, fitted_values
    ):
        predictors = [float(predictor) for predictor, _errors in chunk]
        sorted_errors = sorted(float(value) for value in errors)
        bins.append(
            {
                "raw_uncertainty_min": float(min(predictors)),
                "raw_uncertainty_max": float(max(predictors)),
                "n": int(len(sorted_errors)),
                "mean_abs_error_ha_per_sqrt_atom": float(mean(sorted_errors)),
                "median_abs_error_ha_per_sqrt_atom": float(median(sorted_errors)),
                "q90_abs_error_ha_per_sqrt_atom": _percentile(
                    sorted_errors, 0.90
                ),
                "calibration_quantile": float(value_quantile),
                "raw_calibrated_abs_error_ha_per_sqrt_atom": float(raw_value),
                "calibrated_abs_error_ha_per_sqrt_atom": float(fitted_value),
            }
        )
    return {
        "n_records": int(len(pairs)),
        "usable": bool(len(pairs) >= int(min_bin_records) and bins),
        "monotone": True,
        "estimator": CALIBRATION_ESTIMATOR,
        "quantile": float(value_quantile),
        "bins": bins,
    }


def _frame_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        record["iteration"],
        record["model_version"],
        record["model_set_sha256"],
        record["pointdir"],
        record["seed_id"],
        record["seed_uid"],
        record["prior_mean_contract_sha256"],
        record["environment_generation"],
        record["environment_generation_digest_sha256"],
        record["calibration_context_sha256"],
        record["source_digests"]["sampling_protocol_sha256"],
        record["property"],
    )


def _total_error_records(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_frame_key(record), []).append(record)
    totals: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        expected_atoms = int(group[0]["frame_atom_count"])
        atom_names = [str(record["atom"]) for record in group]
        identity_hashes = {record["frame_atom_identity_sha256"] for record in group}
        if (
            len(group) != expected_atoms
            or len(set(atom_names)) != expected_atoms
            or any(int(record["frame_atom_count"]) != expected_atoms for record in group)
            or len(identity_hashes) != 1
        ):
            continue
        raw_values = [float(record["raw_total_energy_variance"]) for record in group]
        raw_total = raw_values[0]
        if any(
            not math.isclose(value, raw_total, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for value in raw_values[1:]
        ):
            continue
        predicted_total = sum(float(record["predicted_iqa_ha"]) for record in group)
        true_total = sum(float(record["true_iqa_ha"]) for record in group)
        normalised_error = abs(predicted_total - true_total) / math.sqrt(expected_atoms)
        (
            iteration,
            model_version,
            model_set_sha256,
            pointdir,
            seed_id,
            seed_uid,
            prior_digest,
            environment_generation,
            environment_digest,
            context_digest,
            sampling_protocol_digest,
            property_name,
        ) = key
        totals.append(
            {
                "iteration": int(iteration),
                "model_version": int(model_version),
                "model_set_sha256": str(model_set_sha256),
                "pointdir": str(pointdir),
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "prior_mean_contract_sha256": str(prior_digest),
                "environment_generation": int(environment_generation),
                "environment_generation_digest_sha256": str(environment_digest),
                "calibration_context_sha256": str(context_digest),
                "sampling_protocol_sha256": str(sampling_protocol_digest),
                "property": str(property_name),
                "n_atoms": int(expected_atoms),
                "predicted_total_iqa_ha": float(predicted_total),
                "true_total_iqa_ha": float(true_total),
                "abs_error_ha": float(abs(predicted_total - true_total)),
                "abs_error_ha_per_sqrt_atom": float(normalised_error),
                "raw_uncertainty": float(raw_total),
                "raw_total_energy_variance": float(raw_total),
                "raw_total_score": group[0].get("raw_total_score"),
                "landing_policy": str(group[0]["landing_policy"]),
                "sampling_aggressiveness": int(
                    group[0]["sampling_aggressiveness"]
                ),
                "source_record_ids": sorted(str(record["record_id"]) for record in group),
            }
        )
    totals.sort(
        key=lambda record: (
            int(record["iteration"]),
            str(record["pointdir"]),
            int(record["model_version"]),
        )
    )
    return totals


def _normalise_by_model(
    total_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for record in total_records:
        grouped.setdefault(int(record["model_version"]), []).append(record)
    normalised: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for version in sorted(grouped):
        group = grouped[version]
        raw_values = sorted(float(record["raw_uncertainty"]) for record in group)
        positive = [value for value in raw_values if value > 0.0]
        scale = max(float(median(positive)) if positive else 1.0, 1.0e-18)
        realised = sorted(
            float(record["abs_error_ha_per_sqrt_atom"]) for record in group
        )
        diagnostics[str(version)] = {
            "n_total_frames": int(len(group)),
            "raw_uncertainty_median": float(scale),
            "raw_uncertainty_min": float(min(raw_values)),
            "raw_uncertainty_max": float(max(raw_values)),
            "realised_error_median_ha_per_sqrt_atom": float(median(realised)),
        }
        for record in group:
            item = dict(record)
            item["raw_uncertainty_unnormalised"] = float(item["raw_uncertainty"])
            item["raw_uncertainty"] = float(item["raw_uncertainty"]) / scale
            item["model_uncertainty_scale"] = float(scale)
            normalised.append(item)
    return normalised, diagnostics


def _environment_binding(
    records: Sequence[Mapping[str, Any]],
    supplied: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if supplied is not None:
        generation = exact_int(
            supplied.get("generation"), "calibration environment generation"
        )
        generation_digest = digest(
            supplied.get("generation_digest_sha256"),
            "calibration environment digest",
        )
        return {
            "generation": generation,
            "generation_digest_sha256": generation_digest,
        }
    bindings = {
        (
            int(record["environment_generation"]),
            str(record["environment_generation_digest_sha256"]),
        )
        for record in records
    }
    if len(bindings) > 1:
        raise CalibrationContractError(
            "calibration model build requires one active environment generation"
        )
    if bindings:
        generation, generation_digest = next(iter(bindings))
        return {
            "generation": int(generation),
            "generation_digest_sha256": digest(
                generation_digest, "calibration environment digest"
            ),
        }
    return {
        "generation": 0,
        "generation_digest_sha256": UNBOUND_ENVIRONMENT_DIGEST,
    }


def build_calibration_model_v2(
    records: Sequence[Mapping[str, Any]],
    config: Any,
    *,
    iteration: int,
    current_model_version: Optional[int] = None,
    environment_binding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    validated = [validate_record(record) for record in records]
    environment = _environment_binding(validated, environment_binding)
    prior_digest = resolve_ferebus_prior_contract(config).contract_sha256
    context_digest = calibration_context_sha256(
        config,
        prior_mean_contract_sha256=prior_digest,
        environment_generation_digest_sha256=environment[
            "generation_digest_sha256"
        ],
    )
    aggressiveness = int(config.campaign.sampling_aggressiveness)
    settings = estimator_settings(config)
    max_age = int(settings["max_model_age_iterations"])
    age_cutoff = int(iteration) - max_age

    aged = [record for record in validated if int(record["iteration"]) >= age_cutoff]
    prior_matched = [
        record
        for record in aged
        if record["prior_mean_contract_sha256"] == prior_digest
    ]
    aggressiveness_matched = [
        record
        for record in prior_matched
        if int(record["sampling_aggressiveness"]) == aggressiveness
    ]
    environment_matched = [
        record
        for record in aggressiveness_matched
        if int(record["environment_generation"]) == int(environment["generation"])
        and record["environment_generation_digest_sha256"]
        == environment["generation_digest_sha256"]
    ]
    context_matched = [
        record
        for record in environment_matched
        if record["calibration_context_sha256"] == context_digest
    ]
    total_records = _total_error_records(context_matched)
    normalised_totals, normalisation = _normalise_by_model(total_records)
    contributing_versions = sorted(
        {int(record["model_version"]) for record in normalised_totals}
    )
    valid_source_ids = {
        record_id
        for record in total_records
        for record_id in record["source_record_ids"]
    }
    source_records = [
        record for record in context_matched if record["record_id"] in valid_source_ids
    ]
    normalised_atoms: List[Dict[str, Any]] = []
    for record in source_records:
        scale = float(
            normalisation[str(record["model_version"])]["raw_uncertainty_median"]
        )
        item = dict(record)
        item["raw_uncertainty_unnormalised"] = float(item["raw_uncertainty"])
        item["raw_uncertainty"] = float(item["raw_uncertainty"]) / scale
        item["model_uncertainty_scale"] = scale
        normalised_atoms.append(item)

    n_bins = int(settings["n_bins"])
    min_bin_records = int(settings["min_bin_records"])
    quantile = float(settings["quantile"])
    tables: Dict[str, Any] = {
        "global": make_calibration_table(
            normalised_atoms,
            n_bins=n_bins,
            min_bin_records=min_bin_records,
            quantile=quantile,
        ),
        "global_total": make_calibration_table(
            normalised_totals,
            n_bins=n_bins,
            min_bin_records=min_bin_records,
            quantile=quantile,
        ),
    }
    if bool(settings["group_by_atom_type"]):
        atom_types = sorted({record["atom_type"] for record in normalised_atoms})
        for atom_type in atom_types:
            table = make_calibration_table(
                [
                    record
                    for record in normalised_atoms
                    if record["atom_type"] == atom_type
                ],
                n_bins=n_bins,
                min_bin_records=min_bin_records,
                quantile=quantile,
            )
            if table["usable"]:
                tables["atom_type:" + atom_type] = table
    if bool(settings["group_by_landing_policy"]):
        policies = sorted(
            {record["landing_policy"] for record in normalised_atoms}
        )
        for policy in policies:
            table = make_calibration_table(
                [
                    record
                    for record in normalised_atoms
                    if record["landing_policy"] == policy
                ],
                n_bins=n_bins,
                min_bin_records=min_bin_records,
                quantile=quantile,
            )
            if table["usable"]:
                tables["landing_policy:" + policy] = table

    errors = sorted(
        float(record["abs_error_ha_per_sqrt_atom"])
        for record in normalised_totals
    )
    reference_error = _percentile(errors, 0.50) if errors else 0.0
    blockers: List[str] = []
    if len(normalised_totals) < int(settings["min_records_to_apply"]):
        blockers.append("not_enough_total_records")
    if len(contributing_versions) < int(settings["min_model_versions_to_apply"]):
        blockers.append("not_enough_model_versions")
    if not bool(tables["global_total"]["usable"]):
        blockers.append("global_total_unusable")

    source_ids, source_digest = records_digest(source_records)
    records_by_model: Dict[str, int] = {}
    records_by_iteration: Dict[str, int] = {}
    for record in source_records:
        model_key = str(record["model_version"])
        iteration_key = str(record["iteration"])
        records_by_model[model_key] = records_by_model.get(model_key, 0) + 1
        records_by_iteration[iteration_key] = (
            records_by_iteration.get(iteration_key, 0) + 1
        )
    resolved_current_version = current_model_version
    if resolved_current_version is None and validated:
        resolved_current_version = max(
            int(record["model_version"]) for record in validated
        )
    model = {
        "schema_version": CALIBRATION_MODEL_SCHEMA_VERSION,
        "iteration": int(iteration),
        "n_records": int(len(source_records)),
        "n_total_error_records": int(len(normalised_totals)),
        "n_total_records": int(len(validated)),
        "n_records_by_model_version": records_by_model,
        "n_records_by_iteration_window": records_by_iteration,
        "model_policy": CALIBRATION_MODEL_POLICY,
        "uncertainty_axis": "model_normalised",
        "model_uncertainty_normalisation": normalisation,
        "contributing_model_versions": contributing_versions,
        "n_contributing_model_versions": int(len(contributing_versions)),
        "current_model_version": (
            None
            if resolved_current_version is None
            else int(resolved_current_version)
        ),
        "sampling_aggressiveness": int(aggressiveness),
        "prior_mean_contract_sha256": prior_digest,
        "environment_generation": int(environment["generation"]),
        "environment_generation_digest_sha256": environment[
            "generation_digest_sha256"
        ],
        "calibration_context_sha256": context_digest,
        "estimator": CALIBRATION_ESTIMATOR,
        "estimator_settings": settings,
        "estimator_settings_sha256": estimator_settings_sha256(config),
        "source_record_ids": source_ids,
        "source_records_sha256": source_digest,
        "n_age_excluded": int(len(validated) - len(aged)),
        "n_prior_contract_mismatch_excluded": int(len(aged) - len(prior_matched)),
        "n_aggressiveness_mismatch_excluded": int(
            len(prior_matched) - len(aggressiveness_matched)
        ),
        "n_environment_mismatch_excluded": int(
            len(aggressiveness_matched) - len(environment_matched)
        ),
        "n_calibration_context_mismatch_excluded": int(
            len(environment_matched) - len(context_matched)
        ),
        "n_bins": n_bins,
        "min_bin_records": min_bin_records,
        "min_records_to_apply": int(settings["min_records_to_apply"]),
        "min_model_versions_to_apply": int(
            settings["min_model_versions_to_apply"]
        ),
        "max_model_age_iterations": max_age,
        "quantile": quantile,
        "reference_error_ha_per_sqrt_atom": float(reference_error),
        "group_by_atom_type": bool(settings["group_by_atom_type"]),
        "group_by_landing_policy": bool(settings["group_by_landing_policy"]),
        "output_units": CALIBRATION_OUTPUT_UNITS,
        "usable_for_acquisition": not blockers,
        "activation_reason": "usable" if not blockers else blockers[0],
        "activation_blockers": blockers,
        "tables": tables,
    }
    return validate_model(model)


__all__ = ["build_calibration_model_v2", "make_calibration_table"]
