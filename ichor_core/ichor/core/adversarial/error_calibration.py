"""Shared acquisition-facing empirical calibration lookup contract."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


CALIBRATION_MODEL_SCHEMA_VERSION = 2
CALIBRATION_OUTPUT_UNITS = "ha_per_sqrt_atom"


def finite_non_negative(value) -> Optional[float]:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def lookup_calibrated_error(
    model: Optional[Mapping[str, object]],
    raw_uncertainty,
    *,
    application_uncertainty_scale: Optional[float] = None,
) -> Optional[float]:
    """Map raw total variance to total IQA error in ``Ha/sqrt(atom)``.

    Zero is a valid calibrated error. Malformed, stale, or incompatible model
    objects return ``None`` so both seed selection and ARIADNE take the same
    explicit raw-variance fallback path.
    """
    if not isinstance(model, Mapping):
        return None
    if model.get("schema_version") != CALIBRATION_MODEL_SCHEMA_VERSION:
        return None
    if str(model.get("output_units") or "") != CALIBRATION_OUTPUT_UNITS:
        return None
    raw = finite_non_negative(raw_uncertainty)
    if raw is None:
        return None
    if str(model.get("uncertainty_axis") or "") == "model_normalised":
        scale = finite_non_negative(application_uncertainty_scale)
        if scale is None or scale <= 0.0:
            return None
        raw /= scale
    tables = model.get("tables")
    if not isinstance(tables, Mapping):
        return None
    table = tables.get("global_total")
    if not isinstance(table, Mapping):
        return None
    bins = table.get("bins")
    if not isinstance(bins, list) or not bins:
        return None
    last_value = None
    for entry in bins:
        if not isinstance(entry, Mapping):
            return None
        high = finite_non_negative(entry.get("raw_uncertainty_max"))
        value = finite_non_negative(
            entry.get("calibrated_abs_error_ha_per_sqrt_atom")
        )
        if high is None or value is None:
            return None
        last_value = value
        if raw <= high:
            return value
    return last_value


__all__ = [
    "CALIBRATION_MODEL_SCHEMA_VERSION",
    "CALIBRATION_OUTPUT_UNITS",
    "finite_non_negative",
    "lookup_calibrated_error",
]
