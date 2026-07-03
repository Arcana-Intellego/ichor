"""Migrate campaign schema v7 to v8.

Schema v8 replaces bootstrap.external_validation_fraction with the exact
bootstrap.external_validation_size. The initial labelled size remains the total
number of bootstrap structures selected by POLUS, including the external
validation reserve.
"""
from __future__ import annotations

from typing import Any, Dict


def migrate_v7_to_v8(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload)
    bootstrap = data.get("bootstrap")
    if bootstrap is None:
        bootstrap = {}
    if not isinstance(bootstrap, dict):
        raise ValueError("bootstrap block must be a mapping for schema v7->v8")
    bootstrap = dict(bootstrap)
    initial = int(bootstrap.get("initial_labelled_size", 12))
    if "external_validation_size" not in bootstrap:
        fraction = float(bootstrap.pop("external_validation_fraction", 0.2))
        bootstrap["external_validation_size"] = int(round(float(initial) * fraction))
    else:
        bootstrap.pop("external_validation_fraction", None)
    data["bootstrap"] = bootstrap
    data["schema_version"] = 8
    return data


__all__ = ["migrate_v7_to_v8"]
