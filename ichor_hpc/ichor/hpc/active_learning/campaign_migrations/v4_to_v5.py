"""Schema-v4 to schema-v5 campaign migration."""
from __future__ import annotations

import copy
from typing import Any, Dict


def _drop(mapping: Any, *keys: str) -> None:
    if not isinstance(mapping, dict):
        return
    for key in keys:
        mapping.pop(key, None)


def migrate_v4_to_v5(data: Dict[str, Any]) -> Dict[str, Any]:
    migrated = copy.deepcopy(data)
    migrated["schema_version"] = 5

    phase_b = migrated.get("phase_b")
    _drop(phase_b, "min_separation", "min_separation_scaled")

    geometry_novelty = migrated.get("geometry_novelty")
    _drop(geometry_novelty, "score_transform")

    acquisition = migrated.get("acquisition")
    if isinstance(acquisition, dict):
        acquisition.pop("movement_band", None)
        acquisition.pop("movement_utility", None)
        fullspace = acquisition.get("fullspace_confinement")
        _drop(
            fullspace,
            "residual_scale",
            "fixed_residual_scale_ang",
            "rmsd_scale_ang",
            "min_residual_scale_ang",
            "failure_penalty",
        )

    return migrated


__all__ = ["migrate_v4_to_v5"]
