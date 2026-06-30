"""Schema-v2 to schema-v3 campaign migration."""
from __future__ import annotations

import copy
from typing import Any, Dict


class CampaignMigrationError(ValueError):
    """Raised when a campaign payload cannot be migrated safely."""


def _same_config_value(a: Any, b: Any) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip().lower() == str(b).strip().lower()
    return a == b


def _set_migrated_value(
    target: Dict[str, Any],
    key: str,
    value: Any,
    *,
    source_path: str,
    target_path: str,
) -> None:
    if key in target:
        if not _same_config_value(target[key], value):
            raise CampaignMigrationError(
                "Conflicting campaign resource settings: "
                + source_path
                + "="
                + repr(value)
                + " but "
                + target_path
                + "="
                + repr(target[key])
                + ". Use only "
                + target_path
                + "."
            )
        return
    target[key] = value


def migrate_v2_to_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    migrated = copy.deepcopy(data)
    migrated["schema_version"] = 3
    resources = migrated.setdefault("resources", {})
    gaussian = migrated.setdefault("gaussian", {})
    if not isinstance(resources, dict):
        raise CampaignMigrationError("resources must be a mapping")
    if not isinstance(gaussian, dict):
        raise CampaignMigrationError("gaussian must be a mapping")

    if "walltime_hours" in resources:
        _set_migrated_value(
            resources,
            "default_walltime_hours",
            resources.pop("walltime_hours"),
            source_path="resources.walltime_hours",
            target_path="resources.default_walltime_hours",
        )
    if "ntasks" in resources:
        old_ntasks = resources.pop("ntasks")
        try:
            ntasks_int = int(old_ntasks)
        except (TypeError, ValueError) as exc:
            raise CampaignMigrationError(
                "resources.ntasks in schema v2 must be 1 to migrate to schema v3"
            ) from exc
        if ntasks_int != 1:
            raise CampaignMigrationError(
                "resources.ntasks="
                + repr(old_ntasks)
                + " cannot migrate to schema v3; current live backends support "
                "only --ntasks=1"
            )
    if "mem_per_cpu" in resources:
        old_mem = resources.pop("mem_per_cpu")
        for backend in ("polus", "gaussian", "aimall", "ariadne", "ferebus"):
            _set_migrated_value(
                resources,
                backend + "_mem_per_cpu",
                old_mem,
                source_path="resources.mem_per_cpu",
                target_path="resources." + backend + "_mem_per_cpu",
            )
    old_cpus = resources.pop("cpus_per_task", None)
    old_gaussian_nproc = gaussian.pop("nproc", None)
    if old_cpus is not None:
        for backend in ("polus", "ferebus"):
            _set_migrated_value(
                resources,
                backend + "_cpus_per_task",
                old_cpus,
                source_path="resources.cpus_per_task",
                target_path="resources." + backend + "_cpus_per_task",
            )
        if old_gaussian_nproc is None:
            _set_migrated_value(
                resources,
                "gaussian_cpus_per_task",
                old_cpus,
                source_path="resources.cpus_per_task",
                target_path="resources.gaussian_cpus_per_task",
            )
    if old_gaussian_nproc is not None:
        _set_migrated_value(
            resources,
            "gaussian_cpus_per_task",
            old_gaussian_nproc,
            source_path="gaussian.nproc",
            target_path="resources.gaussian_cpus_per_task",
        )
    gaussian_resource_moves = {
        "mem": "gaussian_link0_mem",
        "memory_mode": "gaussian_memory_mode",
        "memory_fraction_of_slurm": "gaussian_memory_fraction_of_slurm",
    }
    for old_key, new_key in gaussian_resource_moves.items():
        if old_key in gaussian:
            _set_migrated_value(
                resources,
                new_key,
                gaussian.pop(old_key),
                source_path="gaussian." + old_key,
                target_path="resources." + new_key,
            )
    return migrated


__all__ = ["migrate_v2_to_v3"]
