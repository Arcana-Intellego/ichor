"""Schema-v3 to schema-v4 campaign migration."""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable


class CampaignMigrationError(ValueError):
    """Raised when a campaign payload cannot be migrated safely."""


BACKENDS = ("polus", "gaussian", "aimall", "ariadne", "ferebus")


def _reject_grouped_v3(resources: Dict[str, Any]) -> None:
    grouped = [name for name in ("defaults", *BACKENDS) if name in resources]
    if grouped:
        raise CampaignMigrationError(
            "schema v3 resources cannot contain schema-v4 grouped keys "
            + repr(sorted(grouped))
        )


def _pop(resources: Dict[str, Any], key: str, default: Any = None) -> Any:
    return resources.pop(key) if key in resources else default


def _leftovers(resources: Dict[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(resources) - set(allowed))
    if extra:
        raise CampaignMigrationError(
            "schema v3 resources contain unknown keys that cannot migrate to v4: "
            + repr(extra)
        )


def migrate_v3_to_v4(data: Dict[str, Any]) -> Dict[str, Any]:
    migrated = copy.deepcopy(data)
    migrated["schema_version"] = 4
    resources = migrated.get("resources")
    if resources is None:
        resources = {}
    if not isinstance(resources, dict):
        raise CampaignMigrationError("resources must be a mapping")
    resources = copy.deepcopy(resources)
    _reject_grouped_v3(resources)

    known_non_backend = {
        "partition",
        "default_walltime_hours",
        "array_concurrency_limit",
        "fail_on_memory_estimate_exceeds_request",
        "gradient_parallel_backend",
    }
    known_backend = {
        backend + suffix
        for backend in BACKENDS
        for suffix in ("_walltime_hours", "_cpus_per_task", "_mem_per_cpu")
    }
    known_gaussian = {
        "gaussian_memory_mode",
        "gaussian_link0_mem",
        "gaussian_memory_fraction_of_slurm",
    }
    _leftovers(resources, known_non_backend | known_backend | known_gaussian)

    new_resources: Dict[str, Any] = {
        "defaults": {
            "partition": _pop(resources, "partition", "multicore"),
            "walltime_hours": _pop(resources, "default_walltime_hours", 24),
            "cpus_per_task": "auto",
            "mem_per_cpu": "auto",
        },
        "array_concurrency_limit": _pop(resources, "array_concurrency_limit", None),
        "fail_on_memory_estimate_exceeds_request": _pop(
            resources,
            "fail_on_memory_estimate_exceeds_request",
            True,
        ),
        "gradient_parallel_backend": _pop(
            resources,
            "gradient_parallel_backend",
            "process",
        ),
    }
    for backend in BACKENDS:
        block: Dict[str, Any] = {
            "partition": None,
            "walltime_hours": _pop(resources, backend + "_walltime_hours", None),
            "cpus_per_task": _pop(resources, backend + "_cpus_per_task", None),
            "mem_per_cpu": _pop(resources, backend + "_mem_per_cpu", None),
        }
        if backend == "gaussian":
            block.update(
                {
                    "memory_mode": _pop(resources, "gaussian_memory_mode", "slurm_env"),
                    "link0_mem": _pop(resources, "gaussian_link0_mem", "8GB"),
                    "memory_fraction_of_slurm": _pop(
                        resources,
                        "gaussian_memory_fraction_of_slurm",
                        0.85,
                    ),
                }
            )
        new_resources[backend] = block
    _leftovers(resources, set())
    migrated["resources"] = new_resources
    return migrated


__all__ = ["migrate_v3_to_v4"]
