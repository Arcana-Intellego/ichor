"""Cluster-profile helpers for live active-learning backends."""
from __future__ import annotations

import os
import importlib
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Optional


class ClusterProfileError(RuntimeError):
    """Raised when a live cluster profile cannot be resolved."""


@dataclass(frozen=True)
class ClusterProfile:
    machine: str
    config: dict


def _load_global_config():
    try:
        global_variables = importlib.import_module("ichor.hpc.global_variables")
    except Exception as exc:
        raise ClusterProfileError(
            "could not load ~/ichor_config.yaml or resolve ICHOR_MACHINE: "
            + type(exc).__name__
            + ": "
            + str(exc)
        ) from exc
    return global_variables


def active_machine() -> Optional[str]:
    """Return the resolved ICHOR machine/profile name, if any."""
    try:
        return getattr(_load_global_config(), "MACHINE", None) or None
    except ClusterProfileError:
        return None


def require_cluster_profile() -> ClusterProfile:
    """Return the active machine profile or raise a clear live-mode error."""
    global_variables = _load_global_config()
    config = getattr(global_variables, "ICHOR_CONFIG", None)
    machine = getattr(global_variables, "MACHINE", None)
    if not isinstance(config, dict) or not config:
        raise ClusterProfileError("~/ichor_config.yaml is missing or empty")
    if not machine:
        raise ClusterProfileError(
            "no active ICHOR machine profile resolved; set ICHOR_MACHINE=csf3 "
            "or use a hostname containing a top-level ~/ichor_config.yaml key"
        )
    if str(machine) == "_default":
        raise ClusterProfileError(
            "_default is a fallback configuration, not a live active-learning "
            "profile; set ICHOR_MACHINE to a real Slurm profile such as csf3 "
            "or csf4"
        )
    if machine not in config:
        raise ClusterProfileError(
            "active ICHOR machine profile "
            + repr(machine)
            + " is not present in ~/ichor_config.yaml"
        )
    profile = ClusterProfile(machine=str(machine), config=config)
    validate_cluster_profile(profile)
    return profile


def _exact_positive_integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ClusterProfileError(label + " must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ClusterProfileError(label + " must be >= " + str(minimum))
    return result


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ClusterProfileError(label + " must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ClusterProfileError(label + " must be finite and > 0")
    return result


def validate_cluster_profile(profile: ClusterProfile) -> None:
    """Validate the profile semantics consumed by every live renderer."""
    machine_profile = profile.config.get(profile.machine)
    if not isinstance(machine_profile, dict):
        raise ClusterProfileError("active machine profile must be a mapping")
    hpc = machine_profile.get("hpc")
    software = machine_profile.get("software")
    if not isinstance(hpc, dict) or not isinstance(software, dict):
        raise ClusterProfileError("active profile requires hpc and software mappings")
    if hpc.get("scheduler") != "slurm":
        raise ClusterProfileError("active-learning live profiles require hpc.scheduler: slurm")
    if hpc.get("jobscript_shebang") != "#!/bin/bash --login":
        raise ClusterProfileError(
            "active-learning Slurm profiles require jobscript_shebang: "
            "#!/bin/bash --login"
        )
    _exact_positive_integer(
        hpc.get("max_array_task_id"),
        "hpc.max_array_task_id",
        minimum=0,
    )
    _exact_positive_integer(
        hpc.get("max_job_log_files_per_directory"),
        "hpc.max_job_log_files_per_directory",
    )
    partitions = hpc.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        raise ClusterProfileError("hpc.partitions must be a non-empty mapping")
    for name, partition in partitions.items():
        label = "hpc.partitions." + str(name)
        if not isinstance(partition, dict):
            raise ClusterProfileError(label + " must be a mapping")
        minimum = _exact_positive_integer(partition.get("min_cpus"), label + ".min_cpus")
        maximum = _exact_positive_integer(partition.get("max_cpus"), label + ".max_cpus")
        if maximum < minimum:
            raise ClusterProfileError(label + ".max_cpus must be >= min_cpus")
        _finite_positive(partition.get("memory_per_core_gb"), label + ".memory_per_core_gb")
        _finite_positive(partition.get("max_walltime_hours"), label + ".max_walltime_hours")
        if not isinstance(partition.get("daemon_supported"), bool):
            raise ClusterProfileError(label + ".daemon_supported must be a Boolean")
    python = software.get("python")
    if not isinstance(python, dict):
        raise ClusterProfileError("software.python must be a mapping")
    raw_python = python.get("python_path")
    if not isinstance(raw_python, str) or not raw_python.strip():
        raise ClusterProfileError("software.python.python_path must be a non-empty string")
    if any(character in raw_python for character in "\r\n\x00"):
        raise ClusterProfileError("software.python.python_path contains control characters")
    expanded_python = os.path.expanduser(os.path.expandvars(raw_python.strip()))
    if not os.path.isabs(expanded_python):
        raise ClusterProfileError("software.python.python_path must resolve absolutely")
    library_path = python.get("library_path")
    if library_path is not None:
        values = [library_path] if isinstance(library_path, str) else library_path
        if not isinstance(values, (list, tuple)) or not values:
            raise ClusterProfileError(
                "software.python.library_path must be a string or non-empty list"
            )
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ClusterProfileError(
                    "software.python.library_path entries must be non-empty strings"
                )
            if any(character in value for character in "\r\n\x00"):
                raise ClusterProfileError(
                    "software.python.library_path contains control characters"
                )
    ferebus = software.get("ferebus")
    if not isinstance(ferebus, dict) or not isinstance(
        ferebus.get("pyferebus_platform"), str
    ) or not ferebus["pyferebus_platform"].strip():
        raise ClusterProfileError(
            "software.ferebus.pyferebus_platform must be explicit for live mode"
        )


def profile_value(*keys: str, default: Any = None, require_profile: bool = False) -> Any:
    """Read a value from the active machine profile.

    When ``require_profile`` is false this returns ``default`` if the active
    profile cannot be resolved. That keeps off-cluster unit tests and generic
    script rendering usable. Live preflight/submit paths call
    :func:`require_cluster_profile` directly.
    """
    try:
        global_variables = _load_global_config()
        config = getattr(global_variables, "ICHOR_CONFIG", None)
        machine = getattr(global_variables, "MACHINE", None)
        getter = getattr(global_variables, "get_param_from_config", None)
        if not isinstance(config, dict) or not machine or getter is None:
            if require_profile:
                require_cluster_profile()
            return default
        return getter(config, machine, *keys, default=default)
    except ClusterProfileError:
        if require_profile:
            raise
        return default


def expanded_profile_value(*keys: str, default: Any = None) -> Any:
    value = profile_value(*keys, default=default)
    if value is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(value)))


__all__ = [
    "ClusterProfile",
    "ClusterProfileError",
    "active_machine",
    "expanded_profile_value",
    "profile_value",
    "require_cluster_profile",
    "validate_cluster_profile",
]
