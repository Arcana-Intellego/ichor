"""Cluster-profile helpers for live active-learning backends."""
from __future__ import annotations

import os
import importlib
from dataclasses import dataclass
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
    if machine not in config:
        raise ClusterProfileError(
            "active ICHOR machine profile "
            + repr(machine)
            + " is not present in ~/ichor_config.yaml"
        )
    return ClusterProfile(machine=str(machine), config=config)


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
