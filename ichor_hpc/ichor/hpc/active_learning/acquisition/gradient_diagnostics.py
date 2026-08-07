"""Small helpers for acquisition-gradient timing diagnostics."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _posterior_diagnostics(acquisition) -> Dict[str, int]:
    posterior = getattr(acquisition, "posterior", None)
    diagnostics = getattr(posterior, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    out: Dict[str, int] = {}
    for key, value in diagnostics.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _acquisition_diagnostics(acquisition) -> Dict[str, int]:
    diagnostics = getattr(acquisition, "performance_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    out: Dict[str, int] = {}
    for key, value in diagnostics.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def static_gradient_diagnostics(acquisition, atoms=None, *, gradient_mode=None, gradient_backend=None) -> Dict[str, Any]:
    mode = "active_fd"
    raw_directions = getattr(acquisition, "mode_directions", ())
    directions = tuple(raw_directions) if raw_directions is not None else ()
    subspace_dim = len(directions)
    natoms = 0
    n_cartesian_dof = 0
    if atoms is not None:
        try:
            natoms = int(len(atoms))
        except Exception:
            natoms = 0
        n_cartesian_dof = 3 * natoms
    live_cartesian_dof = n_cartesian_dof
    estimated_calls = 2 * subspace_dim
    return {
        "schema_version": 1,
        "gradient_mode": mode or None,
        "gradient_backend": None if gradient_backend is None else str(gradient_backend),
        "natoms": int(natoms),
        "n_cartesian_dof": int(n_cartesian_dof),
        "n_live_cartesian_dof": int(live_cartesian_dof),
        "subspace_dim": int(subspace_dim),
        "n_estimated_acquisition_value_calls_per_gradient": (
            None if estimated_calls is None else int(estimated_calls)
        ),
        "posterior_diagnostics": _posterior_diagnostics(acquisition),
        "acquisition_diagnostics": _acquisition_diagnostics(acquisition),
    }


def calculator_gradient_diagnostics(calculator) -> Dict[str, Any]:
    getter = getattr(calculator, "gradient_diagnostics", None)
    if callable(getter):
        try:
            data = getter()
        except Exception:
            data = {}
    else:
        data = {}
    if not isinstance(data, Mapping):
        return {}
    out = dict(data)
    # Keep trace records compact and JSON-safe.
    for key in (
        "gradient_wall_seconds_last",
        "gradient_wall_seconds_total",
        "gradient_wall_seconds_mean",
        "gradient_wall_seconds_max",
        "last_gradient_norm",
        "last_gradient_norm_raw",
        "last_gradient_norm_post_rigid",
        "last_gradient_norm_post_cap",
        "last_gradient_clamp_scale",
        "last_force_norm_ev_per_ang",
    ):
        if key in out:
            out[key] = _finite_float(out.get(key), 0.0)
    for key in (
        "gradient_call_count",
        "natoms",
        "n_cartesian_dof",
        "n_live_cartesian_dof",
        "subspace_dim",
        "n_estimated_acquisition_value_calls_per_gradient",
        "workers_requested",
        "workers_used",
    ):
        if key in out and out[key] is not None:
            out[key] = _finite_int(out.get(key), 0)
    if "inside_gradient_worker" in out:
        out["inside_gradient_worker"] = bool(out.get("inside_gradient_worker"))
    posterior = out.get("posterior_diagnostics")
    if isinstance(posterior, Mapping):
        out["posterior_diagnostics"] = {
            str(k): _finite_int(v, 0) for k, v in posterior.items()
        }
    acquisition = out.get("acquisition_diagnostics")
    if isinstance(acquisition, Mapping):
        out["acquisition_diagnostics"] = {
            str(k): _finite_int(v, 0) for k, v in acquisition.items()
        }
    return out


def flatten_trace_gradient_diagnostics(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact selected nested diagnostics into trace-friendly scalar fields."""
    out: Dict[str, Any] = {}
    for key in (
        "gradient_mode",
        "gradient_objective",
        "gradient_backend",
        "workers_requested",
        "workers_used",
        "inside_gradient_worker",
        "parallel_fallback_reason",
        "natoms",
        "n_cartesian_dof",
        "n_live_cartesian_dof",
        "subspace_dim",
        "n_estimated_acquisition_value_calls_per_gradient",
        "gradient_call_count",
        "gradient_wall_seconds_last",
        "gradient_wall_seconds_total",
        "gradient_wall_seconds_mean",
        "gradient_wall_seconds_max",
        "last_gradient_norm",
        "last_gradient_norm_raw",
        "last_gradient_norm_post_rigid",
        "last_gradient_norm_post_cap",
        "last_gradient_clamp_scale",
        "last_force_norm_ev_per_ang",
    ):
        if key in data:
            out[key] = data[key]
    posterior = data.get("posterior_diagnostics")
    if isinstance(posterior, Mapping):
        for key in (
            "n_means_batched_calls",
            "n_covariance_matrix_batched_calls",
            "n_means_scalar_fallbacks",
            "n_covariance_matrix_scalar_fallbacks",
            "n_mean_scalar_calls",
            "n_covariance_scalar_calls",
            "n_runtime_context_builds",
            "n_factor_validations",
            "n_triangular_solves",
            "n_train_query_builds",
            "n_prepared_component_batches",
            "n_scalar_fallbacks",
            "n_alf_batch_calls",
            "n_alf_scalar_fallbacks",
        ):
            if key in posterior:
                out["posterior_" + key] = posterior[key]
    acquisition = data.get("acquisition_diagnostics")
    if isinstance(acquisition, Mapping):
        for key in (
            "n_prepared_component_batches",
            "n_prepared_component_centres",
            "n_component_cache_hits",
            "n_component_cache_misses",
            "n_atom_diagnostic_cache_hits",
            "n_atom_diagnostic_cache_misses",
            "n_scalar_component_fallbacks",
        ):
            if key in acquisition:
                out["acquisition_" + key] = acquisition[key]
    return out
