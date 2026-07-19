"""Per-seed ARIADNE driver.

Bridges the ICHOR daemon and the ARIADNE oneAPI optimiser. The daemon
submits a SLURM array job for the ARIADNE_ARRAY phase; each array task
invokes python -m ichor.hpc.active_learning.acquisition.ariadne_runner
which (after loading the campaign state) calls in here to actually drive
the adversarial descent for one seed.

Why this lives in its own file rather than alongside ariadne_runner.py:
the runner file holds the public contract -- the dataclasses and the
optimise_seed dispatcher -- and is imported by dry-run code that runs
off-cluster where ariadne is not on PYTHONPATH. Keeping the heavy
import ariadne in this side module lets the contract module stay
lightweight and importable everywhere.

The driver mirrors the pattern from ARIADNE/ariadne_starter_pack/
run_ariadne.py: a two-stage step_py loop (propose then accept/reject)
with energy + gradient supplied by an ASE-style calculator. For our
purposes the calculator is AdversarialASECalculator wrapping the
SeedLocalAdversarialAcquisition we built earlier.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .gradient_diagnostics import (
    calculator_gradient_diagnostics,
    flatten_trace_gradient_diagnostics,
)
from .ariadne_abi import DS_STATUS_LENGTH, TRQN_STATUS_LENGTH, probe_ariadne_module
from .pseudo_units import hartree_ev


# ariadne tags hessian models by integer. mirror the starter pack
# HESSIAN_MODEL_MAP so a config naming a model in caps still resolves.
_HESSIAN_MODEL_MAP = {
    "constant": 0,
    "almlof":   1,
    "lindh":    2,
    "schlegel": 3,
}

# ARIADNE enum values mirrored from the starter pack. Passing these explicitly
# avoids f90wrap optional-argument default drift in live cluster builds.
_ARIADNE_ROT_PRIMITIVE_EXPMAP3 = 1
_ARIADNE_CARTESIAN_RECOVERY_GEODESIC = 1
_ARIADNE_CARTESIAN_RECOVERY_NEWTON = 2
_ARIADNE_GEO_BT_DENSE = 1
_ARIADNE_GEO_BT_MATRIX_FREE = 2
_ARIADNE_TRQN_CONTROLLER_NONE = 0
_ARIADNE_TRQN_STRATEGY_BFGS = 0
_ARIADNE_TRQN_HANDOFF_NONE = 0
_ARIADNE_ROT_V_RESET_THRESH = float(0.9 * np.pi)
_TRQN_EXPLICIT_INIT_KEYS = frozenset({
    "trust0",
    "trust_min",
    "trust_max",
    "controller_mode",
    "q_good",
    "q_okay",
    "q_reject",
    "rho_de_eps",
    "hebden_tol_rel",
    "brent_tol_frac",
    "bt_ic_tol",
    "geo_sol_dt",
    "geo_sol_tol",
    "max_backtransform_iter",
    "max_hebdon_iter",
    "cartesian_recovery_mode",
    "geo_bt_mode",
    "rot_primitive_mode",
    "skip_bfgs_after_rot_reset",
    "freeze_dlc_basis",
    "rot_ref_reset_enabled",
    "rot_v_reset_thresh",
    "rot_gap_reset_rel",
    "rot_gap_reset_abs",
    "hessian_model",
    "usedmax",
    "epsilon_shift",
    "reset_on_bad_hessian",
    "subfrctor",
    "history_trust_scale",
    "max_hessian_updates",
    "enable_force_rebuild",
    "enable_cartesian_fallback",
    "strategy_mode",
    "gediis_history_capacity",
    "gediis_min_history",
    "gediis_rcond",
    "gediis_energy_check",
    "gediis_gradnorm_check",
    "handoff_enabled",
    "handoff_gmax_threshold",
    "handoff_accept_streak_req",
    "handoff_source_code",
    "per_block_damping_enabled",
    "flat_eig_rel_threshold",
    "flat_eig_damping_ratio",
    "s_model_enabled",
    "s_model_delta",
})

# TRQN get_status_py layout. Keep these names close to the Fortran/starter-pack
# contract so result.json diagnostics can be decoded without reading raw tuples.
_TRQN_STATUS_TRUST = 0
_TRQN_STATUS_RUN_IDX = 2
_TRQN_STATUS_PROPOSAL_PENDING = 3
_TRQN_STATUS_TRUST_BEFORE_UPDATE = 21
_TRQN_STATUS_TRUST_AFTER_UPDATE = 22
_TRQN_STATUS_INVALID_REASON = 23
_TRQN_STATUS_STEP_STATE = 35
_TRQN_STATUS_CARTNORM_LAST = 36
_TRQN_STATUS_FORCE_REBUILD = 39
_TRQN_STATUS_FORCE_REBUILD_REASON = 53
_TRQN_STATUS_FORCE_REBUILD_COUNT = 54
_TRQN_STATUS_SKIP_STEP_AFTER_REBUILD = 55
_TRQN_STATUS_LAST_REBUILD_USED_CARTESIAN = 56
_TRQN_STATUS_PROPOSAL_STAGE = 57
_TRQN_STATUS_BT_ENTRY_CODE = 58
_TRQN_STATUS_BT_ATTEMPTED = 59
_TRQN_STATUS_PROPOSAL_READY = 60
_TRQN_STATUS_REBUILD_REQUESTED_LAST = 61
_TRQN_STATUS_REBUILD_APPLIED_LAST = 62
_TRQN_STATUS_IC_SYSTEM_CHANGED_LAST = 63
_TRQN_STATUS_BT_INPUT_DLC_INF = 64
_TRQN_STATUS_BT_INPUT_CARTNORM = 65
_TRQN_STATUS_CONSECUTIVE_BT_FAIL_COUNT = 66
_TRQN_STATUS_PREVIOUS_CYCLE_WAS_REBUILD_SKIP = 71
_TRQN_STATUS_FULLSTEP_BORKED_LAST = 72
_TRQN_STATUS_FINAL_BORKED_LAST = 73
_TRQN_STATUS_FINAL_SOLUTION_KIND_LAST = 74
_DS_STATUS_PROPOSAL_PENDING = 7

_TRQN_INVALID_REASON_LABELS = {
    0: "none",
    1: "internal_solve_fail",
    2: "backtransform_fail",
    3: "unsafe_trial_geometry",
    4: "rotation_refresh_fail",
    5: "proposal_not_built",
}
_TRQN_INVALID_BACKTRANSFORM_FAIL = 2
_TRQN_INVALID_PROPOSAL_NOT_BUILT = 5

_TRQN_STEP_STATE_LABELS = {
    0: "none",
    1: "good",
    2: "okay",
    3: "poor",
    4: "reject",
}
_TRQN_STEP_STATE_REJECT = 4

# when TRQN rejects too many proposals in a row we give up and re-init
# under DS. three is the value the starter pack uses for
# recovery_accepts_to_exit which is the closest equivalent.
_REJECT_STREAK_TRIGGER = 3


@dataclass
class OptimisationResult:
    """Everything _live_optimise_seed needs to build an AriadneRunResult.

    Tracking the per-step trajectories lets the caller see how the descent
    progressed -- handy when reading per-seed result.json files later.
    """
    final_positions_angstrom: np.ndarray
    alpha_trajectory: List[float]
    grad_norm_trajectory: List[float]
    n_evaluations: int
    # return_code: 0 converged, 1 max_iter, 2 error during evaluation
    return_code: int
    wall_seconds: float
    fell_back_to_ds: bool
    rigid_force_clamps: int
    converged: bool
    candidate_positions_angstrom: List[np.ndarray]
    candidate_alphas: List[float]
    candidate_grad_norms: List[float]
    candidate_origins: List[str]
    diagnostics: Dict[str, Any]


def _json_safe_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return value if np.isfinite(value) else str(value)


def _json_safe_status(status) -> List[Any]:
    return [_json_safe_value(value) for value in tuple(status or ())]


def _new_optimiser_diagnostics(optimiser_name: str) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "optimiser_initial": str(optimiser_name),
        "optimiser_final": str(optimiser_name),
        "n_stage0_calls": 0,
        "n_stage1_calls": 0,
        "n_trial_evaluations": 0,
        "n_no_proposal_pending": 0,
        "n_no_proposal_invalid": 0,
        "n_no_proposal_backtransform_fail": 0,
        "n_no_proposal_proposal_not_built": 0,
        "n_no_proposal_reject": 0,
        "n_no_proposal_recoveries": 0,
        "n_skip_step_after_rebuild": 0,
        "n_accepted_steps": 0,
        "n_rejected_steps": 0,
        "n_fallback_to_ds": 0,
        "daemon_convergence_streak": 0,
        "daemon_convergence_last_metrics": None,
        "daemon_convergence_last_checks": None,
        "termination_source": None,
        "ds_init_profile": None,
        "last_return_code_reason": "not_finished",
        "last_no_proposal_reason": None,
        "last_no_proposal_status_summary": None,
        "status_samples_first": [],
        "status_samples_last": [],
    }


def _resolved_convergence(run_config) -> Dict[str, Any]:
    payload = getattr(run_config, "resolved_convergence", None)
    if isinstance(payload, dict):
        return payload
    convergence = getattr(run_config, "convergence", None)
    if str(getattr(convergence, "mode", "fixed")).strip().lower() != "fixed":
        raise RuntimeError(
            "scale-adaptive ARIADNE convergence requires frozen seed-scale evidence"
        )
    from .ariadne_runner import resolve_ariadne_convergence

    return resolve_ariadne_convergence(
        convergence,
        initial_acquisition_score=0.0,
        initial_trust_scale_ang=float(run_config.delta0),
    )


def _convergence_metrics(
    *,
    f_old: float,
    f_current: float,
    gradient_current,
    positions_before: np.ndarray,
    positions_current: np.ndarray,
    objective_scale: float,
) -> Dict[str, float]:
    scale = float(objective_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("objective scale is invalid during convergence evaluation")
    gradient = np.asarray(gradient_current, dtype=np.float64).reshape(-1) / scale
    step = np.asarray(positions_current, dtype=np.float64) - np.asarray(
        positions_before, dtype=np.float64
    )
    if not np.isfinite(gradient).all() or not np.isfinite(step).all():
        raise RuntimeError("non-finite state reached ARIADNE convergence evaluation")
    return {
        "objective_change": abs(float(f_current) - float(f_old)) / scale,
        "gradient_rms_per_ang": float(np.sqrt(np.mean(np.square(gradient)))),
        "gradient_max_per_ang": float(np.max(np.abs(gradient))),
        "step_rms_ang": float(np.sqrt(np.mean(np.square(step)))),
        "step_max_ang": float(np.max(np.abs(step))),
    }


def _convergence_checks(
    metrics: Dict[str, float],
    resolution: Dict[str, Any],
) -> Dict[str, bool]:
    thresholds = resolution.get("effective_thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("ARIADNE convergence thresholds are missing")
    mapping = {
        "objective_change": "objective_change_tolerance",
        "gradient_rms_per_ang": "gradient_rms_tolerance_per_ang",
        "gradient_max_per_ang": "gradient_max_tolerance_per_ang",
        "step_rms_ang": "step_rms_tolerance_ang",
        "step_max_ang": "step_max_tolerance_ang",
    }
    return {
        metric: bool(float(metrics[metric]) <= float(thresholds[threshold]))
        for metric, threshold in mapping.items()
    }


def _advance_convergence_streak(
    current: int,
    *,
    accepted: bool,
    checks: Optional[Dict[str, bool]] = None,
) -> int:
    """Advance only after an accepted step satisfying every criterion."""

    if not accepted:
        return 0
    if not checks:
        raise RuntimeError("accepted ARIADNE step is missing convergence checks")
    return int(current) + 1 if all(bool(value) for value in checks.values()) else 0


def _trqn_backtransform_mode(run_config) -> str:
    mode = str(
        getattr(run_config, "trqn_backtransform_mode", "geodesic") or "geodesic"
    ).strip().lower()
    if mode == "geodesic":
        return "geodesic"
    if mode == "newton":
        return "newton"
    raise ValueError(
        "unknown ariadne.trqn_backtransform_mode " + repr(mode)
        + "; valid: geodesic | newton"
    )


def _trqn_geodesic_bt_mode(run_config) -> str:
    mode = str(
        getattr(run_config, "trqn_geodesic_bt_mode", "dense") or "dense"
    ).strip().lower()
    if mode == "dense":
        return "dense"
    if mode == "matrix_free":
        return "matrix_free"
    raise ValueError(
        "unknown ariadne.trqn_geodesic_bt_mode " + repr(mode)
        + "; valid: dense | matrix_free"
    )


def _ariadne_cartesian_recovery_mode_id(mode: str) -> int:
    return (
        _ARIADNE_CARTESIAN_RECOVERY_GEODESIC
        if str(mode) == "geodesic"
        else _ARIADNE_CARTESIAN_RECOVERY_NEWTON
    )


def _ariadne_geo_bt_mode_id(mode: str) -> int:
    return (
        _ARIADNE_GEO_BT_MATRIX_FREE
        if str(mode) == "matrix_free"
        else _ARIADNE_GEO_BT_DENSE
    )


def _status_int(status, index: int, default: Optional[int] = None) -> Optional[int]:
    if status is None or len(status) <= int(index):
        return default
    try:
        return int(status[index])
    except (TypeError, ValueError, OverflowError):
        return default


def _status_float(status, index: int, default: Optional[float] = None) -> Optional[float]:
    if status is None or len(status) <= int(index):
        return default
    try:
        value = float(status[index])
    except (TypeError, ValueError, OverflowError):
        return default
    return value if np.isfinite(value) else default


def _status_bool(status, index: int, default: bool = False) -> bool:
    value = _status_int(status, index, None)
    if value is None:
        return default
    return bool(value)


def _strict_status_flag(status, index: int, label: str) -> bool:
    if status is None or len(status) <= int(index):
        raise RuntimeError("ARIADNE status is missing " + label)
    value = status[int(index)]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not isinstance(value, Integral):
        raise RuntimeError("ARIADNE status flag " + label + " is not integer 0/1")
    parsed = int(value)
    if parsed not in (0, 1):
        raise RuntimeError("ARIADNE status flag " + label + " is not integer 0/1")
    return bool(parsed)


def _fortran_logical(value: Any, label: str) -> bool:
    """Decode a direct f90wrap Fortran ``LOGICAL`` scalar.

    Intel Fortran conventionally exposes ``.TRUE.`` as ``-1`` through
    f90wrap, while other builds use ``1``.  This contract is deliberately
    separate from ``get_status_py()``, whose integer flags are explicitly
    normalised by ARIADNE to ``0`` or ``1``.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value) != 0
    raise RuntimeError(
        "ARIADNE Fortran LOGICAL "
        + label
        + " is not an exact boolean or integer scalar"
    )


def _validate_status_shape(status, *, optimiser: str) -> tuple:
    values = tuple(status or ())
    expected = TRQN_STATUS_LENGTH if optimiser == "trqn" else DS_STATUS_LENGTH
    if len(values) != expected:
        raise RuntimeError(
            "ARIADNE "
            + optimiser.upper()
            + " status ABI mismatch: expected "
            + str(expected)
            + " values, received "
            + str(len(values))
        )
    return values


def _trqn_status_summary(status) -> Dict[str, Any]:
    if not status:
        return {}
    invalid_reason = _status_int(status, _TRQN_STATUS_INVALID_REASON, 0)
    step_state = _status_int(status, _TRQN_STATUS_STEP_STATE, 0)
    summary = {
        "trust": _status_float(status, _TRQN_STATUS_TRUST, None),
        "run_idx": _status_int(status, _TRQN_STATUS_RUN_IDX, None),
        "proposal_pending": _status_bool(
            status, _TRQN_STATUS_PROPOSAL_PENDING, False,
        ),
        "invalid_reason": invalid_reason,
        "invalid_reason_label": _TRQN_INVALID_REASON_LABELS.get(
            invalid_reason, "unknown_" + str(invalid_reason),
        ),
        "step_state": step_state,
        "step_state_label": _TRQN_STEP_STATE_LABELS.get(
            step_state, "unknown_" + str(step_state),
        ),
        "trust_before_update": _status_float(
            status, _TRQN_STATUS_TRUST_BEFORE_UPDATE, None,
        ),
        "trust_after_update": _status_float(
            status, _TRQN_STATUS_TRUST_AFTER_UPDATE, None,
        ),
        "cartnorm_last": _status_float(status, _TRQN_STATUS_CARTNORM_LAST, None),
        "force_rebuild": _status_bool(status, _TRQN_STATUS_FORCE_REBUILD, False),
        "force_rebuild_reason": _status_int(
            status, _TRQN_STATUS_FORCE_REBUILD_REASON, None,
        ),
        "force_rebuild_count": _status_int(
            status, _TRQN_STATUS_FORCE_REBUILD_COUNT, None,
        ),
        "skip_step_after_rebuild": _status_bool(
            status, _TRQN_STATUS_SKIP_STEP_AFTER_REBUILD, False,
        ),
        "last_rebuild_used_cartesian": _status_bool(
            status, _TRQN_STATUS_LAST_REBUILD_USED_CARTESIAN, False,
        ),
        "proposal_stage": _status_int(status, _TRQN_STATUS_PROPOSAL_STAGE, None),
        "bt_entry_code": _status_int(status, _TRQN_STATUS_BT_ENTRY_CODE, None),
        "bt_attempted": _status_bool(status, _TRQN_STATUS_BT_ATTEMPTED, False),
        "proposal_ready": _status_bool(status, _TRQN_STATUS_PROPOSAL_READY, False),
        "rebuild_requested_last": _status_bool(
            status, _TRQN_STATUS_REBUILD_REQUESTED_LAST, False,
        ),
        "rebuild_applied_last": _status_bool(
            status, _TRQN_STATUS_REBUILD_APPLIED_LAST, False,
        ),
        "ic_system_changed_last": _status_bool(
            status, _TRQN_STATUS_IC_SYSTEM_CHANGED_LAST, False,
        ),
        "bt_input_dlc_inf": _status_float(
            status, _TRQN_STATUS_BT_INPUT_DLC_INF, None,
        ),
        "bt_input_cartnorm": _status_float(
            status, _TRQN_STATUS_BT_INPUT_CARTNORM, None,
        ),
        "consecutive_bt_fail_count": _status_int(
            status, _TRQN_STATUS_CONSECUTIVE_BT_FAIL_COUNT, None,
        ),
        "previous_cycle_was_rebuild_skip": _status_bool(
            status, _TRQN_STATUS_PREVIOUS_CYCLE_WAS_REBUILD_SKIP, False,
        ),
        "fullstep_borked_last": _status_bool(
            status, _TRQN_STATUS_FULLSTEP_BORKED_LAST, False,
        ),
        "final_borked_last": _status_bool(
            status, _TRQN_STATUS_FINAL_BORKED_LAST, False,
        ),
        "final_solution_kind_last": _status_int(
            status, _TRQN_STATUS_FINAL_SOLUTION_KIND_LAST, None,
        ),
    }
    return {k: _json_safe_value(v) for k, v in summary.items()}


def _append_status_sample(
    diagnostics: Dict[str, Any],
    *,
    step_index: int,
    label: str,
    status,
) -> None:
    sample = {
        "step_index": int(step_index),
        "label": str(label),
        "status": _json_safe_status(status),
        "status_summary": _trqn_status_summary(status),
    }
    first = diagnostics.setdefault("status_samples_first", [])
    if len(first) < 5:
        first.append(dict(sample))
    last = diagnostics.setdefault("status_samples_last", [])
    last.append(dict(sample))
    del last[:-5]


def _append_trace_event(trace_path, **payload: Any) -> None:
    if trace_path is None:
        return
    try:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": 1}
        record.update({str(k): _json_safe_value(v) for k, v in payload.items()})
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    except Exception:
        return


def _append_trace_event_with_gradient(trace_path, calculator, **payload: Any) -> None:
    try:
        gradient_payload = flatten_trace_gradient_diagnostics(
            calculator_gradient_diagnostics(calculator)
        )
    except Exception:
        gradient_payload = {}
    gradient_payload.update(payload)
    _append_trace_event(trace_path, **gradient_payload)


def _trace_status(status) -> Dict[str, Any]:
    summary = _trqn_status_summary(status)
    return {
        "trust": summary.get("trust"),
        "reason": summary.get("invalid_reason_label"),
        "proposal_pending": summary.get("proposal_pending"),
    }


def _finalise_optimiser_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    return_code: int,
    converged: bool,
) -> None:
    diagnostics["return_code"] = int(return_code)
    diagnostics["converged"] = bool(converged)
    if diagnostics.get("termination_source") is None:
        diagnostics["termination_source"] = (
            "max_iterations" if int(return_code) == 1 else "optimiser_error"
        )
    if str(diagnostics.get("last_return_code_reason")) != "not_finished":
        return
    if int(return_code) == 0:
        diagnostics["last_return_code_reason"] = "converged"
    elif int(return_code) == 1:
        if int(diagnostics.get("n_trial_evaluations", 0)) == 0:
            diagnostics["last_return_code_reason"] = "max_iterations_no_trial_evaluations"
        elif int(diagnostics.get("n_accepted_steps", 0)) == 0:
            diagnostics["last_return_code_reason"] = "max_iterations_no_accepted_steps"
        else:
            diagnostics["last_return_code_reason"] = "max_iterations"
    else:
        diagnostics["last_return_code_reason"] = "optimiser_error"


def _import_ariadne():
    """Lazy import. Raises a friendly RuntimeError if the oneAPI .so is
    not on PYTHONPATH yet, which is the expected case off-cluster.
    """
    try:
        import ariadne  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "ARIADNE Python module not importable. Build the oneAPI .so "
            "and prepend build-oneapi/python to PYTHONPATH."
        ) from exc
    try:
        probe_ariadne_module(ariadne)
    except Exception as exc:
        raise RuntimeError("ARIADNE Python wrapper ABI is incompatible: " + str(exc)) from exc
    return ariadne


def _symbols_to_atom_list(symbols):
    """Build the S2 byte array ARIADNE expects for atom labels.

    2-char fixed-width, left-aligned, truncated to 2. matches the
    starter pack symbols_to_f90_labels.
    """
    cleaned = []
    for s in symbols:
        s = str(s).strip()
        if len(s) < 1 or len(s) > 2:
            raise ValueError(
                "atom symbols must be 1 or 2 characters; got " + repr(s)
            )
        cleaned.append(s)
    return np.asarray(["{:<2}".format(s)[:2] for s in cleaned], dtype="S2")


def _hessian_model_id(name) -> int:
    """Resolve a hessian-model name to the integer ARIADNE wants."""
    key = (str(name) if name is not None else "schlegel").strip().lower()
    if key not in _HESSIAN_MODEL_MAP:
        raise ValueError(
            "unknown ariadne.hessian_model " + repr(name)
            + "; valid: " + repr(sorted(_HESSIAN_MODEL_MAP))
        )
    return _HESSIAN_MODEL_MAP[key]


def _flatten_xyz(values) -> np.ndarray:
    return np.asfortranarray(np.asarray(values, dtype=np.float64).reshape(-1), dtype=np.float64)


def _eval_energy_gradient(atoms, *, objective_scale: float = 1.0):
    """One calculator evaluation. Returns (f_pseudo, g_flat_pseudo).

    Mirrors evaluate_energy_and_gradient from the ARIADNE starter pack.
    The adversarial calculator returns pseudo-energy + pseudo-forces in ASE
    units (eV and eV/Angstrom). Divide by the same pseudo Hartree scale so
    ARIADNE optimises alpha in acquisition units, then optionally apply a
    positive optimiser-only scale. The scale preserves the acquisition argmax
    but can keep TRQN's internal-coordinate backtransform in a sane numerical
    range.
    """
    scale = float(objective_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("objective_scale must be finite and positive")
    e_ev = float(atoms.get_potential_energy())
    forces_ev = np.asarray(atoms.get_forces(), dtype=np.float64)
    grad_ev = -forces_ev
    pseudo_hartree_ev = hartree_ev()
    e_hartree = (e_ev / pseudo_hartree_ev) * scale
    g_xyz_hartree = np.asfortranarray(
        (grad_ev / pseudo_hartree_ev) * scale,
        dtype=np.float64,
    )
    g_flat_hartree = _flatten_xyz(g_xyz_hartree)
    if not np.isfinite(e_hartree):
        raise NonFiniteCalculatorOutput(
            "calculator returned non-finite pseudo-energy"
        )
    if not np.isfinite(g_flat_hartree).all():
        raise NonFiniteCalculatorOutput(
            "calculator returned non-finite entries in the gradient"
        )
    return e_hartree, g_flat_hartree


def _raw_alpha_from_scaled(f_scaled: float, objective_scale: float) -> float:
    scale = float(objective_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return -float(f_scaled) / scale


def _raw_grad_norm_from_scaled(g_scaled, objective_scale: float) -> float:
    scale = float(objective_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.linalg.norm(g_scaled)) / scale


def _trqn_scale_mode(run_config) -> str:
    mode = str(getattr(run_config, "trqn_scale_mode", "off") or "off").strip().lower()
    if mode not in {"off", "fixed", "adaptive_initial_gradient", "adaptive_initial_gradient_rms"}:
        return "off"
    return mode


def _bounded_objective_scale(run_config, value: float) -> float:
    try:
        lower = float(getattr(run_config, "trqn_min_objective_scale", 1.0e-6))
        upper = float(getattr(run_config, "trqn_max_objective_scale", 1.0))
    except (TypeError, ValueError):
        lower, upper = 1.0e-6, 1.0
    if not np.isfinite(lower) or lower <= 0.0:
        lower = 1.0e-6
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0
    if upper < lower:
        upper = lower
    upper = min(upper, 1.0)
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = upper
    return float(min(max(scale, lower), upper))


def _compute_trqn_objective_scale(
    run_config,
    raw_gradient,
    *,
    retry: bool = False,
    atoms=None,
    under_move_retry: bool = False,
) -> Dict[str, Any]:
    mode = _trqn_scale_mode(run_config)
    raw_array = np.asarray(raw_gradient, dtype=float)
    if raw_array.ndim == 0:
        raw_norm = float(raw_array)
        raw_flat = None
    else:
        raw_flat = raw_array.reshape(-1)
        raw_norm = float(np.linalg.norm(raw_flat))
    raw_rms = None
    n_live = None
    n_rigid = None
    n_effective = None
    gradient_projected_rigid = False
    target_kind = "total_norm"
    if mode == "off":
        target = None
        scale = 1.0
        reason = "disabled"
    elif mode == "fixed":
        target = None
        scale = _bounded_objective_scale(
            run_config,
            float(getattr(run_config, "trqn_fixed_objective_scale", 1.0)),
        )
        reason = "fixed"
    elif mode == "adaptive_initial_gradient":
        attr = (
            "trqn_retry_target_initial_grad_norm"
            if retry else "trqn_target_initial_grad_norm"
        )
        try:
            target = float(getattr(run_config, attr))
        except (TypeError, ValueError):
            target = 0.003 if retry else 0.01
        if not np.isfinite(target) or target <= 0.0:
            target = 0.003 if retry else 0.01
        if not np.isfinite(raw_norm):
            raise RuntimeError("TRQN initial raw gradient norm is non-finite")
        if raw_norm <= 0.0:
            scale = _bounded_objective_scale(
                run_config,
                float(getattr(run_config, "trqn_max_objective_scale", 1.0)),
            )
            reason = "zero_initial_gradient"
        else:
            scale = _bounded_objective_scale(run_config, target / raw_norm)
            reason = "adaptive_initial_gradient"
    else:
        target_kind = "rms"
        if under_move_retry:
            target = 6.0e-4
        else:
            attr = (
                "trqn_retry_target_initial_grad_rms"
                if retry else "trqn_target_initial_grad_rms"
            )
            try:
                target = float(getattr(run_config, attr))
            except (TypeError, ValueError):
                target = 4.0e-4 if retry else 2.0e-4
        if not np.isfinite(target) or target <= 0.0:
            target = 6.0e-4 if under_move_retry else (4.0e-4 if retry else 2.0e-4)
        if raw_flat is not None and atoms is not None:
            try:
                from .rigid_projection import rigid_basis, project_out_rigid

                projected = np.asarray(project_out_rigid(raw_flat.reshape(-1, 3), atoms), dtype=float).reshape(-1)
                n_live = int(projected.size)
                n_rigid = int(rigid_basis(atoms).shape[1])
                n_effective = max(1, int(n_live) - int(n_rigid))
                raw_rms = float(np.linalg.norm(projected) / np.sqrt(float(n_effective)))
                gradient_projected_rigid = True
            except Exception:
                n_live = int(raw_flat.size)
                n_effective = max(1, n_live)
                raw_rms = float(raw_norm / np.sqrt(float(n_effective)))
        else:
            n_live = None if raw_flat is None else int(raw_flat.size)
            n_effective = 1 if raw_flat is None else max(1, int(raw_flat.size))
            raw_rms = float(raw_norm / np.sqrt(float(n_effective)))
        if not np.isfinite(raw_rms):
            raise RuntimeError("TRQN initial raw gradient RMS is non-finite")
        if raw_rms <= 0.0:
            scale = _bounded_objective_scale(
                run_config,
                float(getattr(run_config, "trqn_max_objective_scale", 1.0)),
            )
            reason = "zero_initial_gradient_rms"
        else:
            scale = _bounded_objective_scale(run_config, target / raw_rms)
            reason = "adaptive_initial_gradient_rms"
    try:
        lower_bound = float(getattr(run_config, "trqn_min_objective_scale", 1.0e-8))
        upper_bound = float(getattr(run_config, "trqn_max_objective_scale", 1.0))
    except (TypeError, ValueError):
        lower_bound, upper_bound = 1.0e-8, 1.0
    clamped = (
        np.isfinite(lower_bound)
        and np.isfinite(upper_bound)
        and (
            abs(float(scale) - float(lower_bound)) <= 1.0e-15
            or abs(float(scale) - float(upper_bound)) <= 1.0e-15
        )
    )
    return {
        "mode": mode,
        "scale": float(scale),
        "target": target,
        "target_kind": target_kind,
        "raw_grad_norm": raw_norm,
        "raw_grad_rms": raw_rms,
        "scaled_grad_norm": (
            None if not np.isfinite(raw_norm) else float(raw_norm) * float(scale)
        ),
        "scaled_grad_rms": (
            None if raw_rms is None or not np.isfinite(raw_rms)
            else float(raw_rms) * float(scale)
        ),
        "n_live_cartesian_dof": n_live,
        "n_projected_rigid_dof": n_rigid,
        "n_effective_internal_dof": n_effective,
        "gradient_projected_rigid": bool(gradient_projected_rigid),
        "objective_scale_clamped": bool(clamped),
        "reason": reason,
        "retry": bool(retry),
        "under_move_retry": bool(under_move_retry),
    }


def _sync_state(opt, n_atoms):
    """Pull (q_xyz, g_xyz) out of the ariadne optimiser.

    Fortran side keeps positions and gradient in flat (3N,) arrays.
    We pre-allocate them and let get_state_flat_py fill them.
    """
    n3 = 3 * int(n_atoms)
    q_flat = np.empty(n3, dtype=np.float64, order="F")
    g_flat = np.empty(n3, dtype=np.float64, order="F")
    opt.get_state_flat_py(q_flat, g_flat)
    return q_flat.reshape(n_atoms, 3), g_flat.reshape(n_atoms, 3)


def _status_trqn(opt):
    """Pull (converged, last_step_accepted, f_current) off a TRQN opt.

    The TRQN get_status_py tuple from the starter pack has f_current at
    index 1, converged at index 4 and last_step_accepted at index 5.
    """
    status = _validate_status_shape(opt.get_status_py(), optimiser="trqn")
    converged = _strict_status_flag(status, 4, "TRQN converged")
    accepted = _strict_status_flag(status, 5, "TRQN last_step_accepted")
    f_current = float(status[1])
    if not np.isfinite(f_current):
        raise RuntimeError("ARIADNE TRQN status returned non-finite f_current")
    return converged, accepted, f_current


def _status_ds(opt):
    """Pull (converged, last_step_accepted, f_current) off a DS opt.

    DS get_status_py layout has f_current at index 2, converged at
    index 8 and last_step_accepted at index 9.
    """
    status = _validate_status_shape(opt.get_status_py(), optimiser="ds")
    converged = _strict_status_flag(status, 8, "DS converged")
    accepted = _strict_status_flag(status, 9, "DS last_step_accepted")
    f_current = float(status[2])
    if not np.isfinite(f_current):
        raise RuntimeError("ARIADNE DS status returned non-finite f_current")
    return converged, accepted, f_current


def _status_tuple(opt):
    return tuple(opt.get_status_py())


_OPTIMIZER_ATTRIBUTE_MISSING = object()


def _optimizer_flag(opt, name: str):
    try:
        value = getattr(opt, name)
    except AttributeError as exc:
        try:
            inspect.getattr_static(opt, name)
        except AttributeError:
            return _OPTIMIZER_ATTRIBUTE_MISSING
        raise RuntimeError(
            "ARIADNE optimiser property " + name + " could not be read"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "ARIADNE optimiser property " + name + " could not be read"
        ) from exc
    try:
        value = value() if callable(value) else value
    except Exception as exc:
        raise RuntimeError(
            "ARIADNE optimiser property " + name + " could not be evaluated"
        ) from exc
    return value


def _raise_if_init_failed(opt, label: str) -> None:
    init_ok = _optimizer_flag(opt, "init_ok")
    if init_ok is _OPTIMIZER_ATTRIBUTE_MISSING:
        return
    if not _fortran_logical(init_ok, label + " init_ok"):
        reason = None
        for reason_name in (
            "last_config_error_msg",
            "init_reason",
            "init_message",
        ):
            candidate = _optimizer_flag(opt, reason_name)
            if candidate is not _OPTIMIZER_ATTRIBUTE_MISSING and candidate:
                reason = candidate
                break
        raise RuntimeError(
            "ARIADNE "
            + label
            + " initialisation failed"
            + (": " + str(reason) if reason else "")
        )


def _proposal_pending(opt, status, *, is_trqn: bool) -> bool:
    value = _optimizer_flag(opt, "proposal_pending")
    if value is not _OPTIMIZER_ATTRIBUTE_MISSING:
        return _fortran_logical(value, "proposal_pending")
    index = (
        _TRQN_STATUS_PROPOSAL_PENDING
        if is_trqn
        else _DS_STATUS_PROPOSAL_PENDING
    )
    return _strict_status_flag(status, index, "proposal_pending")


def _skip_step_after_rebuild(opt, status, *, is_trqn: bool) -> bool:
    if not is_trqn:
        return False
    value = _optimizer_flag(opt, "skip_step_after_rebuild")
    if value is not _OPTIMIZER_ATTRIBUTE_MISSING:
        return _fortran_logical(value, "skip_step_after_rebuild")
    return _strict_status_flag(
        status,
        _TRQN_STATUS_SKIP_STEP_AFTER_REBUILD,
        "skip_step_after_rebuild",
    )


def _trqn_no_proposal_failure_reason(status) -> Optional[str]:
    invalid_reason = _status_int(status, _TRQN_STATUS_INVALID_REASON, 0)
    if invalid_reason == _TRQN_INVALID_BACKTRANSFORM_FAIL:
        return "trqn_no_proposal_backtransform_fail"
    if invalid_reason == _TRQN_INVALID_PROPOSAL_NOT_BUILT:
        return "trqn_no_proposal_proposal_not_built"
    step_state = _status_int(status, _TRQN_STATUS_STEP_STATE, 0)
    if step_state == _TRQN_STEP_STATE_REJECT:
        return "trqn_no_proposal_reject"
    return None


def _record_no_proposal_failure(
    diagnostics: Dict[str, Any],
    *,
    reason: str,
    status,
) -> None:
    diagnostics["n_no_proposal_invalid"] = (
        int(diagnostics.get("n_no_proposal_invalid", 0)) + 1
    )
    if reason == "trqn_no_proposal_backtransform_fail":
        key = "n_no_proposal_backtransform_fail"
    elif reason == "trqn_no_proposal_proposal_not_built":
        key = "n_no_proposal_proposal_not_built"
    else:
        key = "n_no_proposal_reject"
    diagnostics[key] = int(diagnostics.get(key, 0)) + 1
    diagnostics["last_no_proposal_reason"] = str(reason)
    diagnostics["last_no_proposal_status_summary"] = _trqn_status_summary(status)


_TRIAL_REASON_CODES = {
    "calc_exception": 7,
    "calc_nonconverged": 8,
    "py_geometry_guard": 9,
    "calc_nonfinite_output": 10,
}


class InvalidTrialReason(Enum):
    CALCULATOR_EXCEPTION = "calc_exception"
    CALCULATOR_NONCONVERGED = "calc_nonconverged"
    PYTHON_GEOMETRY_GUARD = "py_geometry_guard"
    CALCULATOR_NONFINITE_OUTPUT = "calc_nonfinite_output"


class NonFiniteCalculatorOutput(RuntimeError):
    """Calculator completed but returned a non-finite objective or gradient."""


def _classify_trial_exception(exc: BaseException) -> InvalidTrialReason:
    if isinstance(exc, NonFiniteCalculatorOutput):
        return InvalidTrialReason.CALCULATOR_NONFINITE_OUTPUT
    label = (type(exc).__name__ + " " + str(exc)).lower()
    if "nonconverg" in label:
        return InvalidTrialReason.CALCULATOR_NONCONVERGED
    if "geometry" in label and ("guard" in label or "unsafe" in label):
        return InvalidTrialReason.PYTHON_GEOMETRY_GUARD
    return InvalidTrialReason.CALCULATOR_EXCEPTION

_DS_INIT_PROFILE = "daemon_safe_v1"


def _ds_hessian_model_name(value: Any) -> str:
    name = str(value or "schlegel").strip().lower()
    if name not in _HESSIAN_MODEL_MAP:
        raise ValueError(f"unsupported DS hessian_model: {value!r}")
    return name


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer > 0")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return result


def _trqn_control_init_kwargs(run_config) -> Dict[str, Any]:
    """Return ICHOR's explicit TRQN initialisation profile.

    The key list mirrors ARIADNE's starter-pack direct-TRQN opt.init()
    contract. Keeping this profile explicit avoids f90wrap optional-default
    drift on CSF3/CSF4, while campaign.yaml still exposes only the few knobs
    operators realistically tune.
    """
    backtransform_mode = _trqn_backtransform_mode(run_config)
    geodesic_bt_mode = _trqn_geodesic_bt_mode(run_config)
    delta0 = _positive_float(run_config.delta0, name="delta0")
    delta_max = _positive_float(run_config.delta_max, name="delta_max")
    trust_min = _positive_float(
        getattr(run_config, "trqn_trust_min", 1.0e-4),
        name="trqn_trust_min",
    )
    max_backtransform_iter = _positive_int(
        getattr(run_config, "trqn_max_backtransform_iter", 50),
        name="trqn_max_backtransform_iter",
    )
    kwargs: Dict[str, Any] = {
        "trust0": delta0,
        "trust_min": trust_min,
        "trust_max": max(delta_max, delta0, trust_min),
        "controller_mode": _ARIADNE_TRQN_CONTROLLER_NONE,
        "q_good": 0.75,
        "q_okay": 0.25,
        "q_reject": -1.0,
        "rho_de_eps": 1.0e-12,
        "hebden_tol_rel": 1.0e-3,
        "brent_tol_frac": 1.0e-1,
        "bt_ic_tol": _positive_float(
            getattr(run_config, "trqn_bt_ic_tol", 1.0e-6),
            name="trqn_bt_ic_tol",
        ),
        "geo_sol_dt": _positive_float(
            getattr(run_config, "trqn_geodesic_dt", 1.0e-2),
            name="trqn_geodesic_dt",
        ),
        "geo_sol_tol": _positive_float(
            getattr(run_config, "trqn_geodesic_tol", 1.0e-8),
            name="trqn_geodesic_tol",
        ),
        "max_backtransform_iter": max_backtransform_iter,
        "max_hebdon_iter": 50,
        "cartesian_recovery_mode": _ariadne_cartesian_recovery_mode_id(
            backtransform_mode
        ),
        "geo_bt_mode": _ariadne_geo_bt_mode_id(geodesic_bt_mode),
        "rot_primitive_mode": _ARIADNE_ROT_PRIMITIVE_EXPMAP3,
        "skip_bfgs_after_rot_reset": True,
        "freeze_dlc_basis": True,
        "rot_ref_reset_enabled": True,
        "rot_v_reset_thresh": _ARIADNE_ROT_V_RESET_THRESH,
        "rot_gap_reset_rel": 1.0e-6,
        "rot_gap_reset_abs": 1.0e-10,
        "hessian_model": _hessian_model_id(run_config.hessian_model),
        "usedmax": False,
        "epsilon_shift": 1.0e-5,
        "reset_on_bad_hessian": True,
        "subfrctor": 1,
        "history_trust_scale": 1.1,
        "max_hessian_updates": 100,
        "enable_force_rebuild": True,
        "enable_cartesian_fallback": True,
        "strategy_mode": _ARIADNE_TRQN_STRATEGY_BFGS,
        "gediis_history_capacity": 10,
        "gediis_min_history": 2,
        "gediis_rcond": 1.0e-10,
        "gediis_energy_check": True,
        "gediis_gradnorm_check": True,
        "handoff_enabled": False,
        "handoff_gmax_threshold": 1.0e-3,
        "handoff_accept_streak_req": 2,
        "handoff_source_code": _ARIADNE_TRQN_HANDOFF_NONE,
        "per_block_damping_enabled": False,
        "flat_eig_rel_threshold": 1.0e-2,
        "flat_eig_damping_ratio": 0.1,
        "s_model_enabled": False,
        "s_model_delta": 1.0e-3,
    }
    _validate_trqn_init_kwargs(kwargs)
    return kwargs


def _validate_trqn_init_kwargs(kwargs: Dict[str, Any]) -> None:
    missing = sorted(_TRQN_EXPLICIT_INIT_KEYS - set(kwargs))
    extra = sorted(set(kwargs) - _TRQN_EXPLICIT_INIT_KEYS)
    if missing or extra:
        raise ValueError(
            "TRQN init profile key mismatch; missing="
            + repr(missing) + " extra=" + repr(extra)
        )

    for key in (
        "trust0",
        "trust_min",
        "trust_max",
        "rho_de_eps",
        "hebden_tol_rel",
        "brent_tol_frac",
        "bt_ic_tol",
        "geo_sol_dt",
        "geo_sol_tol",
        "epsilon_shift",
        "history_trust_scale",
        "gediis_rcond",
        "handoff_gmax_threshold",
        "flat_eig_rel_threshold",
        "flat_eig_damping_ratio",
        "s_model_delta",
    ):
        _positive_float(kwargs[key], name=key)
    for key in (
        "max_backtransform_iter",
        "max_hebdon_iter",
        "subfrctor",
        "max_hessian_updates",
        "gediis_history_capacity",
        "gediis_min_history",
        "handoff_accept_streak_req",
    ):
        _positive_int(kwargs[key], name=key)
    if not float(kwargs["trust_max"]) >= float(kwargs["trust_min"]):
        raise ValueError("trust_max must be >= trust_min")
    if not float(kwargs["trust_max"]) >= float(kwargs["trust0"]):
        raise ValueError("trust_max must be >= trust0")


def _trqn_init_kwargs(q0_xyz, g0_xyz, atom_list, run_config) -> Dict[str, Any]:
    kwargs = _trqn_control_init_kwargs(run_config)
    kwargs.update(q0_xyz=q0_xyz, g0_xyz=g0_xyz, atom_list=atom_list)
    return kwargs


def _ds_safe_init_kwargs(run_config) -> Dict[str, Any]:
    """Return the daemon-owned DS initialisation profile.

    Several f90wrap builds treat omitted optional scalar arguments as present
    zero values. Passing this profile explicitly keeps DS startup independent of
    wrapper-default behaviour on CSF3/CSF4.
    """
    delta0 = _positive_float(run_config.delta0, name="delta0")
    requested_delta_max = _positive_float(run_config.delta_max, name="delta_max")
    delta_max = max(requested_delta_max, delta0)
    delta_min = min(1.0e-4, delta0)
    gamma = _positive_float(run_config.gamma, name="gamma")
    h = delta0
    h_min = min(1.0e-3, h)
    h_max = max(1.0e-2, h)
    gamma_min = min(1.0e-4, gamma)
    gamma_max = max(5.0e1, gamma)
    finish_gmax_trigger = 1.0e-3
    finish_small_gmax_cap = 5.0e-3
    convergence = _resolved_convergence(run_config)
    thresholds = convergence["effective_thresholds"]

    kwargs: Dict[str, Any] = {
        "gamma": gamma,
        "h": h,
        "f_tol": float(thresholds["objective_change_tolerance"]),
        "gradf_tol": float(thresholds["gradient_max_tolerance_per_ang"]),
        "hessian_model": _ds_hessian_model_name(run_config.hessian_model),
        "auto_params": False,
        "use_nonzero_p0": False,
        "p0_strategy": "zero",
        "ds_controller": "safe",
        "hpos_estimator": "syev",
        "lanczos_k": 4,
        "cartesian_recovery_mode": _ARIADNE_CARTESIAN_RECOVERY_NEWTON,
        "geo_bt_mode": _ARIADNE_GEO_BT_DENSE,
        "rot_primitive_mode": _ARIADNE_ROT_PRIMITIVE_EXPMAP3,
        "geo_sol_dt": 1.0e-2,
        "geo_sol_tol": 1.0e-8,
        "bt_ic_tol": 1.0e-6,
        "max_backtransform_iter": 50,
        "delta0": delta0,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "delta_grow": 1.5,
        "delta_shrink": 0.5,
        "delta_finish": delta_min,
        "h_finish": h_min,
        "gamma_finish": gamma,
        "finish_after_uphill": 3,
        "delta_regrow_step_frac": 0.10,
        "controller_de_tol": 1.0e-6,
        "finish_stall_window": 3,
        "finish_stall_min_iter": 4,
        "finish_stall_de_tol": 2.5e-7,
        "finish_stall_step_tol": 2.5e-3,
        "finish_stall_gmax_cap": 3.5e-4,
        "finish_delta_cap": 8.0e-3,
        "finish_delta_grow": 1.10,
        "finish_delta_shrink": 0.70,
        "finish_no_grow_gmax": 3.5e-4,
        "finish_disp_target": 3.0e-3,
        "finish_disp_hard": 5.0e-3,
        "finish_growth_cooldown_steps": 2,
        "finish_stall_delta_cap": 3.5e-3,
        "finish_entry_accept_streak_req": 3,
        "finish_entry_gmax_trigger": 4.5e-4,
        "finish_entry_dmax_cap": 3.0e-3,
        "finish_stage2_gmax": 3.0e-4,
        "finish_delta_cap_stage1": 8.0e-3,
        "finish_delta_cap_stage2": 4.5e-3,
        "finish_delta_grow_stage1": 1.15,
        "finish_delta_shrink_stage1": 0.70,
        "finish_delta_shrink_stage2": 0.80,
        "h_min": h_min,
        "h_max": h_max,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "eig_abs_floor": 1.0e-6,
        "mu_floor_rel": 1.0e-3,
        "ch": 0.70,
        "cart_step_inf_max": 3.0e-1,
        "cart_step_rms_max": 1.5e-1,
        "cart_atom_step_max": 4.5e-1,
        "cart_min_pair_dist": 5.5e-1,
        "cart_abs_coord_max": 1.0e3,
        "cart_pair_dist_max": 1.0e3,
        "grad_inf_max": 1.0e6,
        "grad_growth_max": 1.0e6,
        "finish_gmax_trigger": finish_gmax_trigger,
        "finish_small_step_streak": 3,
        "finish_step_ratio_trigger": 0.25,
        "finish_grad_ratio_trigger": 1.0e-2,
        "finish_small_gmax_cap": finish_small_gmax_cap,
        "finish_small_step_min_iter": 0,
        "delta_grow_dmax_floor": 0.0,
        "mass_mode": "identity",
        "kinetic_model": "quadratic",
        "delta_rel": 0.0,
        "recovery_accepts_to_exit": 2,
        "recovery_h_scale": 0.5,
        "recovery_gamma_grow": 1.2,
        "recovery_momentum_scale": 0.0,
        "reject_momentum_scale": 0.0,
        "reject_delta_ref_frac": 1.0,
        "reject_h_scale": 0.5,
        "reject_gamma_scale": 1.2,
        "max_reject_streak": 3,
        "mass_diag_abs_floor": 1.0e-12,
        "mass_diag_floor_rel": 1.0e-8,
        "block_metric_eps": 1.0e-12,
        "block_shape_eps": 1.0e-12,
        "block_scale_min": 1.0e-3,
        "block_scale_max": 1.0e3,
        "block_scale_relax": 0.5,
        "block_secant_disp_tol": 0.0,
        "block_secant_curv_floor": 1.0e-8,
        "block_scale_history_len": 4,
        "block_scale_seed_mode": "uniform",
        "block_coupling_mode": "row_gram",
        "block_merge_hysteresis_gap": 0.04,
        "block_transfer_min_row_overlap": 0.50,
        "block_transfer_min_subspace_overlap": 0.60,
        "block_transfer_mode": "weighted",
        "block_transfer_total_weight_floor": 0.0,
        "bond_switching": False,
        "bond_switching_kappa": 8.0,
        "bond_switching_xi0": 1.0,
        "block_scale_metric_aware": True,
        "block_scale_log_domain": True,
        "external_mode_dominance": 0.75,
        "rel_cap_mode": "global",
        "delta_rel_trans_factor": 1.0,
        "delta_rel_rot_factor": 1.0,
        "delta_rel_mixed_factor": 1.0,
        "poincare_monitor": "balanced",
        "poincare_alpha": 1.0,
        "poincare_beta": 1.0,
        "poincare_zeta": 0.5,
        "poincare_tau": 1.0e-3,
        "poincare_lambda": 1.0e-3,
        "poincare_m_min": 1.0e-6,
        "poincare_m_max": 1.0,
        "htvi_p_bregman": 2.0,
        "htvi_gamma_0": 1.0,
        "htvi_max_inner": 8,
        "htvi_tol_abs": 1.0e-10,
        "htvi_tol_rel": 1.0e-8,
        "htvi_tol_step": 1.0e-10,
        "htvi_monitor_eval": "start",
        "htvi_picard_max": 8,
        "htvi_picard_tol_m": 1.0e-10,
        "saddle_active_escape": False,
        "saddle_probe_on_final": True,
        "saddle_probe_on_streak": False,
        "saddle_entry_gmax_trigger": 4.5e-4,
        "saddle_entry_accept_streak_req": 3,
        "saddle_cooldown_steps": 8,
        "saddle_probe_dx_target": 1.0e-4,
        "saddle_lanczos_k_min": 4,
        "saddle_lanczos_k_max": 12,
        "saddle_lanczos_stab_tol": 1.0e-2,
        "saddle_lambda_rel_threshold": 1.0e-3,
        "saddle_lambda_abs_threshold": 1.0e-5,
        "saddle_escape_df_target": 1.0e-3,
        "saddle_probe_seed": 0,
        "inline_gediis_enabled": False,
        "inline_gediis_capacity": 4,
        "inline_gediis_escalate_streak": 3,
        "inline_gediis_e_scale": 1.0e-3,
        "reject_response_aware_enabled": False,
        "reject_h_scale_aggressive_mult": 1.0,
        "reject_gamma_boost_mult": 1.0,
        "reject_gamma_relax_rate": 0.7,
        "reject_gamma_relax_steps": 5,
        "reject_poincare_curvature_mult": 2.0,
        "contact_active": False,
        "contact_xi": 1.10,
        "contact_kappa": 8.0,
        "contact_weight_floor": 0.05,
        "contact_max_edges_per_node": 8,
        "contact_w0_norm": 6.0,
        "contact_aabb_safety_margin": 2.0,
        "contact_mass_gain": 0.25,
        "contact_trust_cap_factor": 1.5,
        "contact_max_modes_total": 16,
        "contact_required_fragments": 0,
        "soft_pulse_active": False,
        "soft_capacity": 8,
        "soft_stagnation_window": 5,
        "soft_max_modes_target": 4,
        "soft_stagnation_ratio_threshold": 0.95,
        "soft_pulse_alpha_frac": 0.5,
        "soft_pulse_kappa_floor": 1.0e-6,
        "soft_pulse_orth_floor": 1.0e-9,
        "soft_pulse_cooldown_after_chart": 1,
        "soft_pulse_momentum_policy": 1,
        "soft_pulse_momentum_damp_factor": 0.25,
        "soft_pulse_htvi": False,
        "soft_pulse_min_gmax_to_fire": 1.0e-3,
        "soft_pulse_allow_in_finish_mode": False,
        "soft_negative_curvature_policy": 1,
        "contact_participation_metric_mode": 2,
    }
    _validate_ds_init_kwargs(kwargs)
    return kwargs


def _validate_ds_init_kwargs(kwargs: Dict[str, Any]) -> None:
    failures: List[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def finite_positive(key: str) -> float:
        value = float(kwargs[key])
        require(np.isfinite(value) and value > 0.0, f"{key} must be finite and > 0")
        return value

    finite_positive("gamma")
    finite_positive("h")
    finite_positive("h_min")
    finite_positive("h_max")
    finite_positive("gamma_min")
    finite_positive("gamma_max")
    require(float(kwargs["h_max"]) >= float(kwargs["h"]), "h_max must cover h")
    require(float(kwargs["gamma_max"]) >= float(kwargs["gamma"]), "gamma_max must cover gamma")
    finite_positive("delta0")
    finite_positive("delta_min")
    finite_positive("delta_max")
    require(float(kwargs["delta_max"]) >= float(kwargs["delta_min"]), "delta_max < delta_min")
    require(float(kwargs["delta0"]) >= float(kwargs["delta_min"]), "delta0 < delta_min")
    require(float(kwargs["delta0"]) <= float(kwargs["delta_max"]), "delta0 > delta_max")
    require(float(kwargs["delta_grow"]) > 1.0, "delta_grow must be > 1")
    require(0.0 < float(kwargs["delta_shrink"]) < 1.0, "delta_shrink must be in (0, 1)")
    require(int(kwargs["lanczos_k"]) >= 1, "lanczos_k must be >= 1")
    require(float(kwargs["htvi_p_bregman"]) >= 2.0, "htvi_p_bregman must be >= 2")
    finite_positive("htvi_gamma_0")
    require(int(kwargs["htvi_max_inner"]) >= 1, "htvi_max_inner must be >= 1")
    require(int(kwargs["htvi_picard_max"]) >= 1, "htvi_picard_max must be >= 1")
    finite_positive("htvi_picard_tol_m")
    for key in (
        "geo_sol_dt",
        "geo_sol_tol",
        "bt_ic_tol",
        "cart_step_inf_max",
        "cart_step_rms_max",
        "cart_atom_step_max",
        "cart_min_pair_dist",
        "cart_abs_coord_max",
        "cart_pair_dist_max",
        "grad_inf_max",
        "grad_growth_max",
        "mass_diag_abs_floor",
        "mass_diag_floor_rel",
        "block_metric_eps",
        "block_shape_eps",
        "block_scale_min",
        "block_secant_curv_floor",
        "saddle_lanczos_stab_tol",
        "contact_xi",
        "contact_kappa",
        "contact_weight_floor",
        "contact_w0_norm",
        "contact_aabb_safety_margin",
        "contact_trust_cap_factor",
        "soft_stagnation_ratio_threshold",
        "soft_pulse_alpha_frac",
        "soft_pulse_kappa_floor",
        "soft_pulse_orth_floor",
        "soft_pulse_momentum_damp_factor",
        "soft_pulse_min_gmax_to_fire",
    ):
        finite_positive(key)
    require(int(kwargs["max_backtransform_iter"]) >= 1, "max_backtransform_iter must be >= 1")
    require(int(kwargs["finish_after_uphill"]) >= 1, "finish_after_uphill must be >= 1")
    require(
        int(kwargs["finish_small_step_streak"]) >= 1,
        "finish_small_step_streak must be >= 1",
    )
    require(
        0.0 < float(kwargs["finish_step_ratio_trigger"]) < 1.0,
        "finish_step_ratio_trigger must be in (0, 1)",
    )
    finite_positive("finish_grad_ratio_trigger")
    finite_positive("finish_gmax_trigger")
    finite_positive("finish_small_gmax_cap")
    require(0.0 < float(kwargs["delta_regrow_step_frac"]) < 1.0, "delta_regrow_step_frac invalid")
    require(float(kwargs["controller_de_tol"]) >= 0.0, "controller_de_tol must be >= 0")
    require(int(kwargs["finish_stall_window"]) >= 1, "finish_stall_window must be >= 1")
    require(int(kwargs["finish_stall_min_iter"]) >= 0, "finish_stall_min_iter must be >= 0")
    finite_positive("finish_stall_de_tol")
    finite_positive("finish_stall_step_tol")
    finite_positive("finish_stall_gmax_cap")
    finite_positive("finish_delta_cap")
    require(float(kwargs["finish_delta_grow"]) > 1.0, "finish_delta_grow must be > 1")
    require(0.0 < float(kwargs["finish_delta_shrink"]) < 1.0, "finish_delta_shrink invalid")
    finite_positive("finish_no_grow_gmax")
    finite_positive("finish_disp_target")
    finite_positive("finish_disp_hard")
    require(
        int(kwargs["finish_growth_cooldown_steps"]) >= 0,
        "finish_growth_cooldown_steps must be >= 0",
    )
    finite_positive("finish_stall_delta_cap")
    require(
        int(kwargs["finish_entry_accept_streak_req"]) >= 1,
        "finish_entry_accept_streak_req must be >= 1",
    )
    finite_positive("finish_entry_gmax_trigger")
    finite_positive("finish_entry_dmax_cap")
    finite_positive("finish_stage2_gmax")
    finite_positive("finish_delta_cap_stage1")
    finite_positive("finish_delta_cap_stage2")
    require(
        float(kwargs["finish_delta_grow_stage1"]) > 1.0,
        "finish_delta_grow_stage1 must be > 1",
    )
    require(
        0.0 < float(kwargs["finish_delta_shrink_stage1"]) < 1.0,
        "finish_delta_shrink_stage1 invalid",
    )
    require(
        0.0 < float(kwargs["finish_delta_shrink_stage2"]) < 1.0,
        "finish_delta_shrink_stage2 invalid",
    )
    require(
        float(kwargs["finish_small_gmax_cap"]) >= float(kwargs["finish_gmax_trigger"]),
        "finish_small_gmax_cap must cover finish_gmax_trigger",
    )
    require(int(kwargs["recovery_accepts_to_exit"]) >= 1, "recovery_accepts_to_exit < 1")
    require(0.0 < float(kwargs["recovery_h_scale"]) <= 1.0, "recovery_h_scale invalid")
    require(float(kwargs["recovery_gamma_grow"]) >= 1.0, "recovery_gamma_grow invalid")
    require(0.0 <= float(kwargs["recovery_momentum_scale"]) <= 1.0, "recovery momentum invalid")
    require(0.0 <= float(kwargs["reject_momentum_scale"]) <= 1.0, "reject momentum invalid")
    require(0.0 < float(kwargs["reject_delta_ref_frac"]) <= 1.0, "reject_delta_ref_frac invalid")
    require(0.0 < float(kwargs["reject_h_scale"]) <= 1.0, "reject_h_scale invalid")
    require(float(kwargs["reject_gamma_scale"]) >= 1.0, "reject_gamma_scale invalid")
    require(int(kwargs["max_reject_streak"]) >= 1, "max_reject_streak must be >= 1")
    require(float(kwargs["block_scale_max"]) >= float(kwargs["block_scale_min"]), "block scale bounds invalid")
    require(0.0 < float(kwargs["block_scale_relax"]) <= 1.0, "block_scale_relax invalid")
    require(int(kwargs["block_scale_history_len"]) >= 1, "block_scale_history_len must be >= 1")
    require(
        str(kwargs["block_scale_seed_mode"]).lower() in {"uniform", "hessian"},
        "block_scale_seed_mode invalid",
    )
    require(
        str(kwargs["block_coupling_mode"]).lower() in {"row_gram", "cart_subspace"},
        "block_coupling_mode invalid",
    )
    require(float(kwargs["block_merge_hysteresis_gap"]) >= 0.0, "block_merge_hysteresis_gap invalid")
    require(
        0.0 <= float(kwargs["block_transfer_min_row_overlap"]) <= 1.0,
        "block_transfer_min_row_overlap invalid",
    )
    require(
        0.0 <= float(kwargs["block_transfer_min_subspace_overlap"]) <= 1.0,
        "block_transfer_min_subspace_overlap invalid",
    )
    require(
        str(kwargs["block_transfer_mode"]).lower() in {"weighted", "binary"},
        "block_transfer_mode invalid",
    )
    require(
        float(kwargs["block_transfer_total_weight_floor"]) >= 0.0,
        "block_transfer_total_weight_floor invalid",
    )
    finite_positive("bond_switching_kappa")
    finite_positive("bond_switching_xi0")
    require(
        0.50 <= float(kwargs["external_mode_dominance"]) <= 0.95,
        "external_mode_dominance invalid",
    )
    require(str(kwargs["rel_cap_mode"]).lower() in {"global", "groupwise"}, "rel_cap_mode invalid")
    require(float(kwargs["delta_rel_trans_factor"]) >= 0.0, "delta_rel_trans_factor invalid")
    require(float(kwargs["delta_rel_rot_factor"]) >= 0.0, "delta_rel_rot_factor invalid")
    require(float(kwargs["delta_rel_mixed_factor"]) >= 0.0, "delta_rel_mixed_factor invalid")
    require(int(kwargs["saddle_lanczos_k_min"]) >= 1, "saddle_lanczos_k_min must be >= 1")
    require(
        int(kwargs["saddle_lanczos_k_max"]) >= int(kwargs["saddle_lanczos_k_min"]),
        "saddle_lanczos_k_max must cover min",
    )
    require(
        str(kwargs["hpos_estimator"]).lower() in {"syev", "lanczos"},
        "hpos_estimator invalid",
    )
    require(
        str(kwargs["hessian_model"]).lower() in _HESSIAN_MODEL_MAP,
        "hessian_model invalid",
    )
    require(int(kwargs["contact_max_edges_per_node"]) >= 1, "contact_max_edges_per_node invalid")
    require(float(kwargs["contact_mass_gain"]) >= 0.0, "contact_mass_gain invalid")
    require(int(kwargs["contact_max_modes_total"]) >= 0, "contact_max_modes_total invalid")
    require(int(kwargs["contact_required_fragments"]) >= 0, "contact_required_fragments invalid")
    require(int(kwargs["soft_capacity"]) >= 1, "soft_capacity invalid")
    require(int(kwargs["soft_stagnation_window"]) >= 1, "soft_stagnation_window invalid")
    require(int(kwargs["soft_max_modes_target"]) >= 1, "soft_max_modes_target invalid")
    require(int(kwargs["soft_pulse_cooldown_after_chart"]) >= 0, "soft_pulse_cooldown_after_chart invalid")
    require(int(kwargs["soft_pulse_momentum_policy"]) in {1, 2, 3}, "soft_pulse_momentum_policy invalid")
    require(int(kwargs["soft_negative_curvature_policy"]) in {1, 2}, "soft_negative_curvature_policy invalid")
    require(int(kwargs["contact_participation_metric_mode"]) in {1, 2}, "contact_participation_metric_mode invalid")
    if failures:
        raise ValueError("invalid DS init defaults: " + "; ".join(failures))


def _set_invalid_trial_reason(opt, reason: InvalidTrialReason) -> None:
    setter = getattr(opt, "set_invalid_trial_reason_py", None)
    if setter is None:
        return
    try:
        setter(int(_TRIAL_REASON_CODES[reason.value]))
    except Exception:
        return


def _push_positions(atoms, q_xyz):
    """Write a (N, 3) Angstrom array back into an ASE Atoms object."""
    atoms.set_positions(np.asarray(q_xyz, dtype=np.float64))


def _build_trqn(ariadne, q0_xyz, g0_xyz, atom_list, run_config):
    """Build and init a trust-region quasi-Newton optimiser.

    Pass a full starter-pack-style init profile explicitly. Several f90wrap
    builds have treated omitted optional scalar arguments as present zero
    values; the geodesic ODE controls are especially sensitive to that.
    """
    opt = ariadne.Geometric_Trqn.trust_region_qn()
    opt.init(**_trqn_init_kwargs(q0_xyz, g0_xyz, atom_list, run_config))
    return opt


def _build_ds(ariadne, q0_xyz, g0_xyz, atom_list, run_config):
    """Build and init a dissipative-symplectic optimiser.

    DS interprets h as its initial step size; we reuse delta0 from the
    config so the two optimisers start from the same trust scale.
    """
    opt = ariadne.Ds_Optimiser.dissipative_symplectic()
    init_kwargs = _ds_safe_init_kwargs(run_config)
    init_kwargs.update(q0_xyz=q0_xyz, g0_xyz=g0_xyz, atom_list=atom_list)
    opt.init(**init_kwargs)
    return opt


def probe_ariadne_runtime(ariadne) -> Dict[str, Any]:
    """Initialise both optimisers and validate the live wrapper contract.

    This probe deliberately stops before ``step_py``.  It catches compiler-
    and wrapper-specific logical representations and status-layout drift at
    preflight time without loading models or evaluating the acquisition.
    """
    from .ariadne_runner import AriadneRunConfig

    probe_ariadne_module(ariadne)
    q0_xyz = np.asfortranarray(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2400, 0.9270, 0.0000],
            [3.0000, 0.0000, 0.0000],
            [3.9572, 0.0000, 0.0000],
            [2.7600, 0.9270, 0.0000],
        ],
        dtype=np.float64,
    )
    g0_xyz = np.asfortranarray(
        np.full((6, 3), 1.0e-4, dtype=np.float64)
    )
    atom_list = _symbols_to_atom_list(["O", "H", "H", "O", "H", "H"])
    run_config = AriadneRunConfig()

    receipts: Dict[str, Any] = {}
    for name, builder in (("trqn", _build_trqn), ("ds", _build_ds)):
        try:
            optimiser = builder(
                ariadne,
                q0_xyz.copy(order="F"),
                g0_xyz.copy(order="F"),
                atom_list.copy(),
                run_config,
            )
            direct_flags: Dict[str, bool] = {}
            raw_init_ok = _optimizer_flag(optimiser, "init_ok")
            if raw_init_ok is _OPTIMIZER_ATTRIBUTE_MISSING:
                if name == "trqn":
                    raise RuntimeError("ARIADNE TRQN has no init_ok property")
            else:
                init_ok = _fortran_logical(
                    raw_init_ok,
                    name.upper() + " init_ok",
                )
                direct_flags["init_ok"] = bool(init_ok)
                if not init_ok:
                    _raise_if_init_failed(optimiser, name.upper())

            status = _validate_status_shape(
                optimiser.get_status_py(),
                optimiser=name,
            )
            if name == "trqn":
                status_flags = {
                    "proposal_pending": _strict_status_flag(
                        status, _TRQN_STATUS_PROPOSAL_PENDING, "TRQN proposal_pending"
                    ),
                    "converged": _strict_status_flag(
                        status, 4, "TRQN converged"
                    ),
                    "last_step_accepted": _strict_status_flag(
                        status, 5, "TRQN last_step_accepted"
                    ),
                    "skip_step_after_rebuild": _strict_status_flag(
                        status,
                        _TRQN_STATUS_SKIP_STEP_AFTER_REBUILD,
                        "TRQN skip_step_after_rebuild",
                    ),
                }
            else:
                status_flags = {
                    "proposal_pending": _strict_status_flag(
                        status, _DS_STATUS_PROPOSAL_PENDING, "DS proposal_pending"
                    ),
                    "converged": _strict_status_flag(status, 8, "DS converged"),
                    "last_step_accepted": _strict_status_flag(
                        status, 9, "DS last_step_accepted"
                    ),
                }

            for property_name in (
                "proposal_pending",
                "skip_step_after_rebuild",
            ):
                raw_value = _optimizer_flag(optimiser, property_name)
                if raw_value is _OPTIMIZER_ATTRIBUTE_MISSING:
                    continue
                direct_flags[property_name] = _fortran_logical(
                    raw_value,
                    name.upper() + " " + property_name,
                )

            q_state, g_state = _sync_state(optimiser, 6)
            if q_state.shape != (6, 3) or g_state.shape != (6, 3):
                raise RuntimeError(
                    "ARIADNE " + name.upper() + " state has the wrong shape"
                )
            if not np.all(np.isfinite(q_state)) or not np.all(np.isfinite(g_state)):
                raise RuntimeError(
                    "ARIADNE " + name.upper() + " state contains non-finite values"
                )
        except Exception as exc:
            raise RuntimeError(
                "ARIADNE " + name.upper() + " initialised runtime probe failed: " + str(exc)
            ) from exc

        receipts[name] = {
            "initialised": True,
            "status_length": int(len(status)),
            "direct_logical_flags": direct_flags,
            "status_flags": status_flags,
            "state_shape": [int(value) for value in q_state.shape],
            "state_finite": True,
        }

    return {
        "probe": "initialise_without_step",
        "geometry": "water_dimer_6_atom",
        **receipts,
    }


def _restart_under_ds(ariadne, atoms, natoms, atom_list, run_config):
    # Re-evaluate at the exact geometry DS will restart from. Gradients from a
    # rejected TRQN proposal or a failed backtransform do not belong to this
    # point and should not be re-used.
    f_here, g_here = _eval_energy_gradient(atoms)
    opt = _build_ds(
        ariadne,
        np.asfortranarray(atoms.get_positions(), dtype=np.float64),
        np.asfortranarray(g_here.reshape(natoms, 3), dtype=np.float64),
        atom_list,
        run_config,
    )
    _raise_if_init_failed(opt, "DS")
    return opt, float(f_here), _flatten_xyz(g_here)


def run_optimisation_against_calculator(
    seed_atoms,
    calculator,
    run_config,
    trace_path=None,
    initial_positions_angstrom=None,
    initial_origin: str = "seed_initial",
    warm_start_records=None,
):
    """Drive ARIADNE for a single seed against the supplied calculator.

    Parameters
    ----------
    seed_atoms
        ASE Atoms at the starting geometry. We work on a copy so the
        caller object is left alone.
    calculator
        ASE Calculator that returns energy + forces. For our case this
        is AdversarialASECalculator wrapping SeedLocalAdversarialAcquisition
        built for this seed.
    run_config
        AriadneRunConfig: optimiser controls plus a frozen convergence policy.

    Returns
    -------
    OptimisationResult
    """
    ariadne = _import_ariadne()
    t0 = time.perf_counter()

    atoms = seed_atoms.copy()
    atoms.calc = calculator
    if initial_positions_angstrom is not None:
        atoms.set_positions(np.asarray(initial_positions_angstrom, dtype=np.float64).reshape(-1, 3))

    natoms = len(atoms)
    symbols = [a.symbol for a in atoms]
    atom_list = _symbols_to_atom_list(symbols)
    q0_xyz_angstrom = np.asfortranarray(
        atoms.get_positions(), dtype=np.float64,
    )

    optimiser_name = (run_config.optimiser or "trust_region_qn").strip().lower()
    diagnostics = _new_optimiser_diagnostics(optimiser_name)
    convergence_resolution = _resolved_convergence(run_config)
    diagnostics["convergence_resolution"] = dict(convergence_resolution)
    warm_start_records = list(warm_start_records or [])

    # one calculator call before the loop -- ariadne needs an initial
    # energy + gradient to seed its internal hessian model. Evaluate once in
    # raw acquisition units, then scale only the optimiser-facing copy when
    # TRQN requests adaptive objective conditioning.
    e0_raw, g0_raw = _eval_energy_gradient(atoms)
    raw_initial_grad_norm = float(np.linalg.norm(g0_raw))
    active_objective_scale = 1.0
    trqn_scale_info = {
        "mode": "not_applicable",
        "scale": 1.0,
        "target": None,
        "raw_grad_norm": raw_initial_grad_norm,
        "scaled_grad_norm": raw_initial_grad_norm,
        "reason": "not_trqn",
        "retry": False,
    }
    n_evaluations = 1
    if optimiser_name in ("trust_region_qn", "trqn"):
        trqn_scale_info = _compute_trqn_objective_scale(
            run_config,
            g0_raw,
            atoms=atoms,
        )
        active_objective_scale = float(trqn_scale_info["scale"])
    e0_hartree = e0_raw * active_objective_scale
    g0_flat = g0_raw * active_objective_scale
    g0_xyz = np.asfortranarray(g0_flat.reshape(natoms, 3), dtype=np.float64)

    if optimiser_name in ("ds", "dissipative_symplectic"):
        opt = _build_ds(ariadne, q0_xyz_angstrom, g0_xyz, atom_list, run_config)
        is_trqn = False
    elif optimiser_name in ("trust_region_qn", "trqn"):
        opt = _build_trqn(ariadne, q0_xyz_angstrom, g0_xyz, atom_list, run_config)
        is_trqn = True
    else:
        raise ValueError(
            "unknown ariadne.optimiser " + repr(run_config.optimiser)
            + "; valid: trust_region_qn | dissipative_symplectic"
        )
    if not is_trqn:
        diagnostics["ds_init_profile"] = _DS_INIT_PROFILE
    diagnostics.update({
        "objective_scale_schema_version": 1,
        "trqn_scale_mode": str(trqn_scale_info["mode"]),
        "trqn_objective_scale": float(trqn_scale_info["scale"]),
        "trqn_scale_reason": str(trqn_scale_info["reason"]),
        "trqn_target_initial_grad_norm": trqn_scale_info["target"],
        "trqn_scale_target_kind": trqn_scale_info.get("target_kind"),
        "trqn_initial_raw_grad_norm": float(trqn_scale_info["raw_grad_norm"]),
        "trqn_initial_raw_grad_rms": trqn_scale_info.get("raw_grad_rms"),
        "trqn_initial_scaled_grad_norm": trqn_scale_info["scaled_grad_norm"],
        "trqn_initial_scaled_grad_rms": trqn_scale_info.get("scaled_grad_rms"),
        "trqn_n_live_cartesian_dof": trqn_scale_info.get("n_live_cartesian_dof"),
        "trqn_n_projected_rigid_dof": trqn_scale_info.get("n_projected_rigid_dof"),
        "trqn_n_effective_internal_dof": trqn_scale_info.get("n_effective_internal_dof"),
        "trqn_gradient_projected_rigid": trqn_scale_info.get("gradient_projected_rigid"),
        "trqn_objective_scale_clamped": trqn_scale_info.get("objective_scale_clamped"),
        "trqn_retry_on_no_proposal": bool(
            getattr(run_config, "trqn_retry_on_no_proposal", False)
        ),
        "trqn_backtransform_mode": _trqn_backtransform_mode(run_config),
        "trqn_geodesic_bt_mode": _trqn_geodesic_bt_mode(run_config),
        "ariadne_cartesian_recovery_mode": _ariadne_cartesian_recovery_mode_id(
            _trqn_backtransform_mode(run_config)
        ),
        "ariadne_geo_bt_mode": _ariadne_geo_bt_mode_id(
            _trqn_geodesic_bt_mode(run_config)
        ),
        "trqn_init_profile": {
            str(k): _json_safe_value(v)
            for k, v in sorted(_trqn_control_init_kwargs(run_config).items())
        },
        "trqn_retry_attempted": False,
        "trqn_retry_objective_scale": None,
        "trqn_retry_target_initial_grad_norm": None,
        "trqn_retry_raw_grad_norm": None,
        "trqn_retry_scaled_grad_norm": None,
        "trqn_retry_reason": None,
        "trqn_retry_succeeded": None,
        "trqn_failed_after_retry": False,
        "gradient_band_warm_start": (
            str(initial_origin) == "gradient_band_warm_start"
        ),
        "gradient_band_warm_start_candidates": list(warm_start_records),
    })

    _raise_if_init_failed(opt, "DS" if not is_trqn else "TRQN")
    _append_trace_event_with_gradient(
        trace_path,
        calculator,
        event="start",
        step=-1,
        optimiser=("trust_region_qn" if is_trqn else "dissipative_symplectic"),
        alpha=_raw_alpha_from_scaled(e0_hartree, active_objective_scale),
        grad_norm=_raw_grad_norm_from_scaled(g0_flat, active_objective_scale),
        objective_scale=float(active_objective_scale),
        raw_grad_norm=float(raw_initial_grad_norm),
        scaled_grad_norm=float(np.linalg.norm(g0_flat)),
        scale_target=trqn_scale_info["target"],
        scale_mode=str(trqn_scale_info["mode"]),
        trqn_backtransform_mode=_trqn_backtransform_mode(run_config),
        trqn_geodesic_bt_mode=_trqn_geodesic_bt_mode(run_config),
        accepted=True,
        wall_seconds=float(time.perf_counter() - t0),
    )

    # alpha is what we want to MAXIMISE (the adversarial acquisition
    # value). the calculator exposes a pseudo-energy so ariadne can
    # minimise, so alpha = -f_pseudo.
    alpha_trajectory = [_raw_alpha_from_scaled(e0_hartree, active_objective_scale)]
    grad_norm_trajectory = [
        _raw_grad_norm_from_scaled(g0_flat, active_objective_scale)
    ]
    candidate_positions = [np.asarray(q0_xyz_angstrom, dtype=np.float64).copy()]
    candidate_alphas = [float(alpha_trajectory[0])]
    candidate_grad_norms = [float(grad_norm_trajectory[0])]
    candidate_origins = [str(initial_origin or "seed_initial")]
    fell_back_to_ds = False
    trqn_retry_attempted = False
    rigid_force_clamps = 0
    consecutive_rejects = 0
    convergence_streak = 0
    no_proposal_failure_reason: Optional[str] = None
    f_current = e0_hartree
    converged = False
    return_code = 1  # max_iter default until we converge or error

    zero_grad = np.zeros(3 * natoms, dtype=np.float64, order="F")

    for step_idx in range(int(run_config.max_iter)):
        f_old = f_current
        positions_before_step = np.asarray(
            atoms.get_positions(), dtype=np.float64
        ).copy()

        # stage 0 -- propose a step. ariadne does not need a trial
        # energy yet so we pass the previous f and a zero gradient.
        try:
            diagnostics["n_stage0_calls"] += 1
            opt.step_py(
                stage=0, f_old=f_old, f_new=f_old, g_xyz_new=zero_grad,
            )
        except Exception as exc:
            return_code = 2
            diagnostics["last_return_code_reason"] = (
                "stage0_exception:" + type(exc).__name__
            )
            break
        try:
            status_after_stage0 = _validate_status_shape(
                _status_tuple(opt),
                optimiser="trqn" if is_trqn else "ds",
            )
        except Exception as exc:
            return_code = 2
            diagnostics["last_return_code_reason"] = (
                "status_abi_mismatch:" + type(exc).__name__ + ":" + str(exc)
            )
            break
        _append_status_sample(
            diagnostics,
            step_index=step_idx,
            label="after_stage0",
            status=status_after_stage0,
        )
        proposal_pending = _proposal_pending(
            opt,
            status_after_stage0,
            is_trqn=is_trqn,
        )
        skip_after_rebuild = _skip_step_after_rebuild(
            opt,
            status_after_stage0,
            is_trqn=is_trqn,
        )
        no_proposal_failure_reason = None
        if (
            is_trqn
            and not proposal_pending
            and not skip_after_rebuild
        ):
            no_proposal_failure_reason = _trqn_no_proposal_failure_reason(
                status_after_stage0,
            )
        if (
            not proposal_pending
            or skip_after_rebuild
        ):
            convergence_streak = 0
            diagnostics["daemon_convergence_streak"] = 0
            if not proposal_pending:
                diagnostics["n_no_proposal_pending"] += 1
            if skip_after_rebuild:
                diagnostics["n_skip_step_after_rebuild"] += 1
            q_current, g_current = _sync_state(opt, natoms)
            _push_positions(atoms, q_current)
            try:
                diagnostics["n_stage1_calls"] += 1
                opt.step_py(
                    stage=1,
                    f_old=f_old,
                    f_new=f_old,
                    g_xyz_new=_flatten_xyz(g_current),
                )
            except Exception as exc:
                return_code = 2
                diagnostics["last_return_code_reason"] = (
                    "no_proposal_stage1_exception:" + type(exc).__name__
                )
                break
            _append_status_sample(
                diagnostics,
                step_index=step_idx,
                label="after_no_proposal_stage1",
                status=_status_tuple(opt),
            )
            trace_extra = _trace_status(status_after_stage0)
            _append_trace_event_with_gradient(
                trace_path,
                calculator,
                event="trqn_no_proposal" if is_trqn else "no_proposal",
                step=step_idx,
                optimiser=(
                    "trust_region_qn" if is_trqn else "dissipative_symplectic"
                ),
                alpha=_raw_alpha_from_scaled(f_current, active_objective_scale),
                grad_norm=_raw_grad_norm_from_scaled(
                    g_current,
                    active_objective_scale,
                ),
                objective_scale=float(active_objective_scale),
                accepted=False,
                wall_seconds=float(time.perf_counter() - t0),
                **trace_extra,
            )
            if is_trqn:
                opt_converged, accepted, f_current = _status_trqn(opt)
            else:
                opt_converged, accepted, f_current = _status_ds(opt)
            alpha_trajectory.append(
                _raw_alpha_from_scaled(f_current, active_objective_scale)
            )
            grad_norm_trajectory.append(
                _raw_grad_norm_from_scaled(g_current, active_objective_scale)
            )
            if opt_converged:
                converged = True
                return_code = 0
                diagnostics["termination_source"] = "native_optimiser"
                break
            if no_proposal_failure_reason is not None:
                _record_no_proposal_failure(
                    diagnostics,
                    reason=no_proposal_failure_reason,
                    status=status_after_stage0,
                )
                consecutive_rejects += 1
                if consecutive_rejects >= _REJECT_STREAK_TRIGGER:
                    if (
                        is_trqn
                        and bool(getattr(run_config, "trqn_retry_on_no_proposal", False))
                        and not trqn_retry_attempted
                    ):
                        try:
                            f_retry_raw, g_retry_raw = _eval_energy_gradient(atoms)
                            n_evaluations += 1
                            retry_info = _compute_trqn_objective_scale(
                                run_config,
                                g_retry_raw,
                                retry=True,
                                atoms=atoms,
                            )
                            active_objective_scale = float(retry_info["scale"])
                            f_current = f_retry_raw * active_objective_scale
                            g_retry = g_retry_raw * active_objective_scale
                            opt = _build_trqn(
                                ariadne,
                                np.asfortranarray(
                                    atoms.get_positions(), dtype=np.float64,
                                ),
                                np.asfortranarray(
                                    g_retry.reshape(natoms, 3), dtype=np.float64,
                                ),
                                atom_list,
                                run_config,
                            )
                            _raise_if_init_failed(opt, "TRQN retry")
                            trqn_retry_attempted = True
                            consecutive_rejects = 0
                            diagnostics["trqn_retry_attempted"] = True
                            diagnostics["trqn_retry_objective_scale"] = float(
                                retry_info["scale"]
                            )
                            diagnostics["trqn_retry_target_initial_grad_norm"] = (
                                retry_info["target"]
                            )
                            diagnostics["trqn_retry_raw_grad_norm"] = float(
                                retry_info["raw_grad_norm"]
                            )
                            diagnostics["trqn_retry_raw_grad_rms"] = (
                                retry_info.get("raw_grad_rms")
                            )
                            diagnostics["trqn_retry_scaled_grad_norm"] = (
                                retry_info["scaled_grad_norm"]
                            )
                            diagnostics["trqn_retry_scaled_grad_rms"] = (
                                retry_info.get("scaled_grad_rms")
                            )
                            diagnostics["trqn_retry_reason"] = str(
                                no_proposal_failure_reason
                            )
                            diagnostics["trqn_retry_succeeded"] = None
                            _append_trace_event_with_gradient(
                                trace_path,
                                calculator,
                                event="trqn_retry",
                                step=step_idx,
                                optimiser="trust_region_qn",
                                alpha=_raw_alpha_from_scaled(
                                    f_current,
                                    active_objective_scale,
                                ),
                                grad_norm=_raw_grad_norm_from_scaled(
                                    g_retry,
                                    active_objective_scale,
                                ),
                                objective_scale=float(active_objective_scale),
                                raw_grad_norm=float(retry_info["raw_grad_norm"]),
                                scaled_grad_norm=retry_info["scaled_grad_norm"],
                                scale_target=retry_info["target"],
                                scale_mode=str(retry_info["mode"]),
                                accepted=False,
                                reason=str(no_proposal_failure_reason),
                                wall_seconds=float(time.perf_counter() - t0),
                            )
                            continue
                        except Exception as exc:
                            trqn_retry_attempted = True
                            diagnostics["trqn_retry_attempted"] = True
                            diagnostics["trqn_retry_succeeded"] = False
                            diagnostics["trqn_failed_after_retry"] = True
                            diagnostics["trqn_retry_reason"] = (
                                "retry_exception:" + type(exc).__name__
                            )
                    if (
                        is_trqn
                        and bool(run_config.fallback_to_ds)
                        and not fell_back_to_ds
                    ):
                        try:
                            fallback_from_scale = float(active_objective_scale)
                            opt, f_current, g_current_flat = _restart_under_ds(
                                ariadne,
                                atoms,
                                natoms,
                                atom_list,
                                run_config,
                            )
                            n_evaluations += 1
                            active_objective_scale = 1.0
                            is_trqn = False
                            fell_back_to_ds = True
                            if bool(diagnostics.get("trqn_retry_attempted")):
                                diagnostics["trqn_retry_succeeded"] = False
                                diagnostics["trqn_failed_after_retry"] = True
                            diagnostics["n_fallback_to_ds"] += 1
                            diagnostics["n_no_proposal_recoveries"] += 1
                            diagnostics["ds_init_profile"] = _DS_INIT_PROFILE
                            diagnostics["optimiser_final"] = (
                                "dissipative_symplectic"
                            )
                            consecutive_rejects = 0
                            diagnostics["fallback_to_ds_step"] = int(step_idx)
                            diagnostics["fallback_to_ds_reason"] = str(
                                no_proposal_failure_reason
                            )
                            diagnostics["trqn_failed_reason"] = str(
                                no_proposal_failure_reason
                            )
                            diagnostics["trqn_no_proposal_reason"] = str(
                                no_proposal_failure_reason
                            )
                            bt_count = (
                                diagnostics.get(
                                    "last_no_proposal_status_summary", {}
                                )
                                or {}
                            ).get("consecutive_bt_fail_count")
                            diagnostics[
                                "trqn_consecutive_backtransform_fail_count"
                            ] = bt_count
                            _append_trace_event_with_gradient(
                                trace_path,
                                calculator,
                                event="fallback_to_ds",
                                step=step_idx,
                                optimiser="dissipative_symplectic",
                                alpha=_raw_alpha_from_scaled(
                                    f_current,
                                    active_objective_scale,
                                ),
                                grad_norm=_raw_grad_norm_from_scaled(
                                    g_current_flat,
                                    active_objective_scale,
                                ),
                                objective_scale=float(active_objective_scale),
                                previous_objective_scale=float(fallback_from_scale),
                                accepted=False,
                                reason=str(no_proposal_failure_reason),
                                wall_seconds=float(time.perf_counter() - t0),
                            )
                        except Exception as exc:
                            return_code = 2
                            diagnostics["last_return_code_reason"] = (
                                "fallback_to_ds_exception:" + type(exc).__name__
                            )
                            break
                    else:
                        return_code = 2
                        diagnostics["last_return_code_reason"] = (
                            no_proposal_failure_reason
                        )
                        break
            continue

        # pull the proposed geometry out and push it into the ASE atoms
        # so the calculator can evaluate the trial point.
        q_proposed, _g_proposed = _sync_state(opt, natoms)
        _push_positions(atoms, q_proposed)

        try:
            f_trial, g_trial = _eval_energy_gradient(
                atoms,
                objective_scale=active_objective_scale,
            )
            n_evaluations += 1
            diagnostics["n_trial_evaluations"] += 1
        except Exception as exc:
            invalid_reason = _classify_trial_exception(exc)
            _set_invalid_trial_reason(opt, invalid_reason)
            diagnostics["last_invalid_trial_reason"] = invalid_reason.value
            diagnostics["last_invalid_trial_reason_code"] = int(
                _TRIAL_REASON_CODES[invalid_reason.value]
            )
            try:
                diagnostics["n_stage1_calls"] += 1
                opt.step_py(
                    stage=1,
                    f_old=f_old,
                    f_new=0.0,
                    g_xyz_new=zero_grad,
                )
            except Exception:
                pass
            return_code = 2
            diagnostics["last_return_code_reason"] = (
                "trial_evaluation_exception:" + type(exc).__name__
            )
            break

        # stage 1 -- ariadne sees the trial energy + gradient and
        # decides whether to accept, reject, or declare convergence.
        try:
            diagnostics["n_stage1_calls"] += 1
            opt.step_py(
                stage=1, f_old=f_old, f_new=float(f_trial),
                g_xyz_new=g_trial,
            )
        except Exception as exc:
            return_code = 2
            diagnostics["last_return_code_reason"] = (
                "stage1_exception:" + type(exc).__name__
            )
            break
        _append_status_sample(
            diagnostics,
            step_index=step_idx,
            label="after_stage1",
            status=_status_tuple(opt),
        )

        q_current, g_current = _sync_state(opt, natoms)
        _push_positions(atoms, q_current)

        if is_trqn:
            opt_converged, accepted, f_current = _status_trqn(opt)
        else:
            opt_converged, accepted, f_current = _status_ds(opt)

        alpha_trajectory.append(
            _raw_alpha_from_scaled(f_current, active_objective_scale)
        )
        current_grad_norm = _raw_grad_norm_from_scaled(
            g_current,
            active_objective_scale,
        )
        trial_grad_norm = _raw_grad_norm_from_scaled(
            g_trial,
            active_objective_scale,
        )
        grad_norm_trajectory.append(
            current_grad_norm
        )

        if accepted:
            diagnostics["n_accepted_steps"] += 1
            consecutive_rejects = 0
            candidate_positions.append(
                np.asarray(atoms.get_positions(), dtype=np.float64).copy()
            )
            candidate_alphas.append(float(alpha_trajectory[-1]))
            candidate_grad_norms.append(float(current_grad_norm))
            candidate_origins.append("accepted_iterate")
            if bool(diagnostics.get("trqn_retry_attempted")) and is_trqn:
                diagnostics["trqn_retry_succeeded"] = True
        else:
            diagnostics["n_rejected_steps"] += 1
            consecutive_rejects += 1
            convergence_streak = _advance_convergence_streak(
                convergence_streak,
                accepted=False,
            )
            diagnostics["daemon_convergence_streak"] = 0
        _append_trace_event_with_gradient(
            trace_path,
            calculator,
            event="accepted_step" if accepted else "rejected_step",
            step=step_idx,
            optimiser=("trust_region_qn" if is_trqn else "dissipative_symplectic"),
            alpha=_raw_alpha_from_scaled(f_current, active_objective_scale),
            grad_norm=float(current_grad_norm),
            trial_alpha=_raw_alpha_from_scaled(f_trial, active_objective_scale),
            trial_grad_norm=float(trial_grad_norm),
            objective_scale=float(active_objective_scale),
            accepted=bool(accepted),
            wall_seconds=float(time.perf_counter() - t0),
        )

        if opt_converged:
            converged = True
            return_code = 0
            diagnostics["termination_source"] = "native_optimiser"
            break

        if accepted:
            try:
                convergence_metrics = _convergence_metrics(
                    f_old=f_old,
                    f_current=f_current,
                    gradient_current=g_current,
                    positions_before=positions_before_step,
                    positions_current=q_current,
                    objective_scale=active_objective_scale,
                )
                convergence_checks = _convergence_checks(
                    convergence_metrics,
                    convergence_resolution,
                )
            except Exception as exc:
                return_code = 2
                diagnostics["last_return_code_reason"] = (
                    "convergence_evaluation_failed:"
                    + type(exc).__name__
                    + ":"
                    + str(exc)
                )
                break
            convergence_streak = _advance_convergence_streak(
                convergence_streak,
                accepted=True,
                checks=convergence_checks,
            )
            diagnostics["daemon_convergence_streak"] = int(convergence_streak)
            diagnostics["daemon_convergence_last_metrics"] = dict(
                convergence_metrics
            )
            diagnostics["daemon_convergence_last_checks"] = dict(
                convergence_checks
            )
            if convergence_streak >= int(
                convergence_resolution["consecutive_accepted_steps"]
            ):
                converged = True
                return_code = 0
                diagnostics["termination_source"] = "daemon_five_criterion"
                diagnostics["last_return_code_reason"] = (
                    "daemon_five_criterion_converged"
                )
                break

        # TRQN -> DS fallback. when too many consecutive proposals
        # get rejected and the config allows it, abandon TRQN and
        # re-init under DS at the current geometry. only one fallback
        # per run.
        if (is_trqn and bool(run_config.fallback_to_ds)
            and not fell_back_to_ds
            and consecutive_rejects >= _REJECT_STREAK_TRIGGER):
            try:
                fallback_from_scale = float(active_objective_scale)
                opt, f_current, g_current_flat = _restart_under_ds(
                    ariadne,
                    atoms,
                    natoms,
                    atom_list,
                    run_config,
                )
                n_evaluations += 1
                active_objective_scale = 1.0
                is_trqn = False
                fell_back_to_ds = True
                if bool(diagnostics.get("trqn_retry_attempted")):
                    diagnostics["trqn_retry_succeeded"] = False
                    diagnostics["trqn_failed_after_retry"] = True
                diagnostics["n_fallback_to_ds"] += 1
                diagnostics["ds_init_profile"] = _DS_INIT_PROFILE
                diagnostics["optimiser_final"] = "dissipative_symplectic"
                consecutive_rejects = 0
                diagnostics["fallback_to_ds_step"] = int(step_idx)
                diagnostics["fallback_to_ds_reason"] = "reject_streak"
                diagnostics["trqn_failed_reason"] = "reject_streak"
                _append_trace_event_with_gradient(
                    trace_path,
                    calculator,
                    event="fallback_to_ds",
                    step=step_idx,
                    optimiser="dissipative_symplectic",
                    alpha=_raw_alpha_from_scaled(f_current, active_objective_scale),
                    grad_norm=_raw_grad_norm_from_scaled(
                        g_current_flat,
                        active_objective_scale,
                    ),
                    objective_scale=float(active_objective_scale),
                    previous_objective_scale=float(fallback_from_scale),
                    accepted=False,
                    reason="reject_streak",
                    wall_seconds=float(time.perf_counter() - t0),
                )
            except Exception as exc:
                return_code = 2
                diagnostics["last_return_code_reason"] = (
                    "fallback_to_ds_exception:" + type(exc).__name__
                )
                break

    # the adversarial calculator subscripts the clamp counter as a dict;
    # per_atom_acquisition_grad is the current key. read the old key too so
    # older calculator shims remain schema-compatible.
    cc = getattr(calculator, "_clamp_counter", None)
    if isinstance(cc, dict):
        try:
            rigid_force_clamps = int(
                cc.get("per_atom_acquisition_grad", cc.get("per_atom_force", 0))
            )
        except (TypeError, ValueError):
            rigid_force_clamps = 0

    _finalise_optimiser_diagnostics(
        diagnostics,
        return_code=int(return_code),
        converged=bool(converged),
    )
    if int(return_code) == 1:
        _append_trace_event_with_gradient(
            trace_path,
            calculator,
            event="max_iterations",
            step=int(run_config.max_iter),
            optimiser=("trust_region_qn" if is_trqn else "dissipative_symplectic"),
            alpha=(float(alpha_trajectory[-1]) if alpha_trajectory else None),
            grad_norm=(
                float(grad_norm_trajectory[-1])
                if grad_norm_trajectory else None
            ),
            objective_scale=float(active_objective_scale),
            accepted=False,
            reason=str(diagnostics.get("last_return_code_reason", "")),
            wall_seconds=float(time.perf_counter() - t0),
        )
    _append_trace_event_with_gradient(
        trace_path,
        calculator,
        event="finish",
        step=int(len(alpha_trajectory) - 1),
        optimiser=("trust_region_qn" if is_trqn else "dissipative_symplectic"),
        alpha=(float(alpha_trajectory[-1]) if alpha_trajectory else None),
        grad_norm=(
            float(grad_norm_trajectory[-1])
            if grad_norm_trajectory else None
        ),
        objective_scale=float(active_objective_scale),
        accepted=bool(converged),
        reason=str(diagnostics.get("last_return_code_reason", "")),
        wall_seconds=float(time.perf_counter() - t0),
    )

    return OptimisationResult(
        final_positions_angstrom=np.asarray(
            atoms.get_positions(), dtype=np.float64,
        ),
        alpha_trajectory=alpha_trajectory,
        grad_norm_trajectory=grad_norm_trajectory,
        n_evaluations=int(n_evaluations),
        return_code=int(return_code),
        wall_seconds=float(time.perf_counter() - t0),
        fell_back_to_ds=bool(fell_back_to_ds),
        rigid_force_clamps=int(rigid_force_clamps),
        converged=bool(converged),
        candidate_positions_angstrom=candidate_positions,
        candidate_alphas=candidate_alphas,
        candidate_grad_norms=candidate_grad_norms,
        candidate_origins=candidate_origins,
        diagnostics=diagnostics,
    )
