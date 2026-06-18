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

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# ASE's Hartree-to-eV scale. Here it is only the reversible pseudo-energy
# scale used by AdversarialASECalculator at the ASE boundary.
_HARTREE_EV = 27.211386245988

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
_ARIADNE_CARTESIAN_RECOVERY_NEWTON = 2
_ARIADNE_GEO_BT_DENSE = 1
_ARIADNE_TRQN_CONTROLLER_NONE = 0

# TRQN get_status_py layout. Keep these names close to the Fortran/starter-pack
# contract so result.json diagnostics can be decoded without reading raw tuples.
_TRQN_STATUS_TRUST = 0
_TRQN_STATUS_RUN_IDX = 2
_TRQN_STATUS_PROPOSAL_PENDING = 3
_TRQN_STATUS_INVALID_REASON = 23
_TRQN_STATUS_STEP_STATE = 35
_TRQN_STATUS_FORCE_REBUILD = 39
_TRQN_STATUS_SKIP_STEP_AFTER_REBUILD = 54
_TRQN_STATUS_CONSECUTIVE_BT_FAIL_COUNT = 65

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
        "schema_version": 1,
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
        "ds_init_profile": None,
        "last_return_code_reason": "not_finished",
        "last_no_proposal_reason": None,
        "last_no_proposal_status_summary": None,
        "status_samples_first": [],
        "status_samples_last": [],
    }


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
        "force_rebuild": _status_bool(status, _TRQN_STATUS_FORCE_REBUILD, False),
        "skip_step_after_rebuild": _status_bool(
            status, _TRQN_STATUS_SKIP_STEP_AFTER_REBUILD, False,
        ),
        "consecutive_bt_fail_count": _status_int(
            status, _TRQN_STATUS_CONSECUTIVE_BT_FAIL_COUNT, None,
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


def _finalise_optimiser_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    return_code: int,
    converged: bool,
) -> None:
    diagnostics["return_code"] = int(return_code)
    diagnostics["converged"] = bool(converged)
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
    key = (str(name) if name is not None else "almlof").strip().lower()
    if key not in _HESSIAN_MODEL_MAP:
        raise ValueError(
            "unknown ariadne.hessian_model " + repr(name)
            + "; valid: " + repr(sorted(_HESSIAN_MODEL_MAP))
        )
    return _HESSIAN_MODEL_MAP[key]


def _flatten_xyz(values) -> np.ndarray:
    return np.asfortranarray(np.asarray(values, dtype=np.float64).reshape(-1), dtype=np.float64)


def _eval_energy_gradient(atoms):
    """One calculator evaluation. Returns (f_pseudo, g_flat_pseudo).

    Mirrors evaluate_energy_and_gradient from the ARIADNE starter pack.
    The adversarial calculator returns pseudo-energy + pseudo-forces in ASE
    units (eV and eV/Angstrom). Divide by the same pseudo Hartree scale so
    ARIADNE optimises alpha in acquisition units.
    """
    e_ev = float(atoms.get_potential_energy())
    forces_ev = np.asarray(atoms.get_forces(), dtype=np.float64)
    grad_ev = -forces_ev
    e_hartree = e_ev / _HARTREE_EV
    g_xyz_hartree = np.asfortranarray(grad_ev / _HARTREE_EV, dtype=np.float64)
    g_flat_hartree = _flatten_xyz(g_xyz_hartree)
    if not np.isfinite(e_hartree):
        raise RuntimeError(
            "calculator returned non-finite pseudo-energy"
        )
    if not np.isfinite(g_flat_hartree).all():
        raise RuntimeError(
            "calculator returned non-finite entries in the gradient"
        )
    return e_hartree, g_flat_hartree


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
    status = opt.get_status_py()
    converged = bool(status[4])
    accepted = bool(status[5])
    f_current = float(status[1])
    return converged, accepted, f_current


def _status_ds(opt):
    """Pull (converged, last_step_accepted, f_current) off a DS opt.

    DS get_status_py layout has f_current at index 2, converged at
    index 8 and last_step_accepted at index 9.
    """
    status = opt.get_status_py()
    converged = bool(status[8])
    accepted = bool(status[9])
    f_current = float(status[2])
    return converged, accepted, f_current


def _status_tuple(opt):
    try:
        return tuple(opt.get_status_py())
    except Exception:
        return ()


def _optimizer_flag(opt, name: str):
    if not hasattr(opt, name):
        return None
    value = getattr(opt, name)
    try:
        value = value() if callable(value) else value
    except Exception:
        return None
    return value


def _raise_if_init_failed(opt, label: str) -> None:
    init_ok = _optimizer_flag(opt, "init_ok")
    if init_ok is not None and not bool(init_ok):
        reason = _optimizer_flag(opt, "last_config_error_msg")
        if reason is None:
            reason = _optimizer_flag(opt, "init_reason")
        if reason is None:
            reason = _optimizer_flag(opt, "init_message")
        raise RuntimeError(
            "ARIADNE "
            + label
            + " initialisation failed"
            + (": " + str(reason) if reason else "")
        )


def _proposal_pending(opt, status) -> bool:
    value = _optimizer_flag(opt, "proposal_pending")
    if value is not None:
        return bool(value)
    return _status_bool(status, _TRQN_STATUS_PROPOSAL_PENDING, True)


def _skip_step_after_rebuild(opt, status) -> bool:
    value = _optimizer_flag(opt, "skip_step_after_rebuild")
    if value is not None:
        return bool(value)
    return _status_bool(status, _TRQN_STATUS_SKIP_STEP_AFTER_REBUILD, False)


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

_DS_INIT_PROFILE = "daemon_safe_v1"


def _ds_hessian_model_name(value: Any) -> str:
    name = str(value or "almlof").strip().lower()
    if name not in _HESSIAN_MODEL_MAP:
        raise ValueError(f"unsupported DS hessian_model: {value!r}")
    return name


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


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

    kwargs: Dict[str, Any] = {
        "gamma": gamma,
        "h": h,
        "f_tol": float(run_config.f_tol),
        "gradf_tol": float(run_config.gradf_tol),
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


def _set_invalid_trial_reason(opt, reason: str) -> None:
    setter = getattr(opt, "set_invalid_trial_reason_py", None)
    if setter is None:
        return
    try:
        setter(int(_TRIAL_REASON_CODES.get(str(reason), 7)))
    except Exception:
        return


def _push_positions(atoms, q_xyz):
    """Write a (N, 3) Angstrom array back into an ASE Atoms object."""
    atoms.set_positions(np.asarray(q_xyz, dtype=np.float64))


def _build_trqn(ariadne, q0_xyz, g0_xyz, atom_list, run_config):
    """Build and init a trust-region quasi-Newton optimiser.

    Minimal-required kwargs only -- everything else falls back to
    ariadne internal defaults. AriadneRunConfig does not expose
    trust_min so we pin a small floor here (1.0e-4 Angstrom), matching
    the starter pack default.
    """
    opt = ariadne.Geometric_Trqn.trust_region_qn()
    opt.init(
        q0_xyz=q0_xyz,
        g0_xyz=g0_xyz,
        atom_list=atom_list,
        trust0=float(run_config.delta0),
        trust_min=1.0e-4,
        trust_max=float(run_config.delta_max),
        controller_mode=_ARIADNE_TRQN_CONTROLLER_NONE,
        cartesian_recovery_mode=_ARIADNE_CARTESIAN_RECOVERY_NEWTON,
        geo_bt_mode=_ARIADNE_GEO_BT_DENSE,
        rot_primitive_mode=_ARIADNE_ROT_PRIMITIVE_EXPMAP3,
        skip_bfgs_after_rot_reset=False,
        hessian_model=_hessian_model_id(run_config.hessian_model),
    )
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


def _restart_under_ds(ariadne, atoms, natoms, atom_list, run_config):
    # Re-evaluate at the exact geometry DS will restart from. Gradients from a
    # rejected TRQN proposal or a failed backtransform do not belong to this
    # point and should not be re-used.
    _f_here, g_here = _eval_energy_gradient(atoms)
    opt = _build_ds(
        ariadne,
        np.asfortranarray(atoms.get_positions(), dtype=np.float64),
        np.asfortranarray(g_here.reshape(natoms, 3), dtype=np.float64),
        atom_list,
        run_config,
    )
    _raise_if_init_failed(opt, "DS")
    return opt


def run_optimisation_against_calculator(
    seed_atoms,
    calculator,
    run_config,
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
        AriadneRunConfig: which optimiser, max_iter, gradf_tol, f_tol,
        delta0, delta_max, gamma, fallback_to_ds, hessian_model.

    Returns
    -------
    OptimisationResult
    """
    ariadne = _import_ariadne()
    t0 = time.perf_counter()

    atoms = seed_atoms.copy()
    atoms.calc = calculator

    natoms = len(atoms)
    symbols = [a.symbol for a in atoms]
    atom_list = _symbols_to_atom_list(symbols)
    q0_xyz_angstrom = np.asfortranarray(
        atoms.get_positions(), dtype=np.float64,
    )

    # one calculator call before the loop -- ariadne needs an initial
    # energy + gradient to seed its internal hessian model.
    e0_hartree, g0_flat = _eval_energy_gradient(atoms)
    g0_xyz = np.asfortranarray(g0_flat.reshape(natoms, 3), dtype=np.float64)
    n_evaluations = 1

    optimiser_name = (run_config.optimiser or "trust_region_qn").strip().lower()
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
    diagnostics = _new_optimiser_diagnostics(optimiser_name)
    if not is_trqn:
        diagnostics["ds_init_profile"] = _DS_INIT_PROFILE

    _raise_if_init_failed(opt, "DS" if not is_trqn else "TRQN")

    # alpha is what we want to MAXIMISE (the adversarial acquisition
    # value). the calculator exposes a pseudo-energy so ariadne can
    # minimise, so alpha = -f_pseudo.
    alpha_trajectory = [-e0_hartree]
    grad_norm_trajectory = [float(np.linalg.norm(g0_flat))]
    candidate_positions = [np.asarray(q0_xyz_angstrom, dtype=np.float64).copy()]
    candidate_alphas = [float(-e0_hartree)]
    candidate_grad_norms = [float(np.linalg.norm(g0_flat))]
    fell_back_to_ds = False
    rigid_force_clamps = 0
    consecutive_rejects = 0
    no_proposal_failure_reason: Optional[str] = None
    f_current = e0_hartree
    converged = False
    return_code = 1  # max_iter default until we converge or error

    zero_grad = np.zeros(3 * natoms, dtype=np.float64, order="F")

    for step_idx in range(int(run_config.max_iter)):
        f_old = f_current

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
        status_after_stage0 = _status_tuple(opt)
        _append_status_sample(
            diagnostics,
            step_index=step_idx,
            label="after_stage0",
            status=status_after_stage0,
        )
        proposal_pending = _proposal_pending(opt, status_after_stage0)
        skip_after_rebuild = _skip_step_after_rebuild(opt, status_after_stage0)
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
            if is_trqn:
                opt_converged, accepted, f_current = _status_trqn(opt)
            else:
                opt_converged, accepted, f_current = _status_ds(opt)
            alpha_trajectory.append(-float(f_current))
            grad_norm_trajectory.append(float(np.linalg.norm(g_current)))
            if opt_converged:
                converged = True
                return_code = 0
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
                        and bool(run_config.fallback_to_ds)
                        and not fell_back_to_ds
                    ):
                        try:
                            opt = _restart_under_ds(
                                ariadne,
                                atoms,
                                natoms,
                                atom_list,
                                run_config,
                            )
                            n_evaluations += 1
                            is_trqn = False
                            fell_back_to_ds = True
                            diagnostics["n_fallback_to_ds"] += 1
                            diagnostics["n_no_proposal_recoveries"] += 1
                            diagnostics["ds_init_profile"] = _DS_INIT_PROFILE
                            diagnostics["optimiser_final"] = (
                                "dissipative_symplectic"
                            )
                            consecutive_rejects = 0
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
            f_trial, g_trial = _eval_energy_gradient(atoms)
            n_evaluations += 1
            diagnostics["n_trial_evaluations"] += 1
        except Exception as exc:
            _set_invalid_trial_reason(opt, type(exc).__name__ + ": " + str(exc))
            try:
                _q_current, g_current = _sync_state(opt, natoms)
                if not np.isfinite(g_current).all():
                    g_current = zero_grad
                diagnostics["n_stage1_calls"] += 1
                opt.step_py(
                    stage=1,
                    f_old=f_old,
                    f_new=f_old,
                    g_xyz_new=_flatten_xyz(g_current),
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

        q_current, _g_current = _sync_state(opt, natoms)
        _push_positions(atoms, q_current)

        if is_trqn:
            opt_converged, accepted, f_current = _status_trqn(opt)
        else:
            opt_converged, accepted, f_current = _status_ds(opt)

        alpha_trajectory.append(-float(f_current))
        grad_norm_trajectory.append(float(np.linalg.norm(g_trial)))

        if accepted:
            diagnostics["n_accepted_steps"] += 1
            consecutive_rejects = 0
            candidate_positions.append(
                np.asarray(atoms.get_positions(), dtype=np.float64).copy()
            )
            candidate_alphas.append(float(-float(f_current)))
            candidate_grad_norms.append(float(np.linalg.norm(g_trial)))
        else:
            diagnostics["n_rejected_steps"] += 1
            consecutive_rejects += 1

        if opt_converged:
            converged = True
            return_code = 0
            break

        # also honour the daemon-side gradf_tol -- ariadne internal
        # convergence is conservative and the daemon may want tighter.
        if accepted and grad_norm_trajectory[-1] < float(run_config.gradf_tol):
            converged = True
            return_code = 0
            break

        # TRQN -> DS fallback. when too many consecutive proposals
        # get rejected and the config allows it, abandon TRQN and
        # re-init under DS at the current geometry. only one fallback
        # per run.
        if (is_trqn and bool(run_config.fallback_to_ds)
            and not fell_back_to_ds
            and consecutive_rejects >= _REJECT_STREAK_TRIGGER):
            try:
                opt = _restart_under_ds(
                    ariadne,
                    atoms,
                    natoms,
                    atom_list,
                    run_config,
                )
                n_evaluations += 1
                is_trqn = False
                fell_back_to_ds = True
                diagnostics["n_fallback_to_ds"] += 1
                diagnostics["ds_init_profile"] = _DS_INIT_PROFILE
                diagnostics["optimiser_final"] = "dissipative_symplectic"
                consecutive_rejects = 0
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
        diagnostics=diagnostics,
    )
