"""Single-seed ARIADNE driver.

optimise_seed(...) is the per-seed entry point used by the daemon adversarial
attack phase. The live ARIADNE driver (which talks to the oneAPI / MKL
ariadne.so) lands alongside the first water-tetramer end-to-end run;
here I provide a fully deterministic `mock=True` mode so the rest of the
pipeline can be tested without the .so.

The mock path mimics the live path:

* Same input contract (Models, seed, trajectory, optional configs).
* Same output dataclass :class: AriadneRunResult.
* Deterministic for a fixed run_config.rng_seed.

Live-path TODO: when the .so is on PYTHONPATH, _import_ariadne() succeeds
and _live_optimise_seed() runs ariadne.Geometric_Trqn.trust_region_qn()
against :class: AdversarialASECalculator, with a fallback to
ariadne.Ds_Optimiser when BFGS curvature gates reject too many steps in
succession. See docs for the full algorithm spec.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
from ichor.core.adversarial.config import AcquisitionConfig
from ichor.core.atoms import Atom, Atoms

from .ase_calculator import AdversarialASECalculator


__all__ = [
    "AriadneRunConfig",
    "AriadneRunResult",
    "optimise_seed",
]


@dataclass(frozen=True)
class AriadneRunConfig:
    """ARIADNE-side configuration for a single-seed run."""

    optimiser: str = "trust_region_qn"
    hessian_model: str = "SCHLEGEL"
    max_iter: int = 200
    gradf_tol: float = 1.0e-4
    f_tol: float = 1.0e-6
    delta0: float = 0.10
    delta_max: float = 0.40
    gamma: float = 0.10
    fallback_to_ds: bool = True
    trqn_scale_mode: str = "adaptive_initial_gradient_rms"
    trqn_target_initial_grad_norm: float = 0.01
    trqn_retry_target_initial_grad_norm: float = 0.003
    trqn_target_initial_grad_rms: float = 2.0e-4
    trqn_retry_target_initial_grad_rms: float = 4.0e-4
    trqn_under_move_target_initial_grad_rms: float = 6.0e-4
    trqn_under_move_retry: bool = True
    trqn_under_move_retry_max: int = 1
    trqn_min_objective_scale: float = 1.0e-8
    trqn_max_objective_scale: float = 1.0
    trqn_fixed_objective_scale: float = 1.0
    trqn_retry_on_no_proposal: bool = True
    trqn_backtransform_mode: str = "geodesic"
    trqn_geodesic_bt_mode: str = "dense"
    trqn_geodesic_dt: float = 1.0e-2
    trqn_geodesic_tol: float = 1.0e-8
    trqn_bt_ic_tol: float = 1.0e-6
    trqn_max_backtransform_iter: int = 50
    trqn_trust_min: float = 1.0e-4
    rng_seed: int = 0
    mock_perturbation_angstrom: float = 1.0e-4


@dataclass
class AriadneRunResult:
    """Structured outcome of one seed ARIADNE optimisation."""

    initial_atoms: Atoms
    final_atoms: Atoms
    seed_atoms: Optional[Atoms] = None
    seed_alpha: Optional[float] = None
    optimiser_initial_atoms: Optional[Atoms] = None
    optimiser_initial_alpha: Optional[float] = None
    optimiser_initial_origin: Optional[str] = None
    warm_start_alpha_delta_from_seed: Optional[float] = None
    alpha_trajectory: List[float] = field(default_factory=list)
    grad_norm_trajectory: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    return_code: int = 0
    wall_seconds: float = 0.0
    fell_back_to_ds: bool = False
    rigid_force_clamps: int = 0
    mock: bool = False
    # distance from the seed to the final geometry, measured in the
    # frozen subspace metric the acquisition built around the seed.
    # zero extra GP evaluations to compute -- the subspace is already
    # in memory at the end of descent. None for the mock runner since
    # the mock has no real subspace.
    whitened_distance_final: Optional[float] = None
    selected_alpha_final: Optional[float] = None
    raw_final_atoms: Optional[Atoms] = None
    raw_alpha_final: Optional[float] = None
    raw_whitened_distance_final: Optional[float] = None
    landing_safety: Optional[Dict[str, Any]] = None
    landing_candidates: List[Dict[str, Any]] = field(default_factory=list)
    selection_diagnostics: Optional[Dict[str, Any]] = None
    optimiser_diagnostics: Optional[Dict[str, Any]] = None
    trqn_backtransform_mode: Optional[str] = None
    trqn_geodesic_bt_mode: Optional[str] = None
    optimiser_converged: Optional[bool] = None
    task_success: Optional[bool] = None
    task_success_reason: Optional[str] = None
    task_exit_code: Optional[int] = None

    @property
    def alpha_initial(self) -> Optional[float]:
        return self.alpha_trajectory[0] if self.alpha_trajectory else None

    @property
    def alpha_final(self) -> Optional[float]:
        if self.selected_alpha_final is not None:
            return self.selected_alpha_final
        return self.alpha_trajectory[-1] if self.alpha_trajectory else None

    def to_dict(self) -> dict:
        seed_atoms = self.seed_atoms if self.seed_atoms is not None else self.initial_atoms
        optimiser_initial_atoms = (
            self.optimiser_initial_atoms
            if self.optimiser_initial_atoms is not None
            else self.initial_atoms
        )
        optimiser_initial_alpha = (
            self.optimiser_initial_alpha
            if self.optimiser_initial_alpha is not None
            else self.alpha_initial
        )
        seed_alpha = (
            self.seed_alpha
            if self.seed_alpha is not None
            else optimiser_initial_alpha
        )
        warm_start_delta = self.warm_start_alpha_delta_from_seed
        if (
            warm_start_delta is None
            and seed_alpha is not None
            and optimiser_initial_alpha is not None
        ):
            warm_start_delta = float(optimiser_initial_alpha) - float(seed_alpha)
        data = {
            "initial_coordinates": np.asarray(self.initial_atoms.coordinates).tolist(),
            "seed_coordinates": np.asarray(seed_atoms.coordinates).tolist(),
            "seed_alpha": None if seed_alpha is None else float(seed_alpha),
            "optimiser_initial_coordinates": (
                np.asarray(optimiser_initial_atoms.coordinates).tolist()
            ),
            "optimiser_initial_alpha": (
                None if optimiser_initial_alpha is None
                else float(optimiser_initial_alpha)
            ),
            "optimiser_initial_origin": str(
                self.optimiser_initial_origin or "seed_fallback"
            ),
            "warm_start_alpha_delta_from_seed": (
                None if warm_start_delta is None else float(warm_start_delta)
            ),
            "final_coordinates": np.asarray(self.final_atoms.coordinates).tolist(),
            "atom_types": [a.type for a in self.initial_atoms],
            "alpha_trajectory": list(self.alpha_trajectory),
            "alpha_initial": self.alpha_initial,
            "alpha_final": self.alpha_final,
            "grad_norm_trajectory": list(self.grad_norm_trajectory),
            "n_evaluations": int(self.n_evaluations),
            "return_code": int(self.return_code),
            "wall_seconds": float(self.wall_seconds),
            "fell_back_to_ds": bool(self.fell_back_to_ds),
            "rigid_force_clamps": int(self.rigid_force_clamps),
            "mock": bool(self.mock),
            "whitened_distance_final": (
                None if self.whitened_distance_final is None
                else float(self.whitened_distance_final)
            ),
        }
        if self.raw_final_atoms is not None:
            data["raw_final_coordinates"] = (
                np.asarray(self.raw_final_atoms.coordinates).tolist()
            )
        if self.raw_alpha_final is not None:
            data["raw_alpha_final"] = float(self.raw_alpha_final)
        if self.alpha_initial is not None and self.alpha_final is not None:
            selected_delta = float(self.alpha_final) - float(self.alpha_initial)
            data["selected_alpha_delta"] = selected_delta
            data["selected_improves_acquisition"] = bool(
                selected_delta >= -_alpha_improvement_tolerance(
                    float(self.alpha_initial)
                )
            )
        if self.alpha_initial is not None and self.raw_alpha_final is not None:
            data["raw_alpha_delta"] = (
                float(self.raw_alpha_final) - float(self.alpha_initial)
            )
        if self.raw_whitened_distance_final is not None:
            data["raw_whitened_distance_final"] = float(
                self.raw_whitened_distance_final
            )
        if self.landing_safety is not None:
            data["landing_safety"] = dict(self.landing_safety)
        if self.landing_candidates:
            data["landing_candidates"] = [dict(c) for c in self.landing_candidates]
        if self.selection_diagnostics is not None:
            data["selection_diagnostics"] = dict(self.selection_diagnostics)
        if self.optimiser_diagnostics is not None:
            data["optimiser_diagnostics"] = dict(self.optimiser_diagnostics)
        if self.trqn_backtransform_mode is not None:
            data["trqn_backtransform_mode"] = str(self.trqn_backtransform_mode)
        if self.trqn_geodesic_bt_mode is not None:
            data["trqn_geodesic_bt_mode"] = str(self.trqn_geodesic_bt_mode)
        if self.optimiser_converged is not None:
            data["optimiser_converged"] = bool(self.optimiser_converged)
        if self.task_success is not None:
            data["task_success"] = bool(self.task_success)
        if self.task_success_reason is not None:
            data["task_success_reason"] = str(self.task_success_reason)
        if self.task_exit_code is not None:
            data["task_exit_code"] = int(self.task_exit_code)
        return data


def _payload_final_coordinates(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        coords = np.asarray(payload.get("final_coordinates"), dtype=float)
    except (TypeError, ValueError):
        return None
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 1:
        return None
    if not np.all(np.isfinite(coords)):
        return None
    return coords


def _optimiser_terminal_reason(payload: Dict[str, Any]) -> str:
    diag = payload.get("optimiser_diagnostics")
    if not isinstance(diag, dict):
        return ""
    for key in (
        "last_return_code_reason",
        "terminal_reason",
        "trqn_no_proposal_reason",
    ):
        value = diag.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _is_salvageable_ariadne_return_code(
    payload: Dict[str, Any],
    return_code: int,
) -> bool:
    """Return whether a non-converged optimiser code may still hand off.

    Code 2 is used by the live TRQN path when it can no longer build a useful
    proposal, including backtransform failure after an already accepted step.
    The selected landing still has to pass the normal finite-geometry and
    landing-safety checks below; this helper only prevents an early hard fail.
    """
    if int(return_code) != 2:
        return False
    reason = _optimiser_terminal_reason(payload).lower()
    return (
        reason.startswith("trqn_no_proposal")
        or reason == "backtransform_fail"
        or reason.startswith("backtransform_fail")
    )


def ariadne_result_usability_payload(
    payload: Dict[str, Any],
    *,
    allow_seed_fallback: bool = False,
) -> Dict[str, Any]:
    """Decide whether a per-seed ARIADNE result is usable by Phase B.

    Optimiser convergence is useful diagnostic information, but the active
    learning hand-off contract is a finite, safety-accepted landing. A
    max-iteration run that selected a safe landing is therefore usable even
    though the optimiser did not prove convergence.
    """
    if not isinstance(payload, dict):
        return {
            "usable": False,
            "reason": "result_payload_not_mapping",
            "task_exit_code": 4,
            "optimiser_converged": False,
        }
    try:
        return_code = int(payload.get("return_code"))
    except (TypeError, ValueError):
        return {
            "usable": False,
            "reason": "return_code_invalid",
            "task_exit_code": 4,
            "optimiser_converged": False,
        }
    optimiser_converged = bool(return_code == 0)
    salvageable_optimiser_failure = _is_salvageable_ariadne_return_code(
        payload,
        return_code,
    )
    if return_code not in (0, 1, 2):
        return {
            "usable": False,
            "reason": "ariadne_return_code_" + str(return_code),
            "task_exit_code": 4,
            "optimiser_converged": optimiser_converged,
        }
    if _payload_final_coordinates(payload) is None:
        return {
            "usable": False,
            "reason": "nonfinite_final_geometry",
            "task_exit_code": 4,
            "optimiser_converged": optimiser_converged,
        }
    safety = payload.get("landing_safety")
    if not isinstance(safety, dict):
        if return_code == 0:
            return {
                "usable": True,
                "reason": "safe_landing_converged_legacy",
                "task_exit_code": 0,
                "optimiser_converged": True,
            }
        return {
            "usable": False,
            "reason": "missing_landing_safety",
            "task_exit_code": 4,
            "optimiser_converged": optimiser_converged,
        }
    if not bool(safety.get("accepted", False)):
        reasons = safety.get("reasons") or ["landing_safety_rejected"]
        reason = ";".join(str(r) for r in reasons)
        return {
            "usable": False,
            "reason": reason,
            "task_exit_code": 4,
            "optimiser_converged": optimiser_converged,
        }
    policy = str(safety.get("policy", "unknown"))
    selected_origin = str(safety.get("selected_origin", ""))
    if (
        not allow_seed_fallback
        and (policy == "seed_fallback" or selected_origin == "seed_fallback")
    ):
        return {
            "usable": False,
            "reason": "seed_fallback_not_allowed",
            "task_exit_code": 4,
            "optimiser_converged": optimiser_converged,
        }
    if return_code == 0:
        reason = "safe_landing_converged"
    elif salvageable_optimiser_failure:
        return {
            "usable": True,
            "reason": "safe_landing_after_backtransform_failure",
            "task_exit_code": 0,
            "optimiser_converged": False,
            "safe_landing_salvaged_after_optimiser_failure": True,
        }
    elif return_code == 2:
        return {
            "usable": False,
            "reason": "ariadne_return_code_2",
            "task_exit_code": 4,
            "optimiser_converged": False,
        }
    else:
        opt_reason = _optimiser_terminal_reason(payload)
        reason = (
            "safe_landing_after_max_iterations"
            if opt_reason.startswith("max_iterations") or not opt_reason
            else "safe_landing_nonconverged"
        )
    return {
        "usable": True,
        "reason": reason,
        "task_exit_code": 0,
        "optimiser_converged": optimiser_converged,
    }


def ariadne_result_usability(
    result: AriadneRunResult,
    *,
    allow_seed_fallback: bool = False,
) -> Dict[str, Any]:
    return ariadne_result_usability_payload(
        result.to_dict(),
        allow_seed_fallback=allow_seed_fallback,
    )


def _import_ariadne():
    """Lazy import of the ARIADNE Python wrapper."""
    try:
        import ariadne  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "ARIADNE Python module not importable. Build the oneAPI .so and "
            "prepend build-oneapi/python to PYTHONPATH."
        ) from exc
    return ariadne


def optimise_seed(
    models,
    seed: Atoms,
    trajectory: Sequence[Atoms],
    *,
    acquisition_config: Optional[AcquisitionConfig] = None,
    run_config: Optional[AriadneRunConfig] = None,
    mock: bool = False,
    seed_frame_id: Optional[int] = None,
    external_reference_scales: Optional[dict] = None,
    error_calibration_model: Optional[dict] = None,
    error_calibration_apply_strength: float = 0.0,
    max_acquisition_grad_per_ang: Optional[float] = None,
    max_force_per_atom_ha_per_ang: Optional[float] = 50.0,
    project_rigid: bool = True,
    gradient_backend: str = "process",
    safety_config: Optional[Any] = None,
    quality_gates: Optional[Any] = None,
    trace_path: Optional[Any] = None,
) -> AriadneRunResult:
    """Drive the adversarial descent for a single seed.

    Two paths:
      * mock=True -- a deterministic synthetic trajectory used by
        plumbing tests. lives entirely in Python; no oneAPI needed.
      * mock=False -- the real path. builds a posterior over the
        trained Models, runs ARIADNE via the local runner module.
        Needs the oneAPI .so on PYTHONPATH.

    The extra kwargs (seed_frame_id, external_reference_scales,
    max_acquisition_grad_per_ang, project_rigid) are passed through
    to _live_optimise_seed when mock=False. They are ignored in the
    mock path -- the mock has no real posterior or subspace to use
    them with.
    """
    run_config = run_config or AriadneRunConfig()
    acquisition_config = acquisition_config or AcquisitionConfig()
    if mock:
        return _mock_optimise_seed(seed, trajectory, run_config)
    return _live_optimise_seed(
        models, seed, trajectory, acquisition_config, run_config,
        seed_frame_id=seed_frame_id,
        external_reference_scales=external_reference_scales,
        error_calibration_model=error_calibration_model,
        error_calibration_apply_strength=error_calibration_apply_strength,
        max_acquisition_grad_per_ang=max_acquisition_grad_per_ang,
        max_force_per_atom_ha_per_ang=max_force_per_atom_ha_per_ang,
        project_rigid=project_rigid,
        gradient_backend=gradient_backend,
        safety_config=safety_config,
        quality_gates=quality_gates,
        trace_path=trace_path,
    )


def _copy_atoms_with_coords(template: Atoms, coords: np.ndarray) -> Atoms:
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(coords) != len(template):
        raise ValueError(f"coord rows {len(coords)} != natoms {len(template)}")
    return Atoms([
        Atom(template[i].type, float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        for i in range(len(template))
    ])


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    return default if config is None else getattr(config, name, default)


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _alpha_improvement_tolerance(alpha_initial: float) -> float:
    return max(1.0e-10, 1.0e-8 * max(abs(float(alpha_initial)), 1.0))


def _annotate_acquisition_improvement(
    candidate: Dict[str, Any],
    alpha_initial: Optional[float],
) -> None:
    metrics = dict(candidate.get("metrics") or {})
    candidate["metrics"] = metrics
    metrics["alpha_initial"] = (
        None if alpha_initial is None else float(alpha_initial)
    )
    alpha = _safe_float_or_none(candidate.get("alpha"))
    if alpha_initial is None:
        metrics["alpha_delta_from_initial"] = None
        metrics["improves_acquisition"] = None
        return
    if alpha is None:
        metrics["alpha_delta_from_initial"] = None
        metrics["improves_acquisition"] = False
        reasons = candidate.setdefault("reasons", [])
        if "ariadne_landing_acquisition_evaluation_failed" not in reasons:
            reasons.append("ariadne_landing_acquisition_not_finite")
        candidate["accepted"] = False
        return
    delta = float(alpha) - float(alpha_initial)
    tolerance = _alpha_improvement_tolerance(float(alpha_initial))
    improves = delta >= -tolerance
    metrics["alpha_delta_from_initial"] = float(delta)
    metrics["alpha_improvement_tolerance"] = float(tolerance)
    metrics["improves_acquisition"] = bool(improves)
    if not improves:
        reasons = candidate.setdefault("reasons", [])
        reasons.append("ariadne_landing_acquisition_not_improved")
        candidate["accepted"] = False


def _coords_array(atoms: Atoms) -> np.ndarray:
    return np.asarray(atoms.coordinates, dtype=float).reshape(-1, 3)


def _geometry_metrics(seed_coords: np.ndarray, coords: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "max_displacement_ang": None,
        "min_pair_distance_ang": None,
    }
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        return metrics
    if seed_coords.shape == coords.shape and np.all(np.isfinite(seed_coords)):
        disp = np.linalg.norm(coords - seed_coords, axis=1)
        metrics["max_displacement_ang"] = float(np.max(disp)) if disp.size else 0.0
    if coords.shape[0] >= 2:
        dmin = None
        for i in range(coords.shape[0]):
            for j in range(i + 1, coords.shape[0]):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                dmin = d if dmin is None else min(dmin, d)
        metrics["min_pair_distance_ang"] = dmin
    return metrics


def _candidate_public(candidate: Dict[str, Any]) -> Dict[str, Any]:
    public = {
        "candidate_index": int(candidate.get("candidate_index", -1)),
        "origin": str(candidate.get("origin", "unknown")),
        "accepted": bool(candidate.get("accepted", False)),
        "reasons": list(candidate.get("reasons") or []),
        "record_only_reasons": list(candidate.get("record_only_reasons") or []),
        "metrics": dict(candidate.get("metrics") or {}),
        "alpha": _safe_float_or_none(candidate.get("alpha")),
        "grad_norm": _safe_float_or_none(candidate.get("grad_norm")),
        "informativeness_score": _safe_float_or_none(
            candidate.get("informativeness_score")
        ),
        "risk_penalty_score": _safe_float_or_none(candidate.get("risk_penalty_score")),
    }
    return public


def _evaluate_landing_candidate(
    *,
    acquisition: SeedLocalAdversarialAcquisition,
    seed_atoms: Atoms,
    seed_mean_energy: float,
    coords: np.ndarray,
    origin: str,
    candidate_index: int,
    alpha: Optional[float],
    grad_norm: Optional[float],
    safety_config: Any,
    quality_gates: Any,
) -> Dict[str, Any]:
    reasons: List[str] = []
    record_only: List[str] = []
    seed_coords = _coords_array(seed_atoms)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    metrics = _geometry_metrics(seed_coords, coords)
    atoms = _copy_atoms_with_coords(seed_atoms, coords)
    try:
        from ichor.core.adversarial.geometry import aligned_mass_weighted_rmsd

        metrics["aligned_mass_weighted_rmsd_ang"] = float(
            aligned_mass_weighted_rmsd(seed_atoms, atoms)
        )
    except Exception:
        metrics["aligned_mass_weighted_rmsd_ang"] = None

    if coords.shape != seed_coords.shape or not np.all(np.isfinite(coords)):
        reasons.append("ariadne_landing_geometry_nonfinite")
    seed_equivalent = _is_seed_equivalent(seed_coords, coords)
    metrics["seed_equivalent"] = bool(seed_equivalent)
    if seed_equivalent:
        if bool(_cfg_value(safety_config, "allow_seed_fallback", False)):
            record_only.append("ariadne_landing_is_seed")
        else:
            reasons.append("ariadne_landing_is_seed")
            reasons.append("seed_fallback_disabled")

    max_disp = _safe_float_or_none(
        _cfg_value(quality_gates, "ariadne_max_displacement_ang", None)
    )
    if (
        max_disp is not None
        and metrics.get("max_displacement_ang") is not None
        and float(metrics["max_displacement_ang"]) > max_disp
    ):
        reasons.append("ariadne_max_displacement_threshold_exceeded")
    min_pair = _safe_float_or_none(
        _cfg_value(quality_gates, "ariadne_min_pair_distance_ang", None)
    )
    if (
        min_pair is not None
        and metrics.get("min_pair_distance_ang") is not None
        and float(metrics["min_pair_distance_ang"]) < min_pair
    ):
        reasons.append("ariadne_min_pair_distance_threshold_exceeded")

    try:
        from ichor.core.adversarial.subspace import whitened_distance_squared

        d_sq = whitened_distance_squared(
            acquisition.subspace,
            atoms,
            acquisition.config.subspace.covariance_regularization,
        )
        metrics["whitened_distance"] = float(math.sqrt(max(0.0, float(d_sq))))
    except Exception:
        metrics["whitened_distance"] = None

    try:
        breakdown = acquisition.components(atoms)
        metrics["mean_energy_ha"] = float(breakdown.mean_energy)
        metrics["predicted_energy_delta_ha"] = float(
            breakdown.mean_energy - seed_mean_energy
        )
        metrics["energy_variance"] = float(breakdown.energy_variance)
        metrics["chemistry_penalty"] = float(breakdown.chemistry_penalty)
        metrics["distance_penalty"] = float(breakdown.distance_penalty)
        metrics["spectral_frequency_risk"] = (
            None if breakdown.spectral_frequency_risk is None
            else float(breakdown.spectral_frequency_risk)
        )
        metrics["legacy_frequency_risk"] = (
            None if breakdown.legacy_frequency_risk is None
            else float(breakdown.legacy_frequency_risk)
        )
        metrics["banded_energy_risk"] = (
            None if breakdown.banded_energy_risk is None
            else float(breakdown.banded_energy_risk)
        )
        metrics["fullspace_residual_distance"] = (
            None if breakdown.fullspace_residual_distance is None
            else float(breakdown.fullspace_residual_distance)
        )
        metrics["fullspace_residual_penalty"] = float(
            breakdown.fullspace_residual_penalty
        )
        metrics["aligned_rmsd_penalty"] = float(breakdown.aligned_rmsd_penalty)
        if breakdown.aligned_rmsd_ang is not None:
            metrics["aligned_rmsd_ang"] = float(breakdown.aligned_rmsd_ang)
        for key in (
            "movement_metric",
            "movement_rmsd_ang",
            "movement_progress_ang",
            "movement_band_min_ang",
            "movement_band_low_ang",
            "movement_band_peak_ang",
            "movement_band_high_ang",
            "movement_band_max_ang",
            "movement_utility_score",
            "movement_band_score",
            "movement_progress_score",
            "movement_direction_source",
            "n_effective_movement_atoms",
        ):
            value = getattr(breakdown, key, None)
            metrics[key] = None if value is None else (
                str(value) if key in ("movement_metric", "movement_direction_source")
                else float(value)
            )
        metrics["weak_mode_penalty_score"] = float(
            getattr(breakdown, "weak_mode_penalty_score", 0.0)
        )
        metrics["anharmonic_risk_raw"] = float(
            getattr(breakdown, "anharmonic_risk_raw", breakdown.anharmonic_risk)
        )
        metrics["anharmonic_risk_capped"] = float(
            getattr(breakdown, "anharmonic_risk_capped", breakdown.anharmonic_risk)
        )
        metrics["observable_score"] = (
            None if breakdown.observable_score is None
            else float(breakdown.observable_score)
        )
        metrics["outlier_penalty_score"] = (
            None if breakdown.outlier_penalty_score is None
            else float(breakdown.outlier_penalty_score)
        )
        metrics["acquisition_fallback_reasons"] = list(breakdown.fallback_reasons)
        metrics["total_score"] = float(breakdown.total)
        alpha_value = float(breakdown.total if alpha is None else alpha)
        informativeness = float(breakdown.informativeness_score)
        risk_penalty = float(breakdown.risk_penalty_score)
    except Exception as exc:
        reasons.append("ariadne_landing_acquisition_evaluation_failed")
        metrics["evaluation_error"] = type(exc).__name__ + ": " + str(exc)
        alpha_value = None
        informativeness = None
        risk_penalty = None

    min_w = float(_cfg_value(safety_config, "min_whitened_distance", 0.0))
    max_w = float(_cfg_value(safety_config, "max_whitened_distance", 10.0))
    enforce_min_w = bool(
        _cfg_value(safety_config, "enforce_min_whitened_distance", False)
    )
    d_w = metrics.get("whitened_distance")
    if d_w is not None:
        if float(d_w) < min_w:
            if enforce_min_w:
                reasons.append("ariadne_landing_below_min_whitened_distance")
            else:
                record_only.append("ariadne_landing_below_min_whitened_distance")
        if float(d_w) > max_w:
            reasons.append("ariadne_landing_above_max_whitened_distance")

    if bool(_cfg_value(safety_config, "enforce_movement_band", True)):
        r_move = _safe_float_or_none(metrics.get("movement_rmsd_ang"))
        r_min = _safe_float_or_none(metrics.get("movement_band_min_ang"))
        r_max = _safe_float_or_none(metrics.get("movement_band_max_ang"))
        if r_move is None or r_min is None or r_max is None:
            reasons.append("ariadne_landing_movement_band_unavailable")
        else:
            if float(r_move) < float(r_min):
                reasons.append("ariadne_landing_under_moved")
            if (
                bool(_cfg_value(safety_config, "reject_over_moved", True))
                and float(r_move) > float(r_max)
            ):
                reasons.append("ariadne_landing_over_moved")

    max_de = _safe_float_or_none(
        _cfg_value(safety_config, "max_predicted_energy_delta_ha", None)
    )
    if (
        max_de is not None
        and metrics.get("predicted_energy_delta_ha") is not None
        and float(metrics["predicted_energy_delta_ha"]) > max_de
    ):
        reasons.append("ariadne_landing_predicted_energy_delta_threshold_exceeded")

    max_var = _safe_float_or_none(
        _cfg_value(safety_config, "max_energy_variance", None)
    )
    if (
        max_var is not None
        and metrics.get("energy_variance") is not None
        and float(metrics["energy_variance"]) > max_var
    ):
        reasons.append("ariadne_landing_energy_variance_threshold_exceeded")

    max_chem = _safe_float_or_none(
        _cfg_value(safety_config, "max_chemistry_penalty", None)
    )
    if (
        max_chem is not None
        and metrics.get("chemistry_penalty") is not None
        and float(metrics["chemistry_penalty"]) > max_chem
    ):
        reasons.append("ariadne_landing_chemistry_penalty_threshold_exceeded")

    return {
        "candidate_index": int(candidate_index),
        "origin": str(origin),
        "atoms": atoms,
        "alpha": alpha_value,
        "grad_norm": grad_norm,
        "informativeness_score": informativeness,
        "risk_penalty_score": risk_penalty,
        "accepted": not reasons,
        "reasons": reasons,
        "record_only_reasons": record_only,
        "metrics": metrics,
    }


def _rank_landing_candidate(candidate: Dict[str, Any]) -> float:
    value = _safe_float_or_none(candidate.get("alpha"))
    if value is None:
        value = _safe_float_or_none(
            candidate.get("metrics", {}).get("total_score")
        )
    if value is None:
        return -math.inf
    metrics = candidate.get("metrics", {}) or {}
    r = _safe_float_or_none(metrics.get("movement_rmsd_ang"))
    peak = _safe_float_or_none(metrics.get("movement_band_peak_ang"))
    low = _safe_float_or_none(metrics.get("movement_band_low_ang"))
    high = _safe_float_or_none(metrics.get("movement_band_high_ang"))
    if r is None or peak is None or low is None or high is None:
        return float(value)
    width = max(float(high) - float(low), 1.0e-12)
    tie = 0.05 * ((float(r) - float(peak)) / width) ** 2
    return float(value) - float(tie)


def _selection_prediction_diagnostics(
    acquisition: SeedLocalAdversarialAcquisition,
    atoms: Atoms,
    *,
    landing_safety: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    breakdown = acquisition.components(atoms)
    per_atom = []
    for atom, diag in acquisition.posterior.atom_diagnostics(atoms).items():
        per_atom.append(
            {
                "atom": str(atom),
                "atom_type": str(diag.get("atom_type", "")),
                "property": str(acquisition.config.property_name),
                "predicted_iqa_ha": float(diag["predicted_iqa_ha"]),
                "raw_variance": float(diag["raw_variance"]),
            }
        )
    safety_metrics = {}
    if isinstance(landing_safety, dict):
        safety_metrics = dict(landing_safety.get("metrics") or {})
    movement_fields = {
        key: safety_metrics.get(key)
        for key in (
            "movement_rmsd_ang",
            "movement_band_min_ang",
            "movement_band_peak_ang",
            "movement_band_max_ang",
            "movement_utility_score",
            "movement_progress_score",
            "movement_direction_source",
        )
        if key in safety_metrics
    }
    return {
        "schema_version": 1,
        "property": str(acquisition.config.property_name),
        "total_predicted_iqa_ha": float(breakdown.mean_energy),
        "total_energy_variance": float(breakdown.energy_variance),
        "raw_total_score": float(breakdown.total),
        "raw_energy_risk": (
            float(breakdown.raw_energy_risk)
            if breakdown.raw_energy_risk is not None
            else float(breakdown.energy_risk)
        ),
        "energy_risk": float(breakdown.energy_risk),
        "banded_energy_risk": (
            None
            if breakdown.banded_energy_risk is None
            else float(breakdown.banded_energy_risk)
        ),
        "calibrated_expected_iqa_error_ha": (
            None
            if breakdown.calibrated_expected_iqa_error_ha is None
            else float(breakdown.calibrated_expected_iqa_error_ha)
        ),
        "calibration_applied": bool(breakdown.calibration_applied),
        "spectral_frequency_risk": (
            None
            if breakdown.spectral_frequency_risk is None
            else float(breakdown.spectral_frequency_risk)
        ),
        "legacy_frequency_risk": (
            None
            if breakdown.legacy_frequency_risk is None
            else float(breakdown.legacy_frequency_risk)
        ),
        "fullspace_residual_distance": (
            None
            if breakdown.fullspace_residual_distance is None
            else float(breakdown.fullspace_residual_distance)
        ),
        "fullspace_residual_penalty": float(breakdown.fullspace_residual_penalty),
        "aligned_rmsd_ang": (
            None
            if breakdown.aligned_rmsd_ang is None
            else float(breakdown.aligned_rmsd_ang)
        ),
        "aligned_rmsd_penalty": float(breakdown.aligned_rmsd_penalty),
        "observable_score": (
            None
            if breakdown.observable_score is None
            else float(breakdown.observable_score)
        ),
        "outlier_penalty_score": (
            None
            if breakdown.outlier_penalty_score is None
            else float(breakdown.outlier_penalty_score)
        ),
        "acquisition_fallback_reasons": list(breakdown.fallback_reasons),
        "spectral_modes": [
            {
                "index": int(mode.index),
                "omega": float(mode.omega),
                "omega_std": float(mode.omega_std),
                "spectral_weight": float(mode.spectral_weight),
                "frequency_observable_score": float(mode.frequency_observable_score),
            }
            for mode in breakdown.mode_evaluations
        ],
        "landing_policy": (
            str(landing_safety.get("policy", "unknown"))
            if isinstance(landing_safety, dict)
            else "unknown"
        ),
        **movement_fields,
        "safety_metrics": safety_metrics,
        "per_atom": per_atom,
    }


def _duplicate_coords(coords: np.ndarray, existing: Sequence[np.ndarray]) -> bool:
    return any(np.allclose(coords, prev, atol=1.0e-12, rtol=1.0e-12) for prev in existing)


def _is_seed_equivalent(seed_coords: np.ndarray, coords: np.ndarray) -> bool:
    coords = np.asarray(coords, dtype=float)
    seed_coords = np.asarray(seed_coords, dtype=float)
    return (
        coords.shape == seed_coords.shape
        and np.all(np.isfinite(coords))
        and np.all(np.isfinite(seed_coords))
        and np.allclose(coords, seed_coords, atol=1.0e-12, rtol=1.0e-12)
    )


def _select_safe_landing(
    *,
    acquisition: SeedLocalAdversarialAcquisition,
    seed_atoms: Atoms,
    raw_final_atoms: Atoms,
    opt_candidate_positions: Sequence[np.ndarray],
    opt_candidate_alphas: Sequence[float],
    opt_candidate_grad_norms: Sequence[float],
    opt_candidate_origins: Optional[Sequence[str]] = None,
    alpha_trajectory: Sequence[float],
    safety_config: Any,
    quality_gates: Any,
) -> Dict[str, Any]:
    if not bool(_cfg_value(safety_config, "enabled", True)):
        return {
            "selected_atoms": raw_final_atoms,
            "selected_alpha": (
                float(alpha_trajectory[-1]) if alpha_trajectory else None
            ),
            "selected_whitened_distance": None,
            "raw_whitened_distance": None,
            "landing_safety": {
                "accepted": True,
                "policy": "disabled",
                "selected_origin": "raw_final",
                "selected_candidate_index": 0,
                "reasons": [],
                "record_only_reasons": ["adversarial_safety_disabled"],
                "metrics": {},
                "raw_final": {},
                "n_candidates_evaluated": 0,
                "n_safe_candidates": 0,
            },
            "landing_candidates": [],
        }

    seed_mean_energy = float(acquisition.posterior.mean(seed_atoms))
    initial_alpha = (
        _safe_float_or_none(alpha_trajectory[0]) if alpha_trajectory else None
    )
    raw_coords = _coords_array(raw_final_atoms)
    candidates: List[Dict[str, Any]] = []
    seen_coords: List[np.ndarray] = []
    idx = 0

    salvage = bool(_cfg_value(safety_config, "salvage_safe_iterate", True))
    for k, coords in enumerate(opt_candidate_positions):
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        if k > 0 and not salvage:
            seen_coords.append(coords.copy())
            continue
        if _duplicate_coords(coords, seen_coords):
            continue
        seen_coords.append(coords.copy())
        origin = (
            str(opt_candidate_origins[k])
            if opt_candidate_origins is not None and k < len(opt_candidate_origins)
            else ("seed_fallback" if k == 0 else "accepted_iterate")
        )
        candidate = _evaluate_landing_candidate(
            acquisition=acquisition,
            seed_atoms=seed_atoms,
            seed_mean_energy=seed_mean_energy,
            coords=coords,
            origin=origin,
            candidate_index=idx,
            alpha=(
                float(opt_candidate_alphas[k])
                if k < len(opt_candidate_alphas) else None
            ),
            grad_norm=(
                float(opt_candidate_grad_norms[k])
                if k < len(opt_candidate_grad_norms) else None
            ),
            safety_config=safety_config,
            quality_gates=quality_gates,
        )
        _annotate_acquisition_improvement(candidate, initial_alpha)
        candidates.append(candidate)
        idx += 1

    raw_candidate = _evaluate_landing_candidate(
        acquisition=acquisition,
        seed_atoms=seed_atoms,
        seed_mean_energy=seed_mean_energy,
        coords=raw_coords,
        origin="raw_final",
        candidate_index=idx,
        alpha=(float(alpha_trajectory[-1]) if alpha_trajectory else None),
        grad_norm=None,
        safety_config=safety_config,
        quality_gates=quality_gates,
    )
    _annotate_acquisition_improvement(raw_candidate, initial_alpha)
    idx += 1
    candidates.append(raw_candidate)
    if not _duplicate_coords(raw_coords, seen_coords):
        seen_coords.append(raw_coords.copy())

    if (
        not bool(raw_candidate.get("accepted"))
        and bool(_cfg_value(safety_config, "backtrack_to_safe_landing", True))
    ):
        seed_coords = _coords_array(seed_atoms)
        n_backtrack = int(_cfg_value(safety_config, "backtrack_points", 16))
        for frac in np.linspace(1.0, 0.0, n_backtrack + 2, dtype=float)[1:-1]:
            coords = seed_coords + float(frac) * (raw_coords - seed_coords)
            if _duplicate_coords(coords, seen_coords):
                continue
            seen_coords.append(coords.copy())
            candidate = _evaluate_landing_candidate(
                acquisition=acquisition,
                seed_atoms=seed_atoms,
                seed_mean_energy=seed_mean_energy,
                coords=coords,
                origin="backtrack",
                candidate_index=idx,
                alpha=None,
                grad_norm=None,
                safety_config=safety_config,
                quality_gates=quality_gates,
            )
            _annotate_acquisition_improvement(candidate, initial_alpha)
            candidates.append(candidate)
            idx += 1

    safe_candidates = [c for c in candidates if bool(c.get("accepted"))]
    if safe_candidates:
        selected = max(
            safe_candidates,
            key=_rank_landing_candidate,
        )
        selected_origin = str(selected.get("origin", "unknown"))
        if bool(selected.get("metrics", {}).get("seed_equivalent", False)):
            policy = "seed_fallback"
            selected_origin = "seed_fallback"
        elif selected_origin == "raw_final":
            policy = "raw_final"
        elif selected_origin == "backtrack":
            policy = "backtracked"
        elif selected_origin == "gradient_band_warm_start":
            policy = "gradient_band_warm_start"
        elif selected_origin == "seed_fallback":
            policy = "seed_fallback"
        else:
            policy = "salvaged_iterate"
        accepted = True
        reasons: List[str] = []
    else:
        selected = raw_candidate
        selected_origin = "raw_final"
        policy = "unsafe_raw_final"
        accepted = not bool(_cfg_value(safety_config, "reject_unsafe_landings", True))
        reasons = ["no_safe_non_seed_landing"]
        if accepted:
            policy = "unsafe_raw_final_record_only"

    metrics = dict(selected.get("metrics") or {})
    selected_alpha = _safe_float_or_none(selected.get("alpha"))
    metrics.setdefault(
        "alpha_initial",
        None if initial_alpha is None else float(initial_alpha),
    )
    metrics.setdefault(
        "selected_alpha_delta",
        None if selected_alpha is None or initial_alpha is None
        else float(selected_alpha) - float(initial_alpha),
    )
    metrics.setdefault(
        "selected_improves_acquisition",
        None if selected_alpha is None or initial_alpha is None
        else bool(
            float(selected_alpha) - float(initial_alpha)
            >= -_alpha_improvement_tolerance(float(initial_alpha))
        ),
    )
    selected_w = _safe_float_or_none(metrics.get("whitened_distance"))
    landing_safety = {
        "accepted": bool(accepted),
        "policy": policy,
        "selected_origin": selected_origin,
        "selected_candidate_index": int(selected.get("candidate_index", -1)),
        "reasons": reasons + list(selected.get("reasons") or []),
        "record_only_reasons": list(selected.get("record_only_reasons") or []),
        "metrics": metrics,
        "raw_final": _candidate_public(raw_candidate),
        "n_candidates_evaluated": int(len(candidates)),
        "n_safe_candidates": int(len(safe_candidates)),
    }
    return {
        "selected_atoms": selected["atoms"],
        "selected_alpha": _safe_float_or_none(selected.get("alpha")),
        "selected_whitened_distance": selected_w,
        "raw_whitened_distance": _safe_float_or_none(
            raw_candidate.get("metrics", {}).get("whitened_distance")
        ),
        "landing_safety": landing_safety,
        "landing_candidates": [_candidate_public(c) for c in candidates],
    }


def _mock_landing_safety(seed: Atoms, final: Atoms) -> Dict[str, Any]:
    seed_coords = _coords_array(seed)
    final_coords = _coords_array(final)
    metrics = _geometry_metrics(seed_coords, final_coords)
    metrics["whitened_distance"] = 0.0
    metrics["spectral_frequency_risk"] = 0.0
    metrics["legacy_frequency_risk"] = 0.0
    metrics["banded_energy_risk"] = None
    metrics["fullspace_residual_distance"] = 0.0
    metrics["fullspace_residual_penalty"] = 0.0
    metrics["aligned_rmsd_penalty"] = 0.0
    move_rmsd = _safe_float_or_none(metrics.get("aligned_mass_weighted_rmsd_ang"))
    if move_rmsd is None:
        move_rmsd = 0.05
    metrics["movement_metric"] = "synthetic_mock_aligned_global_rmsd"
    metrics["movement_rmsd_ang"] = float(move_rmsd)
    metrics["movement_progress_ang"] = float(move_rmsd)
    metrics["movement_band_min_ang"] = 0.0
    metrics["movement_band_low_ang"] = 0.0
    metrics["movement_band_peak_ang"] = float(move_rmsd)
    metrics["movement_band_high_ang"] = max(float(move_rmsd), 0.1)
    metrics["movement_band_max_ang"] = max(float(move_rmsd), 0.2)
    metrics["movement_utility_score"] = 0.0
    metrics["movement_band_score"] = 1.0
    metrics["movement_progress_score"] = 1.0
    metrics["movement_direction_source"] = "synthetic_mock"
    metrics["n_effective_movement_atoms"] = float(len(seed))
    metrics["observable_score"] = 0.0
    metrics["outlier_penalty_score"] = 0.0
    metrics["acquisition_fallback_reasons"] = ["synthetic_mock_acquisition_metrics"]
    candidate = {
        "candidate_index": 0,
        "origin": "mock_final",
        "accepted": True,
        "reasons": [],
        "record_only_reasons": ["synthetic_mock_safety_metrics"],
        "metrics": metrics,
        "alpha": None,
        "grad_norm": None,
        "informativeness_score": None,
        "risk_penalty_score": None,
    }
    return {
        "accepted": True,
        "policy": "mock_final",
        "selected_origin": "mock_final",
        "selected_candidate_index": 0,
        "reasons": [],
        "record_only_reasons": ["synthetic_mock_safety_metrics"],
        "metrics": metrics,
        "raw_final": dict(candidate),
        "n_candidates_evaluated": 1,
        "n_safe_candidates": 1,
    }


def _mock_optimise_seed(
    seed: Atoms,
    trajectory: Sequence[Atoms],
    run_config: AriadneRunConfig,
) -> AriadneRunResult:
    """Deterministic synthetic ARIADNE result for plumbing tests."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(run_config.rng_seed)
    n_iters = max(2, min(int(run_config.max_iter), 12))

    alpha_values = sorted(float(v) for v in rng.uniform(0.5, 2.0, size=n_iters))
    grad_values = sorted(
        (float(v) for v in rng.uniform(1.0e-3, 1.0e-1, size=n_iters)),
        reverse=True,
    )

    seed_coords = np.asarray(seed.coordinates, dtype=float)
    perturbation = rng.normal(0.0, float(run_config.mock_perturbation_angstrom), size=seed_coords.shape)
    final = _copy_atoms_with_coords(seed, seed_coords + perturbation)

    wall = max(time.perf_counter() - t0, 1.0e-6)
    landing_safety = _mock_landing_safety(seed, final)
    landing_candidates = [dict(landing_safety["raw_final"])]
    per_atom = []
    for idx, atom in enumerate(final):
        per_atom.append({
            "atom": str(atom.name),
            "atom_type": str(atom.type),
            "property": "iqa",
            "predicted_iqa_ha": float(-1.0 - 1.0e-4 * idx),
            "raw_variance": float(1.0e-3 + 1.0e-4 * idx),
        })
    selection_diagnostics = {
        "schema_version": 1,
        "property": "iqa",
        "seed_alpha": float(alpha_values[0]),
        "optimiser_initial_alpha": float(alpha_values[0]),
        "optimiser_initial_origin": "seed_fallback",
        "warm_start_alpha_delta_from_seed": 0.0,
        "total_predicted_iqa_ha": float(sum(r["predicted_iqa_ha"] for r in per_atom)),
        "total_energy_variance": float(sum(r["raw_variance"] for r in per_atom)),
        "raw_total_score": float(alpha_values[-1]),
        "raw_energy_risk": float(sum(r["raw_variance"] for r in per_atom)),
        "energy_risk": float(sum(r["raw_variance"] for r in per_atom)),
        "banded_energy_risk": None,
        "calibrated_expected_iqa_error_ha": None,
        "calibration_applied": False,
        "spectral_frequency_risk": 0.0,
        "legacy_frequency_risk": 0.0,
        "fullspace_residual_distance": 0.0,
        "fullspace_residual_penalty": 0.0,
        "aligned_rmsd_ang": 0.0,
        "aligned_rmsd_penalty": 0.0,
        "observable_score": float(alpha_values[-1]),
        "outlier_penalty_score": 0.0,
        "acquisition_fallback_reasons": ["synthetic_mock_acquisition_metrics"],
        "spectral_modes": [],
        "landing_policy": "mock_final",
        "movement_rmsd_ang": metrics["movement_rmsd_ang"],
        "movement_band_min_ang": metrics["movement_band_min_ang"],
        "movement_band_peak_ang": metrics["movement_band_peak_ang"],
        "movement_band_max_ang": metrics["movement_band_max_ang"],
        "movement_utility_score": metrics["movement_utility_score"],
        "movement_progress_score": metrics["movement_progress_score"],
        "movement_direction_source": metrics["movement_direction_source"],
        "safety_metrics": dict(landing_safety.get("metrics") or {}),
        "per_atom": per_atom,
    }
    return AriadneRunResult(
        initial_atoms=seed,
        seed_atoms=seed,
        seed_alpha=float(alpha_values[0]),
        optimiser_initial_atoms=seed,
        optimiser_initial_alpha=float(alpha_values[0]),
        optimiser_initial_origin="seed_fallback",
        warm_start_alpha_delta_from_seed=0.0,
        final_atoms=final,
        alpha_trajectory=alpha_values,
        grad_norm_trajectory=grad_values,
        n_evaluations=2 * n_iters,
        return_code=0,
        wall_seconds=float(wall),
        fell_back_to_ds=False,
        rigid_force_clamps=0,
        mock=True,
        whitened_distance_final=0.0,
        raw_final_atoms=final,
        raw_alpha_final=alpha_values[-1],
        raw_whitened_distance_final=0.0,
        landing_safety=landing_safety,
        landing_candidates=landing_candidates,
        selection_diagnostics=selection_diagnostics,
        trqn_backtransform_mode=run_config.trqn_backtransform_mode,
        trqn_geodesic_bt_mode=run_config.trqn_geodesic_bt_mode,
    )


def _select_safe_gradient_band_warm_start(
    *,
    acquisition: SeedLocalAdversarialAcquisition,
    seed_atoms: Atoms,
    safety_config: Any,
    quality_gates: Any,
) -> tuple[Optional[np.ndarray], str, List[Dict[str, Any]]]:
    probe_method = getattr(acquisition, "gradient_band_probe_atoms", None)
    if not callable(probe_method):
        return None, "seed_fallback", []
    seed_mean_energy = float(acquisition.posterior.mean(seed_atoms))
    records: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for idx, (label, probe_atoms) in enumerate(probe_method()):
        try:
            coords = _coords_array(probe_atoms)
            candidate = _evaluate_landing_candidate(
                acquisition=acquisition,
                seed_atoms=seed_atoms,
                seed_mean_energy=seed_mean_energy,
                coords=coords,
                origin=str(label),
                candidate_index=idx,
                alpha=None,
                grad_norm=None,
                safety_config=safety_config,
                quality_gates=quality_gates,
            )
            public = _candidate_public(candidate)
            records.append(public)
            if bool(candidate.get("accepted")):
                candidates.append(candidate)
        except Exception as exc:
            records.append({
                "candidate_index": int(idx),
                "origin": str(label),
                "accepted": False,
                "reasons": ["gradient_band_warm_start_evaluation_failed"],
                "record_only_reasons": [],
                "metrics": {"evaluation_error": type(exc).__name__ + ": " + str(exc)},
                "alpha": None,
                "grad_norm": None,
                "informativeness_score": None,
                "risk_penalty_score": None,
            })
    if not candidates:
        return None, "seed_fallback", records
    selected = max(candidates, key=_rank_landing_candidate)
    selected_coords = _coords_array(selected["atoms"])
    for record in records:
        record["selected_for_warm_start"] = (
            int(record.get("candidate_index", -1))
            == int(selected.get("candidate_index", -2))
        )
    return selected_coords, "gradient_band_warm_start", records


def _landing_needs_under_move_retry(
    payload: Dict[str, Any],
    *,
    safety_config: Any,
    run_config: AriadneRunConfig,
) -> bool:
    if not bool(_cfg_value(safety_config, "under_move_retry", True)):
        return False
    if not bool(getattr(run_config, "trqn_under_move_retry", True)):
        return False
    if int(getattr(run_config, "trqn_under_move_retry_max", 1)) <= 0:
        return False
    safety = payload.get("landing_safety", {}) if isinstance(payload, dict) else {}
    if bool(safety.get("accepted", False)):
        return False
    candidates = (
        payload.get("landing_candidates")
        or safety.get("landing_candidates")
        or []
    )
    under = 0
    blocking = 0
    for candidate in candidates:
        reasons = set(candidate.get("reasons") or [])
        if "ariadne_landing_under_moved" in reasons:
            under += 1
        other = {
            r for r in reasons
            if r not in {
                "ariadne_landing_under_moved",
                "ariadne_landing_is_seed",
                "seed_fallback_disabled",
            }
        }
        if other:
            blocking += 1
    return under > 0 and blocking == 0


def _live_optimise_seed(
    models,
    seed: Atoms,
    trajectory: Sequence[Atoms],
    acquisition_config: AcquisitionConfig,
    run_config: AriadneRunConfig,
    *,
    seed_frame_id: Optional[int] = None,
    external_reference_scales: Optional[dict] = None,
    error_calibration_model: Optional[dict] = None,
    error_calibration_apply_strength: float = 0.0,
    max_acquisition_grad_per_ang: Optional[float] = None,
    max_force_per_atom_ha_per_ang: Optional[float] = 50.0,
    project_rigid: bool = True,
    gradient_backend: str = "process",
    safety_config: Optional[Any] = None,
    quality_gates: Optional[Any] = None,
    trace_path: Optional[Any] = None,
) -> AriadneRunResult:
    """Run ARIADNE adversarial descent for one seed against a real
    FEREBUS-trained posterior.

    Composition (the whole point of this function):

      1. build SeedLocalAdversarialAcquisition -- which itself builds
         the TotalEnergyPosterior over the trained Models, picks the
         50-frame local neighbourhood around the seed, fits the local
         subspace, and (unless we hand it external_reference_scales)
         computes the per-iteration scale anchors.
      2. wrap it in AdversarialASECalculator so ARIADNE sees an
         ordinary ASE-style get_potential_energy / get_forces API.
      3. hand it to the local runner which drives the two-stage
         ARIADNE step_py loop.
      4. compute the whitened distance using the frozen local subspace.
         this is the real anti-overlap metric, not the alpha-delta
         proxy the dry-run executor uses.
      5. wrap everything into an AriadneRunResult and return.
    """
    from .ariadne_local_runner import run_optimisation_against_calculator
    from .ase_calculator import AdversarialASECalculator

    acquisition = SeedLocalAdversarialAcquisition(
        models=models,
        seed=seed,
        trajectory=trajectory,
        config=acquisition_config,
        seed_frame_id=seed_frame_id,
        external_reference_scales=external_reference_scales,
        error_calibration_model=error_calibration_model,
        error_calibration_apply_strength=error_calibration_apply_strength,
    )
    seed_alpha = float(acquisition.components(acquisition.seed_atoms).total)

    # AdversarialASECalculator subscripts the clamp counter as a dict
    # (per_atom_acquisition_grad key). a plain dict matches that contract exactly --
    # an earlier custom-class shim had a .value attribute that did not.
    counter = {}

    calculator = AdversarialASECalculator(
        acquisition=acquisition,
        project_rigid=project_rigid,
        max_acquisition_grad_per_ang=max_acquisition_grad_per_ang,
        max_force_per_atom_ha_per_ang=max_force_per_atom_ha_per_ang,
        # honour the operator's resources.gradient_parallel_backend rather than hardcoding -- lets
        # them force "serial" on a node without fork, or for debugging (A33).
        gradient_backend=gradient_backend,
        clamp_counter=counter,
    )

    seed_ase = _ichor_to_ase(acquisition.seed_atoms)
    warm_start_positions, initial_origin, warm_start_records = (
        _select_safe_gradient_band_warm_start(
            acquisition=acquisition,
            seed_atoms=acquisition.seed_atoms,
            safety_config=safety_config,
            quality_gates=quality_gates,
        )
    )

    opt_result = run_optimisation_against_calculator(
        seed_atoms=seed_ase,
        calculator=calculator,
        run_config=run_config,
        trace_path=trace_path,
        initial_positions_angstrom=warm_start_positions,
        initial_origin=initial_origin,
        warm_start_records=warm_start_records,
    )

    raw_final_atoms = _make_ichor_from_positions(
        acquisition.seed_atoms, opt_result.final_positions_angstrom,
    )
    landing = _select_safe_landing(
        acquisition=acquisition,
        seed_atoms=acquisition.seed_atoms,
        raw_final_atoms=raw_final_atoms,
        opt_candidate_positions=list(opt_result.candidate_positions_angstrom),
        opt_candidate_alphas=list(opt_result.candidate_alphas),
        opt_candidate_grad_norms=list(opt_result.candidate_grad_norms),
        opt_candidate_origins=list(opt_result.candidate_origins),
        alpha_trajectory=list(opt_result.alpha_trajectory),
        safety_config=safety_config,
        quality_gates=quality_gates,
    )
    if _landing_needs_under_move_retry(
        landing,
        safety_config=safety_config,
        run_config=run_config,
    ):
        retry_config = replace(
            run_config,
            trqn_target_initial_grad_rms=float(
                getattr(run_config, "trqn_under_move_target_initial_grad_rms", 6.0e-4)
            ),
            trqn_scale_mode="adaptive_initial_gradient_rms",
            trqn_under_move_retry_max=0,
        )
        opt_result_retry = run_optimisation_against_calculator(
            seed_atoms=seed_ase,
            calculator=calculator,
            run_config=retry_config,
            trace_path=trace_path,
            initial_positions_angstrom=warm_start_positions,
            initial_origin=initial_origin,
            warm_start_records=warm_start_records,
        )
        raw_final_atoms_retry = _make_ichor_from_positions(
            acquisition.seed_atoms,
            opt_result_retry.final_positions_angstrom,
        )
        landing_retry = _select_safe_landing(
            acquisition=acquisition,
            seed_atoms=acquisition.seed_atoms,
            raw_final_atoms=raw_final_atoms_retry,
            opt_candidate_positions=list(opt_result_retry.candidate_positions_angstrom),
            opt_candidate_alphas=list(opt_result_retry.candidate_alphas),
            opt_candidate_grad_norms=list(opt_result_retry.candidate_grad_norms),
            opt_candidate_origins=list(opt_result_retry.candidate_origins),
            alpha_trajectory=list(opt_result_retry.alpha_trajectory),
            safety_config=safety_config,
            quality_gates=quality_gates,
        )
        opt_result_retry.diagnostics["under_move_retry_attempted"] = True
        opt_result_retry.diagnostics["under_move_retry_target_grad_rms"] = float(
            getattr(run_config, "trqn_under_move_target_initial_grad_rms", 6.0e-4)
        )
        opt_result_retry.diagnostics["under_move_retry_succeeded"] = bool(
            landing_retry.get("landing_safety", {}).get("accepted", False)
        )
        opt_result = opt_result_retry
        raw_final_atoms = raw_final_atoms_retry
        landing = landing_retry
    optimiser_initial_positions = (
        opt_result.candidate_positions_angstrom[0]
        if opt_result.candidate_positions_angstrom
        else np.asarray(acquisition.seed_atoms.coordinates, dtype=float)
    )
    optimiser_initial_atoms = _make_ichor_from_positions(
        acquisition.seed_atoms,
        optimiser_initial_positions,
    )
    optimiser_initial_alpha = (
        float(opt_result.candidate_alphas[0])
        if opt_result.candidate_alphas else None
    )
    optimiser_initial_origin = (
        str(opt_result.candidate_origins[0])
        if opt_result.candidate_origins else "seed_fallback"
    )
    warm_start_alpha_delta = (
        None if optimiser_initial_alpha is None
        else float(optimiser_initial_alpha) - float(seed_alpha)
    )
    try:
        selection_diagnostics = _selection_prediction_diagnostics(
            acquisition,
            landing["selected_atoms"],
            landing_safety=landing["landing_safety"],
        )
        selection_diagnostics["seed_alpha"] = float(seed_alpha)
        selection_diagnostics["optimiser_initial_alpha"] = (
            None if optimiser_initial_alpha is None
            else float(optimiser_initial_alpha)
        )
        selection_diagnostics["optimiser_initial_origin"] = optimiser_initial_origin
        selection_diagnostics["warm_start_alpha_delta_from_seed"] = (
            None if warm_start_alpha_delta is None
            else float(warm_start_alpha_delta)
        )
    except Exception:
        selection_diagnostics = None

    return AriadneRunResult(
        initial_atoms=acquisition.seed_atoms,
        seed_atoms=acquisition.seed_atoms,
        seed_alpha=float(seed_alpha),
        optimiser_initial_atoms=optimiser_initial_atoms,
        optimiser_initial_alpha=optimiser_initial_alpha,
        optimiser_initial_origin=optimiser_initial_origin,
        warm_start_alpha_delta_from_seed=warm_start_alpha_delta,
        final_atoms=landing["selected_atoms"],
        alpha_trajectory=list(opt_result.alpha_trajectory),
        grad_norm_trajectory=list(opt_result.grad_norm_trajectory),
        n_evaluations=int(opt_result.n_evaluations),
        return_code=int(opt_result.return_code),
        wall_seconds=float(opt_result.wall_seconds),
        fell_back_to_ds=bool(opt_result.fell_back_to_ds),
        rigid_force_clamps=int(opt_result.rigid_force_clamps),
        mock=False,
        whitened_distance_final=landing["selected_whitened_distance"],
        selected_alpha_final=landing["selected_alpha"],
        raw_final_atoms=raw_final_atoms,
        raw_alpha_final=(
            float(opt_result.alpha_trajectory[-1])
            if opt_result.alpha_trajectory else None
        ),
        raw_whitened_distance_final=landing["raw_whitened_distance"],
        landing_safety=landing["landing_safety"],
        landing_candidates=landing["landing_candidates"],
        selection_diagnostics=selection_diagnostics,
        optimiser_diagnostics=dict(opt_result.diagnostics or {}),
        trqn_backtransform_mode=run_config.trqn_backtransform_mode,
        trqn_geodesic_bt_mode=run_config.trqn_geodesic_bt_mode,
        optimiser_converged=bool(opt_result.converged),
    )


def _ichor_to_ase(ichor_atoms):
    """Convert ICHOR Atoms -> ASE Atoms for the local runner."""
    from ase import Atoms as ASEAtoms
    symbols = [a.type for a in ichor_atoms]
    positions = [(float(a.x), float(a.y), float(a.z)) for a in ichor_atoms]
    return ASEAtoms(symbols=symbols, positions=positions)


def _make_ichor_from_positions(template_ichor, positions_array):
    """Build an ICHOR Atoms object with same atom types as the
    template but updated coordinates from a (N, 3) array.
    """
    import numpy as _np
    pos = _np.asarray(positions_array, dtype=float).reshape(-1, 3)
    if len(pos) != len(template_ichor):
        raise ValueError(
            "position array length " + str(len(pos))
            + " does not match seed natoms " + str(len(template_ichor))
        )
    return Atoms([
        Atom(template_ichor[i].type, float(pos[i, 0]),
             float(pos[i, 1]), float(pos[i, 2]))
        for i in range(len(template_ichor))
    ])


def _await_exists(check) -> bool:
    """a file just committed on the login node can take a few seconds to show up on a compute node
    over NFS/Lustre. retry the existence check a handful of times before giving up, so a transient
    metadata lag does not kill the seed on the spot (A19). tunable via env -- tests can set the
    delay to 0 so they do not actually sleep."""
    import os as _os
    import time as _time
    tries = max(1, int(_os.environ.get("ICHOR_RUNNER_DEP_RETRIES", "5")))
    delay = float(_os.environ.get("ICHOR_RUNNER_DEP_RETRY_SECONDS", "2"))
    for k in range(tries):
        if check():
            return True
        if k < tries - 1 and delay > 0:
            _time.sleep(delay)
    return False


def main(argv=None) -> int:
    """Command-line entrypoint for a single per-seed ARIADNE run.

    The daemon submits a SLURM array job for the ARIADNE_ARRAY phase;
    each array task invokes this with a different --seed-index.
    The task picks the matching seed out of seeds_picked.json,
    loads the trained FEREBUS Models for the current iteration, builds
    the adversarial acquisition + calculator, drives ARIADNE, and
    writes 7_ACTIVE_LEARNING/iteration-NNNN/pool/seed_NNNN/result.json.

    Exit codes:
      0 -- success, result.json written.
      2 -- bad command line (missing files, malformed args).
      3 -- per-iteration state was not in a runnable shape.
      4 -- ARIADNE descent failed mid-run; we still write result.json.
    """
    import argparse
    import json
    import sys as _sys
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        prog="python -m ichor.hpc.active_learning.acquisition.ariadne_runner",
        description=(
            "Run ARIADNE adversarial descent for a single seed picked at "
            "SEED_SELECT. Writes result.json into the per-seed pool dir."
        ),
    )
    parser.add_argument(
        "--seed-index", type=int, required=True,
        help="Row in seeds_picked.json for this array task.",
    )
    parser.add_argument(
        "--iteration", type=int, required=True,
        help="Campaign iteration this descent belongs to.",
    )
    parser.add_argument(
        "--campaign-dir", type=str, required=True,
        help="Path to the campaign root (where campaign.yaml lives).",
    )
    args = parser.parse_args(argv)

    campaign = _Path(args.campaign_dir).resolve()
    if not campaign.is_dir():
        print(
            "campaign-dir does not exist or is not a directory: "
            + str(campaign), file=_sys.stderr,
        )
        return 2
    cfg_path = campaign / "campaign.yaml"
    if not cfg_path.is_file():
        print(
            "campaign.yaml not found at " + str(cfg_path),
            file=_sys.stderr,
        )
        return 2

    from ..config import CampaignConfig
    from ..acquisition.trajectory_pool import TrajectoryPool
    from ..daemon.state import read_state, DEFAULT_STATE_FILENAME
    from ichor.core.models import Models

    config = CampaignConfig.from_yaml(cfg_path)

    try:
        pool = TrajectoryPool.load(campaign)
    except (FileNotFoundError, ValueError) as exc:
        print(
            "trajectory pool not loadable: " + str(exc),
            file=_sys.stderr,
        )
        return 3

    state_path = (
        campaign / ".DATA"
        / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    )
    if not _await_exists(lambda: state_path.is_file()):
        print(
            "state.json not found at " + str(state_path),
            file=_sys.stderr,
        )
        return 3
    state = read_state(state_path)
    if int(state.models_version) < 0:
        print(
            "models_version is " + str(state.models_version)
            + "; INITIAL_FEREBUS has not committed any models yet.",
            file=_sys.stderr,
        )
        return 3
    models_dir = (
        campaign / "6_TRAINED_MODELS"
        / ("iteration-" + str(int(state.models_version)).zfill(4))
    )
    if not _await_exists(lambda: models_dir.is_dir()):
        print(
            "models directory not found: " + str(models_dir),
            file=_sys.stderr,
        )
        return 3
    models = Models(models_dir)

    iter_dir = (
        campaign / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(args.iteration)).zfill(4))
    )
    sp_path = iter_dir / "seeds_picked.json"
    if not sp_path.is_file():
        print(
            "seeds_picked.json not found at " + str(sp_path)
            + "; SEED_SELECT must run before ARIADNE_ARRAY.",
            file=_sys.stderr,
        )
        return 3
    try:
        from ..handoff_manifests import load_seeds_picked
        sp = load_seeds_picked(iter_dir, expected_iteration=int(args.iteration))
    except Exception as exc:
        print(
            "seeds_picked.json invalid: " + type(exc).__name__ + ": " + str(exc),
            file=_sys.stderr,
        )
        return 3
    # the seed list was picked against a specific pool. frame ids are positional, so if the pool on
    # disk is not the one SEED_SELECT chose against (a re-import, a partial swap, a botched reconcile)
    # then frame N is now a different geometry and we'd attack the wrong point silently. refuse.
    # only checked when the field is present, so older seeds_picked.json still run (A30).
    picked_sha = sp.get("trajectory_sha256")
    if picked_sha and picked_sha != pool.sha256:
        print(
            "pool sha256 mismatch: seeds_picked.json was made against " + str(picked_sha)
            + " but the pool on disk is " + str(pool.sha256)
            + "; the trajectory changed under the campaign -- refusing to attack the wrong frame.",
            file=_sys.stderr,
        )
        return 3
    seed_records = list(sp.get("seed_records") or [])
    if not 0 <= args.seed_index < len(seed_records):
        print(
            "seed-index " + str(args.seed_index)
            + " out of range of " + str(len(seed_records))
            + " picked seeds",
            file=_sys.stderr,
        )
        return 3
    seed_frame_raw = seed_records[args.seed_index].get("frame_id")
    if seed_frame_raw is None:
        print(
            "seed-index " + str(args.seed_index) + " has no frame_id",
            file=_sys.stderr,
        )
        return 3
    seed_frame_id = int(seed_frame_raw)
    seed_atoms = pool.frame(seed_frame_id)

    rs_path = iter_dir / "reference_scales.json"
    external_reference_scales = None
    if rs_path.is_file():
        try:
            with open(rs_path, "r", encoding="utf-8") as f:
                external_reference_scales = json.load(f)
            required = ("energy", "force", "omega", "anh", "anh_std")
            if not isinstance(external_reference_scales, dict):
                raise ValueError("reference_scales.json must contain an object")
            for key in required:
                value = float(external_reference_scales[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        "reference scale " + key + " must be finite and positive"
                    )
        except (OSError, KeyError, ValueError):
            print(
                "reference_scales.json invalid at " + str(rs_path),
                file=_sys.stderr,
            )
            return 3

    acquisition_config = config.to_acquisition_config()
    ariadne_run_config = config.to_ariadne_run_config()
    error_calibration_model = None
    error_calibration_reason = "disabled"
    try:
        from ..daemon.error_calibration import load_calibration_model_for_acquisition

        error_calibration_model, error_calibration_reason = (
            load_calibration_model_for_acquisition(campaign, config)
        )
    except Exception:
        error_calibration_model = None
        error_calibration_reason = "malformed_model"
    error_calibration_strength = (
        float(config.error_calibration.apply_strength)
        if error_calibration_model is not None
        else 0.0
    )
    seed_dir = (
        iter_dir / "pool"
        / ("seed_" + str(int(args.seed_index)).zfill(4))
    )
    trace_path = seed_dir / "ARIADNE_TRACE.jsonl"

    try:
        result = optimise_seed(
            models=models,
            seed=seed_atoms,
            trajectory=pool.to_atoms_list(),
            acquisition_config=acquisition_config,
            run_config=ariadne_run_config,
            mock=False,
            seed_frame_id=seed_frame_id,
            external_reference_scales=external_reference_scales,
            error_calibration_model=error_calibration_model,
            error_calibration_apply_strength=error_calibration_strength,
            max_acquisition_grad_per_ang=config.effective_max_acquisition_grad_per_ang(),
            gradient_backend=str(config.resources.gradient_parallel_backend),
            safety_config=getattr(config, "adversarial_safety", None),
            quality_gates=getattr(config, "quality_gates", None),
            trace_path=trace_path,
        )
    except Exception as exc:
        print(
            "ARIADNE descent raised: " + repr(exc),
            file=_sys.stderr,
        )
        result = AriadneRunResult(
            initial_atoms=seed_atoms,
            final_atoms=seed_atoms,
            return_code=2,
            mock=False,
            trqn_backtransform_mode=ariadne_run_config.trqn_backtransform_mode,
            trqn_geodesic_bt_mode=ariadne_run_config.trqn_geodesic_bt_mode,
        )

    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["seed_frame_id"] = seed_frame_id
    payload["seed_index"] = int(args.seed_index)
    payload["iteration"] = int(args.iteration)
    allow_seed_fallback = bool(
        getattr(getattr(config, "adversarial_safety", None), "allow_seed_fallback", False)
    )
    usability = ariadne_result_usability_payload(
        payload,
        allow_seed_fallback=allow_seed_fallback,
    )
    payload["optimiser_converged"] = bool(usability["optimiser_converged"])
    payload["task_success"] = bool(usability["usable"])
    payload["task_success_reason"] = str(usability["reason"])
    payload["task_exit_code"] = int(usability["task_exit_code"])
    if isinstance(payload.get("landing_safety"), dict):
        payload["selected_landing_policy"] = str(
            payload["landing_safety"].get("policy", "unknown")
        )
        payload["selected_landing_usable"] = bool(usability["usable"])
    if isinstance(payload.get("selection_diagnostics"), dict):
        payload["selection_diagnostics"]["model_version"] = int(state.models_version)
        payload["selection_diagnostics"]["seed_index"] = int(args.seed_index)
        payload["selection_diagnostics"]["seed_frame_id"] = int(seed_frame_id)
        payload["selection_diagnostics"]["error_calibration_model_reason"] = str(
            error_calibration_reason
        )
    if bool(usability.get("safe_landing_salvaged_after_optimiser_failure", False)):
        diag = payload.get("optimiser_diagnostics")
        if not isinstance(diag, dict):
            diag = {}
        diag["safe_landing_salvaged_after_optimiser_failure"] = True
        diag.setdefault(
            "salvaged_after_optimiser_failure_reason",
            str(usability.get("reason", "")),
        )
        payload["optimiser_diagnostics"] = diag
    from ..daemon.state import atomic_write_json

    atomic_write_json(seed_dir / "result.json", payload)
    return int(usability["task_exit_code"])


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
