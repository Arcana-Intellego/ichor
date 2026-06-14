from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.files.xyz import Trajectory
from ichor.core.models.models import Models

from .barrier import ChemistryBarrierState, build_chemistry_barrier_state, chemistry_barrier_value
from .config import AcquisitionConfig
from .geometry import aligned_mass_weighted_rmsd, load_seed_atoms, load_trajectory, select_local_neighbours
from .posterior import TotalEnergyPosterior
from .stencils import (
    directional_cubic_stencil,
    directional_curvature_stencil,
    directional_force_stencil,
    directional_quartic_stencil,
)
from .subspace import (
    LocalSubspace,
    active_coordinates,
    build_local_subspace,
    directional_step_sizes,
    fullspace_residual_distance,
    local_neighbour_residual_scale,
    whitened_distance_squared,
)


@dataclass(frozen=True)
class ModeEvaluation:
    index: int
    force_std: float
    curvature_mean: float
    curvature_std: float
    omega: float
    omega_std: float
    cubic_mean: float
    cubic_std: float
    quartic_mean: float
    quartic_std: float
    anharmonicity: float
    anharmonicity_std: float
    spectral_weight: float = 0.0
    frequency_observable_score: float = 0.0


@dataclass(frozen=True)
class AcquisitionBreakdown:
    total: float
    informativeness_score: float
    risk_penalty_score: float
    energy_risk: float
    force_risk: float
    frequency_risk: float
    anharmonic_risk: float
    distance_penalty: float
    chemistry_penalty: float
    mean_energy: float
    energy_variance: float
    raw_energy_risk: Optional[float] = None
    banded_energy_risk: Optional[float] = None
    calibrated_expected_iqa_error_ha: Optional[float] = None
    calibration_applied: bool = False
    spectral_frequency_risk: Optional[float] = None
    legacy_frequency_risk: Optional[float] = None
    fullspace_residual_distance: Optional[float] = None
    fullspace_residual_penalty: float = 0.0
    aligned_rmsd_ang: Optional[float] = None
    aligned_rmsd_penalty: float = 0.0
    negative_curvature_penalty: float = 0.0
    observable_score: Optional[float] = None
    outlier_penalty_score: Optional[float] = None
    fallback_reasons: Tuple[str, ...] = field(default_factory=tuple)
    mode_evaluations: Tuple[ModeEvaluation, ...] = field(default_factory=tuple)


def _finite_positive_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out) or out <= 0.0:
        return default
    return out


def _lookup_table_error(table: Mapping[str, object], raw_uncertainty: float) -> Optional[float]:
    bins = table.get("bins") if isinstance(table, Mapping) else None
    if not isinstance(bins, list) or not bins:
        return None
    raw = float(raw_uncertainty)
    if not np.isfinite(raw):
        return None
    last_value = None
    for entry in bins:
        if not isinstance(entry, Mapping):
            continue
        value = _finite_positive_float(
            entry.get("calibrated_abs_error_ha"),
            _finite_positive_float(entry.get("median_abs_error_ha")),
        )
        if value is None:
            continue
        last_value = value
        hi = _finite_positive_float(entry.get("raw_uncertainty_max"), None)
        if hi is None:
            continue
        if raw <= hi:
            return value
    return last_value


def _lookup_calibrated_error(
    model: Optional[Mapping[str, object]],
    atom_variances: Optional[Mapping[str, float]],
    atom_types: Sequence[str],
    total_variance: float,
) -> Optional[float]:
    if not isinstance(model, Mapping):
        return None
    tables = model.get("tables")
    if not isinstance(tables, Mapping):
        return None

    total_table = tables.get("global_total")
    if isinstance(total_table, Mapping):
        total_value = _lookup_table_error(total_table, float(total_variance))
        if total_value is not None:
            return float(total_value)

    values: List[float] = []
    if atom_variances:
        for atom, raw in atom_variances.items():
            table = tables.get("global")
            if isinstance(table, Mapping):
                value = _lookup_table_error(table, float(raw))
                if value is not None:
                    values.append(float(value))
    if values:
        return float(np.sqrt(float(np.sum(np.square(values)))))

    global_table = tables.get("global")
    if isinstance(global_table, Mapping):
        return _lookup_table_error(global_table, float(total_variance))
    return None


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    finite = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, float(q)))


def _calibration_error_values(model: Optional[Mapping[str, object]]) -> List[float]:
    if not isinstance(model, Mapping):
        return []
    tables = model.get("tables")
    if not isinstance(tables, Mapping):
        return []
    table = tables.get("global")
    if isinstance(tables.get("global_total"), Mapping):
        table = tables.get("global_total")
    if not isinstance(table, Mapping):
        return []
    bins = table.get("bins")
    if not isinstance(bins, list):
        return []
    values = []
    for entry in bins:
        if not isinstance(entry, Mapping):
            continue
        value = _finite_positive_float(
            entry.get("calibrated_abs_error_ha"),
            _finite_positive_float(entry.get("median_abs_error_ha")),
        )
        if value is not None:
            values.append(float(value))
    return values


def _infer_bands_from_calibration_model(
    model: Optional[Mapping[str, object]],
) -> Optional[Tuple[float, float]]:
    values = _calibration_error_values(model)
    if len(values) < 2:
        return None
    low = _percentile(values, 0.25)
    high = _percentile(values, 0.90)
    if low is None or high is None or not high > low:
        return None
    return float(low), float(high)


class SeedLocalAdversarialAcquisition:
    """Single-model seed-local acquisition built from IQA energy stencils.

    The implementation is deliberately conservative and fully backwards compatible
    with existing ICHOR GP interfaces:

    * no changes are required in `Model` or `Models`
    * approximate total-energy uncertainty is built by summing per-atom IQA posteriors
    * force/curvature/anharmonicity terms are built from linear energy stencils
    * the pseudo-force for ARIADNE is provided by finite differences of the
      acquisition value in Cartesian coordinates or along the local active modes
    """

    def __init__(
        self,
        models: Models,
        seed: Union[str, Atoms],
        trajectory: Union[str, Trajectory, Sequence[Atoms]],
        config: AcquisitionConfig | None = None,
        *,
        seed_frame_id: Optional[int] = None,
        external_reference_scales: Optional[Dict[str, float]] = None,
        error_calibration_model: Optional[Mapping[str, object]] = None,
        error_calibration_apply_strength: float = 0.0,
    ) -> None:
        """seed_frame_id is the stable trajectory pool position the seed
        was sampled from. Optional (defaults to None to preserve backward compatibility 
        with an earlier patch); the daemon wiring will set it explicitly so
        per-pointdir provenance can record where each seed came from.
        """
        self.config = config or AcquisitionConfig()
        self.models = models
        self.error_calibration_model = error_calibration_model
        self.error_calibration_apply_strength = max(
            0.0, min(1.0, float(error_calibration_apply_strength))
        )
        self.seed_atoms = load_seed_atoms(seed)
        self.seed_frame_id = seed_frame_id
        #trajectory may already be a TrajectoryPool (duck-typed); load_trajectory
        #only triggers on path/Trajectory/Sequence inputs. The pool case is forwarded
        #untouched so select_local_neighbours can extract its stable frame IDs.
        if hasattr(trajectory, "frame") and hasattr(trajectory, "frame_ids"):
            self.trajectory = trajectory  #TrajectoryPool-like; preserve identity
        else:
            self.trajectory = load_trajectory(trajectory)
        self.posterior = TotalEnergyPosterior(
            models=models,
            property_name=self.config.property_name,
            scaled=self.config.use_scaled_posterior_covariance,
        )
        neighbours = select_local_neighbours(
            self.seed_atoms,
            self.trajectory,
            max_neighbours=self.config.subspace.neighbour_count,
            deduplicate_rmsd=self.config.subspace.neighbour_deduplicate_rmsd,
        )
        if not neighbours:
            raise ValueError("No local neighbours could be selected around the seed geometry.")
        self.subspace = build_local_subspace(self.seed_atoms, neighbours, self.config.subspace)
        self.mode_directions = self.subspace.cartesian_directions
        self.mode_steps = directional_step_sizes(
            self.subspace,
            self.config.stencils.step_scale,
            self.config.stencils.min_step,
            self.config.stencils.max_step,
        )
        self.barrier_state = build_chemistry_barrier_state(
            self.seed_atoms,
            [item.atoms for item in neighbours],
            self.posterior,
            self.config.barrier,
        )
        #Per-seed cache of autotuned FD step sizes per mode. None
        #until the first _mode_metrics call materialises them from the cubic
        #estimate; thereafter every subsequent call (including all ARIADNE
        #descent iterations on this seed) uses the cached steps directly,
        #paying only 1x stencil cost. Only consulted when
        #config.stencils.autotune_from_cubic is True.
        self.tuned_mode_steps: Optional[np.ndarray] = None
        if external_reference_scales is not None:
            #daemon supplied per-iteration scales; skip the expensive
            #per-seed _build_reference_scales call (~480 GP evals each).
            self.reference_scales = dict(external_reference_scales)
        else:
            self.reference_scales = self._build_reference_scales()

    @property
    def subspace_frame_ids(self):
        """Tuple of stable trajectory pool frame_ids that built this seed's
        local subspace. Only meaningful when 'trajectory' was a
        TrajectoryPool; otherwise the values are enumeration positions in the
        bare list and should be treated as transient.

        Returned as Tuple[int, ...] so the value is hashable and
        JSON-serialisable for direct insertion into provenance sidecars.
        """
        return tuple(int(n.index) for n in self.subspace.neighbours)

    @staticmethod
    def _softplus(x: float) -> float:
        return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))

    def _phi(self, z: float) -> float:
        return float(np.log1p(max(z, 0.0)))

    def _reference_scale(self, name: str, fallback: float) -> float:
        value = self.reference_scales.get(name, fallback)
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float(fallback)
        if not np.isfinite(out) or out <= 0.0:
            out = float(fallback)
        if not np.isfinite(out) or out <= 0.0:
            out = float(self.config.references.floor)
        if name == "residual":
            out = max(out, float(self.config.fullspace_confinement.min_residual_scale_ang))
        return float(max(out, 1.0e-12))

    def _configured_residual_scale(self) -> float:
        cfg = self.config.fullspace_confinement
        minimum = float(cfg.min_residual_scale_ang)
        if cfg.residual_scale == "fixed" and cfg.fixed_residual_scale_ang is not None:
            return float(max(float(cfg.fixed_residual_scale_ang), minimum))
        try:
            return local_neighbour_residual_scale(
                self.subspace,
                floor=self.config.references.floor,
                min_scale=minimum,
            )
        except Exception:
            return float(max(self.config.references.floor, minimum))

    def _band_parameters(self) -> Tuple[Optional[Tuple[float, float, float, float]], Tuple[str, ...]]:
        cfg = self.config.calibrated_energy
        reasons: List[str] = []
        try:
            low = None if cfg.band_low_ha is None else float(cfg.band_low_ha)
        except (TypeError, ValueError):
            low = None
        if low is not None and (not np.isfinite(low) or low < 0.0):
            low = None
        high = _finite_positive_float(cfg.band_high_ha)
        if low is None or high is None:
            inferred = _infer_bands_from_calibration_model(self.error_calibration_model)
            if inferred is not None:
                low, high = inferred
                reasons.append("banded_energy_inferred_from_calibration_model")
        if low is None or high is None:
            return None, tuple(reasons + ["banded_energy_missing_bands"])
        if high <= low:
            return None, tuple(reasons + ["banded_energy_invalid_bands"])
        width = max(high - low, 1.0e-12)
        low_soft = _finite_positive_float(cfg.low_softness_ha, width / 4.0)
        high_soft = _finite_positive_float(cfg.high_softness_ha, width / 4.0)
        return (
            (float(low), float(high), float(low_soft), float(high_soft)),
            tuple(reasons),
        )

    def _energy_utility(self, value: float, scale: float) -> Tuple[float, Optional[float], Tuple[str, ...]]:
        cfg = self.config.calibrated_energy
        log_value = self._phi(float(value) / max(float(scale), 1.0e-12))
        if cfg.utility == "log":
            return float(log_value), None, ("energy_utility_log",)
        if cfg.utility != "banded":
            return float(log_value), None, ("energy_utility_unknown_fallback_log",)

        params, reasons = self._band_parameters()
        if params is None:
            return float(log_value), None, reasons
        low, high, low_soft, high_soft = params
        gate_low = _sigmoid((float(value) - low) / max(low_soft, 1.0e-12))
        gate_high = _sigmoid((high - float(value)) / max(high_soft, 1.0e-12))
        banded = float(log_value * gate_low * gate_high)
        return banded, banded, reasons

    def _fullspace_confinement_metrics(self, atoms: Atoms) -> Dict[str, object]:
        cfg = self.config.fullspace_confinement
        out: Dict[str, object] = {
            "residual_distance": None,
            "residual_scale": None,
            "residual_penalty": 0.0,
            "aligned_rmsd_ang": None,
            "rmsd_scale_ang": None,
            "rmsd_penalty": 0.0,
            "fallback_reasons": [],
        }
        if not bool(cfg.enabled):
            out["fallback_reasons"] = ["fullspace_confinement_disabled"]
            return out
        try:
            residual = float(fullspace_residual_distance(self.subspace, atoms))
            residual_scale = self._reference_scale(
                "residual",
                self._configured_residual_scale(),
            )
            rmsd = float(aligned_mass_weighted_rmsd(self.seed_atoms, atoms))
            rmsd_scale = self._reference_scale(
                "rmsd",
                float(cfg.rmsd_scale_ang),
            )
            out.update({
                "residual_distance": residual,
                "residual_scale": residual_scale,
                "residual_penalty": float((residual / residual_scale) ** 2),
                "aligned_rmsd_ang": rmsd,
                "rmsd_scale_ang": rmsd_scale,
                "rmsd_penalty": float((rmsd / rmsd_scale) ** 2),
            })
        except Exception as exc:
            penalty = float(max(float(cfg.failure_penalty), 1.0e-12))
            out["residual_penalty"] = penalty
            out["rmsd_penalty"] = penalty
            out["fallback_reasons"] = [
                "fullspace_confinement_unavailable:" + type(exc).__name__
            ]
        return out

    def _curvature_floor(self, curvature: float) -> float:
        floor = self.config.stencils.curvature_floor
        beta = self.config.stencils.softplus_scale
        return float(floor + beta * self._softplus(abs(float(curvature)) / beta))

    def _negative_curvature_penalty(self, mode_evals, weights) -> float:
        if str(getattr(self.config.stencils, "negative_curvature_policy", "ignore")) != "penalise":
            return 0.0
        scale = max(float(self.config.stencils.curvature_floor), 1.0e-12)
        total = 0.0
        for weight, mode in zip(weights, mode_evals):
            negative = max(0.0, -float(mode.curvature_mean))
            bounded = 1.0 - float(np.exp(-negative / scale))
            total += float(weight) * bounded
        return float(total)

    def _mode_metrics(self, atoms, mean_energy=None):
        """Compute per-mode metrics. With autotune_from_cubic enabled, the
        first call refines the per-mode FD step from the cubic estimate
        and caches the tuned steps on self; subsequent calls reuse the
        cache. Cost: 1x stencils per call when autotune off, 2x on first
        autotune call, 1x on subsequent autotune calls (cache hit).
        """
        autotune = bool(getattr(self.config.stencils, "autotune_from_cubic", False))
        if not autotune:
            return self._compute_mode_evaluations(atoms, mean_energy)
        if self.tuned_mode_steps is None:
            baseline = self._compute_mode_evaluations(atoms, mean_energy)
            self.tuned_mode_steps = self._refine_steps_from_cubic(baseline)
            return self._compute_mode_evaluations(
                atoms, mean_energy, override_steps=self.tuned_mode_steps,
            )
        return self._compute_mode_evaluations(
            atoms, mean_energy, override_steps=self.tuned_mode_steps,
        )

    def _refine_steps_from_cubic(self, baseline_evaluations):
        """Ppick per-mode eps such that FD truncation error
        eps^2 * |cubic| stays below 1 percent of |gradient| magnitude.
        Clamped to [stencils.min_step, stencils.max_step].
        """
        import math
        tuned = []
        for ev in baseline_evaluations:
            cubic_mag = max(abs(ev.cubic_mean), 1.0e-12)
            grad_mag = max(ev.force_std, 1.0e-12)
            target = math.sqrt(0.01 * grad_mag / cubic_mag)
            tuned.append(float(np.clip(
                target,
                self.config.stencils.min_step,
                self.config.stencils.max_step,
            )))
        return np.asarray(tuned, dtype=float)

    def _compute_mode_evaluations(self, atoms, mean_energy=None, override_steps=None):
        """Inner per-mode evaluation. Uses override_steps if provided
        (autotune path) or self.mode_steps (legacy / autotune-off path)."""
        steps = override_steps if override_steps is not None else self.mode_steps
        evaluations = []
        for idx, (direction, step) in enumerate(zip(self.mode_directions, steps)):
            step_f = float(step)
            force_eval = directional_force_stencil(self.posterior, atoms, direction, step_f)
            curvature_eval = directional_curvature_stencil(self.posterior, atoms, direction, step_f)
            cubic_eval = directional_cubic_stencil(self.posterior, atoms, direction, step_f)
            quartic_eval = directional_quartic_stencil(self.posterior, atoms, direction, step_f)

            k_hat = self._curvature_floor(curvature_eval.mean)
            omega = float(np.sqrt(k_hat))
            omega_std = float(curvature_eval.std / max(2.0 * omega, 1.0e-12))
            anh = float((step_f * abs(cubic_eval.mean) + step_f * step_f * abs(quartic_eval.mean)) / max(k_hat, 1.0e-12))
            anh_std = float((step_f * cubic_eval.std + step_f * step_f * quartic_eval.std) / max(k_hat, 1.0e-12))

            evaluations.append(
                ModeEvaluation(
                    index=idx,
                    force_std=float(force_eval.std),
                    curvature_mean=float(curvature_eval.mean),
                    curvature_std=float(curvature_eval.std),
                    omega=omega,
                    omega_std=omega_std,
                    cubic_mean=float(cubic_eval.mean),
                    cubic_std=float(cubic_eval.std),
                    quartic_mean=float(quartic_eval.mean),
                    quartic_std=float(quartic_eval.std),
                    anharmonicity=anh,
                    anharmonicity_std=anh_std,
                )
            )
        return tuple(evaluations)

    def _build_reference_scales(self) -> Dict[str, float]:
        neighbours = [item.atoms for item in self.subspace.neighbours]
        max_samples = self.config.references.max_reference_samples
        if len(neighbours) > max_samples:
            step = max(1, len(neighbours) // max_samples)
            sample_atoms = neighbours[::step][:max_samples]
        else:
            sample_atoms = neighbours

        force_vals: List[float] = []
        omega_vals: List[float] = []
        anh_vals: List[float] = []
        anh_std_vals: List[float] = []
        energy_vars: List[float] = []
        residual_vals: List[float] = []
        rmsd_vals: List[float] = []

        for atoms in sample_atoms:
            energy_vars.append(self.posterior.variance(atoms))
            try:
                residual_vals.append(fullspace_residual_distance(self.subspace, atoms))
            except Exception:
                pass
            try:
                rmsd_vals.append(aligned_mass_weighted_rmsd(self.seed_atoms, atoms))
            except Exception:
                pass
            for mode_eval in self._mode_metrics(atoms):
                force_vals.append(mode_eval.force_std)
                omega_vals.append(mode_eval.omega_std)
                anh_vals.append(mode_eval.anharmonicity)
                anh_std_vals.append(mode_eval.anharmonicity_std)

        floor = self.config.references.floor

        # guard the empty case. np.median([]) is nan, and a nan reference scale poisons every
        # downstream alpha (each mode metric gets divided by one of these), which then poisons the
        # whole acquisition and the stop-check alpha history. a tiny initial training set, or a seed
        # whose neighbours yield no usable modes, can genuinely leave these lists empty -- so fall
        # back to the floor alone (a sane positive anchor) instead of letting a nan through. (A46)
        def _median_or_zero(vals):
            return float(np.median(vals)) if len(vals) else 0.0

        scales = {
            "energy": _median_or_zero(energy_vars) + floor,
            "force": _median_or_zero(force_vals) + floor,
            "omega": _median_or_zero(omega_vals) + floor,
            "anh": _median_or_zero(anh_vals) + floor,
            "anh_std": _median_or_zero(anh_std_vals) + floor,
            "spectral": _median_or_zero(omega_vals) + floor,
            "residual": _median_or_zero(residual_vals) + floor,
            "rmsd": _median_or_zero(rmsd_vals) + floor,
            "calibrated_error": _finite_positive_float(
                self.error_calibration_model.get("reference_error_ha")
                if isinstance(self.error_calibration_model, Mapping)
                else None,
                _median_or_zero(energy_vars) + floor,
            ),
        }
        for key, value in scales.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    "reference scale " + key + " must be finite and positive"
                )
        return scales


    def _effective_mode_weights(self, mode_evals, policy: Optional[str] = None):
        """Compute mode weights based on the configured policy.

        variance:           w_i = lambda_i / sum_j lambda_j (= subspace.mode_weights)
        inverse_frequency:  w_i = (1 / (omega_i + omega_floor)) / sum_j (...)
        uniform:            w_i = 1/r

        The omega values come from mode_evals (sqrt of curvature, already
        floored in _mode_metrics). inverse_frequency biases the aggregation
        toward low-frequency modes -- exactly where the spectroscopic VACF
        peaks concentrate.
        """
        import numpy as np
        if policy is None:
            policy = getattr(self.config.subspace, "mode_weighting_policy", "variance")
        r = len(mode_evals)
        if r == 0:
            return tuple()
        if policy == "variance":
            return self.subspace.mode_weights
        if policy == "uniform":
            w = 1.0 / float(r)
            return tuple(w for _ in range(r))
        if policy == "inverse_frequency":
            floor = max(self.config.stencils.curvature_floor, 1.0e-12) ** 0.5
            inv = np.array([1.0 / (m.omega + floor) for m in mode_evals], dtype=float)
            total = float(inv.sum())
            if total <= 0.0:
                return tuple(1.0 / r for _ in range(r))
            return tuple(float(x / total) for x in inv)
        raise ValueError("unknown mode_weighting_policy: " + repr(policy))

    def _spectral_mode_weights(self, mode_evals):
        """Cheap observable weights: default to inverse-frequency soft modes."""
        if not mode_evals:
            return tuple()
        cfg = self.config.spectral
        mode_evals = tuple(mode_evals)
        if cfg.max_modes is not None and int(cfg.max_modes) > 0:
            mode_evals = mode_evals[: int(cfg.max_modes)]
        policy = str(getattr(cfg, "mode_weighting", "inverse_frequency"))
        if policy == "variance":
            raw = np.asarray(self.subspace.mode_weights[: len(mode_evals)], dtype=float)
            total = float(raw.sum())
            if not np.isfinite(total) or total <= 0.0:
                return tuple(1.0 / len(mode_evals) for _ in mode_evals)
            return tuple(float(x / total) for x in raw)
        if policy == "uniform":
            base = self._effective_mode_weights(mode_evals, policy=policy)
            return tuple(float(x) for x in base)
        if policy != "inverse_frequency":
            raise ValueError("unknown spectral.mode_weighting: " + repr(policy))
        omega_floor = max(float(cfg.omega_floor), 1.0e-12)
        power = max(float(cfg.low_frequency_power), 0.0)
        inv = np.array(
            [1.0 / ((float(m.omega) + omega_floor) ** power) for m in mode_evals],
            dtype=float,
        )
        total = float(inv.sum())
        if not np.isfinite(total) or total <= 0.0:
            return tuple(1.0 / len(mode_evals) for _ in mode_evals)
        return tuple(float(x / total) for x in inv)

    def components(self, atoms: Atoms) -> AcquisitionBreakdown:
        mean_energy = self.posterior.mean(atoms)
        atom_variances = None
        if self.error_calibration_model and self.error_calibration_apply_strength > 0.0:
            energy_var, atom_variances = self.posterior.variance_components(atoms)
        else:
            energy_var = self.posterior.variance(atoms)
        mode_evals = self._mode_metrics(atoms, mean_energy=mean_energy)

        legacy_weights = self._effective_mode_weights(mode_evals)
        spectral_weights = (
            self._spectral_mode_weights(mode_evals)
            if bool(self.config.spectral.enabled)
            else tuple()
        )
        spectral_by_index = {
            int(mode.index): float(weight)
            for mode, weight in zip(mode_evals, spectral_weights)
        }
        annotated_modes: List[ModeEvaluation] = []
        force_risk = 0.0
        legacy_frequency_risk = 0.0
        spectral_frequency_risk = 0.0
        anh_risk = 0.0
        for weight, mode in zip(legacy_weights, mode_evals):
            force_risk += float(weight) * self._phi(mode.force_std / self.reference_scales["force"])
            legacy_frequency_risk += float(weight) * self._phi(mode.omega_std / self.reference_scales["omega"])
            anh_risk += float(weight) * self._phi(mode.anharmonicity / self.reference_scales["anh"]) * self._phi(mode.anharmonicity_std / self.reference_scales["anh_std"])
        negative_curvature_penalty = self._negative_curvature_penalty(
            mode_evals,
            legacy_weights,
        )
        spectral_scale = self._reference_scale("spectral", self.reference_scales["omega"])
        for mode in mode_evals:
            spectral_weight = spectral_by_index.get(int(mode.index), 0.0)
            observable = spectral_weight * self._phi(mode.omega_std / spectral_scale)
            spectral_frequency_risk += observable
            annotated_modes.append(
                replace(
                    mode,
                    spectral_weight=float(spectral_weight),
                    frequency_observable_score=float(observable),
                )
            )
        mode_evals = tuple(annotated_modes)

        spectral_mode = str(getattr(self.config.spectral, "mode", "blend"))
        fallback_reasons: List[str] = []
        if (
            bool(self.config.spectral.enabled)
            and spectral_mode == "blend"
        ):
            frequency_risk = float(spectral_frequency_risk)
            frequency_contribution = (
                float(self.config.spectral.lambda_spectral) * frequency_risk
            )
        else:
            if not bool(self.config.spectral.enabled) or spectral_mode == "off":
                fallback_reasons.append("spectral_frequency_disabled")
            elif spectral_mode == "record_only":
                fallback_reasons.append("spectral_frequency_record_only")
            else:
                fallback_reasons.append("spectral_frequency_unknown_mode_fallback_legacy")
            frequency_risk = float(legacy_frequency_risk)
            frequency_contribution = (
                self.config.weights.lambda_frequency * frequency_risk
            )

        raw_energy_risk = self._phi(energy_var / self.reference_scales["energy"])
        energy_risk = raw_energy_risk
        banded_energy_risk = None
        calibrated_error = None
        calibration_applied = False
        if self.error_calibration_model and self.error_calibration_apply_strength > 0.0:
            atom_types = [str(a.type) for a in atoms]
            calibrated_error = _lookup_calibrated_error(
                self.error_calibration_model,
                atom_variances,
                atom_types,
                float(energy_var),
            )
            if calibrated_error is not None:
                scale = _finite_positive_float(
                    self.error_calibration_model.get("reference_error_ha")
                    if isinstance(self.error_calibration_model, Mapping)
                    else None,
                    self._reference_scale("calibrated_error", self.reference_scales["energy"]),
                )
                calibrated_risk, banded_energy_risk, energy_reasons = self._energy_utility(
                    float(calibrated_error),
                    float(scale),
                )
                fallback_reasons.extend(energy_reasons)
                strength = float(self.error_calibration_apply_strength)
                energy_risk = (1.0 - strength) * raw_energy_risk + strength * calibrated_risk
                calibration_applied = True
            else:
                fallback_reasons.append("calibrated_energy_missing_lookup")
                if not bool(self.config.calibrated_energy.fallback_to_raw_variance):
                    energy_risk = 0.0
                    fallback_reasons.append("calibrated_energy_raw_variance_fallback_disabled")
        elif self.config.calibrated_energy.utility == "banded":
            fallback_reasons.append("calibrated_energy_not_active")
            if not bool(self.config.calibrated_energy.fallback_to_raw_variance):
                energy_risk = 0.0
                fallback_reasons.append("calibrated_energy_raw_variance_fallback_disabled")
        distance_penalty = whitened_distance_squared(self.subspace, atoms, self.config.subspace.covariance_regularization)
        chemistry_penalty = chemistry_barrier_value(atoms, self.barrier_state, mean_energy)
        fullspace = self._fullspace_confinement_metrics(atoms)
        fallback_reasons.extend(str(r) for r in fullspace.get("fallback_reasons", []) or [])
        residual_penalty = float(fullspace.get("residual_penalty", 0.0) or 0.0)
        rmsd_penalty = float(fullspace.get("rmsd_penalty", 0.0) or 0.0)

        informativeness_score = (
            self.config.weights.lambda_force * force_risk
            + frequency_contribution
            + self.config.weights.lambda_anharmonic * anh_risk
            + self.config.weights.lambda_energy * energy_risk
        )
        risk_penalty_score = (
            self.config.weights.lambda_distance * distance_penalty
            + float(self.config.fullspace_confinement.lambda_residual) * residual_penalty
            + float(self.config.fullspace_confinement.lambda_rmsd) * rmsd_penalty
            + float(self.config.stencils.lambda_negative_curvature) * negative_curvature_penalty
            + chemistry_penalty
        )
        total = informativeness_score - risk_penalty_score
        return AcquisitionBreakdown(
            total=float(total),
            informativeness_score=float(informativeness_score),
            risk_penalty_score=float(risk_penalty_score),
            energy_risk=float(energy_risk),
            raw_energy_risk=float(raw_energy_risk),
            banded_energy_risk=(
                None if banded_energy_risk is None else float(banded_energy_risk)
            ),
            calibrated_expected_iqa_error_ha=(
                None if calibrated_error is None else float(calibrated_error)
            ),
            calibration_applied=bool(calibration_applied),
            force_risk=float(force_risk),
            frequency_risk=float(frequency_risk),
            spectral_frequency_risk=float(spectral_frequency_risk),
            legacy_frequency_risk=float(legacy_frequency_risk),
            anharmonic_risk=float(anh_risk),
            distance_penalty=float(distance_penalty),
            chemistry_penalty=float(chemistry_penalty),
            fullspace_residual_distance=(
                None
                if fullspace.get("residual_distance") is None
                else float(fullspace["residual_distance"])
            ),
            fullspace_residual_penalty=float(residual_penalty),
            aligned_rmsd_ang=(
                None
                if fullspace.get("aligned_rmsd_ang") is None
                else float(fullspace["aligned_rmsd_ang"])
            ),
            aligned_rmsd_penalty=float(rmsd_penalty),
            negative_curvature_penalty=float(negative_curvature_penalty),
            observable_score=float(informativeness_score),
            outlier_penalty_score=float(risk_penalty_score),
            fallback_reasons=tuple(str(r) for r in fallback_reasons),
            mean_energy=float(mean_energy),
            energy_variance=float(energy_var),
            mode_evaluations=mode_evals,
        )

    def value(self, atoms: Atoms) -> float:
        return self.components(atoms).total

    def gradient(self, atoms: Atoms, mode: str | None = None) -> np.ndarray:
        gradient_mode = mode or self.config.gradient.mode
        if gradient_mode == "cartesian_fd":
            return self._cartesian_finite_difference_gradient(atoms)
        if gradient_mode == "active_fd":
            return self._active_finite_difference_gradient(atoms)
        raise ValueError(f"Unknown gradient mode {gradient_mode!r}")

    def _fd_indices_eps(self, atoms):
        """Shared FD setup: the live coordinate indices (ghost DOFs skipped),
        the flat coordinate vector, the step size, and the original shape.
        Pulled out so the serial loop here and the parallel driver in the hpc
        layer compute exactly the same components."""
        gconfig = self.config.gradient
        # optional floor on the FD step: too small a step loses precision to
        # cancellation. default 0.0 lets cartesian_step through unchanged.
        eps = float(gconfig.cartesian_step)
        if gconfig.cartesian_step_floor > 0.0:
            eps = max(eps, float(gconfig.cartesian_step_floor))
        coords = np.asarray(atoms.coordinates, dtype=float)
        flat = coords.reshape(-1)
        # optionally skip DOFs of very light (ghost / dummy) atoms -- their FD
        # contributions are just numerical noise. default threshold 0.0 skips
        # nothing.
        skip = np.zeros(flat.size, dtype=bool)
        if gconfig.ghost_mass_threshold > 0.0:
            atom_skip = np.array(
                [float(getattr(a, "mass", 0.0)) < float(gconfig.ghost_mass_threshold) for a in atoms],
                dtype=bool,
            )
            skip = np.repeat(atom_skip, 3)
        indices = [i for i in range(flat.size) if not skip[i]]
        return indices, flat, float(eps), coords.shape

    def _fd_single(self, i, flat, eps, atoms):
        """Central-difference derivative of the acquisition value along
        Cartesian DOF i. Independent of every other i, which is what lets the
        gradient be evaluated in parallel."""
        disp = np.zeros_like(flat)
        disp[i] = eps
        plus = self._atoms_from_flat(flat + disp, atoms)
        minus = self._atoms_from_flat(flat - disp, atoms)
        return (self.value(plus) - self.value(minus)) / (2.0 * eps)

    def _cartesian_finite_difference_gradient(self, atoms: Atoms) -> np.ndarray:
        indices, flat, eps, shape = self._fd_indices_eps(atoms)
        grad = np.zeros(flat.size, dtype=float)
        for i in indices:
            grad[i] = self._fd_single(i, flat, eps, atoms)
        return grad.reshape(shape)

    def _atoms_from_flat(self, flat: np.ndarray, template: Atoms) -> Atoms:
        from .geometry import coordinates_to_atoms

        return coordinates_to_atoms(template, np.asarray(flat, dtype=float).reshape((-1, 3)))

    def _active_finite_difference_gradient(self, atoms: Atoms) -> np.ndarray:
        """Active-mode FD gradient via Moore-Penrose pseudoinverse projection.

        self.mode_directions is M^{-1/2} * basis where basis is
        orthonormal in mass-weighted space, so the columns of D are NOT
        Cartesian-orthonormal, but that is fine: the formula
        'D (D^T D)^{-1} rhs' is the standard Moore-Penrose pseudoinverse,
        valid for any full-rank D regardless of orthonormality. For square
        D this exactly recovers the Cartesian gradient (we verified this
        numerically against analytic potentials in
        test_active_fd_mass_weighted.py). For subset D the result is the
        Cartesian-least-squares projection of the gradient onto the
        subspace, which is what ARIADNE needs (its descent step lives in
        Cartesian coordinates).

        Historical note: gradient was biased toward heavy-atom DOFs and proposed 
        replacing the Cartesian Gram with M^{1/2} D mass-weighted. That changed the
        projection from Cartesian-least-squares to mass-weighted-least-
        squares, which divides by mass for full-rank bases (verified
        analytically). The original code is
        the correct least-squares projection for ARIADNE's Cartesian
        descent target. Heterogeneous-mass systems get an unbiased
        Cartesian gradient.
        """
        eps = self.config.gradient.active_step
        directional_derivs: List[float] = []
        for direction in self.mode_directions:
            plus = self._atoms_from_flat(np.asarray(atoms.coordinates).reshape(-1) + eps * direction, atoms)
            minus = self._atoms_from_flat(np.asarray(atoms.coordinates).reshape(-1) - eps * direction, atoms)
            directional_derivs.append((self.value(plus) - self.value(minus)) / (2.0 * eps))
        D = np.column_stack(self.mode_directions)
        rhs = np.asarray(directional_derivs, dtype=float)
        gram = D.T @ D + self.config.gradient.regularization * np.eye(D.shape[1])
        coeffs = np.linalg.solve(gram, rhs)
        flat_grad = D @ coeffs
        return flat_grad.reshape(np.asarray(atoms.coordinates).shape)
