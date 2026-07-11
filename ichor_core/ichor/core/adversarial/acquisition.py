from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.files.xyz import Trajectory
from ichor.core.models.models import Models

from .barrier import ChemistryBarrierState, build_chemistry_barrier_state, chemistry_barrier_value
from .config import AcquisitionConfig
from .geometry import (
    aligned_mass_weighted_displacement,
    aligned_mass_weighted_rmsd,
    load_seed_atoms,
    load_trajectory,
    select_local_neighbours,
)
from .posterior import TotalEnergyPosterior
from .stencils import (
    directional_all_stencils,
)
from .subspace import (
    LocalSubspace,
    aligned_active_displacement,
    aligned_active_rmsd,
    active_participation_weights,
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
    weak_mode_reliability: float = 1.0
    raw_anharmonic_score: float = 0.0
    capped_anharmonic_score: float = 0.0
    safe_anharmonic_score: float = 0.0
    weak_mode_penalty: float = 0.0
    omega_low_threshold: float = 0.0
    omega_high_threshold: float = 0.0


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
    movement_metric: Optional[str] = None
    movement_rmsd_ang: Optional[float] = None
    movement_progress_ang: Optional[float] = None
    movement_band_min_ang: Optional[float] = None
    movement_band_low_ang: Optional[float] = None
    movement_band_peak_ang: Optional[float] = None
    movement_band_high_ang: Optional[float] = None
    movement_band_max_ang: Optional[float] = None
    movement_utility_score: float = 0.0
    movement_band_score: float = 0.0
    movement_progress_score: float = 0.0
    movement_direction_source: Optional[str] = None
    movement_band_scale_source: Optional[str] = None
    geometry_novelty_scale_angstrom: Optional[float] = None
    n_effective_movement_atoms: Optional[float] = None
    size_normalisation_mode: Optional[str] = None
    negative_curvature_penalty: float = 0.0
    weak_mode_penalty_score: float = 0.0
    anharmonic_risk_raw: float = 0.0
    anharmonic_risk_capped: float = 0.0
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
    application_uncertainty_scale: Optional[float] = None,
) -> Optional[float]:
    if not isinstance(model, Mapping):
        return None
    tables = model.get("tables")
    if not isinstance(tables, Mapping):
        return None

    total_table = tables.get("global_total")
    if isinstance(total_table, Mapping):
        lookup_variance = float(total_variance)
        if str(model.get("uncertainty_axis", "raw")) == "model_normalised":
            scale = _finite_positive_float(application_uncertainty_scale)
            if scale is None:
                return None
            lookup_variance /= float(scale)
        total_value = _lookup_table_error(total_table, lookup_variance)
        if total_value is not None:
            return float(total_value)
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
        self._movement_band_cache: Optional[Dict[str, float]] = None
        self._movement_direction_cache: Optional[Tuple[np.ndarray, str]] = None
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

    def movement_band(self) -> Dict[str, float]:
        cached = getattr(self, "_movement_band_cache", None)
        if cached is not None:
            return dict(cached)
        cfg = self.config.movement_band

        geometry_scale = getattr(cfg, "geometry_novelty_scale_angstrom", None)
        try:
            geometry_scale_value = float(geometry_scale)
        except (TypeError, ValueError):
            geometry_scale_value = math.nan
        if np.isfinite(geometry_scale_value) and geometry_scale_value > 0.0:
            eps = 1.0e-9
            hard_min = max(0.0, float(cfg.hard_min_fraction) * geometry_scale_value)
            low = max(float(cfg.target_low_fraction) * geometry_scale_value, hard_min + eps)
            peak = max(float(cfg.target_peak_fraction) * geometry_scale_value, low + eps)
            high = max(float(cfg.target_high_fraction) * geometry_scale_value, peak + eps)
            hard_max = max(float(cfg.hard_max_fraction) * geometry_scale_value, high + eps)
            band = {
                "local_rmsd_ang": float(geometry_scale_value),
                "min": float(hard_min),
                "low": float(low),
                "peak": float(peak),
                "high": float(high),
                "max": float(hard_max),
                "scale_source": "geometry_novelty",
                "geometry_novelty_scale_angstrom": float(geometry_scale_value),
            }
            self._movement_band_cache = dict(band)
            return band

        values = []
        for neighbour in self.subspace.neighbours:
            try:
                if str(cfg.metric) == "aligned_global_rmsd":
                    value = aligned_mass_weighted_rmsd(self.seed_atoms, neighbour.atoms)
                else:
                    value = aligned_active_rmsd(self.subspace, neighbour.atoms)
            except Exception:
                continue
            if np.isfinite(value) and float(value) > 0.0:
                values.append(float(value))
        if values:
            q = 0.25 if str(cfg.local_statistic) == "p25" else 0.50
            local = _percentile(values, q) or 0.0
        else:
            local = 0.0
        if not np.isfinite(local) or local <= 0.0:
            local = float(cfg.target_peak_floor_ang) / max(float(cfg.target_peak_fraction), 1.0e-12)

        eps = 1.0e-9
        default_band = (
            float(cfg.hard_min_floor_ang),
            float(cfg.target_low_floor_ang),
            float(cfg.target_peak_floor_ang),
            min(float(cfg.target_high_cap_ang), max(float(cfg.target_peak_floor_ang) + 0.020, 0.120)),
            float(cfg.hard_max_cap_ang),
        )
        hard_min = max(float(cfg.hard_min_floor_ang), float(cfg.hard_min_fraction) * local)
        low = max(float(cfg.target_low_floor_ang), float(cfg.target_low_fraction) * local, hard_min + eps)
        peak = max(float(cfg.target_peak_floor_ang), float(cfg.target_peak_fraction) * local, low + eps)
        high = max(float(cfg.target_high_fraction) * local, peak + eps)
        hard_max = max(float(cfg.hard_max_fraction) * local, high + eps)
        hard_max = min(hard_max, float(cfg.hard_max_cap_ang))
        high = min(high, hard_max - eps, float(cfg.target_high_cap_ang))
        peak = min(peak, high - eps)
        low = min(low, peak - eps)
        hard_min = min(hard_min, low - eps)
        if not (hard_min < low < peak < high < hard_max):
            hard_min, low, peak, high, hard_max = default_band

        band = {
            "local_rmsd_ang": float(local),
            "min": float(hard_min),
            "low": float(low),
            "peak": float(peak),
            "high": float(high),
            "max": float(hard_max),
        }
        self._movement_band_cache = dict(band)
        return band

    def movement_distance(self, atoms: Atoms) -> Tuple[float, str]:
        metric = str(getattr(self.config.movement_band, "metric", "aligned_active_rmsd"))
        if metric == "aligned_global_rmsd":
            return float(aligned_mass_weighted_rmsd(self.seed_atoms, atoms)), metric
        try:
            return float(aligned_active_rmsd(self.subspace, atoms)), metric
        except Exception:
            return float(aligned_mass_weighted_rmsd(self.seed_atoms, atoms)), "aligned_global_rmsd_fallback"

    def _movement_delta_and_weights(self, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray]:
        if str(getattr(self.config.movement_band, "metric", "aligned_active_rmsd")) == "aligned_global_rmsd":
            from .geometry import aligned_mass_weighted_displacement

            disp = aligned_mass_weighted_displacement(self.seed_atoms, atoms)
            masses = np.asarray(self.seed_atoms.masses, dtype=float)
            masses = np.where(np.isfinite(masses) & (masses > 0.0), masses, 1.0)
            delta = disp.reshape(-1, 3) / np.sqrt(masses)[:, None]
            return delta.reshape(-1), np.ones(delta.size, dtype=float)
        try:
            return aligned_active_displacement(self.subspace, atoms)
        except Exception:
            from .geometry import aligned_mass_weighted_displacement

            disp = aligned_mass_weighted_displacement(self.seed_atoms, atoms)
            masses = np.asarray(self.seed_atoms.masses, dtype=float)
            masses = np.where(np.isfinite(masses) & (masses > 0.0), masses, 1.0)
            delta = disp.reshape(-1, 3) / np.sqrt(masses)[:, None]
            return delta.reshape(-1), np.ones(delta.size, dtype=float)

    def _base_value(self, atoms: Atoms) -> float:
        return float(self.components(atoms, include_movement=False).total)

    def _base_cartesian_gradient(self, atoms: Atoms) -> np.ndarray:
        indices, flat, eps, shape = self._fd_indices_eps(atoms)
        grad = np.zeros(flat.size, dtype=float)
        for i in indices:
            disp = np.zeros_like(flat)
            disp[i] = eps
            plus = self._atoms_from_flat(flat + disp, atoms)
            minus = self._atoms_from_flat(flat - disp, atoms)
            grad[i] = (self._base_value(plus) - self._base_value(minus)) / (2.0 * eps)
        return grad.reshape(shape)

    def movement_direction(self) -> Tuple[np.ndarray, str]:
        if self._movement_direction_cache is not None:
            direction, source = self._movement_direction_cache
            return direction.copy(), source
        source = "initial_projected_acquisition_gradient"
        direction = None
        if str(getattr(self.config.movement_utility, "direction", source)) == source:
            try:
                raw = np.asarray(self._base_cartesian_gradient(self.seed_atoms), dtype=float).reshape(-1)
                if self.mode_directions:
                    basis = np.column_stack([
                        np.asarray(v, dtype=float).reshape(-1)
                        for v in self.mode_directions
                    ])
                    if basis.shape[0] == raw.size and basis.shape[1] > 0:
                        reg = max(
                            float(self.config.subspace.covariance_regularization),
                            1.0e-12,
                        )
                        gram = basis.T @ basis + reg * np.eye(basis.shape[1])
                        coeff = np.linalg.solve(gram, basis.T @ raw)
                        raw = basis @ coeff
                norm = float(np.linalg.norm(raw))
                if np.isfinite(norm) and norm > 0.0:
                    direction = raw / norm
            except Exception:
                direction = None
        if direction is None:
            source = "dominant_active_mode"
            if self.mode_directions:
                raw = np.asarray(self.mode_directions[0], dtype=float).reshape(-1)
                norm = float(np.linalg.norm(raw))
                if np.isfinite(norm) and norm > 0.0:
                    direction = raw / norm
        if direction is None:
            source = "movement_direction_unavailable"
            direction = np.zeros(3 * len(self.seed_atoms), dtype=float)
        self._movement_direction_cache = (np.asarray(direction, dtype=float), source)
        return np.asarray(direction, dtype=float).copy(), source

    def movement_metrics(self, atoms: Atoms) -> Dict[str, object]:
        cfg = self.config.movement_utility
        band_cfg = self.config.movement_band
        band = self.movement_band()
        distance, metric = self.movement_distance(atoms)
        delta, weights = self._movement_delta_and_weights(atoms)
        direction, source = self.movement_direction()
        if direction.shape[0] != delta.shape[0]:
            direction = np.zeros_like(delta)
            source = "movement_direction_shape_mismatch"
        coord_weights = np.asarray(weights, dtype=float)
        coord_weights = np.where(np.isfinite(coord_weights) & (coord_weights > 0.0), coord_weights, 0.0)
        weighted_delta = delta * np.sqrt(coord_weights)
        weighted_direction = direction * np.sqrt(coord_weights)
        direction_norm = float(np.linalg.norm(weighted_direction))
        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            progress = 0.0
            source = "movement_direction_zero"
        else:
            progress = float(np.dot(weighted_delta, weighted_direction / direction_norm))

        low_soft = max(float(cfg.low_softness_ang), 1.0e-12)
        high_soft = max(float(cfg.high_softness_ang), 1.0e-12)
        band_score = _sigmoid((float(distance) - band["low"]) / low_soft) * _sigmoid((band["high"] - float(distance)) / high_soft)
        progress_score = float(np.tanh(progress / max(band["low"], 1.0e-12)))
        raw = float(cfg.band_fraction) * float(band_score) + float(cfg.progress_fraction) * float(progress_score)
        score = float(cfg.lambda_move) * raw if bool(cfg.enabled) and bool(band_cfg.enabled) else 0.0
        atom_weights = active_participation_weights(self.subspace)
        denom = float(np.sum(np.square(atom_weights)))
        n_eff = None
        if np.isfinite(denom) and denom > 0.0:
            n_eff = float((float(np.sum(atom_weights)) ** 2) / denom)
        return {
            "movement_metric": metric,
            "movement_rmsd_ang": float(distance),
            "movement_progress_ang": float(progress),
            "movement_band_min_ang": float(band["min"]),
            "movement_band_low_ang": float(band["low"]),
            "movement_band_peak_ang": float(band["peak"]),
            "movement_band_high_ang": float(band["high"]),
            "movement_band_max_ang": float(band["max"]),
            "movement_utility_score": float(score),
            "movement_band_score": float(band_score),
            "movement_progress_score": float(progress_score),
            "movement_direction_source": source,
            "movement_band_scale_source": str(band.get("scale_source", "local_motion")),
            "geometry_novelty_scale_angstrom": (
                None
                if band.get("geometry_novelty_scale_angstrom") is None
                else float(band.get("geometry_novelty_scale_angstrom"))
            ),
            "n_effective_movement_atoms": n_eff,
        }

    def gradient_band_probe_atoms(self) -> Tuple[Tuple[str, Atoms], ...]:
        """Return conservative seed displacements along the movement direction."""
        if not bool(getattr(self.config.movement_band, "enabled", True)):
            return tuple()
        direction, _ = self.movement_direction()
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0.0:
            return tuple()
        unit = np.asarray(direction, dtype=float).reshape(-1) / norm
        flat0 = np.asarray(self.seed_atoms.coordinates, dtype=float).reshape(-1)
        probes = []
        band = self.movement_band()
        for label, radius in (
            ("gradient_band_probe_min", band["min"]),
            ("gradient_band_probe_low", band["low"]),
            ("gradient_band_probe_peak", band["peak"]),
        ):
            target = float(radius)
            if not np.isfinite(target) or target <= 0.0:
                continue
            trial_flat = flat0 + target * unit
            atoms = self._atoms_from_flat(trial_flat, self.seed_atoms)
            for _ in range(2):
                current, _ = self.movement_distance(atoms)
                if not np.isfinite(current) or current <= 1.0e-12:
                    break
                trial_flat = flat0 + unit * (target * target / float(current))
                atoms = self._atoms_from_flat(trial_flat, self.seed_atoms)
            probes.append((label, atoms))
        return tuple(probes)

    def _curvature_floor(self, curvature: float) -> float:
        floor = self.config.stencils.curvature_floor
        beta = self.config.stencils.softplus_scale
        positive_curvature = max(0.0, float(curvature))
        return float(
            floor + beta * self._softplus(positive_curvature / beta)
        )

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

    def _weak_mode_thresholds(self) -> Tuple[float, float]:
        cfg = self.config.stencils
        omega_ref = self._reference_scale(
            "omega",
            self.reference_scales.get("omega", float(cfg.curvature_floor) ** 0.5),
        )
        low = max(
            float(cfg.weak_mode_abs_omega_floor),
            float(cfg.weak_mode_omega_low_fraction) * omega_ref,
        )
        high = max(
            low + 1.0e-12,
            float(cfg.weak_mode_omega_high_fraction) * omega_ref,
        )
        return float(low), float(high)

    @staticmethod
    def _smoothstep01(value: float) -> float:
        x = float(np.clip(float(value), 0.0, 1.0))
        return float(x * x * (3.0 - 2.0 * x))

    def _weak_mode_reliability(self, omega: float) -> Tuple[float, float, float]:
        if not bool(getattr(self.config.stencils, "weak_mode_gating_enabled", True)):
            return 1.0, 0.0, 0.0
        low, high = self._weak_mode_thresholds()
        t = (float(omega) - low) / max(high - low, 1.0e-12)
        return self._smoothstep01(t), low, high

    @staticmethod
    def _saturating_cap(value: float, cap: float) -> float:
        value_f = max(0.0, float(value))
        cap_f = float(cap)
        if not np.isfinite(cap_f) or cap_f <= 0.0:
            return value_f
        if value_f > cap_f * 50.0:
            return cap_f
        return float(cap_f * (1.0 - np.exp(-value_f / cap_f)))

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
            bundle = directional_all_stencils(self.posterior, atoms, direction, step_f)
            force_eval = bundle.force
            curvature_eval = bundle.curvature
            cubic_eval = bundle.cubic
            quartic_eval = bundle.quartic

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
            energy_value = float(self.posterior.variance(atoms))
            if (
                bool(getattr(self.config.size_normalisation, "enabled", True))
                and str(getattr(self.config.size_normalisation, "energy_mode", "per_sqrt_atom"))
                == "per_sqrt_atom"
            ):
                energy_value /= max(float(np.sqrt(max(1, len(atoms)))), 1.0e-12)
            energy_vars.append(energy_value)
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
                if float(mode_eval.curvature_mean) >= 0.0:
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
        stable = np.asarray(
            [float(mode.curvature_mean) >= 0.0 for mode in mode_evals],
            dtype=float,
        )
        policy = str(getattr(cfg, "mode_weighting", "inverse_frequency"))
        if policy == "variance":
            raw = (
                np.asarray(self.subspace.mode_weights[: len(mode_evals)], dtype=float)
                * stable
            )
            total = float(raw.sum())
            if not np.isfinite(total) or total <= 0.0:
                return tuple(0.0 for _ in mode_evals)
            return tuple(float(x / total) for x in raw)
        if policy == "uniform":
            total = float(stable.sum())
            if total <= 0.0:
                return tuple(0.0 for _ in mode_evals)
            return tuple(float(x / total) for x in stable)
        if policy != "inverse_frequency":
            raise ValueError("unknown spectral.mode_weighting: " + repr(policy))
        omega_floor = max(float(cfg.omega_floor), 1.0e-12)
        power = max(float(cfg.low_frequency_power), 0.0)
        inv = np.array(
            [1.0 / ((float(m.omega) + omega_floor) ** power) for m in mode_evals],
            dtype=float,
        ) * stable
        total = float(inv.sum())
        if not np.isfinite(total) or total <= 0.0:
            return tuple(0.0 for _ in mode_evals)
        return tuple(float(x / total) for x in inv)

    def components(
        self,
        atoms: Atoms,
        *,
        include_movement: bool = True,
        objective: str = "full",
    ) -> AcquisitionBreakdown:
        objective = str(objective or "full")
        if objective == "cheap_driver":
            return self._driver_components(atoms, include_movement=include_movement)
        if objective != "full":
            raise ValueError("Unknown acquisition objective " + repr(objective))
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
        anh_risk_raw = 0.0
        anh_risk_capped = 0.0
        weak_mode_penalty_score = 0.0
        mode_diagnostics: Dict[int, Dict[str, float]] = {}
        gating_enabled = bool(
            getattr(self.config.stencils, "weak_mode_gating_enabled", True)
        )
        for weight, mode in zip(legacy_weights, mode_evals):
            force_risk += float(weight) * self._phi(mode.force_std / self.reference_scales["force"])
            curvature_informative = float(mode.curvature_mean) >= 0.0
            if curvature_informative:
                legacy_frequency_risk += float(weight) * self._phi(
                    mode.omega_std / self.reference_scales["omega"]
                )
            reliability, omega_low, omega_high = self._weak_mode_reliability(mode.omega)
            raw_anh = 0.0
            if curvature_informative:
                raw_anh = (
                    self._phi(mode.anharmonicity / self.reference_scales["anh"])
                    * self._phi(mode.anharmonicity_std / self.reference_scales["anh_std"])
                )
            if gating_enabled:
                capped_anh = self._saturating_cap(
                    raw_anh,
                    self.config.stencils.max_anharmonic_mode_score,
                )
                safe_anh = float(reliability) * capped_anh
                weak_penalty = (
                    (1.0 - float(reliability))
                    * float(self.config.stencils.weak_mode_penalty)
                ) if curvature_informative else 0.0
            else:
                capped_anh = float(raw_anh)
                safe_anh = float(raw_anh)
                weak_penalty = 0.0
            anh_risk_raw += float(weight) * float(raw_anh)
            anh_risk_capped += float(weight) * float(safe_anh)
            weak_mode_penalty_score += float(weight) * float(weak_penalty)
            mode_diagnostics[int(mode.index)] = {
                "weak_mode_reliability": float(reliability),
                "raw_anharmonic_score": float(raw_anh),
                "capped_anharmonic_score": float(capped_anh),
                "safe_anharmonic_score": float(safe_anh),
                "weak_mode_penalty": float(weak_penalty),
                "omega_low_threshold": float(omega_low),
                "omega_high_threshold": float(omega_high),
            }
        if gating_enabled:
            anh_risk = self._saturating_cap(
                anh_risk_capped,
                self.config.stencils.max_anharmonic_total_score,
            )
        else:
            anh_risk = float(anh_risk_raw)
        negative_curvature_penalty = self._negative_curvature_penalty(
            mode_evals,
            legacy_weights,
        )
        spectral_scale = self._reference_scale("spectral", self.reference_scales["omega"])
        for mode in mode_evals:
            spectral_weight = spectral_by_index.get(int(mode.index), 0.0)
            mode_diag = mode_diagnostics.get(int(mode.index), {})
            reliability = float(mode_diag.get("weak_mode_reliability", 1.0))
            observable = (
                reliability
                * spectral_weight
                * self._phi(mode.omega_std / spectral_scale)
            ) if float(mode.curvature_mean) >= 0.0 else 0.0
            spectral_frequency_risk += observable
            annotated_modes.append(
                replace(
                    mode,
                    spectral_weight=float(spectral_weight),
                    frequency_observable_score=float(observable),
                    weak_mode_reliability=float(reliability),
                    raw_anharmonic_score=float(
                        mode_diag.get("raw_anharmonic_score", 0.0)
                    ),
                    capped_anharmonic_score=float(
                        mode_diag.get("capped_anharmonic_score", 0.0)
                    ),
                    safe_anharmonic_score=float(
                        mode_diag.get("safe_anharmonic_score", 0.0)
                    ),
                    weak_mode_penalty=float(
                        mode_diag.get("weak_mode_penalty", 0.0)
                    ),
                    omega_low_threshold=float(
                        mode_diag.get("omega_low_threshold", 0.0)
                    ),
                    omega_high_threshold=float(
                        mode_diag.get("omega_high_threshold", 0.0)
                    ),
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

        norm_enabled = bool(getattr(self.config.size_normalisation, "enabled", True))
        n_atoms = max(1, len(atoms))
        energy_norm = float(np.sqrt(float(n_atoms))) if (
            norm_enabled
            and str(getattr(self.config.size_normalisation, "energy_mode", "per_sqrt_atom"))
            == "per_sqrt_atom"
        ) else 1.0
        energy_value = float(energy_var) / max(energy_norm, 1.0e-12)
        energy_scale = self.reference_scales["energy"]
        raw_energy_risk = self._phi(energy_value / max(energy_scale, 1.0e-12))
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
                application_uncertainty_scale=(
                    float(self.reference_scales["energy"]) * float(energy_norm)
                ),
            )
            if calibrated_error is not None:
                scale = _finite_positive_float(
                    self.error_calibration_model.get("reference_error_ha")
                    if isinstance(self.error_calibration_model, Mapping)
                    else None,
                    self._reference_scale("calibrated_error", self.reference_scales["energy"]) / max(energy_norm, 1.0e-12),
                )
                calibrated_risk, banded_energy_risk, energy_reasons = self._energy_utility(
                    float(calibrated_error) / max(energy_norm, 1.0e-12),
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
        if (
            norm_enabled
            and str(getattr(self.config.size_normalisation, "whitened_distance_mode", "per_subspace_dim"))
            == "per_subspace_dim"
        ):
            dimension = getattr(self.subspace, "dimension", None)
            if dimension is None:
                basis = getattr(self.subspace, "basis", None)
                dimension = (
                    int(np.asarray(basis).shape[1])
                    if basis is not None and np.asarray(basis).ndim == 2
                    else 1
                )
            distance_penalty = float(distance_penalty) / max(1, int(dimension))
        chemistry_penalty = chemistry_barrier_value(
            atoms,
            self.barrier_state,
            mean_energy,
            normalisation_mode=(
                str(getattr(self.config.size_normalisation, "chemistry_barrier_mode", "family_mean"))
                if norm_enabled else "raw_sum"
            ),
        )
        fullspace = self._fullspace_confinement_metrics(atoms)
        fallback_reasons.extend(str(r) for r in fullspace.get("fallback_reasons", []) or [])
        residual_penalty = float(fullspace.get("residual_penalty", 0.0) or 0.0)
        rmsd_penalty = float(fullspace.get("rmsd_penalty", 0.0) or 0.0)
        movement = self.movement_metrics(atoms) if include_movement else {}
        movement_score = float(movement.get("movement_utility_score", 0.0) or 0.0)

        informativeness_score = (
            self.config.weights.lambda_force * force_risk
            + frequency_contribution
            + self.config.weights.lambda_anharmonic * anh_risk
            + self.config.weights.lambda_energy * energy_risk
            + movement_score
        )
        risk_penalty_score = (
            self.config.weights.lambda_distance * distance_penalty
            + float(self.config.fullspace_confinement.lambda_residual) * residual_penalty
            + float(self.config.fullspace_confinement.lambda_rmsd) * rmsd_penalty
            + float(self.config.stencils.lambda_negative_curvature) * negative_curvature_penalty
            + float(weak_mode_penalty_score)
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
            movement_metric=(
                None if not movement else str(movement.get("movement_metric"))
            ),
            movement_rmsd_ang=(
                None if not movement else float(movement.get("movement_rmsd_ang"))
            ),
            movement_progress_ang=(
                None if not movement else float(movement.get("movement_progress_ang"))
            ),
            movement_band_min_ang=(
                None if not movement else float(movement.get("movement_band_min_ang"))
            ),
            movement_band_low_ang=(
                None if not movement else float(movement.get("movement_band_low_ang"))
            ),
            movement_band_peak_ang=(
                None if not movement else float(movement.get("movement_band_peak_ang"))
            ),
            movement_band_high_ang=(
                None if not movement else float(movement.get("movement_band_high_ang"))
            ),
            movement_band_max_ang=(
                None if not movement else float(movement.get("movement_band_max_ang"))
            ),
            movement_utility_score=float(movement_score),
            movement_band_score=float(movement.get("movement_band_score", 0.0) or 0.0),
            movement_progress_score=float(movement.get("movement_progress_score", 0.0) or 0.0),
            movement_direction_source=(
                None if not movement else str(movement.get("movement_direction_source"))
            ),
            movement_band_scale_source=(
                None if not movement else str(movement.get("movement_band_scale_source"))
            ),
            geometry_novelty_scale_angstrom=(
                None
                if not movement or movement.get("geometry_novelty_scale_angstrom") is None
                else float(movement.get("geometry_novelty_scale_angstrom"))
            ),
            n_effective_movement_atoms=(
                None
                if not movement or movement.get("n_effective_movement_atoms") is None
                else float(movement.get("n_effective_movement_atoms"))
            ),
            size_normalisation_mode=(
                "enabled" if norm_enabled else "disabled"
            ),
            negative_curvature_penalty=float(negative_curvature_penalty),
            weak_mode_penalty_score=float(weak_mode_penalty_score),
            anharmonic_risk_raw=float(anh_risk_raw),
            anharmonic_risk_capped=float(anh_risk_capped),
            observable_score=float(informativeness_score),
            outlier_penalty_score=float(risk_penalty_score),
            fallback_reasons=tuple(str(r) for r in fallback_reasons),
            mean_energy=float(mean_energy),
            energy_variance=float(energy_var),
            mode_evaluations=mode_evals,
        )

    def _driver_components(
        self,
        atoms: Atoms,
        *,
        include_movement: bool = True,
    ) -> AcquisitionBreakdown:
        """Cheap objective used only to drive ARIADNE steps.

        Full landing selection still calls objective="full". This omits the
        directional force/frequency/anharmonic stencils and keeps only the
        lower-cost energy uncertainty, movement, distance, full-space, and
        chemistry terms.
        """
        cfg = self.config.driver
        if bool(getattr(cfg, "include_stencils", False)):
            return self.components(atoms, include_movement=include_movement, objective="full")
        mean_energy = self.posterior.mean(atoms)
        atom_variances = None
        if self.error_calibration_model and self.error_calibration_apply_strength > 0.0:
            energy_var, atom_variances = self.posterior.variance_components(atoms)
        else:
            energy_var = self.posterior.variance(atoms)

        norm_enabled = bool(getattr(self.config.size_normalisation, "enabled", True))
        n_atoms = max(1, len(atoms))
        energy_norm = float(np.sqrt(float(n_atoms))) if (
            norm_enabled
            and str(getattr(self.config.size_normalisation, "energy_mode", "per_sqrt_atom"))
            == "per_sqrt_atom"
        ) else 1.0
        energy_value = float(energy_var) / max(energy_norm, 1.0e-12)
        energy_scale = self.reference_scales["energy"]
        raw_energy_risk = self._phi(energy_value / max(energy_scale, 1.0e-12))
        energy_risk = raw_energy_risk
        banded_energy_risk = None
        calibrated_error = None
        calibration_applied = False
        fallback_reasons: List[str] = ["cheap_driver_omits_stencils"]
        if self.error_calibration_model and self.error_calibration_apply_strength > 0.0:
            atom_types = [str(a.type) for a in atoms]
            calibrated_error = _lookup_calibrated_error(
                self.error_calibration_model,
                atom_variances,
                atom_types,
                float(energy_var),
                application_uncertainty_scale=(
                    float(self.reference_scales["energy"]) * float(energy_norm)
                ),
            )
            if calibrated_error is not None:
                scale = _finite_positive_float(
                    self.error_calibration_model.get("reference_error_ha")
                    if isinstance(self.error_calibration_model, Mapping)
                    else None,
                    self._reference_scale("calibrated_error", self.reference_scales["energy"]) / max(energy_norm, 1.0e-12),
                )
                calibrated_risk, banded_energy_risk, energy_reasons = self._energy_utility(
                    float(calibrated_error) / max(energy_norm, 1.0e-12),
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

        distance_penalty = whitened_distance_squared(
            self.subspace,
            atoms,
            self.config.subspace.covariance_regularization,
        )
        if (
            norm_enabled
            and str(getattr(self.config.size_normalisation, "whitened_distance_mode", "per_subspace_dim"))
            == "per_subspace_dim"
        ):
            distance_penalty = float(distance_penalty) / max(1, int(self.subspace.dimension))
        include_chemistry_angles = str(getattr(cfg, "gradient_backend", "fd")) != "hybrid_geometry"
        if not include_chemistry_angles:
            fallback_reasons.append("cheap_driver_hybrid_omits_angle_barrier")
        chemistry_penalty = chemistry_barrier_value(
            atoms,
            self.barrier_state,
            mean_energy,
            normalisation_mode=(
                str(getattr(self.config.size_normalisation, "chemistry_barrier_mode", "family_mean"))
                if norm_enabled else "raw_sum"
            ),
            include_angles=include_chemistry_angles,
        )
        fullspace = self._fullspace_confinement_metrics(atoms)
        fallback_reasons.extend(str(r) for r in fullspace.get("fallback_reasons", []) or [])
        residual_penalty = float(fullspace.get("residual_penalty", 0.0) or 0.0)
        rmsd_penalty = float(fullspace.get("rmsd_penalty", 0.0) or 0.0)
        movement = self.movement_metrics(atoms) if include_movement else {}
        movement_score = float(movement.get("movement_utility_score", 0.0) or 0.0)

        fullspace_penalty = (
            float(self.config.fullspace_confinement.lambda_residual) * residual_penalty
            + float(self.config.fullspace_confinement.lambda_rmsd) * rmsd_penalty
        )
        informativeness_score = (
            float(cfg.lambda_energy) * float(energy_risk)
            + float(cfg.lambda_movement) * movement_score
        )
        risk_penalty_score = (
            float(cfg.lambda_distance) * float(distance_penalty)
            + float(cfg.lambda_fullspace) * float(fullspace_penalty)
            + float(cfg.lambda_chemistry) * float(chemistry_penalty)
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
            force_risk=0.0,
            frequency_risk=0.0,
            spectral_frequency_risk=0.0,
            legacy_frequency_risk=0.0,
            anharmonic_risk=0.0,
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
            movement_metric=(
                None if not movement else str(movement.get("movement_metric"))
            ),
            movement_rmsd_ang=(
                None if not movement else float(movement.get("movement_rmsd_ang"))
            ),
            movement_progress_ang=(
                None if not movement else float(movement.get("movement_progress_ang"))
            ),
            movement_band_min_ang=(
                None if not movement else float(movement.get("movement_band_min_ang"))
            ),
            movement_band_low_ang=(
                None if not movement else float(movement.get("movement_band_low_ang"))
            ),
            movement_band_peak_ang=(
                None if not movement else float(movement.get("movement_band_peak_ang"))
            ),
            movement_band_high_ang=(
                None if not movement else float(movement.get("movement_band_high_ang"))
            ),
            movement_band_max_ang=(
                None if not movement else float(movement.get("movement_band_max_ang"))
            ),
            movement_utility_score=float(movement_score),
            movement_band_score=float(movement.get("movement_band_score", 0.0) or 0.0),
            movement_progress_score=float(movement.get("movement_progress_score", 0.0) or 0.0),
            movement_direction_source=(
                None if not movement else str(movement.get("movement_direction_source"))
            ),
            movement_band_scale_source=(
                None if not movement else str(movement.get("movement_band_scale_source"))
            ),
            geometry_novelty_scale_angstrom=(
                None
                if not movement or movement.get("geometry_novelty_scale_angstrom") is None
                else float(movement.get("geometry_novelty_scale_angstrom"))
            ),
            n_effective_movement_atoms=(
                None
                if not movement or movement.get("n_effective_movement_atoms") is None
                else float(movement.get("n_effective_movement_atoms"))
            ),
            size_normalisation_mode=("enabled" if norm_enabled else "disabled"),
            negative_curvature_penalty=0.0,
            weak_mode_penalty_score=0.0,
            anharmonic_risk_raw=0.0,
            anharmonic_risk_capped=0.0,
            observable_score=float(informativeness_score),
            outlier_penalty_score=float(risk_penalty_score),
            fallback_reasons=tuple(str(r) for r in fallback_reasons),
            mean_energy=float(mean_energy),
            energy_variance=float(energy_var),
            mode_evaluations=(),
        )

    def value(self, atoms: Atoms, *, objective: str = "full") -> float:
        return self.components(atoms, objective=objective).total

    def _driver_energy_value(self, atoms: Atoms) -> float:
        """Cheap-driver energy channel only.

        This deliberately keeps the posterior-derived channel finite
        differenced. Geometry penalties use the hybrid analytic path below.
        """
        parts = self._driver_components(atoms, include_movement=False)
        return float(self.config.driver.lambda_energy) * float(parts.energy_risk) - (
            float(self.config.driver.lambda_chemistry)
            * self._energy_cap_barrier_penalty(float(parts.mean_energy))
        )

    def _value_for_objective(self, atoms: Atoms, objective: str = "full") -> float:
        """Call value with objective while preserving old full-objective shims."""
        objective = str(objective or "full")
        try:
            return float(self.value(atoms, objective=objective))
        except TypeError:
            if objective == "full":
                return float(self.value(atoms))
            raise

    def gradient(
        self,
        atoms: Atoms,
        mode: str | None = None,
        *,
        objective: str = "full",
    ) -> np.ndarray:
        gradient_mode = mode or self.config.gradient.mode
        if (
            str(objective or "full") == "cheap_driver"
            and str(getattr(self.config.driver, "gradient_backend", "fd")) == "hybrid_geometry"
            and not bool(getattr(self.config.driver, "include_stencils", False))
        ):
            return self._hybrid_driver_geometry_gradient(atoms, mode=gradient_mode)
        if gradient_mode == "cartesian_fd":
            return self._cartesian_finite_difference_gradient(
                atoms,
                objective=objective,
            )
        if gradient_mode == "active_fd":
            return self._active_finite_difference_gradient(
                atoms,
                objective=objective,
            )
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

    def _fd_single(self, i, flat, eps, atoms, *, objective: str = "full"):
        """Central-difference derivative of the acquisition value along
        Cartesian DOF i. Independent of every other i, which is what lets the
        gradient be evaluated in parallel."""
        disp = np.zeros_like(flat)
        disp[i] = eps
        plus = self._atoms_from_flat(flat + disp, atoms)
        minus = self._atoms_from_flat(flat - disp, atoms)
        return (
            self._value_for_objective(plus, objective=objective)
            - self._value_for_objective(minus, objective=objective)
        ) / (2.0 * eps)

    def _cartesian_finite_difference_gradient(
        self,
        atoms: Atoms,
        *,
        objective: str = "full",
    ) -> np.ndarray:
        indices, flat, eps, shape = self._fd_indices_eps(atoms)
        grad = np.zeros(flat.size, dtype=float)
        for i in indices:
            grad[i] = self._fd_single(i, flat, eps, atoms, objective=objective)
        return grad.reshape(shape)

    def _atoms_from_flat(self, flat: np.ndarray, template: Atoms) -> Atoms:
        from .geometry import coordinates_to_atoms

        return coordinates_to_atoms(template, np.asarray(flat, dtype=float).reshape((-1, 3)))

    def _active_finite_difference_gradient(
        self,
        atoms: Atoms,
        *,
        objective: str = "full",
    ) -> np.ndarray:
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
        directions, flat, eps, shape = self._active_fd_setup(atoms)
        directional_derivs: List[float] = []
        for direction in directions:
            directional_derivs.append(
                self._active_fd_single(direction, flat, eps, atoms, objective=objective)
            )
        return self._active_fd_project(directions, directional_derivs, shape)

    def _active_fd_setup(self, atoms: Atoms):
        """Shared active-FD setup for the serial core path and HPC workers."""
        coords = np.asarray(atoms.coordinates, dtype=float)
        flat = coords.reshape(-1)
        eps = float(self.config.gradient.active_step)
        directions = tuple(
            np.asarray(direction, dtype=float).reshape(-1)
            for direction in self.mode_directions
        )
        return directions, flat, eps, coords.shape

    def _active_fd_single(self, direction, flat, eps, atoms, *, objective: str = "full"):
        """Central-difference derivative along one active Cartesian direction."""
        direction = np.asarray(direction, dtype=float).reshape(-1)
        plus = self._atoms_from_flat(np.asarray(flat, dtype=float) + eps * direction, atoms)
        minus = self._atoms_from_flat(np.asarray(flat, dtype=float) - eps * direction, atoms)
        return (
            self._value_for_objective(plus, objective=objective)
            - self._value_for_objective(minus, objective=objective)
        ) / (2.0 * eps)

    def _active_fd_project(self, directions, directional_derivs, shape) -> np.ndarray:
        """Project active-direction derivatives back to Cartesian coordinates."""
        D = np.column_stack(directions)
        rhs = np.asarray(directional_derivs, dtype=float)
        gram = D.T @ D + self.config.gradient.regularization * np.eye(D.shape[1])
        coeffs = np.linalg.solve(gram, rhs)
        flat_grad = D @ coeffs
        return flat_grad.reshape(shape)

    def _driver_energy_finite_difference_gradient(self, atoms: Atoms, *, mode: str) -> np.ndarray:
        """Finite-difference only the posterior energy-risk driver channel."""
        if mode == "active_fd":
            directions, flat, eps, shape = self._active_fd_setup(atoms)
            directional_derivs: List[float] = []
            for direction in directions:
                plus = self._atoms_from_flat(flat + eps * direction, atoms)
                minus = self._atoms_from_flat(flat - eps * direction, atoms)
                directional_derivs.append(
                    (self._driver_energy_value(plus) - self._driver_energy_value(minus))
                    / (2.0 * eps)
                )
            return self._active_fd_project(directions, directional_derivs, shape)
        if mode == "cartesian_fd":
            indices, flat, eps, shape = self._fd_indices_eps(atoms)
            grad = np.zeros(flat.size, dtype=float)
            for i in indices:
                disp = np.zeros_like(flat)
                disp[i] = eps
                plus = self._atoms_from_flat(flat + disp, atoms)
                minus = self._atoms_from_flat(flat - disp, atoms)
                grad[i] = (
                    self._driver_energy_value(plus) - self._driver_energy_value(minus)
                ) / (2.0 * eps)
            return grad.reshape(shape)
        raise ValueError(f"Unknown gradient mode {mode!r}")

    def _active_project_cartesian_gradient(self, atoms: Atoms, grad: np.ndarray) -> np.ndarray:
        """Project a Cartesian gradient into the configured active-FD basis."""
        directions, _flat, _eps, shape = self._active_fd_setup(atoms)
        flat_grad = np.asarray(grad, dtype=float).reshape(-1)
        directional_derivs = [float(np.dot(flat_grad, direction)) for direction in directions]
        return self._active_fd_project(directions, directional_derivs, shape)

    def _mass_weighted_displacement_and_masses(self, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray]:
        disp = aligned_mass_weighted_displacement(self.seed_atoms, atoms).reshape(-1)
        masses = np.asarray(self.seed_atoms.masses, dtype=float).reshape(-1)
        masses = np.where(np.isfinite(masses) & (masses > 0.0), masses, 1.0)
        return disp, masses

    def _mass_weighted_gradient_to_cartesian(self, grad_mw: np.ndarray, masses: np.ndarray) -> np.ndarray:
        mass_vec = np.repeat(np.sqrt(np.asarray(masses, dtype=float)), 3)
        return (np.asarray(grad_mw, dtype=float).reshape(-1) * mass_vec).reshape((-1, 3))

    def _whitened_distance_gradient(self, atoms: Atoms) -> np.ndarray:
        disp, masses = self._mass_weighted_displacement_and_masses(atoms)
        basis = np.asarray(self.subspace.basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != disp.size or basis.shape[1] == 0:
            return np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        cov = np.asarray(self.subspace.active_covariance, dtype=float)
        eig_max = float(np.max(np.diag(cov))) if int(self.subspace.dimension) else 1.0
        reg = float(self.config.subspace.covariance_regularization) * max(eig_max, 1.0e-12)
        metric = np.linalg.inv(cov + reg * np.eye(cov.shape[0]))
        xi = basis.T @ disp
        grad_mw = 2.0 * (basis @ (metric @ xi))
        norm_enabled = bool(getattr(self.config.size_normalisation, "enabled", True))
        if (
            norm_enabled
            and str(getattr(self.config.size_normalisation, "whitened_distance_mode", "per_subspace_dim"))
            == "per_subspace_dim"
        ):
            grad_mw = grad_mw / max(1, int(self.subspace.dimension))
        return self._mass_weighted_gradient_to_cartesian(grad_mw, masses)

    def _fullspace_confinement_gradient(self, atoms: Atoms) -> np.ndarray:
        cfg = self.config.fullspace_confinement
        if not bool(getattr(cfg, "enabled", True)):
            return np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        disp, masses = self._mass_weighted_displacement_and_masses(atoms)
        basis = np.asarray(self.subspace.basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != disp.size:
            return np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        residual = disp - basis @ (basis.T @ disp)
        total_mass = float(np.sum(masses))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            total_mass = float(max(1, len(masses)))
        residual_scale = self._reference_scale("residual", self._configured_residual_scale())
        rmsd_scale = self._reference_scale("rmsd", float(cfg.rmsd_scale_ang))
        grad_mw = (
            2.0
            * float(cfg.lambda_residual)
            * residual
            / max(total_mass * residual_scale * residual_scale, 1.0e-24)
        )
        coord_delta = (disp.reshape(-1, 3) / np.sqrt(masses)[:, None]).reshape(-1)
        rmsd_grad = (
            2.0
            * float(cfg.lambda_rmsd)
            * np.repeat(masses, 3)
            * coord_delta
            / max(total_mass * rmsd_scale * rmsd_scale, 1.0e-24)
        )
        return self._mass_weighted_gradient_to_cartesian(grad_mw, masses) + rmsd_grad.reshape((-1, 3))

    @staticmethod
    def _softplus_sq_value(u: float, delta: float, cap: float | None) -> float:
        delta = max(float(delta), 1.0e-12)
        arg = float(u) / delta
        if cap is not None:
            try:
                cap_value = float(cap)
            except (TypeError, ValueError):
                cap_value = np.nan
            if np.isfinite(cap_value):
                arg = min(arg, cap_value)
        soft = SeedLocalAdversarialAcquisition._softplus(arg)
        return float(soft * soft)

    def _energy_cap_barrier_penalty(self, mean_energy: float) -> float:
        state = self.barrier_state
        cfg = state.config
        return float(cfg.energy_cap_lambda) * self._softplus_sq_value(
            float(mean_energy) - float(state.seed_energy) - float(state.energy_cap),
            float(cfg.energy_cap_delta),
            cfg.softplus_cap,
        )

    @staticmethod
    def _softplus_sq_derivative(u: float, delta: float, cap: float | None) -> float:
        delta = max(float(delta), 1.0e-12)
        arg = float(u) / delta
        if cap is not None:
            try:
                cap_value = float(cap)
            except (TypeError, ValueError):
                cap_value = np.nan
            if np.isfinite(cap_value) and arg > cap_value:
                return 0.0
        soft = SeedLocalAdversarialAcquisition._softplus(arg)
        return float(2.0 * soft * _sigmoid(arg) / delta)

    def _add_pair_barrier_gradient(
        self,
        grad: np.ndarray,
        atoms: Atoms,
        pair: Tuple[int, int],
        coeff: float,
        u: float,
        delta: float,
        sign_dudr: float,
        scale: float,
    ) -> None:
        coords = np.asarray(atoms.coordinates, dtype=float)
        i, j = int(pair[0]), int(pair[1])
        if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
            return
        diff = coords[i] - coords[j]
        dist = float(np.linalg.norm(diff))
        if not np.isfinite(dist) or dist <= 1.0e-12:
            return
        deriv = (
            float(coeff)
            * self._softplus_sq_derivative(float(u), float(delta), self.barrier_state.config.softplus_cap)
            * float(sign_dudr)
            * float(scale)
        )
        vec = deriv * diff / dist
        grad[i] += vec
        grad[j] -= vec

    def _pair_chemistry_barrier_gradient(self, atoms: Atoms) -> Tuple[np.ndarray, Tuple[str, ...]]:
        state = self.barrier_state
        cfg = state.config
        grad = np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        normalisation_mode = (
            str(getattr(self.config.size_normalisation, "chemistry_barrier_mode", "family_mean"))
            if bool(getattr(self.config.size_normalisation, "enabled", True))
            else "raw_sum"
        )
        family_mean = normalisation_mode == "family_mean"
        clash_scale = 1.0 / max(1, len(state.safe_nonbonded)) if family_mean else 1.0
        expansion_scale = 1.0 / max(1, len(state.nonbonded_upper)) if family_mean else 1.0
        bond_count = 2 * len(state.bonded_pairs)
        bond_scale = 1.0 / max(1, bond_count) if family_mean else 1.0
        coords = np.asarray(atoms.coordinates, dtype=float)
        for pair, safe_distance in state.safe_nonbonded.items():
            dist = float(np.linalg.norm(coords[int(pair[0])] - coords[int(pair[1])]))
            self._add_pair_barrier_gradient(
                grad,
                atoms,
                pair,
                float(cfg.clash_lambda),
                float(safe_distance) - dist,
                float(cfg.clash_delta),
                -1.0,
                clash_scale,
            )
        for pair, upper_distance in state.nonbonded_upper.items():
            dist = float(np.linalg.norm(coords[int(pair[0])] - coords[int(pair[1])]))
            self._add_pair_barrier_gradient(
                grad,
                atoms,
                pair,
                float(cfg.nonbonded_expansion_lambda),
                dist - float(upper_distance),
                float(cfg.nonbonded_expansion_delta),
                1.0,
                expansion_scale,
            )
        if bool(getattr(cfg, "use_connectivity_barrier", True)):
            for pair in state.bonded_pairs:
                dist = float(np.linalg.norm(coords[int(pair[0])] - coords[int(pair[1])]))
                self._add_pair_barrier_gradient(
                    grad,
                    atoms,
                    pair,
                    float(cfg.bond_lambda),
                    float(state.bond_lower[pair]) - dist,
                    float(cfg.bond_delta),
                    -1.0,
                    bond_scale,
                )
                self._add_pair_barrier_gradient(
                    grad,
                    atoms,
                    pair,
                    float(cfg.bond_lambda),
                    dist - float(state.bond_upper[pair]),
                    float(cfg.bond_delta),
                    1.0,
                    bond_scale,
                )
        reasons: List[str] = [
            "angle_barrier_gradient_not_implemented",
            "energy_cap_barrier_gradient_finite_differenced",
        ]
        return grad, tuple(reasons)

    def _movement_utility_gradient(self, atoms: Atoms) -> np.ndarray:
        cfg = self.config.movement_utility
        band_cfg = self.config.movement_band
        if not bool(cfg.enabled) or not bool(band_cfg.enabled):
            return np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        delta, weights = self._movement_delta_and_weights(atoms)
        delta = np.asarray(delta, dtype=float).reshape(-1)
        coord_weights = np.asarray(weights, dtype=float).reshape(-1)
        coord_weights = np.where(np.isfinite(coord_weights) & (coord_weights > 0.0), coord_weights, 0.0)
        if delta.size != coord_weights.size or delta.size != 3 * len(atoms):
            return np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        denom = float(np.sum(coord_weights) / 3.0)
        if not np.isfinite(denom) or denom <= 0.0:
            return np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
        distance = float(np.sqrt(max(0.0, float(np.sum(coord_weights * np.square(delta))) / denom)))
        direction, _source = self.movement_direction()
        direction = np.asarray(direction, dtype=float).reshape(-1)
        if direction.size != delta.size:
            direction = np.zeros_like(delta)
        weighted_direction = direction * np.sqrt(coord_weights)
        direction_norm = float(np.linalg.norm(weighted_direction))
        band = self.movement_band()
        low_soft = max(float(cfg.low_softness_ang), 1.0e-12)
        high_soft = max(float(cfg.high_softness_ang), 1.0e-12)
        s_low = _sigmoid((distance - float(band["low"])) / low_soft)
        s_high = _sigmoid((float(band["high"]) - distance) / high_soft)
        band_score = float(s_low * s_high)
        d_band_dd = band_score * ((1.0 - s_low) / low_soft - (1.0 - s_high) / high_soft)
        if distance > 1.0e-12:
            d_distance = coord_weights * delta / max(denom * distance, 1.0e-24)
        else:
            d_distance = np.zeros_like(delta)
        if np.isfinite(direction_norm) and direction_norm > 0.0:
            d_progress = coord_weights * direction / direction_norm
            progress = float(np.dot(delta, d_progress))
        else:
            d_progress = np.zeros_like(delta)
            progress = 0.0
        progress_scale = max(float(band["low"]), 1.0e-12)
        progress_score = float(np.tanh(progress / progress_scale))
        d_progress_score = (1.0 - progress_score * progress_score) / progress_scale
        grad_delta = float(cfg.lambda_move) * (
            float(cfg.band_fraction) * d_band_dd * d_distance
            + float(cfg.progress_fraction) * d_progress_score * d_progress
        )
        return grad_delta.reshape((-1, 3))

    def _hybrid_driver_geometry_gradient(self, atoms: Atoms, *, mode: str) -> np.ndarray:
        """Hybrid cheap-driver gradient.

        The posterior energy channel remains finite-differenced. Stable
        geometry-only driver terms use local analytic gradients under the
        same fixed-alignment approximation already used by the ALF projection
        helpers. Any failure falls back to the previous finite-difference
        cheap-driver gradient.
        """
        diag: Dict[str, object] = {
            "driver_gradient_backend": "hybrid_geometry",
            "driver_gradient_mode": str(mode),
            "driver_gradient_fallback": False,
            "driver_gradient_terms": [],
            "driver_gradient_reasons": [],
        }
        self._last_driver_gradient_diagnostics = dict(diag)
        driver = self.config.driver
        try:
            grad = np.zeros_like(np.asarray(atoms.coordinates, dtype=float))
            if bool(getattr(driver, "finite_difference_energy", True)):
                grad += self._driver_energy_finite_difference_gradient(atoms, mode=mode)
                diag["driver_gradient_terms"].append("finite_difference_energy")
            geom_grad = np.zeros_like(grad)
            if bool(getattr(driver, "analytic_movement", True)):
                geom_grad += float(driver.lambda_movement) * self._movement_utility_gradient(atoms)
                diag["driver_gradient_terms"].append("analytic_movement")
            if bool(getattr(driver, "analytic_whitened_distance", True)):
                geom_grad -= float(driver.lambda_distance) * self._whitened_distance_gradient(atoms)
                diag["driver_gradient_terms"].append("analytic_whitened_distance")
            if bool(getattr(driver, "analytic_fullspace_rmsd", True)):
                geom_grad -= float(driver.lambda_fullspace) * self._fullspace_confinement_gradient(atoms)
                diag["driver_gradient_terms"].append("analytic_fullspace_rmsd")
            if bool(getattr(driver, "analytic_pair_barriers", True)):
                pair_grad, pair_reasons = self._pair_chemistry_barrier_gradient(atoms)
                geom_grad -= float(driver.lambda_chemistry) * pair_grad
                diag["driver_gradient_terms"].append("analytic_pair_barriers")
                diag["driver_gradient_reasons"].extend(pair_reasons)
            if mode == "active_fd":
                geom_grad = self._active_project_cartesian_gradient(atoms, geom_grad)
                diag["driver_gradient_terms"].append("active_projection")
            grad += geom_grad
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError("non_finite_hybrid_driver_gradient")
            if bool(getattr(driver, "analytic_validation", False)):
                fd = self._active_finite_difference_gradient(atoms, objective="cheap_driver") if mode == "active_fd" else self._cartesian_finite_difference_gradient(atoms, objective="cheap_driver")
                num = float(np.dot(grad.reshape(-1), fd.reshape(-1)))
                den = float(np.linalg.norm(grad.reshape(-1)) * np.linalg.norm(fd.reshape(-1)))
                cosine = 1.0 if den <= 1.0e-24 else num / den
                diag["driver_gradient_validation_cosine"] = float(cosine)
                if cosine < float(getattr(driver, "analytic_validation_tol_cosine", 0.98)):
                    diag["driver_gradient_reasons"].append("analytic_validation_cosine_below_tolerance")
                    diag["driver_gradient_fallback"] = True
                    diag["driver_gradient_fallback_reason"] = "analytic_validation_cosine_below_tolerance"
                    self._last_driver_gradient_diagnostics = dict(diag)
                    return fd
            self._last_driver_gradient_diagnostics = dict(diag)
            return grad
        except Exception as exc:
            diag["driver_gradient_fallback"] = True
            diag["driver_gradient_reasons"].append(
                "hybrid_geometry_fallback:" + type(exc).__name__
            )
            self._last_driver_gradient_diagnostics = dict(diag)
            if mode == "active_fd":
                return self._active_finite_difference_gradient(atoms, objective="cheap_driver")
            if mode == "cartesian_fd":
                return self._cartesian_finite_difference_gradient(atoms, objective="cheap_driver")
            raise
