from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
from ichor.core.atoms import Atoms
from ichor.core.files.xyz import Trajectory
from ichor.core.models.models import Models

from .barrier import ChemistryBarrierState, build_chemistry_barrier_state, chemistry_barrier_value
from .config import AcquisitionConfig
from .geometry import load_seed_atoms, load_trajectory, select_local_neighbours
from .posterior import TotalEnergyPosterior
from .stencils import (
    directional_cubic_stencil,
    directional_curvature_stencil,
    directional_force_stencil,
    directional_quartic_stencil,
)
from .subspace import LocalSubspace, active_coordinates, build_local_subspace, directional_step_sizes, whitened_distance_squared


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
    mode_evaluations: Tuple[ModeEvaluation, ...] = field(default_factory=tuple)


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
    ) -> None:
        """seed_frame_id is the stable trajectory pool position the seed
        was sampled from. Optional (defaults to None to preserve backward compatibility 
        with an earlier patch); the daemon wiring will set it explicitly so
        per-pointdir provenance can record where each seed came from.
        """
        self.config = config or AcquisitionConfig()
        self.models = models
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

    def _curvature_floor(self, curvature: float) -> float:
        floor = self.config.stencils.curvature_floor
        beta = self.config.stencils.softplus_scale
        return float(floor + beta * self._softplus(curvature / beta))

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

        for atoms in sample_atoms:
            energy_vars.append(self.posterior.variance(atoms))
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
        }
        for key, value in scales.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    "reference scale " + key + " must be finite and positive"
                )
        return scales


    def _effective_mode_weights(self, mode_evals):
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

    def components(self, atoms: Atoms) -> AcquisitionBreakdown:
        mean_energy = self.posterior.mean(atoms)
        energy_var = self.posterior.variance(atoms)
        mode_evals = self._mode_metrics(atoms, mean_energy=mean_energy)

        weights = self._effective_mode_weights(mode_evals)
        force_risk = 0.0
        frequency_risk = 0.0
        anh_risk = 0.0
        for weight, mode in zip(weights, mode_evals):
            force_risk += float(weight) * self._phi(mode.force_std / self.reference_scales["force"])
            frequency_risk += float(weight) * self._phi(mode.omega_std / self.reference_scales["omega"])
            anh_risk += float(weight) * self._phi(mode.anharmonicity / self.reference_scales["anh"]) * self._phi(mode.anharmonicity_std / self.reference_scales["anh_std"])

        energy_risk = self._phi(energy_var / self.reference_scales["energy"])
        distance_penalty = whitened_distance_squared(self.subspace, atoms, self.config.subspace.covariance_regularization)
        chemistry_penalty = chemistry_barrier_value(atoms, self.barrier_state, mean_energy)

        informativeness_score = (
            self.config.weights.lambda_force * force_risk
            + self.config.weights.lambda_frequency * frequency_risk
            + self.config.weights.lambda_anharmonic * anh_risk
            + self.config.weights.lambda_energy * energy_risk
        )
        risk_penalty_score = (
            self.config.weights.lambda_distance * distance_penalty
            + chemistry_penalty
        )
        total = informativeness_score - risk_penalty_score
        return AcquisitionBreakdown(
            total=float(total),
            informativeness_score=float(informativeness_score),
            risk_penalty_score=float(risk_penalty_score),
            energy_risk=float(energy_risk),
            force_risk=float(force_risk),
            frequency_risk=float(frequency_risk),
            anharmonic_risk=float(anh_risk),
            distance_penalty=float(distance_penalty),
            chemistry_penalty=float(chemistry_penalty),
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
