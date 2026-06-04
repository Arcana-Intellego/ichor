from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SubspaceConfig:
    """Configuration for building the seed-local dynamical subspace."""

    neighbour_count: int = 50
    neighbour_deduplicate_rmsd: float = 1.0e-3
    variance_capture: float = 0.90
    min_subspace_dim: int = 3
    max_subspace_dim: int = 6
    gaussian_weight_sigma: Optional[float] = None
    covariance_regularization: float = 1.0e-10
    #Fix #1 (opt-in): canonicalise the eigenbasis so degenerate-eigenvalue
    #blocks do not produce visibly different bases across otherwise identical
    #runs. Default False preserves prior behaviour exactly.
    canonicalise_basis: bool = False
    #Two consecutive eigenvalues are treated as degenerate when their gap is
    #below 'degeneracy_tolerance * max(eigenvalues)'. Only consulted when
    #'canonicalise_basis' is True.
    degeneracy_tolerance: float = 1.0e-3
    #Fix #2: how mode weights are recomputed at aggregation time. variance is the
    #legacy behaviour (eigenvalue normalised); inverse_frequency biases toward
    #low-frequency modes (the spectroscopic target); uniform is 1/r.
    mode_weighting_policy: str = "variance"


@dataclass(frozen=True)
class BarrierConfig:
    """Configuration for chemistry-preserving soft barriers."""

    use_connectivity_barrier: bool = True
    nonbonded_clash_scale: float = 0.85
    clash_delta: float = 0.05
    clash_lambda: float = 5.0
    bond_lower_scale: float = 0.80
    bond_upper_scale: float = 1.25
    bond_delta: float = 0.05
    bond_lambda: float = 2.0
    energy_cap_quantile: float = 0.95
    energy_cap_floor: float = 0.05
    energy_cap_delta: float = 0.05
    energy_cap_lambda: float = 1.0
    # cap on the softplus argument (u / delta). without it, very deep
    # clashes make the squared softplus term explode and the whole
    # acquisition starts behaving like a barrier wall instead of a soft
    # penalty. setting cap=10 makes each clash contribute at most
    # softplus(10)**2 ~= 100, a saturating but still meaningful penalty.
    # None means no cap, just plain softplus**2.
    # we default to the safe 10 rather than None -- uncapped is the setting this very comment warns
    # is dangerous, and the adversarial maximiser will happily push into deep clashes, so cap by
    # default and let anyone who really wants the bare wall pass None explicitly. (A57)
    softplus_cap: Optional[float] = 10.0


@dataclass(frozen=True)
class StencilConfig:
    """Configuration for directional energy stencils."""

    step_scale: float = 0.30
    min_step: float = 1.0e-3
    max_step: float = 0.10
    jitter: float = 1.0e-12
    curvature_floor: float = 1.0e-6
    softplus_scale: float = 1.0e-4
    #When True, pick a per-mode FD step eps such that the truncation error
    #eps**2 * |cubic| stays below 1 percent of the gradient magnitude. Cost ~2x
    #stencil evals on iteration 1; subsequent calls re-use the auto-tuned step.
    autotune_from_cubic: bool = False


@dataclass(frozen=True)
class WeightConfig:
    """Relative weighting of the acquisition terms."""

    lambda_force: float = 1.0
    lambda_frequency: float = 1.5
    lambda_anharmonic: float = 1.0
    lambda_energy: float = 0.25
    lambda_distance: float = 1.0


@dataclass(frozen=True)
class GradientConfig:
    """Configuration for the pseudo-force gradient of the acquisition."""

    mode: str = "cartesian_fd"
    cartesian_step: float = 1.0e-4
    active_step: float = 1.0e-3
    regularization: float = 1.0e-10
    #Fix #2a: enforce a minimum FD step magnitude.
    #When > 0, the effective step is max(cartesian_step, cartesian_step_floor).
    #Default 0.0 preserves prior behaviour exactly.
    cartesian_step_floor: float = 0.0
    #Fix #2b (opt-in): skip Cartesian DOF of atoms whose mass is below
    #this threshold (typically ghost / dummy atoms in QM inputs).
    #Default 0.0 preserves prior behaviour exactly.
    ghost_mass_threshold: float = 0.0


@dataclass(frozen=True)
class ReferenceScaleConfig:
    """Configuration for seed-local robust reference scales."""

    max_reference_samples: int = 24
    floor: float = 1.0e-12
    #How often the daemon recomputes the per-iteration reference scales.
    #Valid values: every_iteration | every_n_iterations | never.
    #When external_reference_scales is passed to SeedLocalAdversarialAcquisition,
    #_build_reference_scales is skipped entirely; the daemon-side policy field
    #determines whether the daemon recomputes per iteration.
    refresh_policy: str = "every_n_iterations"
    refresh_period: int = 3


@dataclass(frozen=True)
class AcquisitionConfig:
    """Top-level configuration for the single-model seed-local acquisition."""

    property_name: str = "iqa"
    use_scaled_posterior_covariance: bool = True
    subspace: SubspaceConfig = SubspaceConfig()
    barrier: BarrierConfig = BarrierConfig()
    stencils: StencilConfig = StencilConfig()
    weights: WeightConfig = WeightConfig()
    gradient: GradientConfig = GradientConfig()
    references: ReferenceScaleConfig = ReferenceScaleConfig()
