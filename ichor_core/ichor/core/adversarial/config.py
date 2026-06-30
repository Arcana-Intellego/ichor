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
    # Canonicalise the eigenbasis so degenerate-eigenvalue blocks do not
    # produce visibly different bases across otherwise identical runs.
    canonicalise_basis: bool = True
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
    nonbonded_expansion_scale: float = 1.80
    nonbonded_expansion_delta: float = 0.10
    nonbonded_expansion_lambda: float = 0.5
    bond_lower_scale: float = 0.80
    bond_upper_scale: float = 1.25
    bond_delta: float = 0.05
    bond_lambda: float = 2.0
    angle_lower_scale: float = 0.65
    angle_upper_scale: float = 1.35
    angle_delta: float = 0.10
    angle_lambda: float = 0.5
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
    negative_curvature_policy: str = "ignore"
    lambda_negative_curvature: float = 1.0
    weak_mode_gating_enabled: bool = True
    weak_mode_omega_low_fraction: float = 0.05
    weak_mode_omega_high_fraction: float = 0.15
    weak_mode_abs_omega_floor: float = 1.0e-4
    weak_mode_penalty: float = 2.0
    max_anharmonic_mode_score: float = 6.0
    max_anharmonic_total_score: float = 15.0


@dataclass(frozen=True)
class WeightConfig:
    """Relative weighting of the acquisition terms."""

    lambda_force: float = 1.0
    lambda_frequency: float = 1.0
    lambda_anharmonic: float = 0.75
    lambda_energy: float = 0.15
    lambda_distance: float = 1.0


@dataclass(frozen=True)
class SpectralConfig:
    """Observable-oriented frequency acquisition settings."""

    enabled: bool = True
    mode: str = "blend"
    mode_weighting: str = "inverse_frequency"
    lambda_spectral: float = 1.5
    omega_floor: float = 1.0e-6
    low_frequency_power: float = 1.0
    max_modes: Optional[int] = None


@dataclass(frozen=True)
class CalibratedEnergyConfig:
    """Energy/error utility settings for calibrated active learning."""

    utility: str = "banded"
    band_low_ha: Optional[float] = None
    band_high_ha: Optional[float] = None
    low_softness_ha: Optional[float] = None
    high_softness_ha: Optional[float] = None
    fallback_to_raw_variance: bool = True


@dataclass(frozen=True)
class FullspaceConfinementConfig:
    """Geometry confinement outside the active local subspace."""

    enabled: bool = True
    lambda_residual: float = 0.5
    lambda_rmsd: float = 0.25
    residual_scale: str = "local_neighbour_median"
    fixed_residual_scale_ang: Optional[float] = None
    rmsd_scale_ang: float = 0.50
    min_residual_scale_ang: float = 1.0e-3
    failure_penalty: float = 1.0e6


@dataclass(frozen=True)
class SizeNormalisationConfig:
    """Controls that keep acquisition terms intensive across system sizes."""

    enabled: bool = True
    energy_mode: str = "per_sqrt_atom"
    whitened_distance_mode: str = "per_subspace_dim"
    chemistry_barrier_mode: str = "family_mean"


@dataclass(frozen=True)
class MovementBandConfig:
    """Aligned-RMSD movement band used by ARIADNE landing selection."""

    enabled: bool = True
    metric: str = "aligned_active_rmsd"
    local_statistic: str = "p25"
    hard_min_floor_ang: float = 0.010
    target_low_floor_ang: float = 0.020
    target_peak_floor_ang: float = 0.035
    target_high_cap_ang: float = 0.120
    hard_max_cap_ang: float = 0.180
    hard_min_fraction: float = 0.10
    target_low_fraction: float = 0.25
    target_peak_fraction: float = 0.40
    target_high_fraction: float = 0.75
    hard_max_fraction: float = 1.25
    # Internal HPC-side override. When set, ARIADNE uses the dimensionless
    # fractions above directly against this campaign geometry-novelty scale.
    geometry_novelty_scale_angstrom: Optional[float] = None


@dataclass(frozen=True)
class MovementUtilityConfig:
    """Bounded directional movement utility around the seed."""

    enabled: bool = True
    direction: str = "initial_projected_acquisition_gradient"
    lambda_move: float = 0.75
    band_fraction: float = 0.75
    progress_fraction: float = 0.25
    low_softness_ang: float = 0.005
    high_softness_ang: float = 0.020


@dataclass(frozen=True)
class DriverConfig:
    """Cheap optimiser-driving acquisition objective.

    When disabled, all existing calls use the full acquisition. When enabled
    by the HPC runner, ARIADNE may use objective="cheap_driver" for gradients
    while landing selection still evaluates objective="full".
    """

    enabled: bool = False
    objective: str = "cheap_driver"
    gradient_backend: str = "fd"
    include_stencils: bool = False
    analytic_movement: bool = True
    analytic_whitened_distance: bool = True
    analytic_pair_barriers: bool = True
    analytic_fullspace_rmsd: bool = True
    finite_difference_energy: bool = True
    analytic_validation: bool = False
    analytic_validation_tol_cosine: float = 0.98
    lambda_energy: float = 1.0
    lambda_movement: float = 1.0
    lambda_distance: float = 1.0
    lambda_fullspace: float = 1.0
    lambda_chemistry: float = 1.0


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
    spectral: SpectralConfig = SpectralConfig()
    calibrated_energy: CalibratedEnergyConfig = CalibratedEnergyConfig()
    fullspace_confinement: FullspaceConfinementConfig = FullspaceConfinementConfig()
    size_normalisation: SizeNormalisationConfig = SizeNormalisationConfig()
    movement_band: MovementBandConfig = MovementBandConfig()
    movement_utility: MovementUtilityConfig = MovementUtilityConfig()
    driver: DriverConfig = DriverConfig()
    gradient: GradientConfig = GradientConfig()
    references: ReferenceScaleConfig = ReferenceScaleConfig()
