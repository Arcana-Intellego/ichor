"""Campaign configuration -- schema v9.

Schema v9 expresses bootstrap and iterative point allocation as exact integer
quotas.  Point allocation is therefore an explicit scientific contract rather
than a cumulative fraction inferred later by FEREBUS staging.
"""
from __future__ import annotations

import math
import re
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .campaign_migrations import (
    CURRENT_SCHEMA_VERSION,
    CampaignMigrationError,
    migrate_campaign_payload,
)
from .config_dataclass import DataclassParseError, parse_dataclass_block
from .geometry_protocol import (
    FULLSPACE_FAILURE_PENALTY,
    FULLSPACE_FIXED_RESIDUAL_SCALE_ANGSTROM,
    FULLSPACE_MIN_RESIDUAL_SCALE_ANGSTROM,
    FULLSPACE_RESIDUAL_SCALE,
    FULLSPACE_RMSD_SCALE_MULTIPLIER,
    MOVEMENT_BAND_ENABLED,
    MOVEMENT_BAND_HARD_MAX_FRACTION,
    MOVEMENT_BAND_HARD_MIN_FRACTION,
    MOVEMENT_BAND_LOCAL_STATISTIC,
    MOVEMENT_BAND_METRIC,
    MOVEMENT_BAND_TARGET_HIGH_FRACTION,
    MOVEMENT_BAND_TARGET_LOW_FRACTION,
    MOVEMENT_BAND_TARGET_PEAK_FRACTION,
    MOVEMENT_UTILITY_BAND_FRACTION,
    MOVEMENT_UTILITY_DIRECTION,
    MOVEMENT_UTILITY_ENABLED,
    MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION,
    MOVEMENT_UTILITY_LAMBDA_MOVE,
    MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION,
    MOVEMENT_UTILITY_PROGRESS_FRACTION,
)


__all__ = [
    "CampaignConfig",
    "CampaignIdentityConfigBlock",
    "PointAllocationConfigBlock",
    "SeedSelectionConfigBlock",
    "AntiOverlapConfigBlock",
    "PhaseBConfigBlock",
    "GeometryNoveltyConfigBlock",
    "FerebusConfigBlock",
    "AcquisitionSubspaceBlock",
    "AcquisitionBarrierBlock",
    "AcquisitionStencilsBlock",
    "AcquisitionWeightsBlock",
    "AcquisitionSpectralBlock",
    "AcquisitionCalibratedEnergyBlock",
    "AcquisitionFullspaceConfinementBlock",
    "AcquisitionSizeNormalisationBlock",
    "AcquisitionDriverBlock",
    "AcquisitionGradientBlock",
    "AcquisitionReferencesBlock",
    "AcquisitionConfigBlock",
    "AriadneConfigBlock",
    "AdversarialSafetyConfigBlock",
    "ErrorCalibrationConfigBlock",
    "QualityGatesConfigBlock",
    "RuntimeConfigBlock",
    "StopConfigBlock",
    "ResourceDefaultsBlock",
    "BackendResourceBlock",
    "GaussianResourceBlock",
    "CONFIG_SCHEMA_VERSION",
    "ConfigValidationError",
    "AimallConfigBlock",
    "VALID_WARMSTART",
    "VALID_DESCRIPTORS",
    "VALID_GEOMETRY_NOVELTY_SCALE_SOURCES",
    "VALID_GEOMETRY_NOVELTY_STATISTICS",
    "VALID_SEED_SELECTION_STRATEGIES",
    "VALID_D_OPTIMAL_DEGENERATE_POLICIES",
    "VALID_GRADIENT_MODES",
    "VALID_MODE_WEIGHTING_POLICIES",
    "VALID_SPECTRAL_MODES",
    "VALID_CALIBRATED_ENERGY_UTILITIES",
    "VALID_SIZE_NORMALISATION_ENERGY_MODES",
    "VALID_SIZE_NORMALISATION_DISTANCE_MODES",
    "VALID_SIZE_NORMALISATION_BARRIER_MODES",
    "VALID_ACQUISITION_DRIVER_OBJECTIVES",
    "VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS",
    "VALID_GAUSSIAN_MEMORY_MODES",
    "VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES",
    "VALID_TRQN_SCALE_MODES",
    "VALID_TRQN_BACKTRANSFORM_MODES",
    "VALID_TRQN_GEODESIC_BT_MODES",
    "VALID_NEGATIVE_CURVATURE_POLICIES",
    "VALID_AIMALL_BOAQ_VALUES",
    "VALID_AIMALL_IASMESH_VALUES",
]


CONFIG_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION


class ConfigValidationError(ValueError):
    """Raised when campaign.yaml fails to validate."""


VALID_WARMSTART = frozenset({"always", "never", "adaptive"})
VALID_DESCRIPTORS = frozenset({
    "rmsd_massweight", "hybrid_alf_rmsd", "acquisition_weighted",
})
VALID_GEOMETRY_NOVELTY_SCALE_SOURCES = frozenset({
    "local_motion", "movement_history", "hybrid",
})
VALID_GEOMETRY_NOVELTY_STATISTICS = frozenset({"p25", "median", "p75"})
VALID_SEED_SELECTION_STRATEGIES = frozenset({"hybrid_variance", "d_optimal"})
VALID_D_OPTIMAL_DEGENERATE_POLICIES = frozenset({"fail", "score_backfill"})
VALID_GRADIENT_MODES = frozenset({"cartesian_fd", "active_fd"})
VALID_MODE_WEIGHTING_POLICIES = frozenset({"variance", "inverse_frequency", "uniform"})
VALID_GRADIENT_PARALLEL_BACKENDS = frozenset({"serial", "thread", "process"})
VALID_ERROR_CALIBRATION_MODES = frozenset({"record_only", "apply_to_acquisition"})
VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES = frozenset(
    {"current", "all", "rolling_normalised"}
)
VALID_SPECTRAL_MODES = frozenset({"off", "record_only", "blend"})
VALID_CALIBRATED_ENERGY_UTILITIES = frozenset({"log", "banded"})
VALID_SIZE_NORMALISATION_ENERGY_MODES = frozenset({"raw_total", "per_sqrt_atom"})
VALID_SIZE_NORMALISATION_DISTANCE_MODES = frozenset({"raw", "per_subspace_dim"})
VALID_SIZE_NORMALISATION_BARRIER_MODES = frozenset({"raw_sum", "family_mean"})
VALID_ACQUISITION_DRIVER_OBJECTIVES = frozenset({"cheap_driver", "full"})
VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS = frozenset({"fd", "hybrid_geometry"})
VALID_GAUSSIAN_MEMORY_MODES = frozenset({"slurm_env", "link0"})
VALID_TRQN_SCALE_MODES = frozenset({
    "off",
    "fixed",
    "adaptive_initial_gradient",
    "adaptive_initial_gradient_rms",
})
VALID_TRQN_BACKTRANSFORM_MODES = frozenset({"geodesic", "newton"})
VALID_TRQN_GEODESIC_BT_MODES = frozenset({"dense", "matrix_free"})
VALID_NEGATIVE_CURVATURE_POLICIES = frozenset({"ignore", "penalise"})
VALID_AIMALL_BOAQ_VALUES = frozenset({
    "auto", "auto_gs2", "auto_gs4",
    "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7", "gs8", "gs9", "gs10",
    "gs15", "gs20", "gs25", "gs30", "gs35", "gs40", "gs45", "gs50", "gs55", "gs60",
    "leb23", "leb25", "leb27", "leb29", "leb31", "leb32",
})
VALID_AIMALL_IASMESH_VALUES = frozenset({
    "sparse", "medium", "fine", "veryfine", "superfine",
})

_SYSTEM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SCHEDULER_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SLURM_MEMORY_RE = re.compile(r"^(?:auto|[1-9][0-9]*[KMGT]?)$")
_GAUSSIAN_MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:[KMGT](?:B|W)?)?$")
_MEMORY_PARSE_RE = re.compile(r"^([1-9][0-9]*)([KMGT]?)([BW]?)$")


# Sentinel used by diff_against_defaults to distinguish "default match" from
# a legitimately empty dict. Module-level so the helper compares by identity
# across recursive calls.
_NODIFF_SENTINEL = object()


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            name + " must be a positive integer, got " + type(value).__name__
        )
    if value <= 0:
        raise ConfigValidationError(name + " must be > 0")


def _validate_positive_walltime(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(name + " must be a positive number of hours")
    if float(value) <= 0.0:
        raise ConfigValidationError(name + " must be > 0")


def _validate_positive_int_or_auto(name: str, value: Any) -> None:
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return
        raise ConfigValidationError(name + " must be a positive integer or 'auto'")
    _validate_positive_int(name, value)


def _validate_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            name + " must be a non-negative integer, got " + type(value).__name__
        )
    if value < 0:
        raise ConfigValidationError(name + " must be >= 0")


def _validate_optional_nonnegative_float(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(name + " must be a number or null")
    if float(value) < 0.0:
        raise ConfigValidationError(name + " must be >= 0")


def _validate_token(name: str, value: Any, pattern: re.Pattern, description: str) -> None:
    if not isinstance(value, str):
        raise ConfigValidationError(name + " must be a string")
    if not value:
        raise ConfigValidationError(name + " must be non-empty")
    if value != value.strip():
        raise ConfigValidationError(name + " must not have leading/trailing whitespace")
    if not pattern.fullmatch(value):
        raise ConfigValidationError(name + " must be " + description + ": " + repr(value))


def _validate_memory(name: str, value: Any, pattern: re.Pattern, description: str) -> None:
    if not isinstance(value, str):
        raise ConfigValidationError(name + " must be a string")
    if not pattern.fullmatch(value):
        raise ConfigValidationError(name + " must use " + description + ": " + repr(value))


def _memory_mebibytes(name: str, value: str, *, gaussian: bool) -> float:
    """Parse the memory syntaxes this daemon accepts into MiB.

    SLURM values are plain K/M/G/T suffixes. Gaussian accepts byte-like MB/GB and word-like MW/GW
    suffixes; for the resource cross-check, one Gaussian word is treated as eight bytes.
    Bare numbers are interpreted as MiB because CSF4 SLURM and our campaign examples use explicit
    G/GB in normal operation and this is the least surprising fallback for validation.
    """
    match = _MEMORY_PARSE_RE.fullmatch(str(value).upper())
    if not match:
        raise ConfigValidationError(name + " has unsupported memory syntax: " + repr(value))
    amount = int(match.group(1))
    scale = match.group(2) or "M"
    suffix = match.group(3)
    multiplier_mib = {
        "K": 1.0 / 1024.0,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
    }[scale]
    mib = float(amount) * multiplier_mib
    if gaussian and suffix == "W":
        mib *= 8.0
    return mib


def diff_against_defaults(config) -> Dict[str, Any]:
    """Recursive diff of 'config' (a CampaignConfig instance) against
    the dataclass defaults from 'CampaignConfig()'.

    Returns a sparse dict containing only the leaves where the operator's
    value differs from the default, plus a 'schema_version' key at the top
    so the result is round-trip-loadable (synchonisable) via 'CampaignConfig.from_dict'.

    Nested dataclass blocks are walked recursively. A sub-block is omitted
    from the output entirely when every leaf in it matches the default.
    """
    defaults_dict = asdict(CampaignConfig())
    full_dict = asdict(config)

    def _diff(current: Any, default: Any) -> Any:
        if isinstance(current, dict) and isinstance(default, dict):
            out: Dict[str, Any] = {}
            for k, v in current.items():
                child = _diff(v, default.get(k, _NODIFF_SENTINEL))
                if child is not _NODIFF_SENTINEL:
                    out[k] = child
            return out if out else _NODIFF_SENTINEL
        if current == default:
            return _NODIFF_SENTINEL
        return current

    diff = _diff(full_dict, defaults_dict)
    if diff is _NODIFF_SENTINEL or not isinstance(diff, dict):
        diff = {}
    # Always include schema_version so the output is self-describing.
    diff["schema_version"] = CONFIG_SCHEMA_VERSION
    return diff


@dataclass
class AcquisitionSubspaceBlock:
    neighbour_count: int = 50
    neighbour_deduplicate_rmsd: float = 1.0e-3
    variance_capture: float = 0.90
    min_subspace_dim: int = 3
    max_subspace_dim: int = 6
    gaussian_weight_sigma: Optional[float] = None
    covariance_regularization: float = 1.0e-10
    canonicalise_basis: bool = True
    degeneracy_tolerance: float = 1.0e-3
    mode_weighting_policy: str = "variance"


@dataclass
class AcquisitionBarrierBlock:
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
    # cap the softplus argument by default (10) rather than None. uncapped, a deep clash makes the
    # squared softplus blow up and swamp the whole acquisition -- the core config comment flags that
    # as dangerous. live runs flow this through to_acquisition_config, so we default it safe. (A57)
    softplus_cap: Optional[float] = 10.0


@dataclass
class AcquisitionStencilsBlock:
    step_scale: float = 0.30
    min_step: float = 1.0e-3
    max_step: float = 0.10
    jitter: float = 1.0e-12
    curvature_floor: float = 1.0e-6
    softplus_scale: float = 1.0e-4
    autotune_from_cubic: bool = True
    negative_curvature_policy: str = "ignore"
    lambda_negative_curvature: float = 1.0
    weak_mode_gating_enabled: bool = True
    weak_mode_omega_low_fraction: float = 0.05
    weak_mode_omega_high_fraction: float = 0.15
    weak_mode_abs_omega_floor: float = 1.0e-4
    weak_mode_penalty: float = 2.0
    max_anharmonic_mode_score: float = 6.0
    max_anharmonic_total_score: float = 15.0


@dataclass
class AcquisitionWeightsBlock:
    lambda_force: float = 1.0
    lambda_frequency: float = 1.0
    lambda_anharmonic: float = 0.75
    lambda_energy: float = 0.15
    lambda_distance: float = 1.0


@dataclass
class AcquisitionSpectralBlock:
    enabled: bool = True
    mode: str = "blend"
    mode_weighting: str = "inverse_frequency"
    lambda_spectral: float = 1.5
    omega_floor: float = 1.0e-6
    low_frequency_power: float = 1.0
    max_modes: Optional[int] = None


@dataclass
class AcquisitionCalibratedEnergyBlock:
    utility: str = "banded"
    band_low_ha: Optional[float] = None
    band_high_ha: Optional[float] = None
    low_softness_ha: Optional[float] = None
    high_softness_ha: Optional[float] = None
    fallback_to_raw_variance: bool = True


@dataclass
class AcquisitionFullspaceConfinementBlock:
    enabled: bool = True
    lambda_residual: float = 0.5
    lambda_rmsd: float = 0.25


@dataclass
class AcquisitionSizeNormalisationBlock:
    enabled: bool = True
    energy_mode: str = "per_sqrt_atom"
    whitened_distance_mode: str = "per_subspace_dim"
    chemistry_barrier_mode: str = "family_mean"


@dataclass
class AcquisitionDriverBlock:
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


@dataclass
class AcquisitionGradientBlock:
    mode: str = "cartesian_fd"
    cartesian_step: float = 1.0e-4
    active_step: float = 1.0e-3
    regularization: float = 1.0e-10
    cartesian_step_floor: float = 0.0
    ghost_mass_threshold: float = 0.0
    max_acquisition_grad_per_ang: float = 50.0


@dataclass
class AcquisitionReferencesBlock:
    max_reference_samples: int = 24
    floor: float = 1.0e-12
    refresh_policy: str = "every_n_iterations"
    refresh_period: int = 3


@dataclass
class AcquisitionConfigBlock:
    property_name: str = "iqa"
    use_scaled_posterior_covariance: bool = True
    allow_uniform_posterior_fallback: bool = False
    subspace: AcquisitionSubspaceBlock = field(default_factory=AcquisitionSubspaceBlock)
    barrier: AcquisitionBarrierBlock = field(default_factory=AcquisitionBarrierBlock)
    stencils: AcquisitionStencilsBlock = field(default_factory=AcquisitionStencilsBlock)
    weights: AcquisitionWeightsBlock = field(default_factory=AcquisitionWeightsBlock)
    spectral: AcquisitionSpectralBlock = field(default_factory=AcquisitionSpectralBlock)
    calibrated_energy: AcquisitionCalibratedEnergyBlock = field(default_factory=AcquisitionCalibratedEnergyBlock)
    fullspace_confinement: AcquisitionFullspaceConfinementBlock = field(default_factory=AcquisitionFullspaceConfinementBlock)
    size_normalisation: AcquisitionSizeNormalisationBlock = field(default_factory=AcquisitionSizeNormalisationBlock)
    driver: AcquisitionDriverBlock = field(default_factory=AcquisitionDriverBlock)
    gradient: AcquisitionGradientBlock = field(default_factory=AcquisitionGradientBlock)
    references: AcquisitionReferencesBlock = field(default_factory=AcquisitionReferencesBlock)


@dataclass
class CampaignIdentityConfigBlock:
    # Short label for the molecular system, used to name per-atom FEREBUS
    # dataset files (<system>_<atom>_TRAINING_SET.csv and friends).
    system_name: str = "SYSTEM"
    max_iterations: int = 50
    source_path: str = "pool.xyz"
    anchor_path: str = "anchor.xyz"
    sampling_aggressiveness: int = 5


@dataclass
class PointAllocationConfigBlock:
    bootstrap_training_size: int = 8
    bootstrap_internal_validation_size: int = 2
    bootstrap_external_validation_size: int = 2
    batch_training_size: int = 3
    batch_internal_validation_size: int = 1
    anchor: bool = False

    @property
    def bootstrap_total_size(self) -> int:
        return (
            int(self.bootstrap_training_size)
            + int(self.bootstrap_internal_validation_size)
            + int(self.bootstrap_external_validation_size)
        )

    @property
    def batch_total_size(self) -> int:
        return int(self.batch_training_size) + int(self.batch_internal_validation_size)


@dataclass
class SeedSelectionConfigBlock:
    n_seeds_per_iteration: int = 50
    bulk_fraction: float = 0.5
    variance_chunk_size: int = 512
    strategy: str = "d_optimal"
    d_optimal_pool_multiplier: int = 8
    d_optimal_jitter: float = 1.0e-12
    d_optimal_novelty_floor: float = 1.0e-12
    d_optimal_score_power: float = 1.0
    d_optimal_degenerate_policy: str = "score_backfill"


@dataclass
class AntiOverlapConfigBlock:
    skip_training_seeds: bool = True
    recent_seeds_cooldown: int = 3
    min_post_ariadne_whitened_distance: float = 0.01
    max_post_ariadne_whitened_distance: float = 10.0
    # when true, seeds flagged moved_too_little / moved_too_far are DROPPED before the expensive QM
    # rather than just recorded. default OFF now: the design doc (section 9) is explicit that the
    # whitened-distance check is a FLAG, not a filter -- dropping on it starves the batch early on
    # (poor model -> ARIADNE barely moves -> nearly everything flags moved_too_little) exactly when
    # you most need the points. near-duplicates get removed instead by Phase-B case (d), via
    # the geometry-novelty Phase B threshold. (A43)
    enforce_post_ariadne: bool = False


@dataclass
class PhaseBConfigBlock:
    descriptor: str = "hybrid_alf_rmsd"
    beta: float = 0.3


@dataclass
class GeometryNoveltyConfigBlock:
    enabled: bool = True
    scale_source: str = "local_motion"
    statistic: str = "median"
    scale_floor_angstrom: float = 1.0e-3
    history_window_iterations: int = 5
    fallback_scale_angstrom: float = 0.05


@dataclass
class FerebusConfigBlock:
    # NOT YET IMPLEMENTED. warmstart (reuse the previous iteration's converged hyperparameters as the
    # initial guess) needs a FEREBUS_CPU change to accept a seed-theta -- there is no config-only way
    # in; see Appendix W of the patch plan. these two are still parsed + validated so existing
    # campaign.yaml and the menu keep working, but nothing consumes them yet, so do not expect any
    # warmstart behaviour from them until the FEREBUS-side feature lands. (A34)
    warmstart: str = "adaptive"
    warmstart_streak: int = 5
    kernel: str = "rbfc_per"
    loss: str = "huber"
    nagents: int = 20
    maxiter: int = 200
    is_constant_noise: bool = True
    scaling: bool = True
    full_ARD: bool = True
    properties: List[str] = field(default_factory=lambda: ["iqa"])


@dataclass
class AriadneConfigBlock:
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


@dataclass
class AdversarialSafetyConfigBlock:
    enabled: bool = True
    reject_unsafe_landings: bool = True
    salvage_safe_iterate: bool = True
    backtrack_to_safe_landing: bool = True
    backtrack_points: int = 16
    allow_seed_fallback: bool = False
    accept_legacy_missing_landing_safety: bool = False
    min_whitened_distance: float = 0.0
    max_whitened_distance: float = 10.0
    enforce_min_whitened_distance: bool = False
    max_predicted_energy_delta_ha: Optional[float] = None
    max_energy_variance: Optional[float] = None
    max_chemistry_penalty: Optional[float] = None
    phase_b_filter_enabled: bool = True
    enforce_movement_band: bool = True
    under_move_retry: bool = True
    reject_under_moved_after_retry: bool = True
    reject_over_moved: bool = True


@dataclass
class ErrorCalibrationConfigBlock:
    enabled: bool = True
    mode: str = "record_only"
    min_records_to_apply: int = 100
    min_model_versions_to_apply: int = 2
    n_bins: int = 10
    min_bin_records: int = 8
    max_records: int = 5000
    max_model_age_iterations: int = 10
    monotone_estimator: bool = True
    quantile: float = 0.75
    apply_strength: float = 0.0
    group_by_atom_type: bool = True
    group_by_landing_policy: bool = False
    model_version_policy: str = "rolling_normalised"
    aggressiveness_match_required: bool = True
    output_units: str = "ha"


@dataclass
class QualityGatesConfigBlock:
    require_readable_aimall_geometry: bool = True
    require_finite_iqa: bool = True
    require_finite_integration_error: bool = True
    max_abs_integration_error: Optional[float] = None
    iqa_energy_recovery_tolerance_ha: Optional[float] = None
    ferebus_min_ext_r2: Optional[float] = None
    ferebus_max_ext_rmse_ha: Optional[float] = None
    ferebus_max_condition_number: Optional[float] = None
    ariadne_max_displacement_ang: Optional[float] = 1.25
    ariadne_min_pair_distance_ang: Optional[float] = 0.60


@dataclass
class RuntimeConfigBlock:
    poll_interval_seconds: int = 60
    poll_interval_idle_seconds: int = 120
    poll_sacct_empty_max_ticks: int = 10
    lease_stale_seconds: int = 900
    postprocess_settle_attempts: int = 3
    postprocess_settle_seconds: int = 10
    transient_phase_retry_max: int = 1
    poll_sacct_unknown_max_ticks: int = 3
    poll_sacct_missing_max_ticks: int = 12
    poll_squeue_inconclusive_max_ticks: int = 10
    failure_threshold_fraction: float = 0.5
    halt_on_tick_exception: bool = True


@dataclass
class StopConfigBlock:
    alpha0_streak_threshold: float = 1.0e-2
    alpha0_streak_length: int = 5
    rel_alpha_improvement_min: float = 0.02
    rel_alpha_improvement_window: int = 3
    min_iterations_before_stop: int = 8


@dataclass
class ResourceDefaultsBlock:
    partition: str = "multicore"
    walltime_hours: Union[int, float] = 24
    cpus_per_task: Union[int, str] = "auto"
    mem_per_cpu: str = "auto"


@dataclass
class BackendResourceBlock:
    partition: Optional[str] = None
    walltime_hours: Optional[Union[int, float]] = None
    cpus_per_task: Optional[Union[int, str]] = None
    mem_per_cpu: Optional[str] = None


@dataclass
class GaussianResourceBlock:
    partition: Optional[str] = None
    walltime_hours: Optional[Union[int, float]] = None
    cpus_per_task: Optional[Union[int, str]] = None
    mem_per_cpu: Optional[str] = None
    memory_mode: str = "slurm_env"
    link0_mem: str = "8GB"
    memory_fraction_of_slurm: float = 0.85


@dataclass
class ResourceConfigBlock:
    # Slurm resources for live phases. Schema v4 groups backend-specific
    # overrides while keeping defaults explicit. A backend value of None means
    # "inherit from resources.defaults".
    defaults: ResourceDefaultsBlock = field(default_factory=ResourceDefaultsBlock)
    polus: BackendResourceBlock = field(default_factory=lambda: BackendResourceBlock(walltime_hours=2))
    gaussian: GaussianResourceBlock = field(default_factory=lambda: GaussianResourceBlock(walltime_hours=24))
    aimall: BackendResourceBlock = field(default_factory=BackendResourceBlock)
    ariadne: BackendResourceBlock = field(default_factory=BackendResourceBlock)
    ferebus: BackendResourceBlock = field(default_factory=BackendResourceBlock)

    array_concurrency_limit: Optional[int] = None
    fail_on_memory_estimate_exceeds_request: bool = True
    # "process" -> node-local process pool sized to the task's cpus-per-task.
    # "serial"  -> force single-core (off-cluster / debugging).
    gradient_parallel_backend: str = "process"

    def _backend_block(self, backend: str):
        return getattr(self, backend)

    @staticmethod
    def _inherit(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def backend_for_phase(self, phase_name: str) -> str:
        if phase_name in ("PHASE_A_POLUS", "PHASE_B_POLUS"):
            return "polus"
        if "GAUSSIAN" in phase_name:
            return "gaussian"
        if "AIMALL" in phase_name:
            return "aimall"
        if phase_name == "ARIADNE_ARRAY":
            return "ariadne"
        if phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
            return "ferebus"
        raise ValueError("unknown live backend phase: " + str(phase_name))

    def partition_for(self, phase_name: str) -> str:
        backend = self.backend_for_phase(phase_name)
        return str(
            self._inherit(
                self._backend_block(backend).partition,
                self.defaults.partition,
            )
        )

    def cpus_for(self, phase_name: str) -> Union[int, str]:
        backend = self.backend_for_phase(phase_name)
        return self._inherit(
            self._backend_block(backend).cpus_per_task,
            self.defaults.cpus_per_task,
        )

    def mem_per_cpu_for(self, phase_name: str) -> str:
        backend = self.backend_for_phase(phase_name)
        return str(
            self._inherit(
                self._backend_block(backend).mem_per_cpu,
                self.defaults.mem_per_cpu,
            )
        )

    def walltime_for(self, phase_name: str) -> Union[int, float]:
        backend = self.backend_for_phase(phase_name)
        return self._inherit(
            self._backend_block(backend).walltime_hours,
            self.defaults.walltime_hours,
        )

    def gaussian_memory_mode_for(self) -> str:
        return str(self.gaussian.memory_mode)

    def gaussian_link0_mem_for(self) -> str:
        return str(self.gaussian.link0_mem)

    def gaussian_memory_fraction_of_slurm_for(self) -> float:
        return float(self.gaussian.memory_fraction_of_slurm)

    # Backward-compatible Python properties for internal call sites and older
    # tests. Schema-v4 YAML still rejects these flat keys.
    @property
    def partition(self) -> str:
        return self.defaults.partition

    @partition.setter
    def partition(self, value: str) -> None:
        self.defaults.partition = value

    @property
    def default_walltime_hours(self) -> Union[int, float]:
        return self.defaults.walltime_hours

    @default_walltime_hours.setter
    def default_walltime_hours(self, value: Union[int, float]) -> None:
        self.defaults.walltime_hours = value

    def _get_walltime(self, backend: str):
        return self._backend_block(backend).walltime_hours

    def _set_walltime(self, backend: str, value) -> None:
        self._backend_block(backend).walltime_hours = value

    def _get_cpus(self, backend: str):
        return self._inherit(
            self._backend_block(backend).cpus_per_task,
            self.defaults.cpus_per_task,
        )

    def _set_cpus(self, backend: str, value) -> None:
        self._backend_block(backend).cpus_per_task = value

    def _get_mem(self, backend: str) -> str:
        return str(
            self._inherit(
                self._backend_block(backend).mem_per_cpu,
                self.defaults.mem_per_cpu,
            )
        )

    def _set_mem(self, backend: str, value: str) -> None:
        self._backend_block(backend).mem_per_cpu = value

    @property
    def polus_walltime_hours(self):
        return self._get_walltime("polus")

    @polus_walltime_hours.setter
    def polus_walltime_hours(self, value):
        self._set_walltime("polus", value)

    @property
    def gaussian_walltime_hours(self):
        return self._get_walltime("gaussian")

    @gaussian_walltime_hours.setter
    def gaussian_walltime_hours(self, value):
        self._set_walltime("gaussian", value)

    @property
    def aimall_walltime_hours(self):
        return self._get_walltime("aimall")

    @aimall_walltime_hours.setter
    def aimall_walltime_hours(self, value):
        self._set_walltime("aimall", value)

    @property
    def ariadne_walltime_hours(self):
        return self._get_walltime("ariadne")

    @ariadne_walltime_hours.setter
    def ariadne_walltime_hours(self, value):
        self._set_walltime("ariadne", value)

    @property
    def ferebus_walltime_hours(self):
        return self._get_walltime("ferebus")

    @ferebus_walltime_hours.setter
    def ferebus_walltime_hours(self, value):
        self._set_walltime("ferebus", value)

    @property
    def polus_cpus_per_task(self):
        return self._get_cpus("polus")

    @polus_cpus_per_task.setter
    def polus_cpus_per_task(self, value):
        self._set_cpus("polus", value)

    @property
    def gaussian_cpus_per_task(self):
        return self._get_cpus("gaussian")

    @gaussian_cpus_per_task.setter
    def gaussian_cpus_per_task(self, value):
        self._set_cpus("gaussian", value)

    @property
    def aimall_cpus_per_task(self):
        return self._get_cpus("aimall")

    @aimall_cpus_per_task.setter
    def aimall_cpus_per_task(self, value):
        self._set_cpus("aimall", value)

    @property
    def ariadne_cpus_per_task(self):
        return self._get_cpus("ariadne")

    @ariadne_cpus_per_task.setter
    def ariadne_cpus_per_task(self, value):
        self._set_cpus("ariadne", value)

    @property
    def ferebus_cpus_per_task(self):
        return self._get_cpus("ferebus")

    @ferebus_cpus_per_task.setter
    def ferebus_cpus_per_task(self, value):
        self._set_cpus("ferebus", value)

    @property
    def polus_mem_per_cpu(self):
        return self._get_mem("polus")

    @polus_mem_per_cpu.setter
    def polus_mem_per_cpu(self, value):
        self._set_mem("polus", value)

    @property
    def gaussian_mem_per_cpu(self):
        return self._get_mem("gaussian")

    @gaussian_mem_per_cpu.setter
    def gaussian_mem_per_cpu(self, value):
        self._set_mem("gaussian", value)

    @property
    def aimall_mem_per_cpu(self):
        return self._get_mem("aimall")

    @aimall_mem_per_cpu.setter
    def aimall_mem_per_cpu(self, value):
        self._set_mem("aimall", value)

    @property
    def ariadne_mem_per_cpu(self):
        return self._get_mem("ariadne")

    @ariadne_mem_per_cpu.setter
    def ariadne_mem_per_cpu(self, value):
        self._set_mem("ariadne", value)

    @property
    def ferebus_mem_per_cpu(self):
        return self._get_mem("ferebus")

    @ferebus_mem_per_cpu.setter
    def ferebus_mem_per_cpu(self, value):
        self._set_mem("ferebus", value)

    @property
    def gaussian_memory_mode(self):
        return self.gaussian.memory_mode

    @gaussian_memory_mode.setter
    def gaussian_memory_mode(self, value):
        self.gaussian.memory_mode = value

    @property
    def gaussian_link0_mem(self):
        return self.gaussian.link0_mem

    @gaussian_link0_mem.setter
    def gaussian_link0_mem(self, value):
        self.gaussian.link0_mem = value

    @property
    def gaussian_memory_fraction_of_slurm(self):
        return self.gaussian.memory_fraction_of_slurm

    @gaussian_memory_fraction_of_slurm.setter
    def gaussian_memory_fraction_of_slurm(self, value):
        self.gaussian.memory_fraction_of_slurm = value


@dataclass
class GaussianConfigBlock:
    # level of theory for the ab-initio training-data calculations. the
    # defaults are a reasonable starting point; set these per system in
    # campaign.yaml. extra_keywords is a space-separated string appended to
    # the route line (the staging layer always adds the wfn output itself).
    method: str = "B3LYP"
    basis_set: str = "aug-cc-pVTZ"
    charge: int = 0
    spin_multiplicity: int = 1
    extra_keywords: str = ""


@dataclass
class AimallConfigBlock:
    encomp: int = 3
    nogui: bool = True
    naat: Union[int, str] = "auto"
    boaq: str = "auto"
    iasmesh: str = "fine"


@dataclass
class CampaignConfig:
    """Top-level campaign configuration (schema v10)."""

    schema_version: int = CONFIG_SCHEMA_VERSION

    campaign: CampaignIdentityConfigBlock = field(
        default_factory=CampaignIdentityConfigBlock
    )
    point_allocation: PointAllocationConfigBlock = field(
        default_factory=PointAllocationConfigBlock
    )
    seed_selection: SeedSelectionConfigBlock = field(
        default_factory=SeedSelectionConfigBlock
    )
    anti_overlap: AntiOverlapConfigBlock = field(
        default_factory=AntiOverlapConfigBlock
    )
    phase_b: PhaseBConfigBlock = field(default_factory=PhaseBConfigBlock)
    geometry_novelty: GeometryNoveltyConfigBlock = field(
        default_factory=GeometryNoveltyConfigBlock
    )
    ferebus: FerebusConfigBlock = field(default_factory=FerebusConfigBlock)
    acquisition: AcquisitionConfigBlock = field(
        default_factory=AcquisitionConfigBlock
    )
    ariadne: AriadneConfigBlock = field(default_factory=AriadneConfigBlock)
    adversarial_safety: AdversarialSafetyConfigBlock = field(
        default_factory=AdversarialSafetyConfigBlock
    )
    error_calibration: ErrorCalibrationConfigBlock = field(
        default_factory=ErrorCalibrationConfigBlock
    )
    quality_gates: QualityGatesConfigBlock = field(
        default_factory=QualityGatesConfigBlock
    )
    runtime: RuntimeConfigBlock = field(default_factory=RuntimeConfigBlock)
    stop: StopConfigBlock = field(default_factory=StopConfigBlock)
    resources: ResourceConfigBlock = field(default_factory=ResourceConfigBlock)
    gaussian: GaussianConfigBlock = field(default_factory=GaussianConfigBlock)
    aimall: AimallConfigBlock = field(default_factory=AimallConfigBlock)

    def __init__(self, **kwargs):
        moved_aliases = {
            "system_name": ("campaign", "system_name"),
            "max_iterations": ("campaign", "max_iterations"),
            "poll_interval_seconds": ("runtime", "poll_interval_seconds"),
            "poll_interval_idle_seconds": ("runtime", "poll_interval_idle_seconds"),
            "poll_sacct_empty_max_ticks": ("runtime", "poll_sacct_empty_max_ticks"),
            "failure_threshold_fraction": ("runtime", "failure_threshold_fraction"),
            "max_acquisition_grad_per_ang": (
                "acquisition.gradient",
                "max_acquisition_grad_per_ang",
            ),
            "max_force_per_atom_ha_per_ang": (
                "acquisition.gradient",
                "max_acquisition_grad_per_ang",
            ),
        }
        alias_values = {}
        for alias, target in moved_aliases.items():
            if alias in kwargs:
                alias_values[target] = kwargs.pop(alias)
        field_names = {f.name for f in fields(type(self))}
        unknown = sorted(str(k) for k in kwargs if str(k) not in field_names)
        if unknown:
            raise TypeError(
                "CampaignConfig got unexpected keyword argument(s): "
                + ", ".join(unknown)
            )
        for f in fields(type(self)):
            if f.name in kwargs:
                value = kwargs.pop(f.name)
            elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                value = f.default_factory()  # type: ignore[misc]
            elif f.default is not MISSING:
                value = f.default
            else:
                raise TypeError("CampaignConfig field has no default: " + f.name)
            setattr(self, f.name, value)
        for (block_path, attr), value in alias_values.items():
            if value is None and attr == "max_acquisition_grad_per_ang":
                continue
            target = self
            for part in str(block_path).split("."):
                target = getattr(target, part)
            setattr(target, attr, value)

    @property
    def system_name(self) -> str:
        return self.campaign.system_name

    @system_name.setter
    def system_name(self, value: str) -> None:
        self.campaign.system_name = value

    @property
    def max_iterations(self) -> int:
        return self.campaign.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self.campaign.max_iterations = int(value)

    @property
    def poll_interval_seconds(self) -> int:
        return self.runtime.poll_interval_seconds

    @poll_interval_seconds.setter
    def poll_interval_seconds(self, value: int) -> None:
        self.runtime.poll_interval_seconds = int(value)

    @property
    def poll_interval_idle_seconds(self) -> int:
        return self.runtime.poll_interval_idle_seconds

    @poll_interval_idle_seconds.setter
    def poll_interval_idle_seconds(self, value: int) -> None:
        self.runtime.poll_interval_idle_seconds = int(value)

    @property
    def poll_sacct_empty_max_ticks(self) -> int:
        return self.runtime.poll_sacct_empty_max_ticks

    @poll_sacct_empty_max_ticks.setter
    def poll_sacct_empty_max_ticks(self, value: int) -> None:
        self.runtime.poll_sacct_empty_max_ticks = int(value)

    @property
    def failure_threshold_fraction(self) -> float:
        return self.runtime.failure_threshold_fraction

    @failure_threshold_fraction.setter
    def failure_threshold_fraction(self, value: float) -> None:
        self.runtime.failure_threshold_fraction = float(value)

    @property
    def max_acquisition_grad_per_ang(self) -> float:
        return self.acquisition.gradient.max_acquisition_grad_per_ang

    @max_acquisition_grad_per_ang.setter
    def max_acquisition_grad_per_ang(self, value: float) -> None:
        self.acquisition.gradient.max_acquisition_grad_per_ang = float(value)

    @property
    def max_force_per_atom_ha_per_ang(self) -> float:
        return self.acquisition.gradient.max_acquisition_grad_per_ang

    @max_force_per_atom_ha_per_ang.setter
    def max_force_per_atom_ha_per_ang(self, value: float) -> None:
        self.acquisition.gradient.max_acquisition_grad_per_ang = float(value)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ConfigValidationError(
                "campaign.yaml must be a mapping at the top level"
            )
        try:
            data = migrate_campaign_payload(data)
        except (CampaignMigrationError, ValueError) as exc:
            raise ConfigValidationError(str(exc)) from exc
        try:
            inst = parse_dataclass_block(cls, data)
        except DataclassParseError as exc:
            raise ConfigValidationError(str(exc)) from exc
        inst._validate()
        return inst

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path):
        """Write ONLY user-edited fields (diff against
        CampaignConfig() defaults) plus 'schema_version'.

        'to_yaml(asdict(self))' previously emitted
        all ~108 lines of nested defaults. The next '--preset NAME' run
        deep-merged campaign.yaml ONTO the preset, which then silently
        wins on every key (since every key is explicitly present in
        campaign.yaml). Presets become a no-op the moment the user
        Saves once. Diff-against-defaults fixes the semantic without
        operator-tracking machinery.
        """
        import yaml
        from .daemon.state import atomic_write_text
        diff = diff_against_defaults(self)
        text = yaml.safe_dump(diff, sort_keys=True, default_flow_style=False)
        atomic_write_text(path, text)

    def to_yaml_dense(self, path):
        """Diagnostic mode dump: every nested key serialised, defaults
        included. Used by 'Show current config' menu / debug scripts;
        NOT used by the menu's Save-to-disk path."""
        import yaml
        from .daemon.state import atomic_write_text
        text = yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)
        atomic_write_text(path, text)

    def _validate(self):
        _validate_token(
            "campaign.system_name",
            self.campaign.system_name,
            _SYSTEM_NAME_RE,
            "a filename-safe token matching ^[A-Za-z0-9][A-Za-z0-9_-]*$",
        )
        _validate_token(
            "resources.defaults.partition",
            self.resources.defaults.partition,
            _SCHEDULER_TOKEN_RE,
            "a scheduler token containing only letters, numbers, '.', '_', ':' and '-'",
        )
        _validate_positive_walltime(
            "resources.defaults.walltime_hours",
            self.resources.defaults.walltime_hours,
        )
        _validate_positive_int_or_auto(
            "resources.defaults.cpus_per_task",
            self.resources.defaults.cpus_per_task,
        )
        _validate_memory(
            "resources.defaults.mem_per_cpu",
            self.resources.defaults.mem_per_cpu,
            _SLURM_MEMORY_RE,
            "SLURM memory syntax such as 4G or 4000M, or auto",
        )
        for _backend in ("polus", "gaussian", "aimall", "ariadne", "ferebus"):
            _block = getattr(self.resources, _backend)
            if _block.partition is not None:
                _validate_token(
                    "resources." + _backend + ".partition",
                    _block.partition,
                    _SCHEDULER_TOKEN_RE,
                    "a scheduler token containing only letters, numbers, '.', '_', ':' and '-'",
                )
            if _block.walltime_hours is not None:
                _validate_positive_walltime(
                    "resources." + _backend + ".walltime_hours",
                    _block.walltime_hours,
                )
            if _block.cpus_per_task is not None:
                _validate_positive_int_or_auto(
                    "resources." + _backend + ".cpus_per_task",
                    _block.cpus_per_task,
                )
            if _block.mem_per_cpu is not None:
                _validate_memory(
                    "resources." + _backend + ".mem_per_cpu",
                    _block.mem_per_cpu,
                    _SLURM_MEMORY_RE,
                    "SLURM memory syntax such as 4G or 4000M, or auto",
                )
        if self.resources.array_concurrency_limit is not None:
            _validate_positive_int(
                "resources.array_concurrency_limit",
                self.resources.array_concurrency_limit,
            )
        if self.resources.gradient_parallel_backend not in VALID_GRADIENT_PARALLEL_BACKENDS:
            raise ConfigValidationError(
                "resources.gradient_parallel_backend must be one of "
                + repr(sorted(VALID_GRADIENT_PARALLEL_BACKENDS))
            )
        if self.resources.gaussian.memory_mode not in VALID_GAUSSIAN_MEMORY_MODES:
            raise ConfigValidationError(
                "resources.gaussian.memory_mode must be one of "
                + repr(sorted(VALID_GAUSSIAN_MEMORY_MODES))
            )
        if isinstance(self.resources.gaussian.memory_fraction_of_slurm, bool) or not isinstance(
            self.resources.gaussian.memory_fraction_of_slurm, (int, float)
        ):
            raise ConfigValidationError(
                "resources.gaussian.memory_fraction_of_slurm must be a number"
            )
        if not 0.0 < float(self.resources.gaussian.memory_fraction_of_slurm) <= 1.0:
            raise ConfigValidationError(
                "resources.gaussian.memory_fraction_of_slurm must be in (0, 1]"
            )
        _validate_memory(
            "resources.gaussian.link0_mem",
            self.resources.gaussian.link0_mem,
            _GAUSSIAN_MEMORY_RE,
            "Gaussian memory syntax such as 8GB or 8000MB",
        )
        _validate_positive_int("aimall.encomp", self.aimall.encomp)
        if not isinstance(self.aimall.nogui, bool):
            raise ConfigValidationError("aimall.nogui must be a boolean")
        if isinstance(self.aimall.naat, str):
            if self.aimall.naat.strip().lower() != "auto":
                raise ConfigValidationError("aimall.naat must be 'auto' or a positive integer")
            self.aimall.naat = "auto"
        else:
            _validate_positive_int("aimall.naat", self.aimall.naat)
            aimall_raw_cpus = self.resources.cpus_for("AIMALL")
            if not (
                isinstance(aimall_raw_cpus, str)
                and aimall_raw_cpus.strip().lower() == "auto"
            ) and int(self.aimall.naat) > int(aimall_raw_cpus):
                raise ConfigValidationError(
                    "aimall.naat must be <= effective resources.aimall.cpus_per_task"
                )
        if not isinstance(self.aimall.boaq, str):
            raise ConfigValidationError("aimall.boaq must be a string")
        self.aimall.boaq = self.aimall.boaq.strip().lower()
        if self.aimall.boaq not in VALID_AIMALL_BOAQ_VALUES:
            raise ConfigValidationError(
                "aimall.boaq must be one of " + repr(sorted(VALID_AIMALL_BOAQ_VALUES))
            )
        if not isinstance(self.aimall.iasmesh, str):
            raise ConfigValidationError("aimall.iasmesh must be a string")
        self.aimall.iasmesh = self.aimall.iasmesh.strip().lower()
        if self.aimall.iasmesh not in VALID_AIMALL_IASMESH_VALUES:
            raise ConfigValidationError(
                "aimall.iasmesh must be one of "
                + repr(sorted(VALID_AIMALL_IASMESH_VALUES))
            )
        if self.campaign.max_iterations <= 0:
            raise ConfigValidationError("campaign.max_iterations must be > 0")
        if self.runtime.poll_interval_seconds < 1:
            raise ConfigValidationError("runtime.poll_interval_seconds must be >= 1")
        if self.runtime.poll_interval_idle_seconds < 1:
            raise ConfigValidationError("runtime.poll_interval_idle_seconds must be >= 1")
        if self.runtime.poll_sacct_empty_max_ticks < 0:
            raise ConfigValidationError("runtime.poll_sacct_empty_max_ticks must be >= 0")
        allocation = self.point_allocation
        for allocation_name in (
            "bootstrap_training_size",
            "bootstrap_internal_validation_size",
            "batch_training_size",
        ):
            value = getattr(allocation, allocation_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigValidationError(
                    "point_allocation." + allocation_name + " must be an integer"
                )
            if value <= 0:
                raise ConfigValidationError(
                    "point_allocation." + allocation_name + " must be > 0"
                )
        batch_internal_size = allocation.batch_internal_validation_size
        if (
            isinstance(batch_internal_size, bool)
            or not isinstance(batch_internal_size, int)
        ):
            raise ConfigValidationError(
                "point_allocation.batch_internal_validation_size must be an integer"
            )
        if batch_internal_size < 0:
            raise ConfigValidationError(
                "point_allocation.batch_internal_validation_size must be >= 0"
            )
        external_size = allocation.bootstrap_external_validation_size
        if isinstance(external_size, bool) or not isinstance(external_size, int):
            raise ConfigValidationError(
                "point_allocation.bootstrap_external_validation_size must be an integer"
            )
        if external_size < 0:
            raise ConfigValidationError(
                "point_allocation.bootstrap_external_validation_size must be >= 0"
            )
        if not isinstance(allocation.anchor, bool):
            raise ConfigValidationError("point_allocation.anchor must be a boolean")
        for path_name in ("source_path", "anchor_path"):
            value = getattr(self.campaign, path_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError(
                    "campaign." + path_name + " must be a non-empty path string"
                )
            if "\x00" in value:
                raise ConfigValidationError(
                    "campaign." + path_name + " must not contain NUL characters"
                )
        if isinstance(self.campaign.sampling_aggressiveness, bool) or not isinstance(
            self.campaign.sampling_aggressiveness,
            int,
        ):
            raise ConfigValidationError(
                "campaign.sampling_aggressiveness must be an integer in [1, 10]"
            )
        if not 1 <= self.campaign.sampling_aggressiveness <= 10:
            raise ConfigValidationError(
                "campaign.sampling_aggressiveness must be in [1, 10]"
            )
        if self.seed_selection.n_seeds_per_iteration <= 0:
            raise ConfigValidationError(
                "seed_selection.n_seeds_per_iteration must be > 0"
            )
        if (
            int(self.seed_selection.n_seeds_per_iteration)
            < int(allocation.batch_total_size)
        ):
            raise ConfigValidationError(
                "seed_selection.n_seeds_per_iteration must be >= "
                "point_allocation.batch_training_size + "
                "point_allocation.batch_internal_validation_size"
            )
        if not 0.0 <= self.seed_selection.bulk_fraction <= 1.0:
            raise ConfigValidationError(
                "seed_selection.bulk_fraction must be in [0, 1]"
            )
        _validate_positive_int(
            "seed_selection.variance_chunk_size",
            self.seed_selection.variance_chunk_size,
        )
        if self.seed_selection.strategy not in VALID_SEED_SELECTION_STRATEGIES:
            raise ConfigValidationError(
                "seed_selection.strategy must be one of "
                + repr(sorted(VALID_SEED_SELECTION_STRATEGIES))
            )
        if (
            self.seed_selection.d_optimal_degenerate_policy
            not in VALID_D_OPTIMAL_DEGENERATE_POLICIES
        ):
            raise ConfigValidationError(
                "seed_selection.d_optimal_degenerate_policy must be one of "
                + repr(sorted(VALID_D_OPTIMAL_DEGENERATE_POLICIES))
            )
        _validate_positive_int(
            "seed_selection.d_optimal_pool_multiplier",
            self.seed_selection.d_optimal_pool_multiplier,
        )
        for name, value in (
            ("seed_selection.d_optimal_jitter", self.seed_selection.d_optimal_jitter),
            ("seed_selection.d_optimal_novelty_floor", self.seed_selection.d_optimal_novelty_floor),
            ("seed_selection.d_optimal_score_power", self.seed_selection.d_optimal_score_power),
        ):
            _validate_optional_nonnegative_float(name, value)
        if float(self.seed_selection.d_optimal_jitter) <= 0.0:
            raise ConfigValidationError("seed_selection.d_optimal_jitter must be > 0")
        if not 0.0 <= float(self.runtime.failure_threshold_fraction) <= 1.0:
            raise ConfigValidationError(
                "runtime.failure_threshold_fraction must be in [0, 1]"
            )
        _validate_positive_int("runtime.lease_stale_seconds", self.runtime.lease_stale_seconds)
        _validate_positive_int(
            "runtime.postprocess_settle_attempts",
            self.runtime.postprocess_settle_attempts,
        )
        _validate_nonnegative_int(
            "runtime.postprocess_settle_seconds",
            self.runtime.postprocess_settle_seconds,
        )
        _validate_nonnegative_int(
            "runtime.transient_phase_retry_max",
            self.runtime.transient_phase_retry_max,
        )
        _validate_nonnegative_int(
            "runtime.poll_sacct_unknown_max_ticks",
            self.runtime.poll_sacct_unknown_max_ticks,
        )
        _validate_nonnegative_int(
            "runtime.poll_sacct_missing_max_ticks",
            self.runtime.poll_sacct_missing_max_ticks,
        )
        _validate_nonnegative_int(
            "runtime.poll_squeue_inconclusive_max_ticks",
            self.runtime.poll_squeue_inconclusive_max_ticks,
        )
        if not isinstance(self.runtime.halt_on_tick_exception, bool):
            raise ConfigValidationError(
                "runtime.halt_on_tick_exception must be a boolean"
            )
        if not isinstance(self.resources.fail_on_memory_estimate_exceeds_request, bool):
            raise ConfigValidationError(
                "resources.fail_on_memory_estimate_exceeds_request must be a boolean"
            )
        if self.anti_overlap.recent_seeds_cooldown < 0:
            raise ConfigValidationError(
                "anti_overlap.recent_seeds_cooldown must be >= 0"
            )
        if self.anti_overlap.min_post_ariadne_whitened_distance < 0.0:
            raise ConfigValidationError(
                "anti_overlap.min_post_ariadne_whitened_distance must be >= 0"
            )
        if (
            self.anti_overlap.max_post_ariadne_whitened_distance
            <= self.anti_overlap.min_post_ariadne_whitened_distance
        ):
            raise ConfigValidationError(
                "anti_overlap.max_post_ariadne_whitened_distance must be > min"
            )
        if self.phase_b.descriptor not in VALID_DESCRIPTORS:
            raise ConfigValidationError(
                "phase_b.descriptor must be one of " + repr(sorted(VALID_DESCRIPTORS))
            )
        if not 0.0 <= self.phase_b.beta <= 1.0:
            raise ConfigValidationError(
                "phase_b.beta must be in [0, 1]"
            )
        if not isinstance(self.geometry_novelty.enabled, bool):
            raise ConfigValidationError(
                "geometry_novelty.enabled must be a boolean"
            )
        if self.geometry_novelty.scale_source not in VALID_GEOMETRY_NOVELTY_SCALE_SOURCES:
            raise ConfigValidationError(
                "geometry_novelty.scale_source must be one of "
                + repr(sorted(VALID_GEOMETRY_NOVELTY_SCALE_SOURCES))
            )
        if self.geometry_novelty.statistic not in VALID_GEOMETRY_NOVELTY_STATISTICS:
            raise ConfigValidationError(
                "geometry_novelty.statistic must be one of "
                + repr(sorted(VALID_GEOMETRY_NOVELTY_STATISTICS))
            )
        if self.geometry_novelty.scale_floor_angstrom <= 0.0:
            raise ConfigValidationError(
                "geometry_novelty.scale_floor_angstrom must be > 0"
            )
        if self.geometry_novelty.history_window_iterations < 0:
            raise ConfigValidationError(
                "geometry_novelty.history_window_iterations must be >= 0"
            )
        if self.geometry_novelty.fallback_scale_angstrom <= 0.0:
            raise ConfigValidationError(
                "geometry_novelty.fallback_scale_angstrom must be > 0"
            )
        if self.ferebus.warmstart not in VALID_WARMSTART:
            raise ConfigValidationError(
                "ferebus.warmstart must be one of " + repr(sorted(VALID_WARMSTART))
            )
        try:
            from ichor.core.common.constants import multipole_names
            valid_ferebus_props = {"iqa", *multipole_names}
        except Exception as exc:
            raise ConfigValidationError(
                "could not load ichor multipole names for ferebus.properties: "
                + type(exc).__name__ + ": " + str(exc)
            ) from exc
        props = self.ferebus.properties
        if not isinstance(props, list) or not props:
            raise ConfigValidationError("ferebus.properties must be a non-empty list")
        seen_props = set()
        for prop in props:
            if not isinstance(prop, str):
                raise ConfigValidationError("ferebus.properties entries must be strings")
            if not prop.strip():
                raise ConfigValidationError("ferebus.properties entries must be non-empty")
            if prop in seen_props:
                raise ConfigValidationError(
                    "ferebus.properties contains duplicate " + repr(prop)
                )
            if prop not in valid_ferebus_props:
                raise ConfigValidationError(
                    "ferebus.properties contains unsupported property "
                    + repr(prop)
                    + "; expected one of "
                    + repr(sorted(valid_ferebus_props))
                )
            seen_props.add(prop)
        if self.acquisition.property_name not in seen_props:
            raise ConfigValidationError(
                "acquisition.property_name "
                + repr(self.acquisition.property_name)
                + " must be present in ferebus.properties"
            )
        if self.acquisition.property_name != "iqa":
            raise ConfigValidationError(
                "acquisition.property_name must be 'iqa' for this daemon patch series; "
                "multipoles may be trained in ferebus.properties but are not valid acquisition targets yet"
            )
        qg = self.quality_gates
        for bool_name in (
            "quality_gates.require_readable_aimall_geometry",
            "quality_gates.require_finite_iqa",
            "quality_gates.require_finite_integration_error",
        ):
            block, field_name = bool_name.split(".", 1)
            value = getattr(getattr(self, block), field_name)
            if not isinstance(value, bool):
                raise ConfigValidationError(bool_name + " must be a boolean")
        for name, value in (
            ("quality_gates.max_abs_integration_error", qg.max_abs_integration_error),
            ("quality_gates.iqa_energy_recovery_tolerance_ha", qg.iqa_energy_recovery_tolerance_ha),
            ("quality_gates.ferebus_max_ext_rmse_ha", qg.ferebus_max_ext_rmse_ha),
            ("quality_gates.ferebus_max_condition_number", qg.ferebus_max_condition_number),
            ("quality_gates.ariadne_max_displacement_ang", qg.ariadne_max_displacement_ang),
            ("quality_gates.ariadne_min_pair_distance_ang", qg.ariadne_min_pair_distance_ang),
        ):
            _validate_optional_nonnegative_float(name, value)
        if qg.ferebus_min_ext_r2 is not None:
            if isinstance(qg.ferebus_min_ext_r2, bool) or not isinstance(
                qg.ferebus_min_ext_r2, (int, float)
            ):
                raise ConfigValidationError(
                    "quality_gates.ferebus_min_ext_r2 must be a number or null"
                )
        safety = self.adversarial_safety
        for bool_name in (
            "adversarial_safety.enabled",
            "adversarial_safety.reject_unsafe_landings",
            "adversarial_safety.salvage_safe_iterate",
            "adversarial_safety.backtrack_to_safe_landing",
            "adversarial_safety.allow_seed_fallback",
            "adversarial_safety.accept_legacy_missing_landing_safety",
            "adversarial_safety.enforce_min_whitened_distance",
            "adversarial_safety.phase_b_filter_enabled",
            "adversarial_safety.enforce_movement_band",
            "adversarial_safety.under_move_retry",
            "adversarial_safety.reject_under_moved_after_retry",
            "adversarial_safety.reject_over_moved",
        ):
            block, field_name = bool_name.split(".", 1)
            value = getattr(getattr(self, block), field_name)
            if not isinstance(value, bool):
                raise ConfigValidationError(bool_name + " must be a boolean")
        _validate_positive_int(
            "adversarial_safety.backtrack_points",
            safety.backtrack_points,
        )
        for name, value in (
            ("adversarial_safety.min_whitened_distance", safety.min_whitened_distance),
            ("adversarial_safety.max_whitened_distance", safety.max_whitened_distance),
            ("adversarial_safety.max_predicted_energy_delta_ha", safety.max_predicted_energy_delta_ha),
            ("adversarial_safety.max_energy_variance", safety.max_energy_variance),
            ("adversarial_safety.max_chemistry_penalty", safety.max_chemistry_penalty),
        ):
            _validate_optional_nonnegative_float(name, value)
        if (
            float(safety.max_whitened_distance)
            <= float(safety.min_whitened_distance)
        ):
            raise ConfigValidationError(
                "adversarial_safety.max_whitened_distance must be > min_whitened_distance"
            )
        calib = self.error_calibration
        if not isinstance(calib.enabled, bool):
            raise ConfigValidationError("error_calibration.enabled must be a boolean")
        if calib.mode not in VALID_ERROR_CALIBRATION_MODES:
            raise ConfigValidationError(
                "error_calibration.mode must be one of "
                + repr(sorted(VALID_ERROR_CALIBRATION_MODES))
            )
        _validate_positive_int(
            "error_calibration.min_records_to_apply",
            calib.min_records_to_apply,
        )
        _validate_positive_int(
            "error_calibration.min_model_versions_to_apply",
            calib.min_model_versions_to_apply,
        )
        _validate_positive_int("error_calibration.n_bins", calib.n_bins)
        _validate_positive_int(
            "error_calibration.min_bin_records",
            calib.min_bin_records,
        )
        _validate_positive_int("error_calibration.max_records", calib.max_records)
        _validate_nonnegative_int(
            "error_calibration.max_model_age_iterations",
            calib.max_model_age_iterations,
        )
        if not isinstance(calib.monotone_estimator, bool):
            raise ConfigValidationError(
                "error_calibration.monotone_estimator must be a boolean"
            )
        if isinstance(calib.quantile, bool) or not isinstance(
            calib.quantile, (int, float)
        ):
            raise ConfigValidationError("error_calibration.quantile must be a number")
        if not 0.0 < float(calib.quantile) <= 1.0:
            raise ConfigValidationError(
                "error_calibration.quantile must be in (0, 1]"
            )
        if isinstance(calib.apply_strength, bool) or not isinstance(
            calib.apply_strength, (int, float)
        ):
            raise ConfigValidationError("error_calibration.apply_strength must be a number")
        if not 0.0 <= float(calib.apply_strength) <= 1.0:
            raise ConfigValidationError(
                "error_calibration.apply_strength must be in [0, 1]"
            )
        if not isinstance(calib.group_by_atom_type, bool):
            raise ConfigValidationError(
                "error_calibration.group_by_atom_type must be a boolean"
            )
        if not isinstance(calib.group_by_landing_policy, bool):
            raise ConfigValidationError(
                "error_calibration.group_by_landing_policy must be a boolean"
            )
        if calib.output_units != "ha":
            raise ConfigValidationError("error_calibration.output_units must be 'ha'")
        if calib.model_version_policy not in VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES:
            raise ConfigValidationError(
                "error_calibration.model_version_policy must be one of "
                + repr(sorted(VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES))
            )
        if not isinstance(calib.aggressiveness_match_required, bool):
            raise ConfigValidationError(
                "error_calibration.aggressiveness_match_required must be a boolean"
            )
        max_acquisition_grad = (
            self.acquisition.gradient.max_acquisition_grad_per_ang
        )
        if isinstance(max_acquisition_grad, bool) or not isinstance(
            max_acquisition_grad,
            (int, float),
        ):
            raise ConfigValidationError(
                "acquisition.gradient.max_acquisition_grad_per_ang must be a number"
            )
        if not math.isfinite(float(max_acquisition_grad)):
            raise ConfigValidationError(
                "acquisition.gradient.max_acquisition_grad_per_ang must be finite"
            )
        if float(max_acquisition_grad) <= 0.0:
            raise ConfigValidationError(
                "acquisition.gradient.max_acquisition_grad_per_ang must be > 0"
            )
        ariadne = self.ariadne
        if str(ariadne.trqn_scale_mode) not in VALID_TRQN_SCALE_MODES:
            raise ConfigValidationError(
                "ariadne.trqn_scale_mode must be one of "
                + repr(sorted(VALID_TRQN_SCALE_MODES))
            )
        for name, value in (
            (
                "ariadne.trqn_target_initial_grad_norm",
                ariadne.trqn_target_initial_grad_norm,
            ),
            (
                "ariadne.trqn_retry_target_initial_grad_norm",
                ariadne.trqn_retry_target_initial_grad_norm,
            ),
            (
                "ariadne.trqn_target_initial_grad_rms",
                ariadne.trqn_target_initial_grad_rms,
            ),
            (
                "ariadne.trqn_retry_target_initial_grad_rms",
                ariadne.trqn_retry_target_initial_grad_rms,
            ),
            (
                "ariadne.trqn_under_move_target_initial_grad_rms",
                ariadne.trqn_under_move_target_initial_grad_rms,
            ),
            ("ariadne.trqn_min_objective_scale", ariadne.trqn_min_objective_scale),
            ("ariadne.trqn_max_objective_scale", ariadne.trqn_max_objective_scale),
            ("ariadne.trqn_fixed_objective_scale", ariadne.trqn_fixed_objective_scale),
            ("ariadne.trqn_geodesic_dt", ariadne.trqn_geodesic_dt),
            ("ariadne.trqn_geodesic_tol", ariadne.trqn_geodesic_tol),
            ("ariadne.trqn_bt_ic_tol", ariadne.trqn_bt_ic_tol),
            ("ariadne.trqn_trust_min", ariadne.trqn_trust_min),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigValidationError(name + " must be a number")
            if not float(value) > 0.0:
                raise ConfigValidationError(name + " must be > 0")
        _validate_positive_int(
            "ariadne.trqn_max_backtransform_iter",
            ariadne.trqn_max_backtransform_iter,
        )
        _validate_nonnegative_int(
            "ariadne.trqn_under_move_retry_max",
            ariadne.trqn_under_move_retry_max,
        )
        if (
            float(ariadne.trqn_retry_target_initial_grad_norm)
            > float(ariadne.trqn_target_initial_grad_norm)
        ):
            raise ConfigValidationError(
                "ariadne.trqn_retry_target_initial_grad_norm must be <= "
                "ariadne.trqn_target_initial_grad_norm"
            )
        if float(ariadne.trqn_min_objective_scale) > float(
            ariadne.trqn_max_objective_scale
        ):
            raise ConfigValidationError(
                "ariadne.trqn_min_objective_scale must be <= "
                "ariadne.trqn_max_objective_scale"
            )
        if float(ariadne.trqn_max_objective_scale) > 1.0:
            raise ConfigValidationError(
                "ariadne.trqn_max_objective_scale must be <= 1"
            )
        if str(ariadne.trqn_scale_mode) == "fixed" and not (
            float(ariadne.trqn_min_objective_scale)
            <= float(ariadne.trqn_fixed_objective_scale)
            <= float(ariadne.trqn_max_objective_scale)
        ):
            raise ConfigValidationError(
                "ariadne.trqn_fixed_objective_scale must be inside the "
                "configured TRQN objective scale bounds"
            )
        if not isinstance(ariadne.trqn_retry_on_no_proposal, bool):
            raise ConfigValidationError(
                "ariadne.trqn_retry_on_no_proposal must be a boolean"
            )
        if not isinstance(ariadne.trqn_under_move_retry, bool):
            raise ConfigValidationError(
                "ariadne.trqn_under_move_retry must be a boolean"
            )
        if str(ariadne.trqn_backtransform_mode) not in VALID_TRQN_BACKTRANSFORM_MODES:
            raise ConfigValidationError(
                "ariadne.trqn_backtransform_mode must be one of "
                + repr(sorted(VALID_TRQN_BACKTRANSFORM_MODES))
            )
        if str(ariadne.trqn_geodesic_bt_mode) not in VALID_TRQN_GEODESIC_BT_MODES:
            raise ConfigValidationError(
                "ariadne.trqn_geodesic_bt_mode must be one of "
                + repr(sorted(VALID_TRQN_GEODESIC_BT_MODES))
            )
        # Subspace-dim cross-validation. These catch configurations
        #that pass field-by-field validation but blow up later inside PCA.
        if self.acquisition.subspace.neighbour_count < 1:
            raise ConfigValidationError(
                "acquisition.subspace.neighbour_count must be >= 1"
            )
        if (
            self.acquisition.subspace.min_subspace_dim
            > self.acquisition.subspace.max_subspace_dim
        ):
            raise ConfigValidationError(
                "acquisition.subspace.min_subspace_dim must be <= max_subspace_dim"
            )
        if (
            self.acquisition.subspace.neighbour_count
            < self.acquisition.subspace.max_subspace_dim
        ):
            raise ConfigValidationError(
                "acquisition.subspace.neighbour_count must be >= max_subspace_dim "
                "(PCA needs at least max_subspace_dim neighbours to fill the subspace)"
            )
        if self.acquisition.gradient.mode not in VALID_GRADIENT_MODES:
            raise ConfigValidationError(
                "acquisition.gradient.mode must be one of "
                + repr(sorted(VALID_GRADIENT_MODES))
            )
        if not isinstance(self.acquisition.allow_uniform_posterior_fallback, bool):
            raise ConfigValidationError(
                "acquisition.allow_uniform_posterior_fallback must be a boolean"
            )
        ba = self.acquisition.barrier
        for name, value in (
            ("acquisition.barrier.nonbonded_clash_scale", ba.nonbonded_clash_scale),
            ("acquisition.barrier.clash_delta", ba.clash_delta),
            ("acquisition.barrier.clash_lambda", ba.clash_lambda),
            ("acquisition.barrier.nonbonded_expansion_scale", ba.nonbonded_expansion_scale),
            ("acquisition.barrier.nonbonded_expansion_delta", ba.nonbonded_expansion_delta),
            ("acquisition.barrier.nonbonded_expansion_lambda", ba.nonbonded_expansion_lambda),
            ("acquisition.barrier.bond_lower_scale", ba.bond_lower_scale),
            ("acquisition.barrier.bond_upper_scale", ba.bond_upper_scale),
            ("acquisition.barrier.bond_delta", ba.bond_delta),
            ("acquisition.barrier.bond_lambda", ba.bond_lambda),
            ("acquisition.barrier.angle_lower_scale", ba.angle_lower_scale),
            ("acquisition.barrier.angle_upper_scale", ba.angle_upper_scale),
            ("acquisition.barrier.angle_delta", ba.angle_delta),
            ("acquisition.barrier.angle_lambda", ba.angle_lambda),
            ("acquisition.barrier.energy_cap_quantile", ba.energy_cap_quantile),
            ("acquisition.barrier.energy_cap_floor", ba.energy_cap_floor),
            ("acquisition.barrier.energy_cap_delta", ba.energy_cap_delta),
            ("acquisition.barrier.energy_cap_lambda", ba.energy_cap_lambda),
        ):
            _validate_optional_nonnegative_float(name, value)
        if ba.bond_upper_scale <= ba.bond_lower_scale:
            raise ConfigValidationError(
                "acquisition.barrier.bond_upper_scale must be > bond_lower_scale"
            )
        if ba.angle_upper_scale <= ba.angle_lower_scale:
            raise ConfigValidationError(
                "acquisition.barrier.angle_upper_scale must be > angle_lower_scale"
            )
        if not 0.0 <= float(ba.energy_cap_quantile) <= 1.0:
            raise ConfigValidationError(
                "acquisition.barrier.energy_cap_quantile must be in [0, 1]"
            )
        _validate_optional_nonnegative_float(
            "acquisition.barrier.softplus_cap", ba.softplus_cap
        )
        if (
            self.acquisition.subspace.mode_weighting_policy
            not in VALID_MODE_WEIGHTING_POLICIES
        ):
            raise ConfigValidationError(
                "acquisition.subspace.mode_weighting_policy must be one of "
                + repr(sorted(VALID_MODE_WEIGHTING_POLICIES))
            )
        spectral = self.acquisition.spectral
        if not isinstance(spectral.enabled, bool):
            raise ConfigValidationError("acquisition.spectral.enabled must be a boolean")
        if spectral.mode not in VALID_SPECTRAL_MODES:
            raise ConfigValidationError(
                "acquisition.spectral.mode must be one of "
                + repr(sorted(VALID_SPECTRAL_MODES))
            )
        if spectral.mode_weighting not in VALID_MODE_WEIGHTING_POLICIES:
            raise ConfigValidationError(
                "acquisition.spectral.mode_weighting must be one of "
                + repr(sorted(VALID_MODE_WEIGHTING_POLICIES))
            )
        for name, value in (
            ("acquisition.spectral.lambda_spectral", spectral.lambda_spectral),
            ("acquisition.spectral.omega_floor", spectral.omega_floor),
            ("acquisition.spectral.low_frequency_power", spectral.low_frequency_power),
        ):
            _validate_optional_nonnegative_float(name, value)
        if spectral.lambda_spectral < 0.0:
            raise ConfigValidationError("acquisition.spectral.lambda_spectral must be >= 0")
        if spectral.omega_floor <= 0.0:
            raise ConfigValidationError("acquisition.spectral.omega_floor must be > 0")
        if spectral.max_modes is not None:
            _validate_positive_int("acquisition.spectral.max_modes", spectral.max_modes)

        cal_energy = self.acquisition.calibrated_energy
        if cal_energy.utility not in VALID_CALIBRATED_ENERGY_UTILITIES:
            raise ConfigValidationError(
                "acquisition.calibrated_energy.utility must be one of "
                + repr(sorted(VALID_CALIBRATED_ENERGY_UTILITIES))
            )
        if not isinstance(cal_energy.fallback_to_raw_variance, bool):
            raise ConfigValidationError(
                "acquisition.calibrated_energy.fallback_to_raw_variance must be a boolean"
            )
        for name, value in (
            ("acquisition.calibrated_energy.band_low_ha", cal_energy.band_low_ha),
            ("acquisition.calibrated_energy.band_high_ha", cal_energy.band_high_ha),
            ("acquisition.calibrated_energy.low_softness_ha", cal_energy.low_softness_ha),
            ("acquisition.calibrated_energy.high_softness_ha", cal_energy.high_softness_ha),
        ):
            _validate_optional_nonnegative_float(name, value)
        if (
            cal_energy.band_low_ha is not None
            and cal_energy.band_high_ha is not None
            and float(cal_energy.band_high_ha) <= float(cal_energy.band_low_ha)
        ):
            raise ConfigValidationError(
                "acquisition.calibrated_energy.band_high_ha must be > band_low_ha"
            )
        if cal_energy.band_high_ha is not None and float(cal_energy.band_high_ha) <= 0.0:
            raise ConfigValidationError(
                "acquisition.calibrated_energy.band_high_ha must be > 0"
            )
        for name, value in (
            ("acquisition.calibrated_energy.low_softness_ha", cal_energy.low_softness_ha),
            ("acquisition.calibrated_energy.high_softness_ha", cal_energy.high_softness_ha),
        ):
            if value is not None and float(value) <= 0.0:
                raise ConfigValidationError(name + " must be > 0")

        fullspace = self.acquisition.fullspace_confinement
        if not isinstance(fullspace.enabled, bool):
            raise ConfigValidationError(
                "acquisition.fullspace_confinement.enabled must be a boolean"
            )
        for name, value in (
            ("acquisition.fullspace_confinement.lambda_residual", fullspace.lambda_residual),
            ("acquisition.fullspace_confinement.lambda_rmsd", fullspace.lambda_rmsd),
        ):
            _validate_optional_nonnegative_float(name, value)
        norm = self.acquisition.size_normalisation
        if not isinstance(norm.enabled, bool):
            raise ConfigValidationError(
                "acquisition.size_normalisation.enabled must be a boolean"
            )
        if norm.energy_mode not in VALID_SIZE_NORMALISATION_ENERGY_MODES:
            raise ConfigValidationError(
                "acquisition.size_normalisation.energy_mode must be one of "
                + repr(sorted(VALID_SIZE_NORMALISATION_ENERGY_MODES))
            )
        if norm.whitened_distance_mode not in VALID_SIZE_NORMALISATION_DISTANCE_MODES:
            raise ConfigValidationError(
                "acquisition.size_normalisation.whitened_distance_mode must be one of "
                + repr(sorted(VALID_SIZE_NORMALISATION_DISTANCE_MODES))
            )
        if norm.chemistry_barrier_mode not in VALID_SIZE_NORMALISATION_BARRIER_MODES:
            raise ConfigValidationError(
                "acquisition.size_normalisation.chemistry_barrier_mode must be one of "
                + repr(sorted(VALID_SIZE_NORMALISATION_BARRIER_MODES))
            )
        driver = self.acquisition.driver
        if not isinstance(driver.enabled, bool):
            raise ConfigValidationError(
                "acquisition.driver.enabled must be a boolean"
            )
        if driver.objective not in VALID_ACQUISITION_DRIVER_OBJECTIVES:
            raise ConfigValidationError(
                "acquisition.driver.objective must be one of "
                + repr(sorted(VALID_ACQUISITION_DRIVER_OBJECTIVES))
            )
        if driver.gradient_backend not in VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS:
            raise ConfigValidationError(
                "acquisition.driver.gradient_backend must be one of "
                + repr(sorted(VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS))
            )
        if not isinstance(driver.include_stencils, bool):
            raise ConfigValidationError(
                "acquisition.driver.include_stencils must be a boolean"
            )
        for name, value in (
            ("acquisition.driver.analytic_movement", driver.analytic_movement),
            ("acquisition.driver.analytic_whitened_distance", driver.analytic_whitened_distance),
            ("acquisition.driver.analytic_pair_barriers", driver.analytic_pair_barriers),
            ("acquisition.driver.analytic_fullspace_rmsd", driver.analytic_fullspace_rmsd),
            ("acquisition.driver.finite_difference_energy", driver.finite_difference_energy),
            ("acquisition.driver.analytic_validation", driver.analytic_validation),
        ):
            if not isinstance(value, bool):
                raise ConfigValidationError(name + " must be a boolean")
        if not (0.0 <= float(driver.analytic_validation_tol_cosine) <= 1.0):
            raise ConfigValidationError(
                "acquisition.driver.analytic_validation_tol_cosine must be in [0, 1]"
            )
        for name, value in (
            ("acquisition.driver.lambda_energy", driver.lambda_energy),
            ("acquisition.driver.lambda_movement", driver.lambda_movement),
            ("acquisition.driver.lambda_distance", driver.lambda_distance),
            ("acquisition.driver.lambda_fullspace", driver.lambda_fullspace),
            ("acquisition.driver.lambda_chemistry", driver.lambda_chemistry),
        ):
            if value is None:
                raise ConfigValidationError(name + " must be a number")
            _validate_optional_nonnegative_float(name, value)
        stencils = self.acquisition.stencils
        if stencils.negative_curvature_policy not in VALID_NEGATIVE_CURVATURE_POLICIES:
            raise ConfigValidationError(
                "acquisition.stencils.negative_curvature_policy must be one of "
                + repr(sorted(VALID_NEGATIVE_CURVATURE_POLICIES))
            )
        _validate_optional_nonnegative_float(
            "acquisition.stencils.lambda_negative_curvature",
            stencils.lambda_negative_curvature,
        )
        if not isinstance(stencils.weak_mode_gating_enabled, bool):
            raise ConfigValidationError(
                "acquisition.stencils.weak_mode_gating_enabled must be a boolean"
            )
        for name, value in (
            (
                "acquisition.stencils.weak_mode_omega_low_fraction",
                stencils.weak_mode_omega_low_fraction,
            ),
            (
                "acquisition.stencils.weak_mode_omega_high_fraction",
                stencils.weak_mode_omega_high_fraction,
            ),
            (
                "acquisition.stencils.weak_mode_abs_omega_floor",
                stencils.weak_mode_abs_omega_floor,
            ),
            ("acquisition.stencils.weak_mode_penalty", stencils.weak_mode_penalty),
            (
                "acquisition.stencils.max_anharmonic_mode_score",
                stencils.max_anharmonic_mode_score,
            ),
            (
                "acquisition.stencils.max_anharmonic_total_score",
                stencils.max_anharmonic_total_score,
            ),
        ):
            _validate_optional_nonnegative_float(name, value)
        if (
            float(stencils.weak_mode_omega_high_fraction)
            <= float(stencils.weak_mode_omega_low_fraction)
        ):
            raise ConfigValidationError(
                "acquisition.stencils.weak_mode_omega_high_fraction must be > "
                "weak_mode_omega_low_fraction"
            )
        for name, value in (
            (
                "acquisition.stencils.weak_mode_abs_omega_floor",
                stencils.weak_mode_abs_omega_floor,
            ),
            (
                "acquisition.stencils.max_anharmonic_mode_score",
                stencils.max_anharmonic_mode_score,
            ),
            (
                "acquisition.stencils.max_anharmonic_total_score",
                stencils.max_anharmonic_total_score,
            ),
        ):
            if float(value) <= 0.0:
                raise ConfigValidationError(name + " must be > 0")
        gaussian_mem_per_cpu = self.resources.mem_per_cpu_for("GAUSSIAN")
        gaussian_cpus_per_task = self.resources.cpus_for("GAUSSIAN")
        if (
            self.resources.gaussian.memory_mode == "link0"
            and str(gaussian_mem_per_cpu).strip().lower() != "auto"
            and not (
                isinstance(gaussian_cpus_per_task, str)
                and gaussian_cpus_per_task.strip().lower() == "auto"
            )
        ):
            gaussian_mem_mib = _memory_mebibytes(
                "resources.gaussian.link0_mem",
                self.resources.gaussian.link0_mem,
                gaussian=True,
            )
            slurm_mem_mib = _memory_mebibytes(
                "resources.gaussian.mem_per_cpu",
                gaussian_mem_per_cpu,
                gaussian=False,
            ) * float(gaussian_cpus_per_task)
            limit_mib = (
                float(self.resources.gaussian.memory_fraction_of_slurm)
                * slurm_mem_mib
            )
            if gaussian_mem_mib > limit_mib:
                raise ConfigValidationError(
                    "resources.gaussian.link0_mem ("
                    + str(self.resources.gaussian.link0_mem)
                    + ") exceeds "
                    + str(self.resources.gaussian.memory_fraction_of_slurm)
                    + " of the Gaussian Slurm allocation resources.gaussian.mem_per_cpu "
                    "* resources.gaussian.cpus_per_task ("
                    + str(gaussian_mem_per_cpu)
                    + " * "
                    + str(gaussian_cpus_per_task)
                    + ")"
                )

    def effective_max_acquisition_grad_per_ang(self) -> float:
        return float(self.acquisition.gradient.max_acquisition_grad_per_ang)

    def to_acquisition_config(self):
        """Materialise an ichor.core AcquisitionConfig from the nested
        acquisition block. Enforces the subspace-dim guard (plan trap #3).
        """
        if (
            self.acquisition.subspace.max_subspace_dim > 6
            and self.acquisition.gradient.mode == "cartesian_fd"
        ):
            raise ConfigValidationError(
                "acquisition.subspace.max_subspace_dim "
                + str(self.acquisition.subspace.max_subspace_dim)
                + " > 6 with gradient.mode == cartesian_fd would explode the"
                " Cartesian FD gradient cost. Pin gradient.mode = active_fd"
                " or drop max_subspace_dim to <= 6."
            )
        from ichor.core.adversarial.config import (
            AcquisitionConfig,
            BarrierConfig,
            CalibratedEnergyConfig,
            DriverConfig,
            FullspaceConfinementConfig,
            GradientConfig,
            MovementBandConfig,
            MovementUtilityConfig,
            ReferenceScaleConfig,
            SizeNormalisationConfig,
            SpectralConfig,
            StencilConfig,
            SubspaceConfig,
            WeightConfig,
        )
        sb = self.acquisition.subspace
        ba = self.acquisition.barrier
        st = self.acquisition.stencils
        we = self.acquisition.weights
        sp = self.acquisition.spectral
        ce = self.acquisition.calibrated_energy
        fs = self.acquisition.fullspace_confinement
        sn = self.acquisition.size_normalisation
        dr = self.acquisition.driver
        gr = self.acquisition.gradient
        re = self.acquisition.references
        return AcquisitionConfig(
            property_name=self.acquisition.property_name,
            use_scaled_posterior_covariance=self.acquisition.use_scaled_posterior_covariance,
            subspace=SubspaceConfig(
                neighbour_count=sb.neighbour_count,
                neighbour_deduplicate_rmsd=sb.neighbour_deduplicate_rmsd,
                variance_capture=sb.variance_capture,
                min_subspace_dim=sb.min_subspace_dim,
                max_subspace_dim=sb.max_subspace_dim,
                gaussian_weight_sigma=sb.gaussian_weight_sigma,
                covariance_regularization=sb.covariance_regularization,
                canonicalise_basis=sb.canonicalise_basis,
                degeneracy_tolerance=sb.degeneracy_tolerance,
                mode_weighting_policy=sb.mode_weighting_policy,
            ),
            barrier=BarrierConfig(
                use_connectivity_barrier=ba.use_connectivity_barrier,
                nonbonded_clash_scale=ba.nonbonded_clash_scale,
                clash_delta=ba.clash_delta,
                clash_lambda=ba.clash_lambda,
                nonbonded_expansion_scale=ba.nonbonded_expansion_scale,
                nonbonded_expansion_delta=ba.nonbonded_expansion_delta,
                nonbonded_expansion_lambda=ba.nonbonded_expansion_lambda,
                bond_lower_scale=ba.bond_lower_scale,
                bond_upper_scale=ba.bond_upper_scale,
                bond_delta=ba.bond_delta,
                bond_lambda=ba.bond_lambda,
                angle_lower_scale=ba.angle_lower_scale,
                angle_upper_scale=ba.angle_upper_scale,
                angle_delta=ba.angle_delta,
                angle_lambda=ba.angle_lambda,
                energy_cap_quantile=ba.energy_cap_quantile,
                energy_cap_floor=ba.energy_cap_floor,
                energy_cap_delta=ba.energy_cap_delta,
                energy_cap_lambda=ba.energy_cap_lambda,
                softplus_cap=ba.softplus_cap,
            ),
            stencils=StencilConfig(
                step_scale=st.step_scale,
                min_step=st.min_step,
                max_step=st.max_step,
                jitter=st.jitter,
                curvature_floor=st.curvature_floor,
                softplus_scale=st.softplus_scale,
                autotune_from_cubic=st.autotune_from_cubic,
                negative_curvature_policy=st.negative_curvature_policy,
                lambda_negative_curvature=st.lambda_negative_curvature,
                weak_mode_gating_enabled=st.weak_mode_gating_enabled,
                weak_mode_omega_low_fraction=st.weak_mode_omega_low_fraction,
                weak_mode_omega_high_fraction=st.weak_mode_omega_high_fraction,
                weak_mode_abs_omega_floor=st.weak_mode_abs_omega_floor,
                weak_mode_penalty=st.weak_mode_penalty,
                max_anharmonic_mode_score=st.max_anharmonic_mode_score,
                max_anharmonic_total_score=st.max_anharmonic_total_score,
            ),
            weights=WeightConfig(
                lambda_force=we.lambda_force,
                lambda_frequency=we.lambda_frequency,
                lambda_anharmonic=we.lambda_anharmonic,
                lambda_energy=we.lambda_energy,
                lambda_distance=we.lambda_distance,
            ),
            spectral=SpectralConfig(
                enabled=sp.enabled,
                mode=sp.mode,
                mode_weighting=sp.mode_weighting,
                lambda_spectral=sp.lambda_spectral,
                omega_floor=sp.omega_floor,
                low_frequency_power=sp.low_frequency_power,
                max_modes=sp.max_modes,
            ),
            calibrated_energy=CalibratedEnergyConfig(
                utility=ce.utility,
                band_low_ha=ce.band_low_ha,
                band_high_ha=ce.band_high_ha,
                low_softness_ha=ce.low_softness_ha,
                high_softness_ha=ce.high_softness_ha,
                fallback_to_raw_variance=ce.fallback_to_raw_variance,
            ),
            fullspace_confinement=FullspaceConfinementConfig(
                enabled=fs.enabled,
                lambda_residual=fs.lambda_residual,
                lambda_rmsd=fs.lambda_rmsd,
                residual_scale=FULLSPACE_RESIDUAL_SCALE,
                fixed_residual_scale_ang=FULLSPACE_FIXED_RESIDUAL_SCALE_ANGSTROM,
                rmsd_scale_ang=FULLSPACE_RMSD_SCALE_MULTIPLIER
                * self.geometry_novelty.fallback_scale_angstrom,
                min_residual_scale_ang=FULLSPACE_MIN_RESIDUAL_SCALE_ANGSTROM,
                failure_penalty=FULLSPACE_FAILURE_PENALTY,
            ),
            size_normalisation=SizeNormalisationConfig(
                enabled=sn.enabled,
                energy_mode=sn.energy_mode,
                whitened_distance_mode=sn.whitened_distance_mode,
                chemistry_barrier_mode=sn.chemistry_barrier_mode,
            ),
            movement_band=MovementBandConfig(
                enabled=MOVEMENT_BAND_ENABLED,
                metric=MOVEMENT_BAND_METRIC,
                local_statistic=MOVEMENT_BAND_LOCAL_STATISTIC,
                hard_min_floor_ang=(
                    MOVEMENT_BAND_HARD_MIN_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
                target_low_floor_ang=(
                    MOVEMENT_BAND_TARGET_LOW_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
                target_peak_floor_ang=(
                    MOVEMENT_BAND_TARGET_PEAK_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
                target_high_cap_ang=(
                    MOVEMENT_BAND_TARGET_HIGH_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
                hard_max_cap_ang=(
                    MOVEMENT_BAND_HARD_MAX_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
                hard_min_fraction=MOVEMENT_BAND_HARD_MIN_FRACTION,
                target_low_fraction=MOVEMENT_BAND_TARGET_LOW_FRACTION,
                target_peak_fraction=MOVEMENT_BAND_TARGET_PEAK_FRACTION,
                target_high_fraction=MOVEMENT_BAND_TARGET_HIGH_FRACTION,
                hard_max_fraction=MOVEMENT_BAND_HARD_MAX_FRACTION,
                geometry_novelty_scale_angstrom=None,
            ),
            movement_utility=MovementUtilityConfig(
                enabled=MOVEMENT_UTILITY_ENABLED,
                direction=MOVEMENT_UTILITY_DIRECTION,
                lambda_move=MOVEMENT_UTILITY_LAMBDA_MOVE,
                band_fraction=MOVEMENT_UTILITY_BAND_FRACTION,
                progress_fraction=MOVEMENT_UTILITY_PROGRESS_FRACTION,
                low_softness_ang=(
                    MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
                high_softness_ang=(
                    MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION
                    * self.geometry_novelty.fallback_scale_angstrom
                ),
            ),
            driver=DriverConfig(
                enabled=dr.enabled,
                objective=dr.objective,
                gradient_backend=dr.gradient_backend,
                include_stencils=dr.include_stencils,
                analytic_movement=dr.analytic_movement,
                analytic_whitened_distance=dr.analytic_whitened_distance,
                analytic_pair_barriers=dr.analytic_pair_barriers,
                analytic_fullspace_rmsd=dr.analytic_fullspace_rmsd,
                finite_difference_energy=dr.finite_difference_energy,
                analytic_validation=dr.analytic_validation,
                analytic_validation_tol_cosine=dr.analytic_validation_tol_cosine,
                lambda_energy=dr.lambda_energy,
                lambda_movement=dr.lambda_movement,
                lambda_distance=dr.lambda_distance,
                lambda_fullspace=dr.lambda_fullspace,
                lambda_chemistry=dr.lambda_chemistry,
            ),
            gradient=GradientConfig(
                mode=gr.mode,
                cartesian_step=gr.cartesian_step,
                active_step=gr.active_step,
                regularization=gr.regularization,
                cartesian_step_floor=gr.cartesian_step_floor,
                ghost_mass_threshold=gr.ghost_mass_threshold,
            ),
            references=ReferenceScaleConfig(
                max_reference_samples=re.max_reference_samples,
                floor=re.floor,
                refresh_policy=re.refresh_policy,
                refresh_period=re.refresh_period,
            ),
        )

    def to_ariadne_run_config(self):
        """Materialise an AriadneRunConfig from the ariadne block."""
        from .acquisition.ariadne_runner import AriadneRunConfig
        a = self.ariadne
        return AriadneRunConfig(
            optimiser=a.optimiser,
            hessian_model=a.hessian_model,
            max_iter=a.max_iter,
            gradf_tol=a.gradf_tol,
            f_tol=a.f_tol,
            delta0=a.delta0,
            delta_max=a.delta_max,
            gamma=a.gamma,
            fallback_to_ds=a.fallback_to_ds,
            trqn_scale_mode=a.trqn_scale_mode,
            trqn_target_initial_grad_norm=a.trqn_target_initial_grad_norm,
            trqn_retry_target_initial_grad_norm=a.trqn_retry_target_initial_grad_norm,
            trqn_target_initial_grad_rms=a.trqn_target_initial_grad_rms,
            trqn_retry_target_initial_grad_rms=a.trqn_retry_target_initial_grad_rms,
            trqn_under_move_target_initial_grad_rms=a.trqn_under_move_target_initial_grad_rms,
            trqn_under_move_retry=a.trqn_under_move_retry,
            trqn_under_move_retry_max=a.trqn_under_move_retry_max,
            trqn_min_objective_scale=a.trqn_min_objective_scale,
            trqn_max_objective_scale=a.trqn_max_objective_scale,
            trqn_fixed_objective_scale=a.trqn_fixed_objective_scale,
            trqn_retry_on_no_proposal=a.trqn_retry_on_no_proposal,
            trqn_backtransform_mode=a.trqn_backtransform_mode,
            trqn_geodesic_bt_mode=a.trqn_geodesic_bt_mode,
            trqn_geodesic_dt=a.trqn_geodesic_dt,
            trqn_geodesic_tol=a.trqn_geodesic_tol,
            trqn_bt_ic_tol=a.trqn_bt_ic_tol,
            trqn_max_backtransform_iter=a.trqn_max_backtransform_iter,
            trqn_trust_min=a.trqn_trust_min,
        )
