"""Campaign configuration for the clean-break schema 13 contract."""
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
    "DiversityResourceBlock",
    "GaussianResourceBlock",
    "RetentionConfigBlock",
    "CONFIG_SCHEMA_VERSION",
    "ConfigValidationError",
    "AimallConfigBlock",
    "normalise_gaussian_route_keywords",
    "VALID_DESCRIPTORS",
    "VALID_GEOMETRY_NOVELTY_SCALE_SOURCES",
    "VALID_GEOMETRY_NOVELTY_STATISTICS",
    "VALID_SEED_SELECTION_STRATEGIES",
    "VALID_D_OPTIMAL_DEGENERATE_POLICIES",
    "VALID_MODE_WEIGHTING_POLICIES",
    "VALID_SPECTRAL_MODES",
    "VALID_CALIBRATED_ENERGY_UTILITIES",
    "VALID_TRQN_SCALE_MODES",
    "VALID_TRQN_BACKTRANSFORM_MODES",
    "VALID_TRQN_GEODESIC_BT_MODES",
    "VALID_NEGATIVE_CURVATURE_POLICIES",
    "VALID_AIMALL_BOAQ_VALUES",
    "VALID_AIMALL_IASMESH_VALUES",
    "VALID_FEREBUS_KERNELS",
    "VALID_FEREBUS_PRIOR_MEAN_STRATEGIES",
]


_SAFE_GAUSSIAN_ROUTE_TOKEN_RE = re.compile(r"^[A-Za-z0-9(),=._+\-]+$")
_SAFE_GAUSSIAN_ROUTE_HEADS = frozenset({"scf", "int", "integral"})
_FORBIDDEN_GAUSSIAN_ROUTE_FRAGMENTS = (
    "restart",
    "checkpoint",
    "chk",
    "geom",
    "guess=read",
    "opt",
    "freq",
    "force",
    "gradient",
    "output",
    "link1",
)


def normalise_gaussian_route_keywords(values: Any) -> List[str]:
    """Validate optional route tokens that preserve the fixed force/WFN job."""
    if not isinstance(values, list):
        raise ConfigValidationError(
            "gaussian.extra_route_keywords must be a list of strings"
        )
    normalised: List[str] = []
    for index, value in enumerate(values):
        label = "gaussian.extra_route_keywords[" + str(index) + "]"
        if not isinstance(value, str) or not value:
            raise ConfigValidationError(label + " must be a non-empty string")
        if value != value.strip() or any(character.isspace() for character in value):
            raise ConfigValidationError(label + " must contain exactly one route token")
        if not _SAFE_GAUSSIAN_ROUTE_TOKEN_RE.fullmatch(value):
            raise ConfigValidationError(label + " contains unsafe route syntax")
        lowered = value.lower()
        head = re.split(r"[=(]", lowered, maxsplit=1)[0]
        if head not in _SAFE_GAUSSIAN_ROUTE_HEADS:
            raise ConfigValidationError(
                label
                + " is not allowlisted; only SCF and integral-accuracy controls are supported"
            )
        if any(fragment in lowered for fragment in _FORBIDDEN_GAUSSIAN_ROUTE_FRAGMENTS):
            raise ConfigValidationError(
                label + " conflicts with the daemon's fixed force/WFN calculation"
            )
        if lowered.count("(") != lowered.count(")"):
            raise ConfigValidationError(label + " has unbalanced parentheses")
        normalised.append(value)
    return normalised


CONFIG_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION


class ConfigValidationError(ValueError):
    """Raised when campaign.yaml fails to validate."""


VALID_DESCRIPTORS = frozenset({"rmsd_massweight", "hybrid_alf_rmsd"})
VALID_GEOMETRY_NOVELTY_SCALE_SOURCES = frozenset({
    "local_motion", "movement_history", "hybrid",
})
VALID_GEOMETRY_NOVELTY_STATISTICS = frozenset({"p25", "median", "p75"})
VALID_SEED_SELECTION_STRATEGIES = frozenset({"hybrid_variance", "d_optimal"})
VALID_D_OPTIMAL_DEGENERATE_POLICIES = frozenset({"fail", "score_backfill"})
VALID_MODE_WEIGHTING_POLICIES = frozenset({"variance", "inverse_frequency", "uniform"})
VALID_GRADIENT_PARALLEL_BACKENDS = frozenset({"serial", "process"})
VALID_ERROR_CALIBRATION_MODES = frozenset({"record_only", "apply_to_acquisition"})
VALID_SPECTRAL_MODES = frozenset({"off", "record_only", "blend"})
VALID_CALIBRATED_ENERGY_UTILITIES = frozenset({"log", "banded"})
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
VALID_FEREBUS_KERNELS = frozenset({"periodic_rbf", "rbf"})
MIN_FEREBUS_ROWS_PER_SPLIT = 2
VALID_FEREBUS_PRIOR_MEAN_STRATEGIES = frozenset({
    "physical_atomic_iqa",
    "zero",
    "training_mean",
    "training_median",
})

_SYSTEM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SCHEDULER_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SLURM_MEMORY_RE = re.compile(r"^(?:auto|[1-9][0-9]*[KMGT]?)$")


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


_SIGNED_NUMERIC_PATHS = frozenset({
    "gaussian.charge",
    "quality_gates.ferebus_min_ext_r2",
})

_STRICTLY_POSITIVE_NUMERIC_PATHS = frozenset({
    "campaign.max_iterations",
    "point_allocation.bootstrap_training_size",
    "point_allocation.bootstrap_internal_validation_size",
    "point_allocation.bootstrap_external_validation_size",
    "point_allocation.batch_training_size",
    "seed_selection.n_seeds_per_iteration",
    "seed_selection.variance_chunk_size",
    "seed_selection.d_optimal_pool_multiplier",
    "seed_selection.d_optimal_jitter",
    "ferebus.nagents",
    "ferebus.maxiter",
    "ferebus.physical_prior_scale",
    "acquisition.subspace.neighbour_count",
    "acquisition.subspace.variance_capture",
    "acquisition.subspace.min_subspace_dim",
    "acquisition.subspace.max_subspace_dim",
    "acquisition.subspace.covariance_regularization",
    "acquisition.subspace.degeneracy_tolerance",
    "acquisition.stencils.step_scale",
    "acquisition.stencils.min_step",
    "acquisition.stencils.max_step",
    "acquisition.stencils.jitter",
    "acquisition.stencils.curvature_floor",
    "acquisition.stencils.softplus_scale",
    "acquisition.spectral.omega_floor",
    "acquisition.gradient.active_step",
    "acquisition.gradient.max_acquisition_grad_per_ang",
    "acquisition.references.max_reference_samples",
    "acquisition.references.floor",
    "acquisition.references.refresh_period",
    "ariadne.max_iter",
    "ariadne.gradf_tol",
    "ariadne.f_tol",
    "ariadne.delta0",
    "ariadne.delta_max",
    "ariadne.gamma",
    "ariadne.trqn_target_initial_grad_norm",
    "ariadne.trqn_retry_target_initial_grad_norm",
    "ariadne.trqn_target_initial_grad_rms",
    "ariadne.trqn_retry_target_initial_grad_rms",
    "ariadne.trqn_under_move_target_initial_grad_rms",
    "ariadne.trqn_min_objective_scale",
    "ariadne.trqn_max_objective_scale",
    "ariadne.trqn_fixed_objective_scale",
    "ariadne.trqn_geodesic_dt",
    "ariadne.trqn_geodesic_tol",
    "ariadne.trqn_bt_ic_tol",
    "ariadne.trqn_max_backtransform_iter",
    "ariadne.trqn_trust_min",
    "adversarial_safety.backtrack_points",
    "error_calibration.min_records_to_apply",
    "error_calibration.min_model_versions_to_apply",
    "error_calibration.n_bins",
    "error_calibration.min_bin_records",
    "error_calibration.max_records",
    "error_calibration.quantile",
    "quality_gates.ferebus_regression_abs_tolerance_ha",
    "runtime.poll_interval_seconds",
    "runtime.poll_interval_idle_seconds",
    "runtime.poll_sacct_error_max_ticks",
    "runtime.lease_stale_seconds",
    "runtime.postprocess_settle_attempts",
    "runtime.scheduler_command_timeout_seconds",
    "runtime.cancellation_confirmation_timeout_seconds",
    "runtime.journal_max_bytes",
    "runtime.journal_retained_files",
    "runtime.lease_heartbeat_seconds",
    "runtime.lease_heartbeat_failure_max",
    "runtime.background_readiness_timeout_seconds",
    "runtime.ledger_lock_timeout_seconds",
    "resources.memory_estimate_safety_factor",
    "resources.scheduler_usage_history_limit",
    "resources.diversity.auto_max_workers",
    "resources.diversity.target_pairs_per_worker",
    "resources.gaussian.memory_fraction_of_slurm",
    "gaussian.spin_multiplicity",
    "aimall.encomp",
    "retention.checkpoint_every_iterations",
})


def _walk_config_leaves(value: Any, prefix: str = ""):
    if is_dataclass(value):
        for config_field in fields(value):
            path = config_field.name if not prefix else prefix + "." + config_field.name
            yield from _walk_config_leaves(getattr(value, config_field.name), path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_config_leaves(item, prefix + "[" + str(index) + "]")
        return
    yield prefix, value


def _validate_numeric_domains(config: Any) -> None:
    """Apply the campaign-wide finite and baseline sign contracts."""
    for path, value in _walk_config_leaves(config):
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigValidationError(path + " must be finite")
        if isinstance(value, (int, float)):
            numeric = float(value)
            if path not in _SIGNED_NUMERIC_PATHS and numeric < 0.0:
                raise ConfigValidationError(path + " must be >= 0")
            if path in _STRICTLY_POSITIVE_NUMERIC_PATHS and numeric <= 0.0:
                raise ConfigValidationError(path + " must be > 0")


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
    nonbonded_expansion_selection_margin_ratio: float = 0.10
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


@dataclass
class AcquisitionSpectralBlock:
    mode: str = "blend"
    mode_weighting: str = "inverse_frequency"
    omega_floor: float = 1.0e-6
    low_frequency_power: float = 1.0
    max_modes: Optional[int] = None


@dataclass
class AcquisitionCalibratedEnergyBlock:
    utility: str = "banded"
    band_low_ha_per_sqrt_atom: Optional[float] = None
    band_high_ha_per_sqrt_atom: Optional[float] = None
    low_softness_ha_per_sqrt_atom: Optional[float] = None
    high_softness_ha_per_sqrt_atom: Optional[float] = None
    fallback_to_raw_variance: bool = True


@dataclass
class AcquisitionGradientBlock:
    active_step: float = 1.0e-3
    regularization: float = 1.0e-10
    max_acquisition_grad_per_ang: float = 50.0


@dataclass
class AcquisitionReferencesBlock:
    max_reference_samples: int = 24
    floor: float = 1.0e-12
    refresh_policy: str = "every_n_iterations"
    refresh_period: int = 3


@dataclass
class AcquisitionConfigBlock:
    subspace: AcquisitionSubspaceBlock = field(default_factory=AcquisitionSubspaceBlock)
    barrier: AcquisitionBarrierBlock = field(default_factory=AcquisitionBarrierBlock)
    stencils: AcquisitionStencilsBlock = field(default_factory=AcquisitionStencilsBlock)
    weights: AcquisitionWeightsBlock = field(default_factory=AcquisitionWeightsBlock)
    spectral: AcquisitionSpectralBlock = field(default_factory=AcquisitionSpectralBlock)
    calibrated_energy: AcquisitionCalibratedEnergyBlock = field(default_factory=AcquisitionCalibratedEnergyBlock)
    gradient: AcquisitionGradientBlock = field(default_factory=AcquisitionGradientBlock)
    references: AcquisitionReferencesBlock = field(default_factory=AcquisitionReferencesBlock)


@dataclass
class CampaignIdentityConfigBlock:
    # Short label for the molecular system, used to name per-atom FEREBUS
    # dataset files (<system>_<atom>_TRAINING_SET.csv and friends).
    system_name: str = "SYSTEM"
    max_iterations: int = 50
    sampling_aggressiveness: int = 5
    random_seed: int = 0
    custom_bootstrap: bool = False


@dataclass
class PointAllocationConfigBlock:
    bootstrap_training_size: int = 8
    bootstrap_internal_validation_size: int = 2
    bootstrap_external_validation_size: int = 2
    batch_training_size: int = 3
    batch_internal_validation_size: int = 1

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
    exclude_committed_seed_frames: bool = True
    recent_seed_cooldown_iterations: int = 1


@dataclass
class AntiOverlapConfigBlock:
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
    beta: float = 0.5


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
    kernel: str = "periodic_rbf"
    nagents: int = 20
    maxiter: int = 200
    prior_mean_strategy: str = "physical_atomic_iqa"
    prior_mean_level_of_theory: str = "auto"
    physical_prior_scale: float = 1.0
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
    quantile: float = 0.75
    apply_strength: float = 0.0
    group_by_atom_type: bool = True
    group_by_landing_policy: bool = False


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
    ferebus_max_aggregate_ext_rmse_increase_fraction: float = 0.05
    ferebus_max_task_ext_rmse_increase_fraction: float = 0.20
    ferebus_regression_abs_tolerance_ha: float = 1.0e-6
    ariadne_max_displacement_ang: Optional[float] = 1.25
    ariadne_min_pair_distance_ang: Optional[float] = 0.60


@dataclass
class RuntimeConfigBlock:
    poll_interval_seconds: int = 60
    poll_interval_idle_seconds: int = 120
    poll_sacct_empty_max_ticks: int = 10
    poll_sacct_error_max_ticks: int = 10
    lease_stale_seconds: int = 900
    postprocess_settle_attempts: int = 3
    postprocess_settle_seconds: int = 10
    transient_phase_retry_max: int = 1
    poll_sacct_unknown_max_ticks: int = 3
    poll_sacct_missing_max_ticks: int = 12
    poll_squeue_inconclusive_max_ticks: int = 10
    failure_threshold_fraction: float = 0.5
    halt_on_tick_exception: bool = True
    scheduler_command_timeout_seconds: int = 60
    cancellation_confirmation_timeout_seconds: int = 120
    journal_max_bytes: int = 67_108_864
    journal_retained_files: int = 8
    lease_heartbeat_seconds: int = 30
    lease_heartbeat_failure_max: int = 3
    clock_skew_tolerance_seconds: int = 60
    background_readiness_timeout_seconds: int = 60
    ledger_lock_timeout_seconds: int = 30


@dataclass
class RetentionConfigBlock:
    checkpoint_destination: Optional[str] = None
    checkpoint_every_iterations: int = 1
    checkpoint_required: bool = False
    checkpoint_verify_after_write: bool = True


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
class DiversityResourceBlock(BackendResourceBlock):
    """ICHOR diversity worker and condensed-distance storage limits."""

    auto_max_workers: int = 16
    target_pairs_per_worker: int = 5_000_000
    in_memory_distance_store_fraction: float = 0.35


@dataclass
class GaussianResourceBlock:
    partition: Optional[str] = None
    walltime_hours: Optional[Union[int, float]] = None
    cpus_per_task: Optional[Union[int, str]] = None
    mem_per_cpu: Optional[str] = None
    memory_fraction_of_slurm: float = 0.85


@dataclass
class ResourceConfigBlock:
    # Slurm resources for live phases. Schema v4 groups backend-specific
    # overrides while keeping defaults explicit. A backend value of None means
    # "inherit from resources.defaults".
    defaults: ResourceDefaultsBlock = field(default_factory=ResourceDefaultsBlock)
    diversity: DiversityResourceBlock = field(
        default_factory=lambda: DiversityResourceBlock(walltime_hours=2)
    )
    gaussian: GaussianResourceBlock = field(default_factory=lambda: GaussianResourceBlock(walltime_hours=24))
    aimall: BackendResourceBlock = field(default_factory=BackendResourceBlock)
    ariadne: BackendResourceBlock = field(default_factory=BackendResourceBlock)
    ferebus: BackendResourceBlock = field(default_factory=BackendResourceBlock)

    array_concurrency_limit: Optional[int] = None
    memory_estimate_safety_factor: float = 1.25
    scheduler_usage_telemetry: bool = True
    scheduler_usage_history_limit: int = 5000
    # "process" -> node-local process pool sized to the task's cpus-per-task.
    # "serial"  -> force single-core (off-cluster / debugging).
    gradient_parallel_backend: str = "process"

    def _backend_block(self, backend: str):
        return getattr(self, backend)

    @staticmethod
    def _inherit(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def backend_for_phase(self, phase_name: str) -> str:
        if phase_name in ("PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"):
            return "diversity"
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
    def diversity_walltime_hours(self):
        return self._get_walltime("diversity")

    @diversity_walltime_hours.setter
    def diversity_walltime_hours(self, value):
        self._set_walltime("diversity", value)

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
    def diversity_cpus_per_task(self):
        return self._get_cpus("diversity")

    @diversity_cpus_per_task.setter
    def diversity_cpus_per_task(self, value):
        self._set_cpus("diversity", value)

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
    def diversity_mem_per_cpu(self):
        return self._get_mem("diversity")

    @diversity_mem_per_cpu.setter
    def diversity_mem_per_cpu(self, value):
        self._set_mem("diversity", value)

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
    def gaussian_memory_fraction_of_slurm(self):
        return self.gaussian.memory_fraction_of_slurm

    @gaussian_memory_fraction_of_slurm.setter
    def gaussian_memory_fraction_of_slurm(self, value):
        self.gaussian.memory_fraction_of_slurm = value


@dataclass
class GaussianConfigBlock:
    # level of theory for the ab-initio training-data calculations. the
    # defaults are a reasonable starting point; set these per system in
    # campaign.yaml. Route additions are parsed as individual allowlisted
    # tokens by the staging layer.
    method: str = "B3LYP"
    basis_set: str = "aug-cc-pVTZ"
    charge: int = 0
    spin_multiplicity: int = 1
    extra_route_keywords: List[str] = field(default_factory=list)


@dataclass
class AimallConfigBlock:
    encomp: int = 3
    nogui: bool = True
    naat: Union[int, str] = "auto"
    boaq: str = "auto"
    iasmesh: str = "fine"


@dataclass
class CampaignConfig:
    """Top-level campaign configuration (schema 13)."""

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
    retention: RetentionConfigBlock = field(default_factory=RetentionConfigBlock)
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
        from .strict_yaml import StrictYamlError, load_yaml_strict

        try:
            data = load_yaml_strict(path)
        except StrictYamlError as exc:
            raise ConfigValidationError(str(exc)) from exc
        return cls.from_dict(data)

    def to_yaml(self, path):
        """Write a validated sparse campaign configuration atomically."""
        import yaml
        from .daemon.state import atomic_write_text
        self._validate()
        diff = diff_against_defaults(self)
        text = yaml.safe_dump(diff, sort_keys=True, default_flow_style=False)
        atomic_write_text(path, text)

    def to_yaml_dense(self, path):
        """Diagnostic mode dump: every nested key serialised, defaults
        included. Used by 'Show current config' menu / debug scripts;
        NOT used by the menu's Save-to-disk path."""
        import yaml
        from .daemon.state import atomic_write_text
        self._validate()
        text = yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)
        atomic_write_text(path, text)

    def _validate(self):
        _validate_numeric_domains(self)
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
        for _backend in ("diversity", "gaussian", "aimall", "ariadne", "ferebus"):
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
        _validate_positive_int(
            "resources.diversity.auto_max_workers",
            self.resources.diversity.auto_max_workers,
        )
        _validate_positive_int(
            "resources.diversity.target_pairs_per_worker",
            self.resources.diversity.target_pairs_per_worker,
        )
        if isinstance(
            self.resources.diversity.in_memory_distance_store_fraction, bool
        ) or not isinstance(
            self.resources.diversity.in_memory_distance_store_fraction,
            (int, float),
        ):
            raise ConfigValidationError(
                "resources.diversity.in_memory_distance_store_fraction must be a number"
            )
        if not 0.0 < float(
            self.resources.diversity.in_memory_distance_store_fraction
        ) <= 1.0:
            raise ConfigValidationError(
                "resources.diversity.in_memory_distance_store_fraction must be in (0, 1]"
            )
        if isinstance(self.resources.memory_estimate_safety_factor, bool) or not isinstance(
            self.resources.memory_estimate_safety_factor, (int, float)
        ):
            raise ConfigValidationError(
                "resources.memory_estimate_safety_factor must be a number"
            )
        safety_factor = float(self.resources.memory_estimate_safety_factor)
        if not math.isfinite(safety_factor) or safety_factor < 1.0:
            raise ConfigValidationError(
                "resources.memory_estimate_safety_factor must be finite and >= 1.0"
            )
        if not isinstance(self.resources.scheduler_usage_telemetry, bool):
            raise ConfigValidationError(
                "resources.scheduler_usage_telemetry must be a boolean"
            )
        _validate_positive_int(
            "resources.scheduler_usage_history_limit",
            self.resources.scheduler_usage_history_limit,
        )
        if self.resources.gradient_parallel_backend not in VALID_GRADIENT_PARALLEL_BACKENDS:
            raise ConfigValidationError(
                "resources.gradient_parallel_backend must be one of "
                + repr(sorted(VALID_GRADIENT_PARALLEL_BACKENDS))
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
        _validate_positive_int(
            "runtime.poll_sacct_error_max_ticks",
            self.runtime.poll_sacct_error_max_ticks,
        )
        allocation = self.point_allocation
        for allocation_name in (
            "bootstrap_training_size",
            "bootstrap_internal_validation_size",
        ):
            value = getattr(allocation, allocation_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigValidationError(
                    "point_allocation." + allocation_name + " must be an integer"
                )
            if value < MIN_FEREBUS_ROWS_PER_SPLIT:
                raise ConfigValidationError(
                    "point_allocation."
                    + allocation_name
                    + " must be >= "
                    + str(MIN_FEREBUS_ROWS_PER_SPLIT)
                    + " because native FEREBUS requires at least two rows "
                    "in every initial dataset"
                )
        _validate_positive_int(
            "point_allocation.batch_training_size",
            allocation.batch_training_size,
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
        if external_size < MIN_FEREBUS_ROWS_PER_SPLIT:
            raise ConfigValidationError(
                "point_allocation.bootstrap_external_validation_size must be >= "
                + str(MIN_FEREBUS_ROWS_PER_SPLIT)
                + " because native FEREBUS quality requires at least two "
                "external-validation rows"
            )
        if not isinstance(self.campaign.custom_bootstrap, bool):
            raise ConfigValidationError(
                "campaign.custom_bootstrap must be a boolean"
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
        _validate_nonnegative_int(
            "campaign.random_seed",
            self.campaign.random_seed,
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
        if not isinstance(self.seed_selection.exclude_committed_seed_frames, bool):
            raise ConfigValidationError(
                "seed_selection.exclude_committed_seed_frames must be a boolean"
            )
        _validate_nonnegative_int(
            "seed_selection.recent_seed_cooldown_iterations",
            self.seed_selection.recent_seed_cooldown_iterations,
        )
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
        for name in (
            "scheduler_command_timeout_seconds",
            "cancellation_confirmation_timeout_seconds",
            "journal_max_bytes",
            "journal_retained_files",
            "lease_heartbeat_seconds",
            "lease_heartbeat_failure_max",
            "background_readiness_timeout_seconds",
            "ledger_lock_timeout_seconds",
        ):
            _validate_positive_int("runtime." + name, getattr(self.runtime, name))
        _validate_nonnegative_int(
            "runtime.clock_skew_tolerance_seconds",
            self.runtime.clock_skew_tolerance_seconds,
        )
        heartbeat_failure_window = (
            int(self.runtime.lease_heartbeat_seconds)
            * int(self.runtime.lease_heartbeat_failure_max)
        )
        if heartbeat_failure_window >= int(self.runtime.lease_stale_seconds):
            raise ConfigValidationError(
                "runtime lease heartbeat failure window must be shorter than "
                "runtime.lease_stale_seconds"
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
        if self.ferebus.kernel not in VALID_FEREBUS_KERNELS:
            raise ConfigValidationError(
                "ferebus.kernel must be one of "
                + repr(sorted(VALID_FEREBUS_KERNELS))
            )
        if self.ferebus.prior_mean_strategy not in VALID_FEREBUS_PRIOR_MEAN_STRATEGIES:
            raise ConfigValidationError(
                "ferebus.prior_mean_strategy must be one of "
                + repr(sorted(VALID_FEREBUS_PRIOR_MEAN_STRATEGIES))
            )
        _validate_positive_int("ferebus.nagents", self.ferebus.nagents)
        _validate_positive_int("ferebus.maxiter", self.ferebus.maxiter)
        if (
            isinstance(self.ferebus.physical_prior_scale, bool)
            or not isinstance(self.ferebus.physical_prior_scale, (int, float))
            or not math.isfinite(float(self.ferebus.physical_prior_scale))
            or float(self.ferebus.physical_prior_scale) <= 0.0
        ):
            raise ConfigValidationError(
                "ferebus.physical_prior_scale must be a finite positive number"
            )
        try:
            from .ferebus_prior import (
                FerebusPriorError,
                resolve_ferebus_prior_contract,
            )

            resolve_ferebus_prior_contract(self)
        except FerebusPriorError as exc:
            raise ConfigValidationError(str(exc)) from exc
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
        if "iqa" not in seen_props:
            raise ConfigValidationError(
                "ferebus.properties must contain 'iqa'; active acquisition is fixed to IQA"
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
            if float(qg.ferebus_min_ext_r2) > 1.0:
                raise ConfigValidationError(
                    "quality_gates.ferebus_min_ext_r2 must be <= 1"
                )
        for name in (
            "ferebus_max_aggregate_ext_rmse_increase_fraction",
            "ferebus_max_task_ext_rmse_increase_fraction",
            "ferebus_regression_abs_tolerance_ha",
        ):
            _validate_optional_nonnegative_float(
                "quality_gates." + name,
                getattr(qg, name),
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
        if float(ariadne.delta_max) < float(ariadne.delta0):
            raise ConfigValidationError(
                "ariadne.delta_max must be >= ariadne.delta0"
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
        if self.acquisition.subspace.max_subspace_dim > 6:
            raise ConfigValidationError(
                "acquisition.subspace.max_subspace_dim must be <= 6"
            )
        if (
            self.acquisition.subspace.neighbour_count
            < self.acquisition.subspace.max_subspace_dim
        ):
            raise ConfigValidationError(
                "acquisition.subspace.neighbour_count must be >= max_subspace_dim "
                "(PCA needs at least max_subspace_dim neighbours to fill the subspace)"
            )
        if not 0.0 < float(self.acquisition.subspace.variance_capture) <= 1.0:
            raise ConfigValidationError(
                "acquisition.subspace.variance_capture must be in (0, 1]"
            )
        if self.acquisition.subspace.gaussian_weight_sigma is not None and float(
            self.acquisition.subspace.gaussian_weight_sigma
        ) <= 0.0:
            raise ConfigValidationError(
                "acquisition.subspace.gaussian_weight_sigma must be > 0 or null"
            )
        stencils = self.acquisition.stencils
        if float(stencils.max_step) < float(stencils.min_step):
            raise ConfigValidationError(
                "acquisition.stencils.max_step must be >= acquisition.stencils.min_step"
            )
        weights = self.acquisition.weights
        if not any(
            float(value) > 0.0
            for value in (
                weights.lambda_force,
                weights.lambda_frequency,
                weights.lambda_anharmonic,
                weights.lambda_energy,
            )
        ):
            raise ConfigValidationError(
                "at least one informative acquisition weight must be > 0"
            )
        ba = self.acquisition.barrier
        for name, value in (
            ("acquisition.barrier.nonbonded_clash_scale", ba.nonbonded_clash_scale),
            ("acquisition.barrier.clash_delta", ba.clash_delta),
            ("acquisition.barrier.clash_lambda", ba.clash_lambda),
            ("acquisition.barrier.nonbonded_expansion_scale", ba.nonbonded_expansion_scale),
            ("acquisition.barrier.nonbonded_expansion_selection_margin_ratio", ba.nonbonded_expansion_selection_margin_ratio),
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
            ("acquisition.spectral.omega_floor", spectral.omega_floor),
            ("acquisition.spectral.low_frequency_power", spectral.low_frequency_power),
        ):
            _validate_optional_nonnegative_float(name, value)
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
            ("acquisition.calibrated_energy.band_low_ha_per_sqrt_atom", cal_energy.band_low_ha_per_sqrt_atom),
            ("acquisition.calibrated_energy.band_high_ha_per_sqrt_atom", cal_energy.band_high_ha_per_sqrt_atom),
            ("acquisition.calibrated_energy.low_softness_ha_per_sqrt_atom", cal_energy.low_softness_ha_per_sqrt_atom),
            ("acquisition.calibrated_energy.high_softness_ha_per_sqrt_atom", cal_energy.high_softness_ha_per_sqrt_atom),
        ):
            _validate_optional_nonnegative_float(name, value)
        if (
            cal_energy.band_low_ha_per_sqrt_atom is not None
            and cal_energy.band_high_ha_per_sqrt_atom is not None
            and float(cal_energy.band_high_ha_per_sqrt_atom) <= float(cal_energy.band_low_ha_per_sqrt_atom)
        ):
            raise ConfigValidationError(
                "acquisition.calibrated_energy.band_high_ha_per_sqrt_atom must be > band_low_ha_per_sqrt_atom"
            )
        if cal_energy.band_high_ha_per_sqrt_atom is not None and float(cal_energy.band_high_ha_per_sqrt_atom) <= 0.0:
            raise ConfigValidationError(
                "acquisition.calibrated_energy.band_high_ha_per_sqrt_atom must be > 0"
            )
        for name, value in (
            ("acquisition.calibrated_energy.low_softness_ha_per_sqrt_atom", cal_energy.low_softness_ha_per_sqrt_atom),
            ("acquisition.calibrated_energy.high_softness_ha_per_sqrt_atom", cal_energy.high_softness_ha_per_sqrt_atom),
        ):
            if value is not None and float(value) <= 0.0:
                raise ConfigValidationError(name + " must be > 0")

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
        if self.retention.checkpoint_destination is not None:
            destination = str(self.retention.checkpoint_destination)
            if not destination.strip():
                raise ConfigValidationError(
                    "retention.checkpoint_destination must be a non-empty path or null"
                )
            if destination != destination.strip():
                raise ConfigValidationError(
                    "retention.checkpoint_destination must not have surrounding whitespace"
                )
        _validate_positive_int(
            "retention.checkpoint_every_iterations",
            self.retention.checkpoint_every_iterations,
        )
        if not isinstance(self.retention.checkpoint_required, bool):
            raise ConfigValidationError(
                "retention.checkpoint_required must be a boolean"
            )
        if not isinstance(self.retention.checkpoint_verify_after_write, bool):
            raise ConfigValidationError(
                "retention.checkpoint_verify_after_write must be a boolean"
            )
        if self.retention.checkpoint_required and self.retention.checkpoint_destination is None:
            raise ConfigValidationError(
                "retention.checkpoint_destination is required when checkpoint_required is true"
            )
        self.gaussian.extra_route_keywords = normalise_gaussian_route_keywords(
            self.gaussian.extra_route_keywords
        )
    def effective_max_acquisition_grad_per_ang(self) -> float:
        return float(self.acquisition.gradient.max_acquisition_grad_per_ang)

    def to_acquisition_config(self):
        """Materialise the mandatory mature IQA acquisition contract."""
        from ichor.core.adversarial.config import (
            AcquisitionConfig,
            BarrierConfig,
            CalibratedEnergyConfig,
            FullspaceConfinementConfig,
            GradientConfig,
            MovementBandConfig,
            MovementUtilityConfig,
            ReferenceScaleConfig,
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
        gr = self.acquisition.gradient
        re = self.acquisition.references
        return AcquisitionConfig(
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
                nonbonded_expansion_selection_margin_ratio=ba.nonbonded_expansion_selection_margin_ratio,
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
            ),
            spectral=SpectralConfig(
                mode=sp.mode,
                mode_weighting=sp.mode_weighting,
                omega_floor=sp.omega_floor,
                low_frequency_power=sp.low_frequency_power,
                max_modes=sp.max_modes,
            ),
            calibrated_energy=CalibratedEnergyConfig(
                utility=ce.utility,
                band_low_ha_per_sqrt_atom=ce.band_low_ha_per_sqrt_atom,
                band_high_ha_per_sqrt_atom=ce.band_high_ha_per_sqrt_atom,
                low_softness_ha_per_sqrt_atom=ce.low_softness_ha_per_sqrt_atom,
                high_softness_ha_per_sqrt_atom=ce.high_softness_ha_per_sqrt_atom,
                fallback_to_raw_variance=ce.fallback_to_raw_variance,
            ),
            fullspace_confinement=FullspaceConfinementConfig(
                residual_scale=FULLSPACE_RESIDUAL_SCALE,
                fixed_residual_scale_ang=FULLSPACE_FIXED_RESIDUAL_SCALE_ANGSTROM,
                rmsd_scale_ang=FULLSPACE_RMSD_SCALE_MULTIPLIER
                * self.geometry_novelty.fallback_scale_angstrom,
                min_residual_scale_ang=FULLSPACE_MIN_RESIDUAL_SCALE_ANGSTROM,
                failure_penalty=FULLSPACE_FAILURE_PENALTY,
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
            gradient=GradientConfig(
                active_step=gr.active_step,
                regularization=gr.regularization,
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
