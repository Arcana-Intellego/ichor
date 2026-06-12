"""Campaign configuration -- schema v2 (clean break from earlier v1).

Schema v2 organises the config fields into nested blocks that mirror
the logical structure of the active sampling pipeline. Every previously inactive field in v1
should now be either 
(a) wired to its downstream consumer
via the to_acquisition_config / to_ariadne_run_config translators, or (
b) explicitly deprecated.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config_dataclass import DataclassParseError, parse_dataclass_block


__all__ = [
    "CampaignConfig",
    "TrajectoryPoolConfigBlock",
    "OutlierFilterConfigBlock",
    "BatchSizingConfigBlock",
    "SeedSelectionConfigBlock",
    "AntiOverlapConfigBlock",
    "PhaseBConfigBlock",
    "SplitConfigBlock",
    "FerebusConfigBlock",
    "AcquisitionSubspaceBlock",
    "AcquisitionBarrierBlock",
    "AcquisitionStencilsBlock",
    "AcquisitionWeightsBlock",
    "AcquisitionGradientBlock",
    "AcquisitionReferencesBlock",
    "AcquisitionConfigBlock",
    "AriadneConfigBlock",
    "AdversarialSafetyConfigBlock",
    "QualityGatesConfigBlock",
    "RuntimeConfigBlock",
    "StopConfigBlock",
    "CONFIG_SCHEMA_VERSION",
    "ConfigValidationError",
    "AimallConfigBlock",
    "VALID_BATCH_POLICIES",
    "VALID_WARMSTART",
    "VALID_DESCRIPTORS",
    "VALID_SPLITS",
    "VALID_GRADIENT_MODES",
    "VALID_MODE_WEIGHTING_POLICIES",
]


CONFIG_SCHEMA_VERSION = 2


class ConfigValidationError(ValueError):
    """Raised when campaign.yaml fails to validate."""


VALID_BATCH_POLICIES = frozenset({"linear", "sqrt", "fixed"})
VALID_WARMSTART = frozenset({"always", "never", "adaptive"})
VALID_DESCRIPTORS = frozenset({
    "rmsd_massweight", "hybrid_alf_rmsd", "acquisition_weighted",
})
VALID_SPLITS = frozenset({
    "stratified_with_holdout", "random_80_20", "pure_top_k",
})
VALID_GRADIENT_MODES = frozenset({"cartesian_fd", "active_fd"})
VALID_MODE_WEIGHTING_POLICIES = frozenset({"variance", "inverse_frequency", "uniform"})
VALID_GRADIENT_PARALLEL_BACKENDS = frozenset({"serial", "thread", "process"})

_SYSTEM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SCHEDULER_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SLURM_MEMORY_RE = re.compile(r"^[1-9][0-9]*[KMGT]?$")
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
    canonicalise_basis: bool = False
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


@dataclass
class AcquisitionWeightsBlock:
    lambda_force: float = 1.0
    lambda_frequency: float = 1.5
    lambda_anharmonic: float = 1.0
    lambda_energy: float = 0.25
    lambda_distance: float = 1.0


@dataclass
class AcquisitionGradientBlock:
    mode: str = "cartesian_fd"
    cartesian_step: float = 1.0e-4
    active_step: float = 1.0e-3
    regularization: float = 1.0e-10
    cartesian_step_floor: float = 0.0
    ghost_mass_threshold: float = 0.0


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
    gradient: AcquisitionGradientBlock = field(default_factory=AcquisitionGradientBlock)
    references: AcquisitionReferencesBlock = field(default_factory=AcquisitionReferencesBlock)


@dataclass
class TrajectoryPoolConfigBlock:
    source_path: str = ""


@dataclass
class OutlierFilterConfigBlock:
    """Pre-Phase-A trajectory outlier filter.

    Applied at TrajectoryPool.import_from time to reject frames whose
    geometry or (if energies are available) energy lies far from the
    distribution mean. Output: rejected.json next to pool.manifest.json,
    plus a trajectory_pool_filtered journal event.
    """
    enabled: bool = True
    energy_z_threshold: float = 3.0
    per_atom_rmsd_z_threshold: float = 4.0


@dataclass
class BatchSizingConfigBlock:
    policy: str = "linear"
    floor: int = 5
    cap: int = 30


@dataclass
class SeedSelectionConfigBlock:
    n_seeds_per_iteration: int = 50
    bulk_fraction: float = 0.5
    variance_chunk_size: int = 512


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
    # phase_b.min_separation below. (A43)
    enforce_post_ariadne: bool = False


@dataclass
class PhaseBConfigBlock:
    descriptor: str = "hybrid_alf_rmsd"
    beta: float = 0.3
    # how close (in aligned mass-weighted RMSD) a candidate may sit to ANY existing training point
    # before we drop it from Phase-B. this is anti-overlap case (d) -- the DESIGNED way to weed out
    # near-duplicates before they reach expensive QM -- and it is on by default now (0.05) so it
    # does that job, rather than leaning on the post-ARIADNE quality flag (see
    # anti_overlap.enforce_post_ariadne). units: angstrom, or whatever the trajectory uses.
    # CALIBRATE per system -- too large and it over-drops genuinely new points. (A43)
    min_separation: float = 0.05


@dataclass
class SplitConfigBlock:
    strategy: str = "stratified_with_holdout"
    train_fraction: float = 0.75
    val_mid_fraction: float = 0.15
    high_holdout_fraction: float = 0.10


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
    # how the committed training set is carved into the three csvs FEREBUS reads
    # each retrain: training, internal validation (used during the fit), and
    # external validation (held-out benchmark). fractions of the current set.
    train_fraction: float = 0.8
    int_val_fraction: float = 0.1
    ext_val_fraction: float = 0.1


@dataclass
class AriadneConfigBlock:
    optimiser: str = "trust_region_qn"
    hessian_model: str = "ALMLOF"
    max_iter: int = 200
    gradf_tol: float = 1.0e-4
    f_tol: float = 1.0e-6
    delta0: float = 0.10
    delta_max: float = 0.40
    gamma: float = 0.10
    fallback_to_ds: bool = True


@dataclass
class AdversarialSafetyConfigBlock:
    enabled: bool = True
    reject_unsafe_landings: bool = True
    salvage_safe_iterate: bool = True
    backtrack_to_safe_landing: bool = True
    backtrack_points: int = 16
    allow_seed_fallback: bool = False
    min_whitened_distance: float = 0.0
    max_whitened_distance: float = 10.0
    enforce_min_whitened_distance: bool = False
    max_predicted_energy_delta_ha: Optional[float] = None
    max_energy_variance: Optional[float] = None
    max_chemistry_penalty: Optional[float] = None
    phase_b_filter_enabled: bool = True


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
    lease_stale_seconds: int = 900
    postprocess_settle_attempts: int = 3
    postprocess_settle_seconds: int = 10
    transient_phase_retry_max: int = 1
    poll_sacct_unknown_max_ticks: int = 3


@dataclass
class StopConfigBlock:
    alpha0_streak_threshold: float = 1.0e-2
    alpha0_streak_length: int = 5
    rel_alpha_improvement_min: float = 0.02
    rel_alpha_improvement_window: int = 3
    min_iterations_before_stop: int = 8


@dataclass
class ResourceConfigBlock:
    # SLURM resources for the sbatch phases. defaults suit the small CSF4 smoke
    # jobs; bump them per campaign. ARIADNE gets its own core count because its
    # array tasks parallelise the acquisition gradient across those cores.
    partition: str = "multicore"
    walltime_hours: int = 24
    # per-core memory, NOT per-job. CSF4 multicore hands memory out per core and a job-level
    # --mem gets rejected/ignored there, so we emit this as --mem-per-cpu. the total a task gets
    # is roughly this * cpus_per_task. keep the sbatch unit style (4G, 8G) -- gaussian.mem below
    # uses gaussian's own "8GB" style, the two are deliberately different conventions.
    mem_per_cpu: str = "8G"
    cpus_per_task: int = 1
    ntasks: int = 1
    ariadne_cpus_per_task: int = 8
    # "process" -> node-local process pool sized to the task's cpus-per-task.
    # "serial"  -> force single-core (off-cluster / debugging).
    gradient_parallel_backend: str = "process"

    def cpus_for(self, phase_name: str) -> int:
        if phase_name == "ARIADNE_ARRAY":
            return int(self.ariadne_cpus_per_task)
        return int(self.cpus_per_task)


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
    nproc: int = 1
    mem: str = "8GB"


@dataclass
class AimallConfigBlock:
    encomp: int = 3
    nogui: bool = True


@dataclass
class CampaignConfig:
    """Top-level campaign configuration (schema v2)."""

    schema_version: int = CONFIG_SCHEMA_VERSION

    # short label for the molecular system, used to name the per-atom FEREBUS
    # dataset files (<system>_<atom>_TRAINING_SET.csv and friends). keep it a
    # bare token, no spaces.
    system_name: str = "SYSTEM"

    max_iterations: int = 50
    poll_interval_seconds: int = 60
    poll_interval_idle_seconds: int = 120
    #Number of consecutive ticks of sacct returning n_tasks=0 before
    #the daemon escalates a pending job to UNKNOWN/failure. SLURM accounting
    #records can age out after ~24h on CSF4; without this the
    #daemon would poll indefinitely. Set to 0 to disable the escalation.
    poll_sacct_empty_max_ticks: int = 10

    initial_train_size: int = 250
    initial_val_size: int = 50

    failure_threshold_fraction: float = 0.5
    max_acquisition_grad_per_ang: Optional[float] = None
    max_force_per_atom_ha_per_ang: float = 50.0

    trajectory_pool: TrajectoryPoolConfigBlock = field(
        default_factory=TrajectoryPoolConfigBlock
    )
    outlier_filter: OutlierFilterConfigBlock = field(
        default_factory=OutlierFilterConfigBlock
    )
    batch_sizing: BatchSizingConfigBlock = field(
        default_factory=BatchSizingConfigBlock
    )
    seed_selection: SeedSelectionConfigBlock = field(
        default_factory=SeedSelectionConfigBlock
    )
    anti_overlap: AntiOverlapConfigBlock = field(
        default_factory=AntiOverlapConfigBlock
    )
    phase_b: PhaseBConfigBlock = field(default_factory=PhaseBConfigBlock)
    split: SplitConfigBlock = field(default_factory=SplitConfigBlock)
    ferebus: FerebusConfigBlock = field(default_factory=FerebusConfigBlock)
    acquisition: AcquisitionConfigBlock = field(
        default_factory=AcquisitionConfigBlock
    )
    ariadne: AriadneConfigBlock = field(default_factory=AriadneConfigBlock)
    adversarial_safety: AdversarialSafetyConfigBlock = field(
        default_factory=AdversarialSafetyConfigBlock
    )
    quality_gates: QualityGatesConfigBlock = field(
        default_factory=QualityGatesConfigBlock
    )
    runtime: RuntimeConfigBlock = field(default_factory=RuntimeConfigBlock)
    stop: StopConfigBlock = field(default_factory=StopConfigBlock)
    resources: ResourceConfigBlock = field(default_factory=ResourceConfigBlock)
    gaussian: GaussianConfigBlock = field(default_factory=GaussianConfigBlock)
    aimall: AimallConfigBlock = field(default_factory=AimallConfigBlock)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ConfigValidationError(
                "campaign.yaml must be a mapping at the top level"
            )
        schema = int(data.get("schema_version", -1))
        if schema != CONFIG_SCHEMA_VERSION:
            raise ConfigValidationError(
                "campaign.yaml schema_version " + str(schema)
                + " != " + str(CONFIG_SCHEMA_VERSION)
            )
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
            "system_name",
            self.system_name,
            _SYSTEM_NAME_RE,
            "a filename-safe token matching ^[A-Za-z0-9][A-Za-z0-9_-]*$",
        )
        _validate_token(
            "resources.partition",
            self.resources.partition,
            _SCHEDULER_TOKEN_RE,
            "a scheduler token containing only letters, numbers, '.', '_', ':' and '-'",
        )
        _validate_positive_int("resources.walltime_hours", self.resources.walltime_hours)
        _validate_positive_int("resources.cpus_per_task", self.resources.cpus_per_task)
        _validate_positive_int("resources.ntasks", self.resources.ntasks)
        _validate_positive_int(
            "resources.ariadne_cpus_per_task",
            self.resources.ariadne_cpus_per_task,
        )
        _validate_memory(
            "resources.mem_per_cpu",
            self.resources.mem_per_cpu,
            _SLURM_MEMORY_RE,
            "SLURM memory syntax such as 4G or 4000M",
        )
        if self.resources.gradient_parallel_backend not in VALID_GRADIENT_PARALLEL_BACKENDS:
            raise ConfigValidationError(
                "resources.gradient_parallel_backend must be one of "
                + repr(sorted(VALID_GRADIENT_PARALLEL_BACKENDS))
            )
        _validate_positive_int("gaussian.nproc", self.gaussian.nproc)
        _validate_memory(
            "gaussian.mem",
            self.gaussian.mem,
            _GAUSSIAN_MEMORY_RE,
            "Gaussian memory syntax such as 8GB or 8000MB",
        )
        _validate_positive_int("aimall.encomp", self.aimall.encomp)
        if not isinstance(self.aimall.nogui, bool):
            raise ConfigValidationError("aimall.nogui must be a boolean")
        if self.max_iterations <= 0:
            raise ConfigValidationError("max_iterations must be > 0")
        if self.poll_interval_seconds < 1:
            raise ConfigValidationError("poll_interval_seconds must be >= 1")
        if self.poll_interval_idle_seconds < 1:
            raise ConfigValidationError("poll_interval_idle_seconds must be >= 1")
        if self.poll_sacct_empty_max_ticks < 0:
            raise ConfigValidationError("poll_sacct_empty_max_ticks must be >= 0")
        if self.initial_train_size <= 0 or self.initial_val_size < 0:
            raise ConfigValidationError("initial train/val sizes must be positive")
        if self.batch_sizing.policy not in VALID_BATCH_POLICIES:
            raise ConfigValidationError(
                "batch_sizing.policy must be one of " + repr(sorted(VALID_BATCH_POLICIES))
            )
        if (
            self.batch_sizing.floor < 1
            or self.batch_sizing.cap < self.batch_sizing.floor
        ):
            raise ConfigValidationError(
                "batch_sizing.cap must be >= batch_sizing.floor >= 1"
            )
        if self.seed_selection.n_seeds_per_iteration <= 0:
            raise ConfigValidationError(
                "seed_selection.n_seeds_per_iteration must be > 0"
            )
        if not 0.0 <= self.seed_selection.bulk_fraction <= 1.0:
            raise ConfigValidationError(
                "seed_selection.bulk_fraction must be in [0, 1]"
            )
        _validate_positive_int(
            "seed_selection.variance_chunk_size",
            self.seed_selection.variance_chunk_size,
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
        if self.phase_b.min_separation < 0.0:
            raise ConfigValidationError(
                "phase_b.min_separation must be >= 0"
            )
        if self.split.strategy not in VALID_SPLITS:
            raise ConfigValidationError(
                "split.strategy must be one of " + repr(sorted(VALID_SPLITS))
            )
        for frac_name, frac in (
            ("split.train_fraction", self.split.train_fraction),
            ("split.val_mid_fraction", self.split.val_mid_fraction),
            ("split.high_holdout_fraction", self.split.high_holdout_fraction),
            ("failure_threshold_fraction", self.failure_threshold_fraction),
        ):
            if not 0.0 <= frac <= 1.0:
                raise ConfigValidationError(frac_name + " must be in [0, 1]")
        if float(self.split.train_fraction) + float(self.split.val_mid_fraction) > 1.0:
            raise ConfigValidationError(
                "split.train_fraction + split.val_mid_fraction must be <= 1"
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
        for frac_name, frac in (
            ("ferebus.train_fraction", self.ferebus.train_fraction),
            ("ferebus.int_val_fraction", self.ferebus.int_val_fraction),
            ("ferebus.ext_val_fraction", self.ferebus.ext_val_fraction),
        ):
            if not 0.0 <= float(frac) <= 1.0:
                raise ConfigValidationError(frac_name + " must be in [0, 1]")
        ferebus_sum = (
            float(self.ferebus.train_fraction)
            + float(self.ferebus.int_val_fraction)
            + float(self.ferebus.ext_val_fraction)
        )
        if abs(ferebus_sum - 1.0) > 1.0e-9:
            raise ConfigValidationError(
                "ferebus train/internal/external fractions must sum to 1.0"
            )
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
            "adversarial_safety.enforce_min_whitened_distance",
            "adversarial_safety.phase_b_filter_enabled",
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
        if self.max_acquisition_grad_per_ang is not None:
            _validate_optional_nonnegative_float(
                "max_acquisition_grad_per_ang",
                self.max_acquisition_grad_per_ang,
            )
            if float(self.max_acquisition_grad_per_ang) <= 0.0:
                raise ConfigValidationError(
                    "max_acquisition_grad_per_ang must be > 0"
                )
        if self.max_force_per_atom_ha_per_ang <= 0:
            raise ConfigValidationError(
                "max_force_per_atom_ha_per_ang must be > 0"
            )
        if (
            self.max_acquisition_grad_per_ang is not None
            and float(self.max_force_per_atom_ha_per_ang) != 50.0
            and abs(
                float(self.max_acquisition_grad_per_ang)
                - float(self.max_force_per_atom_ha_per_ang)
            ) > 1.0e-12
        ):
            raise ConfigValidationError(
                "max_acquisition_grad_per_ang conflicts with deprecated "
                "max_force_per_atom_ha_per_ang; set only one clamp field"
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
        # gaussian's %NProcShared and the SLURM --cpus-per-task it runs under are separate knobs with
        # no link, so an operator can set gaussian.nproc=8 while cpus_per_task stays 1 -> 8 threads on
        # 1 allocated core: oversubscription, or a cgroup kill. cross-check so the mismatch is caught
        # at load, not on the cluster (A9).
        if int(self.gaussian.nproc) > int(self.resources.cpus_per_task):
            raise ConfigValidationError(
                "gaussian.nproc (" + str(self.gaussian.nproc) + ") must be <= "
                "resources.cpus_per_task (" + str(self.resources.cpus_per_task) + ") -- "
                "%NProcShared threads would oversubscribe the cores SLURM gives the task"
            )
        gaussian_mem_mib = _memory_mebibytes("gaussian.mem", self.gaussian.mem, gaussian=True)
        slurm_mem_mib = _memory_mebibytes(
            "resources.mem_per_cpu", self.resources.mem_per_cpu, gaussian=False
        ) * float(self.resources.cpus_per_task)
        if gaussian_mem_mib > slurm_mem_mib:
            raise ConfigValidationError(
                "gaussian.mem ("
                + str(self.gaussian.mem)
                + ") exceeds SLURM allocation resources.mem_per_cpu * cpus_per_task ("
                + str(self.resources.mem_per_cpu)
                + " * "
                + str(self.resources.cpus_per_task)
                + ")"
            )

    def effective_max_acquisition_grad_per_ang(self) -> float:
        if self.max_acquisition_grad_per_ang is not None:
            return float(self.max_acquisition_grad_per_ang)
        return float(self.max_force_per_atom_ha_per_ang)

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
            GradientConfig,
            ReferenceScaleConfig,
            StencilConfig,
            SubspaceConfig,
            WeightConfig,
        )
        sb = self.acquisition.subspace
        ba = self.acquisition.barrier
        st = self.acquisition.stencils
        we = self.acquisition.weights
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
            ),
            weights=WeightConfig(
                lambda_force=we.lambda_force,
                lambda_frequency=we.lambda_frequency,
                lambda_anharmonic=we.lambda_anharmonic,
                lambda_energy=we.lambda_energy,
                lambda_distance=we.lambda_distance,
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
        )
