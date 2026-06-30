"""Campaign configuration lock and guarded recovery checks.

The active-learning daemon is allowed to restart after crashes, scheduler
failures, and deliberate operator stops. It must not silently continue a
campaign after a protocol-changing edit to ``campaign.yaml``. This module
stores a canonical, default-expanded config snapshot and classifies later
edits before ``reconcile --apply`` promotes a proposed recovery state.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..campaign_migrations import migrate_campaign_payload
from ..config import CampaignConfig
from ..versioning.training_set import TrainingSetVersioning
from .artifact_contracts import (
    verify_committed_model_version,
    verify_committed_training_version,
)
from .state import CampaignPhase, CampaignState, atomic_write_json


CONFIG_LOCK_SCHEMA_VERSION = 1
CONFIG_LOCK_FILENAME = "config_lock.json"
CONFIG_LOCK_POLICY_VERSION = 1


@dataclass(frozen=True)
class ConfigChange:
    path: str
    old: Any
    new: Any
    category: str
    allowed: bool
    reason: str


@dataclass
class ConfigLockReview:
    lock_path: Path
    lock_existed: bool
    allowed_changes: List[ConfigChange] = field(default_factory=list)
    blocked_changes: List[ConfigChange] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return not self.blocked_changes

    @property
    def changed(self) -> bool:
        return bool(self.allowed_changes or self.blocked_changes)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_lock_path(campaign_dir: Union[str, Path]) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / CONFIG_LOCK_FILENAME
    )


def canonical_config(config: CampaignConfig) -> Dict[str, Any]:
    return config.to_dict()


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def config_fingerprint(config_payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config_payload).encode("utf-8")).hexdigest()


def _lock_payload(config: CampaignConfig, *, created_at_iso: Optional[str] = None) -> Dict[str, Any]:
    payload = canonical_config(config)
    now = _now_iso()
    return {
        "schema_version": CONFIG_LOCK_SCHEMA_VERSION,
        "campaign_schema_version": int(payload.get("schema_version", -1)),
        "created_at_iso": created_at_iso or now,
        "last_checked_at_iso": now,
        "canonical_config": payload,
        "fingerprint_sha256": config_fingerprint(payload),
        "field_policy_version": CONFIG_LOCK_POLICY_VERSION,
    }


def write_config_lock(campaign_dir: Union[str, Path], config: CampaignConfig) -> Path:
    path = config_lock_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = None
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            created = str(existing.get("created_at_iso") or "") or None
        except Exception:
            created = None
    atomic_write_json(path, _lock_payload(config, created_at_iso=created))
    return path


def ensure_config_lock(campaign_dir: Union[str, Path], config: CampaignConfig) -> Path:
    path = config_lock_path(campaign_dir)
    if not path.is_file():
        return write_config_lock(campaign_dir, config)
    return path


def _flatten(payload: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            path = (prefix + "." + str(key)) if prefix else str(key)
            out.update(_flatten(value, path))
        return out
    return {prefix: payload}


def _diff(old: Dict[str, Any], new: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    old_flat = _flatten(old)
    new_flat = _flatten(new)
    paths = sorted(set(old_flat) | set(new_flat))
    return [
        (path, old_flat.get(path), new_flat.get(path))
        for path in paths
        if old_flat.get(path) != new_flat.get(path)
    ]


def _matches(path: str, exact: Iterable[str], prefixes: Iterable[str]) -> bool:
    return path in set(exact) or any(path.startswith(prefix) for prefix in prefixes)


ALWAYS_SAFE_EXACT = {
    "max_iterations",
    "poll_interval_seconds",
    "poll_interval_idle_seconds",
    "poll_sacct_empty_max_ticks",
}
ALWAYS_SAFE_PREFIXES = {
    "runtime.",
    "stop.",
    "error_calibration.",
}
ALWAYS_SAFE_RESOURCE_EXACT = {
    "resources.defaults.walltime_hours",
    "resources.polus.walltime_hours",
    "resources.gaussian.walltime_hours",
    "resources.aimall.walltime_hours",
    "resources.ariadne.walltime_hours",
    "resources.ferebus.walltime_hours",
    "resources.array_concurrency_limit",
    "resources.gradient_parallel_backend",
    "resources.fail_on_memory_estimate_exceeds_request",
}

FUTURE_SAFE_PREFIXES = {
    "seed_selection.",
    "batch_sizing.",
    "anti_overlap.",
    "phase_b.",
    "adversarial_safety.",
    "ariadne.",
    "acquisition.subspace.",
    "acquisition.barrier.",
    "acquisition.stencils.",
    "acquisition.weights.",
    "acquisition.spectral.",
    "acquisition.calibrated_energy.",
    "acquisition.fullspace_confinement.",
    "acquisition.size_normalisation.",
    "acquisition.movement_band.",
    "acquisition.movement_utility.",
    "acquisition.gradient.",
    "acquisition.references.",
    "acquisition.driver.",
}
FUTURE_SAFE_EXACT = {
    "resources.defaults.partition",
    "resources.defaults.cpus_per_task",
    "resources.defaults.mem_per_cpu",
    "resources.polus.partition",
    "resources.gaussian.partition",
    "resources.aimall.partition",
    "resources.ariadne.partition",
    "resources.ferebus.partition",
    "resources.polus.cpus_per_task",
    "resources.gaussian.cpus_per_task",
    "resources.aimall.cpus_per_task",
    "resources.ariadne.cpus_per_task",
    "resources.ferebus.cpus_per_task",
    "resources.polus.mem_per_cpu",
    "resources.gaussian.mem_per_cpu",
    "resources.aimall.mem_per_cpu",
    "resources.ariadne.mem_per_cpu",
    "resources.ferebus.mem_per_cpu",
    "resources.gaussian.memory_mode",
    "resources.gaussian.link0_mem",
    "resources.gaussian.memory_fraction_of_slurm",
    "acquisition.use_scaled_posterior_covariance",
    "acquisition.allow_uniform_posterior_fallback",
}

ARIADNE_OUTPUT_INTERPRETATION_PREFIXES = {
    "adversarial_safety.",
    "ariadne.",
    "acquisition.subspace.",
    "acquisition.barrier.",
    "acquisition.stencils.",
    "acquisition.weights.",
    "acquisition.spectral.",
    "acquisition.calibrated_energy.",
    "acquisition.fullspace_confinement.",
    "acquisition.size_normalisation.",
    "acquisition.movement_band.",
    "acquisition.movement_utility.",
    "acquisition.gradient.",
    "acquisition.references.",
    "acquisition.driver.",
}
ARIADNE_OUTPUT_INTERPRETATION_EXACT = {
    "acquisition.use_scaled_posterior_covariance",
    "acquisition.allow_uniform_posterior_fallback",
    "quality_gates.ariadne_max_displacement_ang",
    "quality_gates.ariadne_min_pair_distance_ang",
    "max_acquisition_grad_per_ang",
    "max_force_per_atom_ha_per_ang",
}

PHASE_B_OUTPUT_INTERPRETATION_PREFIXES = {
    "anti_overlap.",
    "phase_b.",
}
PHASE_B_OUTPUT_INTERPRETATION_EXACT = {
    "adversarial_safety.phase_b_filter_enabled",
}

PHASE_LOCAL_EXACT = {
    "ferebus.warmstart",
    "ferebus.warmstart_streak",
    "ferebus.kernel",
    "ferebus.loss",
    "ferebus.nagents",
    "ferebus.maxiter",
    "ferebus.is_constant_noise",
    "ferebus.scaling",
    "ferebus.full_ARD",
    "quality_gates.ferebus_min_ext_r2",
    "quality_gates.ferebus_max_ext_rmse_ha",
    "quality_gates.ferebus_max_condition_number",
    "aimall.encomp",
    "aimall.nogui",
    "aimall.naat",
    "aimall.boaq",
    "aimall.iasmesh",
}

COMMITTED_LOCKED_EXACT = {
    "trajectory_pool.source_path",
    "outlier_filter.enabled",
    "outlier_filter.energy_z_threshold",
    "outlier_filter.per_atom_rmsd_z_threshold",
    "split.strategy",
    "split.train_fraction",
    "split.val_mid_fraction",
    "split.high_holdout_fraction",
    "ferebus.properties",
    "ferebus.train_fraction",
    "ferebus.int_val_fraction",
    "ferebus.ext_val_fraction",
    "acquisition.property_name",
    "quality_gates.require_readable_aimall_geometry",
    "quality_gates.require_finite_iqa",
    "quality_gates.require_finite_integration_error",
    "quality_gates.max_abs_integration_error",
    "quality_gates.iqa_energy_recovery_tolerance_ha",
    "quality_gates.ariadne_max_displacement_ang",
    "quality_gates.ariadne_min_pair_distance_ang",
}

CAMPAIGN_LOCKED_EXACT = {
    "schema_version",
    "system_name",
    "initial_train_size",
    "initial_val_size",
    "gaussian.method",
    "gaussian.basis_set",
    "gaussian.charge",
    "gaussian.spin_multiplicity",
    "gaussian.extra_keywords",
    "failure_threshold_fraction",
    "max_acquisition_grad_per_ang",
    "max_force_per_atom_ha_per_ang",
}


def _committed_model_exists(campaign_dir: Union[str, Path], version: int) -> bool:
    models = Path(campaign_dir) / "6_TRAINED_MODELS"
    if not models.is_dir():
        return False
    try:
        committed = TrainingSetVersioning(models).list_committed_versions()
    except Exception:
        return False
    return int(version) in {int(v) for v in committed}


def _phase_intent_exists(
    campaign_dir: Union[str, Path],
    phase: CampaignPhase,
    iteration: int,
) -> bool:
    try:
        from . import submission_intent as _submission_intent

        path = _submission_intent.intent_path(campaign_dir, phase.value, int(iteration))
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return True
        status = str(payload.get("status") or "")
        return status not in {"SUPERSEDED", "FAILED"}
    except Exception:
        # A malformed intent is still evidence that this phase has begun.
        return True


def _iteration_dir(campaign_dir: Union[str, Path], iteration: int) -> Path:
    return (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )


def _ariadne_outputs_exist(campaign_dir: Union[str, Path], proposed_state: CampaignState) -> bool:
    iteration = int(getattr(proposed_state, "iteration", 0))
    iter_dir = _iteration_dir(campaign_dir, iteration)
    if _phase_intent_exists(campaign_dir, CampaignPhase.ARIADNE_ARRAY, iteration):
        return True
    for name in (
        "ARIADNE_RESULTS.json",
        "ARIADNE_LANDING_AUDIT.json",
        "ACQUISITION_MATURITY_AUDIT.json",
        "ERROR_CALIBRATION_AUDIT.json",
    ):
        if (iter_dir / name).is_file():
            return True
    pool = iter_dir / "pool"
    if pool.is_dir():
        for result in pool.glob("seed_*/result.json"):
            if result.is_file():
                return True
    return False


def _phase_b_outputs_exist(campaign_dir: Union[str, Path], proposed_state: CampaignState) -> bool:
    iteration = int(getattr(proposed_state, "iteration", 0))
    iter_dir = _iteration_dir(campaign_dir, iteration)
    if _phase_intent_exists(campaign_dir, CampaignPhase.PHASE_B_POLUS, iteration):
        return True
    for name in (
        "PHASE_B_SELECTION.json",
        "phase_b_SAMPLE.xyz",
        "phase_b_SAMPLE_raw.xyz",
        "phase_b_dedup.json",
    ):
        if (iter_dir / name).exists():
            return True
    return False


def _active_iteration_committed(proposed_state: CampaignState, iteration: int) -> bool:
    """Return true when active-loop artefacts for ``iteration`` are committed.

    The initial diverse set is version 0. Active iteration ``i`` appends a new
    training/model version ``i + 1``, so ARIADNE/Phase B outputs for iteration
    ``i`` are historical only once both versions have reached that value.
    """
    try:
        training_version = int(getattr(proposed_state, "training_set_version", -1))
        models_version = int(getattr(proposed_state, "models_version", -1))
    except (TypeError, ValueError):
        return False
    return min(training_version, models_version) >= int(iteration) + 1


def _phase_outputs_lock_change(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
    phase: CampaignPhase,
    exists_fn,
) -> bool:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if not bool(exists_fn(campaign_dir, proposed_state)):
        return False
    if proposed_state.phase is phase:
        return True
    if proposed_state.phase in {CampaignPhase.HALTED, CampaignPhase.STOP_CHECK}:
        return not _active_iteration_committed(proposed_state, iteration)
    return False


def _blocks_existing_phase_outputs(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
    path: str,
) -> Optional[str]:
    if (
        _matches(
            path,
            ARIADNE_OUTPUT_INTERPRETATION_EXACT,
            ARIADNE_OUTPUT_INTERPRETATION_PREFIXES,
        )
        and _phase_outputs_lock_change(
            campaign_dir,
            proposed_state,
            CampaignPhase.ARIADNE_ARRAY,
            _ariadne_outputs_exist,
        )
    ):
        return "field affects in-flight or uncommitted ARIADNE outputs; discard or rerun the phase before changing it"
    if (
        _matches(
            path,
            PHASE_B_OUTPUT_INTERPRETATION_EXACT,
            PHASE_B_OUTPUT_INTERPRETATION_PREFIXES,
        )
        and _phase_outputs_lock_change(
            campaign_dir,
            proposed_state,
            CampaignPhase.PHASE_B_POLUS,
            _phase_b_outputs_exist,
        )
    ):
        return "field affects in-flight or uncommitted Phase B outputs; discard or rerun the phase before changing it"
    return None


def _phase_local_allowed(
    campaign_dir: Union[str, Path],
    path: str,
    proposed_state: CampaignState,
) -> Tuple[bool, str]:
    phase = proposed_state.phase
    if path.startswith("ferebus.") or path.startswith("quality_gates.ferebus_"):
        if phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
            return False, "FEREBUS settings may change only when re-entering a FEREBUS phase"
        target = int(proposed_state.training_set_version)
        if _committed_model_exists(campaign_dir, target):
            return False, "target FEREBUS model version is already committed"
        return True, "allowed for uncommitted FEREBUS re-entry"
    if path.startswith("gaussian."):
        if phase not in (CampaignPhase.INITIAL_GAUSSIAN, CampaignPhase.GAUSSIAN):
            return False, "Gaussian runtime settings may change only when re-entering Gaussian"
        return True, "allowed for Gaussian re-entry before acceptance"
    if path.startswith("aimall."):
        if phase not in (CampaignPhase.INITIAL_AIMALL, CampaignPhase.AIMALL):
            return False, "AIMAll settings may change only when re-entering AIMAll"
        return True, "allowed for AIMAll re-entry before acceptance"
    return False, "phase-local field is not recognised for this re-entry phase"


def _classify_change(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
    path: str,
    old: Any,
    new: Any,
) -> ConfigChange:
    if _matches(path, ALWAYS_SAFE_EXACT | ALWAYS_SAFE_RESOURCE_EXACT, ALWAYS_SAFE_PREFIXES):
        return ConfigChange(path, old, new, "always_safe", True, "safe runtime/diagnostic change")
    if _matches(path, FUTURE_SAFE_EXACT, FUTURE_SAFE_PREFIXES):
        blocked_reason = _blocks_existing_phase_outputs(campaign_dir, proposed_state, path)
        if blocked_reason:
            return ConfigChange(
                path,
                old,
                new,
                "postprocess_locked",
                False,
                blocked_reason,
            )
        return ConfigChange(path, old, new, "future_safe", True, "allowed for future phase execution")
    if path in PHASE_LOCAL_EXACT:
        allowed, reason = _phase_local_allowed(campaign_dir, path, proposed_state)
        return ConfigChange(path, old, new, "phase_local", allowed, reason)
    if path in COMMITTED_LOCKED_EXACT:
        return ConfigChange(path, old, new, "committed_locked", False, "field affects committed artefact contracts")
    if path in CAMPAIGN_LOCKED_EXACT:
        return ConfigChange(path, old, new, "campaign_locked", False, "field is locked after campaign start")
    return ConfigChange(path, old, new, "unclassified_locked", False, "field has no configured mid-campaign change policy")


def review_config_changes(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    proposed_state: CampaignState,
    *,
    initialise_missing: bool = False,
) -> ConfigLockReview:
    path = config_lock_path(campaign_dir)
    if not path.is_file():
        review = ConfigLockReview(lock_path=path, lock_existed=False)
        if initialise_missing:
            write_config_lock(campaign_dir, config)
            review.notes.append("config lock was missing and has been initialised from current campaign.yaml")
        else:
            review.notes.append("config lock is missing")
        return review
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
        old_config = lock.get("canonical_config")
        if not isinstance(old_config, dict):
            raise ValueError("canonical_config missing")
    except Exception as exc:
        review = ConfigLockReview(lock_path=path, lock_existed=True)
        review.blocked_changes.append(
            ConfigChange(
                "config_lock",
                "<unreadable>",
                "<current>",
                "lock_invalid",
                False,
                "config lock is unreadable: " + type(exc).__name__ + ": " + str(exc)[:160],
            )
        )
        return review

    try:
        old_config = CampaignConfig.from_dict(
            migrate_campaign_payload(old_config)
        ).to_dict()
        new_config = CampaignConfig.from_dict(
            migrate_campaign_payload(canonical_config(config))
        ).to_dict()
    except Exception as exc:
        review = ConfigLockReview(lock_path=path, lock_existed=True)
        review.blocked_changes.append(
            ConfigChange(
                "config_lock",
                "<unmigrated>",
                "<current>",
                "lock_invalid",
                False,
                "config lock migration failed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:160],
            )
        )
        return review
    review = ConfigLockReview(lock_path=path, lock_existed=True)
    for dotted, old, new in _diff(old_config, new_config):
        change = _classify_change(campaign_dir, proposed_state, dotted, old, new)
        if change.allowed:
            review.allowed_changes.append(change)
        else:
            review.blocked_changes.append(change)
    return review


def assert_config_unchanged_for_start(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    state: Optional[CampaignState],
) -> ConfigLockReview:
    if state is None:
        ensure_config_lock(campaign_dir, config)
        return ConfigLockReview(lock_path=config_lock_path(campaign_dir), lock_existed=True)
    review = review_config_changes(campaign_dir, config, state, initialise_missing=True)
    if review.changed:
        return review
    if review.lock_existed:
        write_config_lock(campaign_dir, config)
    return review


def _ensure_inside_campaign(campaign_dir: Path, target: Path) -> None:
    campaign = campaign_dir.resolve()
    resolved = target.resolve()
    if resolved != campaign and campaign not in resolved.parents:
        raise ValueError("refusing to clean path outside campaign: " + str(target))


def clean_reentry_staging(campaign_dir: Union[str, Path], phase: CampaignPhase) -> List[str]:
    campaign = Path(campaign_dir)
    removed: List[str] = []
    if phase in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        target = campaign / "6_TRAINED_MODELS" / "iteration-staging"
        if target.exists():
            _ensure_inside_campaign(campaign, target)
            shutil.rmtree(target)
            removed.append(str(target))
    scripts = campaign / ".DATA" / "SCRIPTS"
    if scripts.is_dir():
        _ensure_inside_campaign(campaign, scripts)
        for script in sorted(scripts.glob("*.sh")):
            if script.is_file():
                script.unlink()
                removed.append(str(script))
    return removed


def archive_scripts_for_reconcile(campaign_dir: Union[str, Path]) -> List[str]:
    campaign = Path(campaign_dir)
    scripts = campaign / ".DATA" / "SCRIPTS"
    if not scripts.exists():
        return []
    if scripts.is_symlink():
        raise ValueError("refusing to archive symlinked .DATA/SCRIPTS")
    if not scripts.is_dir():
        raise ValueError(".DATA/SCRIPTS is not a directory")
    children = [p for p in scripts.iterdir() if p.name not in (".", "..")]
    if not children:
        return []
    _ensure_inside_campaign(campaign, scripts)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = scripts.with_name(scripts.name + ".before-reconcile-" + stamp)
    suffix = 1
    while target.exists():
        target = scripts.with_name(
            scripts.name + ".before-reconcile-" + stamp + "." + str(suffix)
        )
        suffix += 1
    scripts.rename(target)
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "OUTPUTS").mkdir(exist_ok=True)
    (scripts / "ERRORS").mkdir(exist_ok=True)
    return [str(target)]


def clean_model_iteration_staging_for_reconcile(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> List[str]:
    campaign = Path(campaign_dir)
    target = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    if not target.exists():
        return []
    if target.is_symlink():
        raise ValueError("refusing to remove symlinked model iteration-staging")
    if not target.is_dir():
        raise ValueError("model iteration-staging is not a directory")
    _ensure_inside_campaign(campaign, target)
    try:
        from .live_executor import validate_ferebus_completed

        ok, reason = validate_ferebus_completed(target)
    except Exception:
        ok, reason = False, "validation unavailable"
    if ok:
        raise ValueError(
            "refusing to remove completed FEREBUS iteration-staging; "
            "postprocess or commit it first"
        )
    if proposed_state.phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        try:
            model_version = int(proposed_state.models_version)
        except (TypeError, ValueError) as exc:
            raise ValueError("models_version is not an integer") from exc
        if model_version < 0:
            raise ValueError("models_version is negative; cannot verify committed model")
        verify_committed_model_version(campaign, model_version)
    shutil.rmtree(target)
    return [str(target)]


def ferebus_reentry_can_archive_data_staging(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Tuple[bool, str]:
    if proposed_state.phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        return False, "only FEREBUS re-entry may archive .DATA/STAGING"
    if proposed_state.pending_jobs:
        return False, "proposed state still has pending jobs"
    try:
        training_version = int(proposed_state.training_set_version)
    except (TypeError, ValueError):
        return False, "training_set_version is not an integer"
    if training_version < 0:
        return False, "training_set_version is negative"
    try:
        verify_committed_training_version(campaign_dir, training_version)
    except Exception as exc:
        return False, "committed training version is invalid: " + str(exc)[:180]
    return True, "verified committed training exists for FEREBUS re-entry"


def archive_data_staging_for_ferebus_reentry(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> List[str]:
    ok, reason = ferebus_reentry_can_archive_data_staging(campaign_dir, proposed_state)
    if not ok:
        raise ValueError(reason)
    campaign = Path(campaign_dir)
    staging = campaign / ".DATA" / "STAGING"
    if not staging.is_dir():
        return []
    children = [p for p in staging.iterdir() if p.name not in (".", "..")]
    if not children:
        return []
    _ensure_inside_campaign(campaign, staging)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = staging.with_name(staging.name + ".before-reconcile-" + stamp)
    suffix = 1
    while target.exists():
        target = staging.with_name(
            staging.name + ".before-reconcile-" + stamp + "." + str(suffix)
        )
        suffix += 1
    staging.rename(target)
    staging.mkdir(parents=True, exist_ok=True)
    return [str(target)]


def archive_data_staging_for_operator_reconcile(
    campaign_dir: Union[str, Path],
) -> List[str]:
    """Archive ``.DATA/STAGING`` for an explicitly approved reconcile repair.

    Safety checks that depend on daemon state, locks, intents, or scheduler
    visibility live in the CLI before this mutating helper is called. This
    helper only validates the filesystem contract and performs a lossless
    rename; it never deletes staging contents.
    """
    campaign = Path(campaign_dir)
    staging = campaign / ".DATA" / "STAGING"
    if not staging.exists():
        return []
    if staging.is_symlink():
        raise ValueError("refusing to archive symlinked .DATA/STAGING")
    if not staging.is_dir():
        raise ValueError(".DATA/STAGING exists but is not a directory")
    children = [p for p in staging.iterdir() if p.name not in (".", "..")]
    if not children:
        return []
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "refusing to archive .DATA/STAGING containing symlink: "
                + str(path)
            )
    _ensure_inside_campaign(campaign, staging)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = staging.with_name(staging.name + ".archived-" + stamp)
    suffix = 1
    while target.exists():
        target = staging.with_name(
            staging.name + ".archived-" + stamp + "." + str(suffix)
        )
        suffix += 1
    staging.rename(target)
    staging.mkdir(parents=True, exist_ok=True)
    return [str(target)]


def apply_config_lock_update(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
) -> Path:
    return write_config_lock(campaign_dir, config)


def restore_config_from_lock_proposal(campaign_dir: Union[str, Path]) -> Path:
    """Write campaign.yaml.proposed from the locked canonical config.

    This is deliberately proposal-only.  The operator must inspect and promote
    the file manually because campaign.yaml is the protocol contract.
    """
    campaign = Path(campaign_dir)
    campaign_yaml = campaign / "campaign.yaml"
    if campaign_yaml.exists():
        raise FileExistsError(
            "campaign.yaml already exists; refusing to overwrite it"
        )
    path = config_lock_path(campaign)
    if not path.is_file():
        raise FileNotFoundError("config lock not found at " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            "config lock is unreadable: "
            + type(exc).__name__
            + ": "
            + str(exc)[:160]
        ) from exc
    config_payload = payload.get("canonical_config")
    if not isinstance(config_payload, dict):
        raise ValueError("config lock does not contain canonical_config")
    config = CampaignConfig.from_dict(config_payload)
    target = campaign / "campaign.yaml.proposed"
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archive = target.with_name(target.name + ".before-" + stamp)
        suffix = 1
        while archive.exists():
            archive = target.with_name(
                target.name + ".before-" + stamp + "." + str(suffix)
            )
            suffix += 1
        target.rename(archive)
    config.to_yaml_dense(target)
    return target


def training_staging_can_archive_for_reconcile(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Tuple[bool, str]:
    campaign = Path(campaign_dir)
    training = campaign / "5_TRAINING"
    if not training.is_dir():
        return False, "5_TRAINING is missing"
    tv = TrainingSetVersioning(training)
    dangling = tv.list_dangling_staging()
    if not dangling:
        return True, "no dangling training staging exists"
    try:
        training_version = int(proposed_state.training_set_version)
    except (TypeError, ValueError):
        return False, "training_set_version is not an integer"
    if training_version < 0:
        return False, "training_set_version is negative"
    try:
        verify_committed_training_version(campaign, training_version)
    except Exception as exc:
        return False, "committed training version is invalid: " + str(exc)[:180]
    try:
        model_version = int(proposed_state.models_version)
    except (TypeError, ValueError):
        model_version = -1
    if model_version >= 0:
        try:
            verify_committed_model_version(campaign, model_version)
        except Exception as exc:
            return False, "committed model version is invalid: " + str(exc)[:180]
    training_resolved = training.resolve()
    for path in dangling:
        if path.is_symlink():
            return False, "refusing to archive symlinked training staging: " + str(path)
        if not path.is_dir():
            return False, "training staging is not a directory: " + str(path)
        _ensure_inside_campaign(campaign, path)
        resolved = path.resolve()
        if training_resolved not in resolved.parents:
            return False, "training staging is outside 5_TRAINING: " + str(path)
    return True, "dangling training staging can be archived"


def archive_training_staging_for_reconcile(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> List[str]:
    ok, reason = training_staging_can_archive_for_reconcile(
        campaign_dir,
        proposed_state,
    )
    if not ok:
        raise ValueError(reason)
    campaign = Path(campaign_dir)
    training = campaign / "5_TRAINING"
    tv = TrainingSetVersioning(training)
    archived: List[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for staging in tv.list_dangling_staging():
        target = staging.with_name(staging.name + ".before-reconcile-" + stamp)
        suffix = 1
        while target.exists():
            target = staging.with_name(
                staging.name
                + ".before-reconcile-"
                + stamp
                + "."
                + str(suffix)
            )
            suffix += 1
        staging.rename(target)
        archived.append(str(target))
    return archived


def format_config_review(review: ConfigLockReview) -> str:
    lines: List[str] = []
    if review.notes:
        lines.extend(review.notes)
    if review.allowed_changes:
        lines.append("Allowed config changes:")
        for change in review.allowed_changes:
            lines.append(
                "  - "
                + change.path
                + ": "
                + repr(change.old)
                + " -> "
                + repr(change.new)
                + " ("
                + change.reason
                + ")"
            )
    if review.blocked_changes:
        lines.append("Blocked config changes:")
        for change in review.blocked_changes:
            lines.append(
                "  - "
                + change.path
                + ": "
                + repr(change.old)
                + " -> "
                + repr(change.new)
                + " ("
                + change.reason
                + ")"
            )
    return "\n".join(lines)
