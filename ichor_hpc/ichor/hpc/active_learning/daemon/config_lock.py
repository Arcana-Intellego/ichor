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
CONFIG_LOCK_POLICY_VERSION = 2


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


@dataclass(frozen=True)
class ConfigFieldPolicy:
    category: str
    lock_kind: str
    description: str


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


RUNTIME_SAFE_EXACT = {
    "campaign.max_iterations",
}
RUNTIME_SAFE_PREFIXES = {
    "runtime.",
    "stop.",
    "error_calibration.",
}

RESOURCE_FUTURE_EXACT = {
    "resources.defaults.partition",
    "resources.defaults.walltime_hours",
    "resources.defaults.cpus_per_task",
    "resources.defaults.mem_per_cpu",
    "resources.polus.partition",
    "resources.polus.walltime_hours",
    "resources.polus.cpus_per_task",
    "resources.polus.mem_per_cpu",
    "resources.gaussian.partition",
    "resources.gaussian.walltime_hours",
    "resources.gaussian.cpus_per_task",
    "resources.gaussian.mem_per_cpu",
    "resources.gaussian.memory_mode",
    "resources.gaussian.link0_mem",
    "resources.gaussian.memory_fraction_of_slurm",
    "resources.aimall.partition",
    "resources.aimall.walltime_hours",
    "resources.aimall.cpus_per_task",
    "resources.aimall.mem_per_cpu",
    "resources.ariadne.partition",
    "resources.ariadne.walltime_hours",
    "resources.ariadne.cpus_per_task",
    "resources.ariadne.mem_per_cpu",
    "resources.ferebus.partition",
    "resources.ferebus.walltime_hours",
    "resources.ferebus.cpus_per_task",
    "resources.ferebus.mem_per_cpu",
    "resources.array_concurrency_limit",
    "resources.fail_on_memory_estimate_exceeds_request",
}

IMMUTABLE_EXACT = {"schema_version"}

PRE_POOL_EXACT = {"trajectory_pool.source_path"}
PRE_PHASE_A_EXACT = {"bootstrap.initial_labelled_size"}
PRE_GAUSSIAN_PREFIXES = {"gaussian."}
PRE_AIMALL_PREFIXES = {"aimall."}
PRE_FEREBUS_FIRST_EXACT = {
    "campaign.system_name",
    "bootstrap.external_validation_size",
    "ferebus.properties",
    "ferebus.train_fraction",
    "ferebus.internal_validation_fraction",
    "acquisition.property_name",
}
FUTURE_FEREBUS_EXACT = {
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
}
PRE_SEED_SELECT_EXACT = {
    "anti_overlap.skip_training_seeds",
    "anti_overlap.recent_seeds_cooldown",
}
PRE_SEED_SELECT_PREFIXES = {"seed_selection."}
PRE_ARIADNE_EXACT = {
    "resources.gradient_parallel_backend",
    "quality_gates.ariadne_max_displacement_ang",
    "quality_gates.ariadne_min_pair_distance_ang",
}
PRE_ARIADNE_PREFIXES = {
    "acquisition.",
    "ariadne.",
    "adversarial_safety.",
    "anti_overlap.",
}
PRE_SAMPLING_PROTOCOL_PREFIXES = {
    "sampling_protocol.",
    "geometry_novelty.",
}
PRE_PHASE_B_PREFIXES = {
    "active_batch.",
    "phase_b.",
}
PRE_SPLIT_PREFIXES: set[str] = set()
PRE_AIMALL_QUALITY_EXACT = {
    "quality_gates.require_readable_aimall_geometry",
    "quality_gates.require_finite_iqa",
    "quality_gates.require_finite_integration_error",
    "quality_gates.max_abs_integration_error",
    "quality_gates.iqa_energy_recovery_tolerance_ha",
}


# Compatibility names retained for the CLI and older tests. New code should go
# through ``field_policy_for_path`` / ``describe_field_editability``.
ALWAYS_SAFE_EXACT = set(RUNTIME_SAFE_EXACT)
ALWAYS_SAFE_PREFIXES = set(RUNTIME_SAFE_PREFIXES)
ALWAYS_SAFE_RESOURCE_EXACT = set(RESOURCE_FUTURE_EXACT)
FUTURE_SAFE_EXACT = set(RESOURCE_FUTURE_EXACT)
FUTURE_SAFE_PREFIXES = (
    set(PRE_SEED_SELECT_PREFIXES)
    | set(PRE_PHASE_B_PREFIXES)
    | set(PRE_SAMPLING_PROTOCOL_PREFIXES)
)

ARIADNE_OUTPUT_INTERPRETATION_PREFIXES = {
    "seed_selection.",
    "sampling_protocol.",
}
ARIADNE_OUTPUT_INTERPRETATION_EXACT = {
    "acquisition.gradient.max_acquisition_grad_per_ang",
}

PHASE_B_OUTPUT_INTERPRETATION_PREFIXES = {
    "active_batch.",
    "sampling_protocol.",
}
PHASE_B_OUTPUT_INTERPRETATION_EXACT = set()

PHASE_LOCAL_EXACT = set(FUTURE_FEREBUS_EXACT)
COMMITTED_LOCKED_EXACT = (
    set(PRE_POOL_EXACT)
    | set(PRE_FEREBUS_FIRST_EXACT)
    | set(PRE_AIMALL_QUALITY_EXACT)
)
CAMPAIGN_LOCKED_EXACT = set(IMMUTABLE_EXACT)


_POLICIES_EXACT: Dict[str, ConfigFieldPolicy] = {
    **{
        path: ConfigFieldPolicy("immutable", "immutable", "never editable mid-campaign")
        for path in IMMUTABLE_EXACT
    },
    **{
        path: ConfigFieldPolicy("runtime_safe", "runtime", "safe runtime/future daemon change")
        for path in RUNTIME_SAFE_EXACT
    },
    **{
        path: ConfigFieldPolicy("resource_future", "resource_future", "applies to future submissions only")
        for path in RESOURCE_FUTURE_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_pool", "pre_pool", "editable until trajectory pool import")
        for path in PRE_POOL_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_phase_a", "pre_phase_a", "editable until Phase A POLUS begins")
        for path in PRE_PHASE_A_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_ferebus", "pre_ferebus_first", "editable until first FEREBUS staging/submission")
        for path in PRE_FEREBUS_FIRST_EXACT
    },
    **{
        path: ConfigFieldPolicy("future_ferebus", "future_ferebus", "editable for future unsubmitted FEREBUS fits")
        for path in FUTURE_FEREBUS_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_seed_select", "pre_seed_select", "editable until seed selection for this iteration")
        for path in PRE_SEED_SELECT_EXACT
    },
    **{
        path: ConfigFieldPolicy(
            "pre_ariadne",
            "pre_ariadne",
            "editable until ARIADNE consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks",
        )
        for path in PRE_ARIADNE_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_aimall_quality", "pre_aimall_first", "editable until first AIMAll quality postprocess")
        for path in PRE_AIMALL_QUALITY_EXACT
    },
}

_POLICIES_PREFIX: Tuple[Tuple[str, ConfigFieldPolicy], ...] = (
    ("runtime.", ConfigFieldPolicy("runtime_safe", "runtime", "safe runtime/future daemon change")),
    ("stop.", ConfigFieldPolicy("runtime_safe", "runtime", "safe future STOP_CHECK change")),
    ("error_calibration.", ConfigFieldPolicy("runtime_safe", "runtime", "safe future calibration/acquisition change; current ARIADNE reuse may require force-resubmitting the array")),
    ("gaussian.", ConfigFieldPolicy("pre_gaussian", "pre_gaussian_first", "editable until first Gaussian staging/submission; current Gaussian array edits require --force-resubmit-array-tasks")),
    ("aimall.", ConfigFieldPolicy("pre_aimall", "pre_aimall_first", "editable until first AIMAll staging/submission; current AIMAll array edits require --force-resubmit-array-tasks")),
    ("seed_selection.", ConfigFieldPolicy("pre_seed_select", "pre_seed_select", "editable until seed selection for this iteration")),
    ("sampling_protocol.", ConfigFieldPolicy("pre_sampling_protocol", "pre_sampling_protocol", "editable until ARIADNE/Phase B consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("active_batch.", ConfigFieldPolicy("pre_phase_b", "pre_phase_b", "editable until Phase B consumes this iteration")),
    ("phase_b.", ConfigFieldPolicy("pre_phase_b", "pre_phase_b", "editable until Phase B consumes this iteration")),
    ("geometry_novelty.", ConfigFieldPolicy("pre_sampling_protocol", "pre_sampling_protocol", "editable until ARIADNE/Phase B consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("acquisition.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("ariadne.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("adversarial_safety.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE/Phase B consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("anti_overlap.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE consumes this iteration")),
)


def _committed_model_exists(campaign_dir: Union[str, Path], version: int) -> bool:
    models = Path(campaign_dir) / "6_TRAINED_MODELS"
    if not models.is_dir():
        return False
    try:
        committed = TrainingSetVersioning(models).list_committed_versions()
    except Exception:
        return False
    return int(version) in {int(v) for v in committed}


def _any_committed_model_exists(campaign_dir: Union[str, Path]) -> bool:
    models = Path(campaign_dir) / "6_TRAINED_MODELS"
    if not models.is_dir():
        return False
    try:
        return bool(TrainingSetVersioning(models).list_committed_versions())
    except Exception:
        return False


def field_policy_for_path(path: str) -> Optional[ConfigFieldPolicy]:
    policy = _POLICIES_EXACT.get(str(path))
    if policy is not None:
        return policy
    for prefix, prefix_policy in _POLICIES_PREFIX:
        if str(path).startswith(prefix):
            return prefix_policy
    return None


def describe_field_editability(path: str) -> str:
    policy = field_policy_for_path(path)
    if policy is None:
        return "unclassified lock policy"
    return policy.description


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


def _phase_intent_file_exists(
    campaign_dir: Union[str, Path],
    phases: Iterable[CampaignPhase],
    *,
    iteration: Optional[int] = None,
) -> Optional[str]:
    try:
        from . import submission_intent as _submission_intent

        if iteration is not None:
            for phase in phases:
                path = _submission_intent.intent_path(
                    campaign_dir,
                    phase.value,
                    int(iteration),
                )
                if path.is_file():
                    return str(path)
            return None
        root = _submission_intent.intent_dir(campaign_dir)
        if not root.is_dir():
            return None
        phase_names = {phase.value for phase in phases}
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                phase_name = str(payload.get("phase") or "")
            except Exception:
                # A malformed intent is still durable evidence that submission
                # reached this phase family.
                return str(path)
            if phase_name in phase_names:
                return str(path)
    except Exception:
        return "<submission intent lookup failed>"
    return None


def _first_existing_path(campaign_dir: Union[str, Path], patterns: Iterable[str]) -> Optional[str]:
    campaign = Path(campaign_dir)
    for pattern in patterns:
        matches = sorted(campaign.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _trajectory_pool_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    return _first_existing_path(
        campaign_dir,
        (
            ".DATA/TRAJECTORY/pool.xyz",
            ".DATA/TRAJECTORY/pool.manifest.json",
        ),
    )


def _phase_a_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.PHASE_A_POLUS,),
    )
    if intent:
        return "Phase A submission intent exists: " + intent
    path = _first_existing_path(
        campaign_dir,
        (
            "3_DIVERSITY_SAMPLING/initial/PHASE_A_SAMPLE.json",
            "3_DIVERSITY_SAMPLING/initial/*SAMPLE*.xyz",
            "3_DIVERSITY_SAMPLING/initial/*INDEX*.dat",
        ),
    )
    if path:
        return "Phase A output exists: " + path
    return None


def _gaussian_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.INITIAL_GAUSSIAN, CampaignPhase.GAUSSIAN),
    )
    if intent:
        return "Gaussian submission intent exists: " + intent
    path = _first_existing_path(
        campaign_dir,
        (
            ".DATA/STAGING/**/*.pointdir/input.gjf",
            ".DATA/STAGING/**/*.pointdir/input.wfn",
            ".DATA/STAGING/**/accepted_pointdirs*.json",
        ),
    )
    if path:
        return "Gaussian staging or handoff exists: " + path
    return None


def _aimall_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.INITIAL_AIMALL, CampaignPhase.AIMALL),
    )
    if intent:
        return "AIMAll submission intent exists: " + intent
    path = _first_existing_path(
        campaign_dir,
        (
            ".DATA/STAGING/**/*.pointdir/AIMALL_TASK.json",
            ".DATA/STAGING/**/*.pointdir/*.int",
            ".DATA/STAGING/**/quantum_quality.json",
            ".DATA/STAGING/**/accepted_pointdirs.INITIAL_AIMALL.json",
            ".DATA/STAGING/**/accepted_pointdirs.AIMALL.json",
        ),
    )
    if path:
        return "AIMAll staging, output, or quality manifest exists: " + path
    return None


def _ferebus_first_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS),
    )
    if intent:
        return "FEREBUS submission intent exists: " + intent
    path = _first_existing_path(
        campaign_dir,
        (
            "6_TRAINED_MODELS/iteration-staging/FEREBUS_TASKS.json",
            "6_TRAINED_MODELS/iteration-staging/**/*.model",
            "6_TRAINED_MODELS/iteration-*/**/*.model",
            "6_TRAINED_MODELS/current",
        ),
    )
    if path:
        return "FEREBUS staging or committed model exists: " + path
    if _any_committed_model_exists(campaign_dir):
        return "at least one FEREBUS model version is committed"
    return None


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
        "SAMPLING_SCALE_MODEL.json",
        "SAMPLING_PROTOCOL_RESOLVED.json",
        "SAMPLING_PROTOCOL_AUDIT.json",
    ):
        if (iter_dir / name).is_file():
            return True
    pool = iter_dir / "pool"
    if pool.is_dir():
        for result in pool.glob("seed_*/result.json"):
            if result.is_file():
                return True
    return False


def _ariadne_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.ARIADNE_ARRAY,),
        iteration=iteration,
    )
    if intent:
        return "ARIADNE submission intent exists: " + intent
    iter_dir = _iteration_dir(campaign_dir, iteration)
    for name in (
        "ARIADNE_RESULTS.json",
        "ARIADNE_LANDING_AUDIT.json",
        "ACQUISITION_MATURITY_AUDIT.json",
        "ERROR_CALIBRATION_AUDIT.json",
        "SAMPLING_SCALE_MODEL.json",
        "SAMPLING_PROTOCOL_RESOLVED.json",
        "SAMPLING_PROTOCOL_AUDIT.json",
    ):
        path = iter_dir / name
        if path.is_file():
            return "ARIADNE output exists: " + str(path)
    pool = iter_dir / "pool"
    if pool.is_dir():
        for result in sorted(pool.glob("seed_*/result.json")):
            if result.is_file():
                return "ARIADNE seed result exists: " + str(result)
    return None


def _seed_select_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    iter_dir = _iteration_dir(campaign_dir, iteration)
    for name in ("seeds_picked.json", "SEED_SELECTION_DIAGNOSTICS.json"):
        path = iter_dir / name
        if path.is_file():
            return "seed-selection output exists: " + str(path)
    ariadne_reason = _ariadne_consumed_reason(campaign_dir, proposed_state)
    if ariadne_reason:
        return "ARIADNE already consumed seed selection: " + ariadne_reason
    return None


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


def _phase_b_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.PHASE_B_POLUS,),
        iteration=iteration,
    )
    if intent:
        return "Phase B submission intent exists: " + intent
    iter_dir = _iteration_dir(campaign_dir, iteration)
    for name in (
        "PHASE_B_SELECTION.json",
        "phase_b_SAMPLE.xyz",
        "phase_b_SAMPLE_raw.xyz",
        "phase_b_dedup.json",
    ):
        path = iter_dir / name
        if path.exists():
            return "Phase B output exists: " + str(path)
    return None


def _split_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    path = _iteration_dir(campaign_dir, iteration) / "split.json"
    if path.is_file():
        return "split output exists: " + str(path)
    return None


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
    if _active_iteration_committed(proposed_state, iteration):
        return False
    # Any current-iteration output that has not yet been appended and retrained
    # is part of the live handoff contract, even if the daemon state has already
    # advanced past the phase that produced it.
    return True


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


def _current_ferebus_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    phase = proposed_state.phase
    if phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        return None
    iteration = int(getattr(proposed_state, "iteration", 0))
    intent = _phase_intent_file_exists(campaign_dir, (phase,), iteration=iteration)
    if intent:
        return "target FEREBUS submission intent exists: " + intent
    staging = Path(campaign_dir) / "6_TRAINED_MODELS" / "iteration-staging"
    if staging.exists():
        return "target FEREBUS staging exists: " + str(staging)
    target = int(getattr(proposed_state, "training_set_version", -1))
    if target >= 0 and _committed_model_exists(campaign_dir, target):
        return "target FEREBUS model version is already committed"
    return None


def _block_if_uncommitted(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
    reason: Optional[str],
    *,
    label: str,
) -> Optional[str]:
    if not reason:
        return None
    iteration = int(getattr(proposed_state, "iteration", 0))
    if _active_iteration_committed(proposed_state, iteration):
        return None
    return (
        "field affects in-flight or uncommitted "
        + label
        + " outputs; "
        + reason
    )


def _consumption_block_reason(
    policy: ConfigFieldPolicy,
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
    *,
    force_resubmit_array_phase: Optional[CampaignPhase] = None,
    path: str = "",
) -> Optional[str]:
    kind = policy.lock_kind
    if force_resubmit_array_phase is not None:
        phase = force_resubmit_array_phase
        dotted = str(path)
        if phase in (CampaignPhase.INITIAL_GAUSSIAN, CampaignPhase.GAUSSIAN):
            if dotted.startswith("gaussian.") or dotted in RESOURCE_FUTURE_EXACT:
                return None
        if phase in (CampaignPhase.INITIAL_AIMALL, CampaignPhase.AIMALL):
            if (
                dotted.startswith("aimall.")
                or dotted in PRE_AIMALL_QUALITY_EXACT
                or dotted in RESOURCE_FUTURE_EXACT
            ):
                return None
        if phase is CampaignPhase.ARIADNE_ARRAY:
            if (
                dotted.startswith("acquisition.")
                or dotted.startswith("ariadne.")
                or dotted.startswith("adversarial_safety.")
                or dotted.startswith("sampling_protocol.")
                or dotted.startswith("geometry_novelty.")
                or dotted.startswith("error_calibration.")
                or dotted in PRE_ARIADNE_EXACT
                or dotted in RESOURCE_FUTURE_EXACT
            ):
                return None
    if kind == "immutable":
        return "field is immutable once a campaign config lock exists"
    if kind == "runtime":
        return None
    if kind == "resource_future":
        return None
    if kind == "pre_pool":
        reason = _trajectory_pool_consumed(campaign_dir)
        return None if reason is None else "trajectory pool has already been imported: " + reason
    if kind == "pre_phase_a":
        return _phase_a_consumed(campaign_dir)
    if kind == "pre_gaussian_first":
        return _gaussian_consumed(campaign_dir)
    if kind == "pre_aimall_first":
        return _aimall_consumed(campaign_dir)
    if kind == "pre_ferebus_first":
        return _ferebus_first_consumed(campaign_dir)
    if kind == "future_ferebus":
        return _current_ferebus_consumed_reason(campaign_dir, proposed_state)
    if kind == "pre_seed_select":
        return _block_if_uncommitted(
            campaign_dir,
            proposed_state,
            _seed_select_consumed_reason(campaign_dir, proposed_state),
            label="seed-selection/ARIADNE",
        )
    if kind == "pre_ariadne":
        return _block_if_uncommitted(
            campaign_dir,
            proposed_state,
            _ariadne_consumed_reason(campaign_dir, proposed_state),
            label="ARIADNE",
        )
    if kind == "pre_sampling_protocol":
        ariadne_reason = _block_if_uncommitted(
            campaign_dir,
            proposed_state,
            _ariadne_consumed_reason(campaign_dir, proposed_state),
            label="ARIADNE",
        )
        if ariadne_reason:
            return ariadne_reason
        return _block_if_uncommitted(
            campaign_dir,
            proposed_state,
            _phase_b_consumed_reason(campaign_dir, proposed_state),
            label="Phase B",
        )
    if kind == "pre_phase_b":
        return _block_if_uncommitted(
            campaign_dir,
            proposed_state,
            _phase_b_consumed_reason(campaign_dir, proposed_state),
            label="Phase B",
        )
    if kind == "pre_split":
        return _block_if_uncommitted(
            campaign_dir,
            proposed_state,
            _split_consumed_reason(campaign_dir, proposed_state),
            label="SPLIT",
        )
    return "field has no configured mid-campaign change policy"


def _classify_change(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
    path: str,
    old: Any,
    new: Any,
    *,
    force_resubmit_array_phase: Optional[CampaignPhase] = None,
) -> ConfigChange:
    policy = field_policy_for_path(path)
    if policy is None:
        return ConfigChange(
            path,
            old,
            new,
            "unclassified_locked",
            False,
            "field has no configured mid-campaign change policy",
        )
    blocked_reason = _consumption_block_reason(
        policy,
        campaign_dir,
        proposed_state,
        force_resubmit_array_phase=force_resubmit_array_phase,
        path=path,
    )
    if blocked_reason:
        category = (
            "postprocess_locked"
            if policy.lock_kind.startswith("pre_")
            or policy.lock_kind == "future_ferebus"
            else policy.category
        )
        return ConfigChange(path, old, new, category, False, blocked_reason)
    if policy.category == "runtime_safe":
        reason = policy.description
    elif policy.category == "resource_future":
        reason = "allowed for future submissions; already queued/running jobs keep submitted resources"
    elif policy.lock_kind.startswith("pre_"):
        reason = policy.description
    elif policy.lock_kind == "future_ferebus":
        reason = policy.description
    else:
        reason = policy.description
    return ConfigChange(path, old, new, policy.category, True, reason)


def review_config_changes(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    proposed_state: CampaignState,
    *,
    initialise_missing: bool = False,
    force_resubmit_array_phase: Optional[CampaignPhase] = None,
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
        change = _classify_change(
            campaign_dir,
            proposed_state,
            dotted,
            old,
            new,
            force_resubmit_array_phase=force_resubmit_array_phase,
        )
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
