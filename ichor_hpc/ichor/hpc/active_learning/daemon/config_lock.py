"""Campaign configuration lock and guarded recovery checks.

The active-learning daemon is allowed to restart after crashes, scheduler
failures, and deliberate operator stops. It must not silently continue a
campaign after a protocol-changing edit to ``campaign.yaml``. This module
stores a canonical, default-expanded config snapshot and classifies later
edits before ``reconcile --apply`` promotes a proposed recovery state.
"""
from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..config import CampaignConfig
from ..versioning.reference_data import ReferenceDataVersioning
from ..versioning.trained_models import (
    TrainedModelVersioning,
    resolve_trained_model_set,
    trained_models_commit_lock,
)
from ..layout import trained_models_dir
from .artifact_contracts import (
    verify_committed_model_version,
    verify_committed_reference_data_version,
)
from .filesystem import campaign_owned_path
from .state import CampaignPhase, CampaignState, atomic_write_json


CONFIG_LOCK_SCHEMA_VERSION = 3
CONFIG_LOCK_FILENAME = "config_lock.json"
CONFIG_LOCK_POLICY_VERSION = 4
CONFIG_LOCK_HISTORY_SCHEMA_VERSION = 2
CONFIG_LOCK_HISTORY_DIRNAME = "config_lock_history"


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
    from .filesystem import operational_path

    return operational_path(campaign_dir, CONFIG_LOCK_FILENAME)


def config_lock_history_dir(campaign_dir: Union[str, Path]) -> Path:
    return config_lock_path(campaign_dir).parent / CONFIG_LOCK_HISTORY_DIRNAME


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


def _lock_payload(
    config: CampaignConfig,
    *,
    campaign_uid: Optional[str],
    created_at_iso: Optional[str] = None,
    history_sequence: Optional[int] = None,
    history_entry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
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
        "campaign_uid": None if not campaign_uid else str(campaign_uid),
        "history_sequence": history_sequence,
        "history_entry_sha256": history_entry_sha256,
    }


def _history_entry_sha256(payload: Dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("entry_sha256", None)
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " is not a regular file: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError(label + " must contain a JSON object")
    return payload


def _validate_lock_payload(
    payload: Dict[str, Any],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        schema_version = int(payload.get("schema_version", -1))
        campaign_schema_version = int(payload.get("campaign_schema_version", -1))
        policy_version = int(payload.get("field_policy_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("config lock version fields are invalid") from exc
    if schema_version != CONFIG_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported config lock schema version: " + str(schema_version))
    if policy_version != CONFIG_LOCK_POLICY_VERSION:
        raise ValueError("unsupported config lock field-policy version: " + str(policy_version))
    canonical = payload.get("canonical_config")
    if not isinstance(canonical, dict):
        raise ValueError("config lock canonical_config is missing")
    if campaign_schema_version != int(canonical.get("schema_version", -2)):
        raise ValueError("config lock campaign schema version mismatch")
    expected_fingerprint = config_fingerprint(canonical)
    if str(payload.get("fingerprint_sha256") or "") != expected_fingerprint:
        raise ValueError("config lock canonical fingerprint mismatch")
    lock_uid = str(payload.get("campaign_uid") or "")
    if expected_campaign_uid is not None:
        expected_uid = str(expected_campaign_uid)
        if lock_uid and lock_uid != expected_uid:
            raise ValueError("config lock campaign UID mismatch")
    return payload


def read_config_lock(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _read_json_object(config_lock_path(campaign_dir), "config lock")
    validated = _validate_lock_payload(
        payload,
        expected_campaign_uid=expected_campaign_uid,
    )
    if int(validated.get("schema_version", -1)) == CONFIG_LOCK_SCHEMA_VERSION:
        try:
            sequence = int(validated.get("history_sequence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("config lock history sequence is invalid") from exc
        entry_sha = str(validated.get("history_entry_sha256") or "")
        matches = list(
            config_lock_history_dir(campaign_dir).glob(
                f"{sequence:08d}-" + entry_sha + ".json"
            )
        )
        if len(matches) != 1:
            raise ValueError("config lock history reference is missing or ambiguous")
        entry = _read_json_object(matches[0], "config lock history entry")
        if int(entry.get("schema_version", -1)) != CONFIG_LOCK_HISTORY_SCHEMA_VERSION:
            raise ValueError("unsupported config lock history schema")
        if str(entry.get("entry_sha256") or "") != _history_entry_sha256(entry):
            raise ValueError("config lock history entry digest mismatch")
        if str(entry.get("entry_sha256") or "") != entry_sha:
            raise ValueError("config lock history reference digest mismatch")
        if str(entry.get("fingerprint_sha256") or "") != str(
            validated.get("fingerprint_sha256") or ""
        ):
            raise ValueError("config lock and history fingerprints differ")
        if str(entry.get("campaign_uid") or "") != str(
            validated.get("campaign_uid") or ""
        ):
            raise ValueError("config lock and history campaign UIDs differ")
    return validated


def restore_config_lock_from_history(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: str,
) -> Path:
    """Restore a missing current lock from one complete linear history chain."""
    path = config_lock_path(campaign_dir)
    if path.exists():
        raise FileExistsError("current config lock already exists")
    root = config_lock_history_dir(campaign_dir)
    entries: Dict[str, Dict[str, Any]] = {}
    for candidate in sorted(root.glob("*.json")):
        entry = _read_json_object(candidate, "config lock history entry")
        if int(entry.get("schema_version", -1)) != CONFIG_LOCK_HISTORY_SCHEMA_VERSION:
            raise ValueError("unsupported config lock history schema")
        digest = str(entry.get("entry_sha256") or "")
        if not digest or digest != _history_entry_sha256(entry):
            raise ValueError("config lock history entry digest mismatch")
        if digest in entries:
            raise ValueError("duplicate config lock history entry digest")
        entries[digest] = entry
    if not entries:
        raise FileNotFoundError("no config lock history entries are available")
    children: Dict[Optional[str], List[str]] = {}
    for digest, entry in entries.items():
        previous = entry.get("previous_entry_sha256")
        previous_key = None if previous in (None, "") else str(previous)
        if previous_key is not None and previous_key not in entries:
            raise ValueError("config lock history chain has a missing predecessor")
        children.setdefault(previous_key, []).append(digest)
    if len(children.get(None, [])) != 1:
        raise ValueError("config lock history does not have one root")
    for digests in children.values():
        if len(digests) > 1:
            raise ValueError("config lock history is forked")
    current = children[None][0]
    visited = set()
    while current is not None:
        if current in visited:
            raise ValueError("config lock history contains a cycle")
        visited.add(current)
        next_values = children.get(current, [])
        current = next_values[0] if next_values else None
    if visited != set(entries):
        raise ValueError("config lock history contains disconnected entries")
    latest_sha = max(
        entries,
        key=lambda digest: int(entries[digest].get("sequence", -1)),
    )
    latest = entries[latest_sha]
    if str(latest.get("campaign_uid") or "") != str(expected_campaign_uid):
        raise ValueError("latest config lock history campaign UID mismatch")
    config_payload = latest.get("canonical_config")
    if not isinstance(config_payload, dict):
        raise ValueError("latest config lock history has no canonical config")
    config = CampaignConfig.from_dict(config_payload)
    atomic_write_json(
        path,
        _lock_payload(
            config,
            campaign_uid=str(expected_campaign_uid),
            history_sequence=int(latest["sequence"]),
            history_entry_sha256=latest_sha,
        ),
    )
    read_config_lock(campaign_dir, expected_campaign_uid=expected_campaign_uid)
    return path


def _write_history_entry(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    *,
    campaign_uid: Optional[str],
    previous: Optional[Dict[str, Any]],
    reason: str,
) -> Tuple[int, str]:
    root = config_lock_history_dir(campaign_dir)
    root.mkdir(parents=True, exist_ok=True)
    previous_sequence = None if previous is None else previous.get("history_sequence")
    previous_sha = None if previous is None else previous.get("history_entry_sha256")
    sequence = 0 if previous_sequence is None else int(previous_sequence) + 1
    entry: Dict[str, Any] = {
        "schema_version": CONFIG_LOCK_HISTORY_SCHEMA_VERSION,
        "sequence": sequence,
        "campaign_uid": None if not campaign_uid else str(campaign_uid),
        "campaign_schema_version": int(config.schema_version),
        "field_policy_version": CONFIG_LOCK_POLICY_VERSION,
        "canonical_config": canonical_config(config),
        "fingerprint_sha256": config_fingerprint(canonical_config(config)),
        "previous_entry_sha256": previous_sha,
        "reason": str(reason),
        "created_at_iso": _now_iso(),
    }
    entry["entry_sha256"] = _history_entry_sha256(entry)
    path = root / (f"{sequence:08d}-" + str(entry["entry_sha256"]) + ".json")
    if path.exists():
        existing = _read_json_object(path, "config lock history entry")
        if existing != entry:
            raise ValueError("config lock history identity collision")
    else:
        atomic_write_json(path, entry)
    return sequence, str(entry["entry_sha256"])


def write_config_lock(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    *,
    campaign_uid: Optional[str] = None,
    reason: str = "config_lock_update",
) -> Path:
    path = config_lock_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = None
    existing: Optional[Dict[str, Any]] = None
    if path.is_file():
        try:
            existing = read_config_lock(campaign_dir)
            created = str(existing.get("created_at_iso") or "") or None
        except Exception:
            raise ValueError("refusing to overwrite an invalid config lock")
    inferred_uid = None
    if campaign_uid is None:
        state_path = path.parent / "state.json"
        if state_path.is_file() and not state_path.is_symlink():
            try:
                state_payload = _read_json_object(state_path, "campaign state")
                inferred_uid = str(state_payload.get("campaign_uid") or "") or None
            except Exception:
                inferred_uid = None
    effective_uid = str(
        campaign_uid
        or inferred_uid
        or (existing or {}).get("campaign_uid")
        or ""
    ) or None
    new_fingerprint = config_fingerprint(canonical_config(config))
    if (
        existing is not None
        and str(existing.get("fingerprint_sha256") or "") == new_fingerprint
        and str(existing.get("campaign_uid") or "") == str(effective_uid or "")
        and int(existing.get("schema_version", -1)) == CONFIG_LOCK_SCHEMA_VERSION
    ):
        refreshed = dict(existing)
        refreshed["last_checked_at_iso"] = _now_iso()
        atomic_write_json(path, refreshed)
        return path
    history_previous = existing
    sequence, entry_sha = _write_history_entry(
        campaign_dir,
        config,
        campaign_uid=effective_uid,
        previous=history_previous,
        reason=reason,
    )
    atomic_write_json(
        path,
        _lock_payload(
            config,
            campaign_uid=effective_uid,
            created_at_iso=created,
            history_sequence=sequence,
            history_entry_sha256=entry_sha,
        ),
    )
    return path


def ensure_config_lock(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    *,
    campaign_uid: Optional[str] = None,
) -> Path:
    path = config_lock_path(campaign_dir)
    if not path.is_file():
        return write_config_lock(
            campaign_dir,
            config,
            campaign_uid=campaign_uid,
            reason="initial_config_lock",
        )
    read_config_lock(campaign_dir, expected_campaign_uid=campaign_uid)
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
    "resources.scheduler_usage_telemetry",
    "resources.scheduler_usage_history_limit",
    "retention.checkpoint_destination",
    "retention.checkpoint_every_iterations",
    "retention.checkpoint_verify_after_write",
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
    "resources.diversity.partition",
    "resources.diversity.walltime_hours",
    "resources.diversity.cpus_per_task",
    "resources.diversity.mem_per_cpu",
    "resources.diversity.auto_max_workers",
    "resources.diversity.target_pairs_per_worker",
    "resources.diversity.in_memory_distance_store_fraction",
    "resources.gaussian.partition",
    "resources.gaussian.walltime_hours",
    "resources.gaussian.cpus_per_task",
    "resources.gaussian.mem_per_cpu",
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
    "resources.memory_estimate_safety_factor",
}

IMMUTABLE_EXACT = {
    "schema_version",
    "campaign.system_name",
    "campaign.random_seed",
    "retention.checkpoint_required",
}

PRE_PHASE_A_EXACT = {
    "campaign.custom_bootstrap",
    "point_allocation.bootstrap_training_size",
    "point_allocation.bootstrap_internal_validation_size",
    "point_allocation.bootstrap_external_validation_size",
}
PRE_GAUSSIAN_PREFIXES = {"gaussian."}
PRE_AIMALL_PREFIXES = {"aimall."}
PRE_FEREBUS_FIRST_EXACT = {
    "ferebus.properties",
    "ferebus.prior_mean_strategy",
    "ferebus.prior_mean_level_of_theory",
    "ferebus.physical_prior_scale",
    "acquisition.property_name",
}
FUTURE_FEREBUS_EXACT = {
    "ferebus.kernel",
    "ferebus.nagents",
    "ferebus.maxiter",
}
PRE_SEED_SELECT_EXACT = {
    "seed_selection.exclude_committed_seed_frames",
    "seed_selection.recent_seed_cooldown_iterations",
}
SUBMISSION_SNAPSHOT_EXACT = {
    "runtime.failure_threshold_fraction",
    "quality_gates.require_readable_aimall_geometry",
    "quality_gates.require_finite_iqa",
    "quality_gates.require_finite_integration_error",
    "quality_gates.max_abs_integration_error",
    "quality_gates.iqa_energy_recovery_tolerance_ha",
    "quality_gates.ferebus_min_ext_r2",
    "quality_gates.ferebus_max_ext_rmse_ha",
    "quality_gates.ferebus_max_condition_number",
    "quality_gates.ferebus_max_aggregate_ext_rmse_increase_fraction",
    "quality_gates.ferebus_max_task_ext_rmse_increase_fraction",
    "quality_gates.ferebus_regression_abs_tolerance_ha",
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
    "geometry_novelty.",
}
PRE_SAMPLING_PROTOCOL_EXACT = {"campaign.sampling_aggressiveness"}
PRE_SAMPLING_PROTOCOL_CALIBRATION_EXACT = {
    "error_calibration.enabled",
    "error_calibration.mode",
    "error_calibration.min_records_to_apply",
    "error_calibration.min_model_versions_to_apply",
    "error_calibration.apply_strength",
    "error_calibration.max_model_age_iterations",
    "error_calibration.model_version_policy",
    "error_calibration.aggressiveness_match_required",
}
PRE_PHASE_B_PREFIXES = {
    "phase_b.",
}
PRE_PHASE_B_EXACT = {
    "point_allocation.batch_training_size",
    "point_allocation.batch_internal_validation_size",
}
PRE_SPLIT_PREFIXES: set[str] = set()
PRE_AIMALL_QUALITY_EXACT: set[str] = set()


# Fields whose interpretation is already embedded in current-iteration output.
ARIADNE_OUTPUT_INTERPRETATION_PREFIXES = {"seed_selection."}
ARIADNE_OUTPUT_INTERPRETATION_EXACT = {
    "acquisition.gradient.max_acquisition_grad_per_ang",
    "campaign.sampling_aggressiveness",
}
PHASE_B_OUTPUT_INTERPRETATION_PREFIXES: set[str] = set()
PHASE_B_OUTPUT_INTERPRETATION_EXACT = (
    set(PRE_PHASE_B_EXACT) | set(PRE_SAMPLING_PROTOCOL_EXACT)
)


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
        path: ConfigFieldPolicy(
            "submission_snapshot",
            "submission_snapshot",
            "applies only to future submissions; active attempts retain their snapshot",
        )
        for path in SUBMISSION_SNAPSHOT_EXACT
    },
    **{
        path: ConfigFieldPolicy("resource_future", "resource_future", "applies to future submissions only")
        for path in RESOURCE_FUTURE_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_phase_a", "pre_phase_a", "editable until Phase A diversity begins")
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
        path: ConfigFieldPolicy(
            "pre_phase_b",
            "pre_phase_b",
            "editable until Phase B allocates the current iteration",
        )
        for path in PRE_PHASE_B_EXACT
    },
    **{
        path: ConfigFieldPolicy("pre_aimall_quality", "pre_aimall_first", "editable until first AIMAll quality postprocess")
        for path in PRE_AIMALL_QUALITY_EXACT
    },
    **{
        path: ConfigFieldPolicy(
            "pre_sampling_protocol",
            "pre_sampling_protocol",
            "editable until ARIADNE/Phase B consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks",
        )
        for path in (
            set(PRE_SAMPLING_PROTOCOL_EXACT)
            | set(PRE_SAMPLING_PROTOCOL_CALIBRATION_EXACT)
        )
    },
}

_POLICIES_PREFIX: Tuple[Tuple[str, ConfigFieldPolicy], ...] = (
    ("runtime.", ConfigFieldPolicy("runtime_safe", "runtime", "safe runtime/future daemon change")),
    ("stop.", ConfigFieldPolicy("runtime_safe", "runtime", "safe future STOP_CHECK change")),
    ("error_calibration.", ConfigFieldPolicy("runtime_safe", "runtime", "safe future calibration/acquisition change; current ARIADNE reuse may require force-resubmitting the array")),
    ("gaussian.", ConfigFieldPolicy("pre_gaussian", "pre_gaussian_first", "editable until first Gaussian staging/submission; current Gaussian array edits require --force-resubmit-array-tasks")),
    ("aimall.", ConfigFieldPolicy("pre_aimall", "pre_aimall_first", "editable until first AIMAll staging/submission; current AIMAll array edits require --force-resubmit-array-tasks")),
    ("seed_selection.", ConfigFieldPolicy("pre_seed_select", "pre_seed_select", "editable until seed selection for this iteration")),
    ("phase_b.", ConfigFieldPolicy("pre_phase_b", "pre_phase_b", "editable until Phase B consumes this iteration")),
    ("geometry_novelty.", ConfigFieldPolicy("pre_sampling_protocol", "pre_sampling_protocol", "editable until ARIADNE/Phase B consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("acquisition.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("ariadne.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("adversarial_safety.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE/Phase B consumes this iteration; current ARIADNE array edits require --force-resubmit-array-tasks")),
    ("anti_overlap.", ConfigFieldPolicy("pre_ariadne", "pre_ariadne", "editable until ARIADNE consumes this iteration")),
)


def _committed_model_exists(campaign_dir: Union[str, Path], version: int) -> bool:
    models = trained_models_dir(campaign_dir)
    if not models.is_dir():
        return False
    try:
        committed = TrainedModelVersioning(models).list_committed_versions()
    except Exception:
        return False
    return int(version) in {int(v) for v in committed}


def _any_committed_model_exists(campaign_dir: Union[str, Path]) -> bool:
    models = trained_models_dir(campaign_dir)
    if not models.is_dir():
        return False
    try:
        return bool(TrainedModelVersioning(models).list_committed_versions())
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


def _phase_a_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.PHASE_A_DIVERSITY,),
    )
    if intent:
        return "Phase A submission intent exists: " + intent
    path = _first_existing_path(
        campaign_dir,
        (
            ".DATA/ACTIVE_LEARNING/CUSTOM_BOOTSTRAP.json",
            ".DATA/BOOTSTRAP/selection/SELECTION.json",
            ".DATA/BOOTSTRAP/selection/selected.xyz",
            ".DATA/BOOTSTRAP/selection/selected_indices.dat",
            ".DATA/BOOTSTRAP/allocation/POINT_ALLOCATION.json",
        ),
    )
    if path:
        return "Phase A output exists: " + path
    return None


def _gaussian_consumed(campaign_dir: Union[str, Path]) -> Optional[str]:
    intent = _phase_intent_file_exists(
        campaign_dir,
        (
            CampaignPhase.INITIAL_GAUSSIAN,
            CampaignPhase.GAUSSIAN,
            CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
            CampaignPhase.REPLACEMENT_GAUSSIAN,
        ),
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
        (
            CampaignPhase.INITIAL_AIMALL,
            CampaignPhase.AIMALL,
            CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
            CampaignPhase.REPLACEMENT_AIMALL,
        ),
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
            ".DATA/ACTIVE_LEARNING/bootstrap_inputs/MODEL_BOOTSTRAP.json",
            ".DATA/ACTIVE_LEARNING/ferebus_split_assignments.json",
            ".DATA/ACTIVE_LEARNING/bootstrap_external_validation.json",
            "TRAINED_MODELS/iteration-staging/FEREBUS_TASKS.json",
            "TRAINED_MODELS/iteration-staging/**/*.model",
            "TRAINED_MODELS/iteration-*/**/*.model",
            "TRAINED_MODELS/current",
        ),
    )
    if path:
        return "FEREBUS staging or committed model exists: " + path
    if _any_committed_model_exists(campaign_dir):
        return "at least one FEREBUS model version is committed"
    return None


def _iteration_dir(campaign_dir: Union[str, Path], iteration: int) -> Path:
    from ..layout import active_iteration_dir

    return active_iteration_dir(campaign_dir, int(iteration))


def _ariadne_outputs_exist(campaign_dir: Union[str, Path], proposed_state: CampaignState) -> bool:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if iteration < 1:
        return False
    iter_dir = _iteration_dir(campaign_dir, iteration)
    if _phase_intent_exists(campaign_dir, CampaignPhase.ARIADNE_ARRAY, iteration):
        return True
    from ..layout import active_ariadne_dir, active_calibration_dir, active_protocol_dir, ariadne_seeds_dir

    for path in (
        active_ariadne_dir(iter_dir) / "RESULTS.json",
        active_ariadne_dir(iter_dir) / "AUDIT.json",
        active_ariadne_dir(iter_dir) / "TASK_MAP.json",
        active_calibration_dir(iter_dir) / "ERROR_CALIBRATION_AUDIT.json",
        active_protocol_dir(iter_dir) / "SAMPLING_SCALE_MODEL.json",
        active_protocol_dir(iter_dir) / "SAMPLING_PROTOCOL_RESOLVED.json",
        active_protocol_dir(iter_dir) / "SAMPLING_PROTOCOL_AUDIT.json",
    ):
        if path.is_file():
            return True
    seeds = ariadne_seeds_dir(iter_dir)
    if seeds.is_dir():
        for result in seeds.glob("seed-*/result.json"):
            if result.is_file():
                return True
    return False


def _ariadne_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if iteration < 1:
        return None
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.ARIADNE_ARRAY,),
        iteration=iteration,
    )
    if intent:
        return "ARIADNE submission intent exists: " + intent
    iter_dir = _iteration_dir(campaign_dir, iteration)
    from ..layout import active_ariadne_dir, active_calibration_dir, active_protocol_dir, ariadne_seeds_dir

    for path in (
        active_ariadne_dir(iter_dir) / "RESULTS.json",
        active_ariadne_dir(iter_dir) / "AUDIT.json",
        active_ariadne_dir(iter_dir) / "TASK_MAP.json",
        active_calibration_dir(iter_dir) / "ERROR_CALIBRATION_AUDIT.json",
        active_protocol_dir(iter_dir) / "SAMPLING_SCALE_MODEL.json",
        active_protocol_dir(iter_dir) / "SAMPLING_PROTOCOL_RESOLVED.json",
        active_protocol_dir(iter_dir) / "SAMPLING_PROTOCOL_AUDIT.json",
    ):
        if path.is_file():
            return "ARIADNE output exists: " + str(path)
    seeds = ariadne_seeds_dir(iter_dir)
    if seeds.is_dir():
        for result in sorted(seeds.glob("seed-*/result.json")):
            if result.is_file():
                return "ARIADNE seed result exists: " + str(result)
    return None


def _seed_select_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if iteration < 1:
        return None
    iter_dir = _iteration_dir(campaign_dir, iteration)
    from ..handoff_manifests import seeds_picked_path

    path = seeds_picked_path(iter_dir)
    if path.is_file():
        return "seed-selection output exists: " + str(path)
    ariadne_reason = _ariadne_consumed_reason(campaign_dir, proposed_state)
    if ariadne_reason:
        return "ARIADNE already consumed seed selection: " + ariadne_reason
    return None


def _phase_b_outputs_exist(campaign_dir: Union[str, Path], proposed_state: CampaignState) -> bool:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if iteration < 1:
        return False
    iter_dir = _iteration_dir(campaign_dir, iteration)
    if _phase_intent_exists(campaign_dir, CampaignPhase.PHASE_B_DIVERSITY, iteration):
        return True
    from ..layout import active_allocation_dir, active_phase_b_dir

    for path in (
        active_phase_b_dir(iter_dir) / "SELECTION.json",
        active_phase_b_dir(iter_dir) / "selected.xyz",
        active_phase_b_dir(iter_dir) / "considered_candidates.xyz",
        active_allocation_dir(iter_dir) / "POINT_ALLOCATION.json",
    ):
        if path.exists():
            return True
    return False


def _phase_b_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if iteration < 1:
        return None
    intent = _phase_intent_file_exists(
        campaign_dir,
        (CampaignPhase.PHASE_B_DIVERSITY,),
        iteration=iteration,
    )
    if intent:
        return "Phase B submission intent exists: " + intent
    iter_dir = _iteration_dir(campaign_dir, iteration)
    from ..layout import active_allocation_dir, active_phase_b_dir

    for path in (
        active_phase_b_dir(iter_dir) / "SELECTION.json",
        active_phase_b_dir(iter_dir) / "selected.xyz",
        active_phase_b_dir(iter_dir) / "considered_candidates.xyz",
        active_allocation_dir(iter_dir) / "POINT_ALLOCATION.json",
    ):
        if path.exists():
            return "Phase B output exists: " + str(path)
    return None


def _split_consumed_reason(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    iteration = int(getattr(proposed_state, "iteration", 0))
    if iteration < 1:
        return None
    from ..layout import active_allocation_dir

    path = active_allocation_dir(
        _iteration_dir(campaign_dir, iteration)
    ) / "SPLIT_RECEIPT.json"
    if path.is_file():
        return "split output exists: " + str(path)
    return None


def _active_iteration_committed(proposed_state: CampaignState, iteration: int) -> bool:
    """Return true when active-loop artefacts for ``iteration`` are committed.

    Bootstrap is version 0. Active iteration ``i`` consumes version ``i - 1``
    and commits reference-data/model version ``i``.
    """
    try:
        reference_data_version = int(getattr(proposed_state, "reference_data_version", -1))
        models_version = int(getattr(proposed_state, "models_version", -1))
    except (TypeError, ValueError):
        return False
    return min(reference_data_version, models_version) >= int(iteration)


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
            CampaignPhase.PHASE_B_DIVERSITY,
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
        target = int(proposed_state.reference_data_version)
        if _committed_model_exists(campaign_dir, target):
            return False, "target FEREBUS model version is already committed"
        return True, "allowed for uncommitted FEREBUS re-entry"
    if path.startswith("gaussian."):
        if phase not in (
            CampaignPhase.INITIAL_GAUSSIAN,
            CampaignPhase.GAUSSIAN,
            CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
            CampaignPhase.REPLACEMENT_GAUSSIAN,
        ):
            return False, "Gaussian runtime settings may change only when re-entering Gaussian"
        return True, "allowed for Gaussian re-entry before acceptance"
    if path.startswith("aimall."):
        if phase not in (
            CampaignPhase.INITIAL_AIMALL,
            CampaignPhase.AIMALL,
            CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
            CampaignPhase.REPLACEMENT_AIMALL,
        ):
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
    staging = trained_models_dir(campaign_dir) / "iteration-staging"
    if staging.exists():
        return "target FEREBUS staging exists: " + str(staging)
    target = int(getattr(proposed_state, "reference_data_version", -1))
    if target >= 0 and _committed_model_exists(campaign_dir, target):
        return "target FEREBUS model version is already committed"
    return None


def _ferebus_quality_can_be_reevaluated(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> bool:
    """Return true only for complete, immutable, non-running FEREBUS output."""
    phase = proposed_state.phase
    if phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        return False
    try:
        from . import submission_intent as _submission_intent
        from .ferebus_quality import validate_ferebus_quality_evidence
        from .live_executor import validate_ferebus_completed

        intent = _submission_intent.load_intent(
            campaign_dir,
            phase.value,
            int(proposed_state.iteration),
            expected_campaign_uid=str(proposed_state.campaign_uid),
        )
        if intent is not None and str(intent.get("status") or "") not in {
            "FAILED",
            "SUPERSEDED",
            "COMPLETED",
        }:
            return False
        staging = trained_models_dir(campaign_dir) / "iteration-staging"
        ok, _ = validate_ferebus_completed(staging)
        if not ok:
            return False
        validate_ferebus_quality_evidence(staging)
        return True
    except Exception:
        return False


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
    force_retrain_ferebus: bool = False,
    path: str = "",
) -> Optional[str]:
    kind = policy.lock_kind
    if (
        force_retrain_ferebus
        and proposed_state.phase in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS)
        and kind == "future_ferebus"
    ):
        return None
    if force_resubmit_array_phase is not None:
        phase = force_resubmit_array_phase
        dotted = str(path)
        if phase in (
            CampaignPhase.INITIAL_GAUSSIAN,
            CampaignPhase.GAUSSIAN,
            CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
            CampaignPhase.REPLACEMENT_GAUSSIAN,
        ):
            if dotted.startswith("gaussian.") or dotted in RESOURCE_FUTURE_EXACT:
                return None
        if phase in (
            CampaignPhase.INITIAL_AIMALL,
            CampaignPhase.AIMALL,
            CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
            CampaignPhase.REPLACEMENT_AIMALL,
        ):
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
                or dotted == "campaign.sampling_aggressiveness"
                or dotted.startswith("geometry_novelty.")
                or dotted.startswith("error_calibration.")
                or dotted in PRE_ARIADNE_EXACT
                or dotted in RESOURCE_FUTURE_EXACT
            ):
                return None
    if kind == "immutable":
        return "field is immutable once a campaign config lock exists"
    if kind == "unsupported":
        return "field is unsupported and has no runtime effect"
    if kind == "runtime":
        return None
    if kind == "resource_future":
        return None
    if kind == "submission_snapshot":
        try:
            from . import submission_intent as _submission_intent

            intent = _submission_intent.load_intent(
                campaign_dir,
                proposed_state.phase.value,
                int(proposed_state.iteration),
                expected_campaign_uid=str(proposed_state.campaign_uid),
            )
        except FileNotFoundError:
            intent = None
        except Exception as exc:
            return (
                "submission ownership is inconclusive while checking the "
                "snapshotted field: " + type(exc).__name__ + ": " + str(exc)[:120]
            )
        if isinstance(intent, dict) and str(intent.get("status") or "") in (
            _submission_intent.ACTIVE_STATUSES
        ):
            return (
                "field is snapshotted by active submission intent "
                + str(intent.get("attempt_id") or intent.get("submission_identity") or "")
            )
        return None
    if kind == "pre_phase_a":
        return _phase_a_consumed(campaign_dir)
    if kind == "pre_gaussian_first":
        return _gaussian_consumed(campaign_dir)
    if kind == "pre_aimall_first":
        return _aimall_consumed(campaign_dir)
    if kind == "pre_ferebus_first":
        return _ferebus_first_consumed(campaign_dir)
    if kind == "future_ferebus":
        if (
            str(path).startswith("quality_gates.ferebus_")
            and _ferebus_quality_can_be_reevaluated(campaign_dir, proposed_state)
        ):
            return None
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
    force_retrain_ferebus: bool = False,
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
        force_retrain_ferebus=force_retrain_ferebus,
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
    if policy.category != "resource_future":
        reason += "; requires reconcile and daemon restart; current process unchanged"
    return ConfigChange(path, old, new, policy.category, True, reason)


def review_config_changes(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    proposed_state: CampaignState,
    *,
    initialise_missing: bool = False,
    force_resubmit_array_phase: Optional[CampaignPhase] = None,
    force_retrain_ferebus: bool = False,
) -> ConfigLockReview:
    path = config_lock_path(campaign_dir)
    if not path.is_file():
        review = ConfigLockReview(lock_path=path, lock_existed=False)
        review.blocked_changes.append(
            ConfigChange(
                "config_lock",
                "<missing>",
                "<current>",
                "lock_missing",
                False,
                "config lock is missing for an existing campaign; run reconcile to restore it from verified lock history",
            )
        )
        return review
    try:
        lock = read_config_lock(
            campaign_dir,
            expected_campaign_uid=str(proposed_state.campaign_uid),
        )
        old_config = lock.get("canonical_config")
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
        old_config = CampaignConfig.from_dict(old_config).to_dict()
        new_config = CampaignConfig.from_dict(canonical_config(config)).to_dict()
    except Exception as exc:
        review = ConfigLockReview(lock_path=path, lock_existed=True)
        review.blocked_changes.append(
            ConfigChange(
                "config_lock",
                "<invalid>",
                "<current>",
                "lock_invalid",
                False,
                "config lock validation failed: "
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
            force_retrain_ferebus=force_retrain_ferebus,
        )
        if change.allowed:
            review.allowed_changes.append(change)
        else:
            review.blocked_changes.append(change)
    return review


def archive_ferebus_iteration_staging_for_retrain(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Optional[str]:
    """Losslessly archive complete FEREBUS output before explicit retraining."""
    if proposed_state.phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        raise ValueError("--retrain-ferebus requires FEREBUS recovery phase")
    if any(value for value in proposed_state.pending_jobs.values()):
        raise ValueError("cannot retrain FEREBUS while state records pending jobs")
    campaign = Path(campaign_dir)
    target = trained_models_dir(campaign) / "iteration-staging"
    if not target.exists():
        return None
    if target.is_symlink() or not target.is_dir():
        raise ValueError("refusing invalid FEREBUS iteration-staging")
    from .ferebus_quality import validate_ferebus_quality_evidence
    from .live_executor import validate_ferebus_completed

    ok, reason = validate_ferebus_completed(target)
    if not ok:
        raise ValueError("FEREBUS staging is not complete: " + str(reason))
    validate_ferebus_quality_evidence(target)
    archive = _reconcile_archive_target(
        campaign,
        _timestamped_reconcile_sibling(target),
    )
    target.rename(archive)
    return str(archive)


def assert_config_unchanged_for_start(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    state: Optional[CampaignState],
) -> ConfigLockReview:
    if state is None:
        ensure_config_lock(campaign_dir, config)
        return ConfigLockReview(lock_path=config_lock_path(campaign_dir), lock_existed=True)
    review = review_config_changes(campaign_dir, config, state, initialise_missing=False)
    if review.changed:
        return review
    if review.lock_existed:
        write_config_lock(
            campaign_dir,
            config,
            campaign_uid=str(state.campaign_uid),
            reason="start_validation",
        )
    return review


def _ensure_inside_campaign(campaign_dir: Path, target: Path) -> None:
    campaign_owned_path(campaign_dir, target)


def _reconcile_archive_target(campaign: Path, target: Path) -> Path:
    """Validate every existing destination ancestor before a repair move."""
    return campaign_owned_path(campaign, target)


def clean_reentry_staging(campaign_dir: Union[str, Path], phase: CampaignPhase) -> List[str]:
    campaign = Path(campaign_dir)
    archived: List[str] = []
    if phase in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        target = trained_models_dir(campaign) / "iteration-staging"
        if target.exists():
            _ensure_inside_campaign(campaign, target)
            if target.is_symlink() or not target.is_dir():
                raise ValueError(
                    "refusing invalid model iteration-staging: " + str(target)
                )
            destination = _reconcile_archive_target(
                campaign,
                _timestamped_reconcile_sibling(target),
            )
            target.rename(destination)
            archived.append(str(destination))
    archived.extend(archive_scripts_for_reconcile(campaign))
    return archived


def archive_scripts_for_reconcile(campaign_dir: Union[str, Path]) -> List[str]:
    """Archive legacy flat scripts without touching immutable job bundles."""
    campaign = Path(campaign_dir)
    scripts = campaign / ".DATA" / "SCRIPTS"
    if not scripts.exists():
        return []
    if scripts.is_symlink():
        raise ValueError("refusing to archive symlinked .DATA/SCRIPTS")
    if not scripts.is_dir():
        raise ValueError(".DATA/SCRIPTS is not a directory")
    legacy_children = [
        path
        for path in scripts.iterdir()
        if path.name != "JOBS" and (
            path.suffix == ".sh" or path.name in {"OUTPUTS", "ERRORS"}
        )
    ]
    if not legacy_children:
        return []
    _ensure_inside_campaign(campaign, scripts)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_root = _reconcile_archive_target(
        campaign,
        scripts / "LEGACY_BEFORE_RECONCILE",
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    target = _reconcile_archive_target(campaign, archive_root / stamp)
    suffix = 1
    while target.exists():
        target = _reconcile_archive_target(
            campaign,
            archive_root / (stamp + "." + str(suffix)),
        )
        suffix += 1
    target.mkdir(parents=True, exist_ok=False)
    try:
        target.chmod(0o700)
    except OSError as exc:
        raise OSError(
            "could not enforce private reconcile archive permissions: "
            + str(target)
        ) from exc
    for child in legacy_children:
        _ensure_inside_campaign(campaign, child)
        child.rename(target / child.name)
    return [str(target)]


def _timestamped_reconcile_sibling(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = path.with_name(path.name + ".before-reconcile-" + stamp)
    suffix = 1
    while target.exists():
        target = path.with_name(
            path.name + ".before-reconcile-" + stamp + "." + str(suffix)
        )
        suffix += 1
    return target


def _staged_model_hashes_by_task(root: Path) -> Dict[Tuple[str, str], str]:
    from . import input_staging as _stg
    from ..versioning.manifest import sha256_file

    manifest = _stg.read_ferebus_manifest(root)
    hashes: Dict[Tuple[str, str], str] = {}
    for task in manifest.get("tasks", []):
        key = (str(task["property"]), str(task["atom"]))
        model = _stg.resolve_ferebus_task_path(
            root,
            task["expected_model_path"],
            "expected_model_path",
        )
        if not model.is_file() or model.is_symlink():
            raise ValueError("staged FEREBUS model is missing: " + str(model))
        hashes[key] = sha256_file(model)
    return hashes


def _archive_completed_model_iteration_staging(
    campaign: Path,
    target: Path,
    proposed_state: CampaignState,
) -> Path:
    try:
        model_version = int(proposed_state.models_version)
    except (TypeError, ValueError) as exc:
        raise ValueError("models_version is not an integer") from exc
    if model_version < 0:
        raise ValueError(
            "completed FEREBUS iteration-staging exists but no committed model "
            "version is recorded"
        )
    committed_set = resolve_trained_model_set(
        campaign,
        model_version,
        verification="deep",
    )
    committed = committed_set.root
    verify_committed_model_version(campaign, model_version)
    staged_models = _staged_model_hashes_by_task(target)
    committed_models = {
        task.key: task.model.sha256 for task in committed_set.tasks
    }
    if not staged_models:
        raise ValueError("completed FEREBUS iteration-staging has no model files")
    if staged_models != committed_models:
        raise ValueError(
            "completed FEREBUS iteration-staging does not match committed model "
            "version "
            + str(model_version)
            + ": staged="
            + repr(sorted(staged_models))
            + " committed="
            + repr(sorted(committed_models))
        )
    archive = _reconcile_archive_target(
        campaign,
        _timestamped_reconcile_sibling(target),
    )
    target.rename(archive)
    return archive


def clean_model_iteration_staging_for_reconcile(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> List[str]:
    campaign = Path(campaign_dir)
    with trained_models_commit_lock(campaign):
        return _clean_model_iteration_staging_locked(campaign, proposed_state)


def _clean_model_iteration_staging_locked(
    campaign: Path,
    proposed_state: CampaignState,
) -> List[str]:
    model_versioning = TrainedModelVersioning(trained_models_dir(campaign))
    archived_paths: List[str] = []
    for dangling in model_versioning.list_dangling_staging():
        if dangling.is_symlink() or not dangling.is_dir():
            raise ValueError(
                "refusing invalid trained-model version staging: " + str(dangling)
            )
        _ensure_inside_campaign(campaign, dangling)
        archive = _reconcile_archive_target(
            campaign,
            _timestamped_reconcile_sibling(dangling),
        )
        dangling.rename(archive)
        archived_paths.append(str(archive))

    target = model_versioning.parent / "iteration-staging"
    if not target.exists():
        return archived_paths
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
        archived = _archive_completed_model_iteration_staging(
            campaign,
            target,
            proposed_state,
        )
        return archived_paths + [str(archived)]
    if proposed_state.phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        try:
            model_version = int(proposed_state.models_version)
        except (TypeError, ValueError) as exc:
            raise ValueError("models_version is not an integer") from exc
        if model_version < 0:
            raise ValueError("models_version is negative; cannot verify committed model")
        verify_committed_model_version(campaign, model_version)
    archive = _reconcile_archive_target(
        campaign,
        _timestamped_reconcile_sibling(target),
    )
    target.rename(archive)
    return archived_paths + [str(archive)]


def ferebus_reentry_can_archive_data_staging(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Tuple[bool, str]:
    if proposed_state.phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
        return False, "only FEREBUS re-entry may archive .DATA/STAGING"
    if proposed_state.pending_jobs:
        return False, "proposed state still has pending jobs"
    try:
        reference_data_version = int(proposed_state.reference_data_version)
    except (TypeError, ValueError):
        return False, "reference_data_version is not an integer"
    if reference_data_version < 0:
        return False, "reference_data_version is negative"
    try:
        verify_committed_reference_data_version(campaign_dir, reference_data_version)
    except Exception as exc:
        return False, "committed reference-data version is invalid: " + str(exc)[:180]
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
    target = _reconcile_archive_target(
        campaign,
        staging.with_name(staging.name + ".before-reconcile-" + stamp),
    )
    suffix = 1
    while target.exists():
        target = _reconcile_archive_target(
            campaign,
            staging.with_name(
                staging.name + ".before-reconcile-" + stamp + "." + str(suffix)
            ),
        )
        suffix += 1
    staging.rename(target)
    _reconcile_archive_target(campaign, staging).mkdir(parents=True, exist_ok=True)
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
    target = _reconcile_archive_target(
        campaign,
        staging.with_name(staging.name + ".archived-" + stamp),
    )
    suffix = 1
    while target.exists():
        target = _reconcile_archive_target(
            campaign,
            staging.with_name(
                staging.name + ".archived-" + stamp + "." + str(suffix)
            ),
        )
        suffix += 1
    staging.rename(target)
    _reconcile_archive_target(campaign, staging).mkdir(parents=True, exist_ok=True)
    return [str(target)]


def apply_config_lock_update(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    *,
    campaign_uid: Optional[str] = None,
) -> Path:
    return write_config_lock(
        campaign_dir,
        config,
        campaign_uid=campaign_uid,
        reason="reconcile_config_update",
    )


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
        payload = read_config_lock(campaign)
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
    target = _reconcile_archive_target(campaign, campaign / "campaign.yaml.proposed")
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archive = _reconcile_archive_target(
            campaign,
            target.with_name(target.name + ".before-" + stamp),
        )
        suffix = 1
        while archive.exists():
            archive = _reconcile_archive_target(
                campaign,
                target.with_name(
                    target.name + ".before-" + stamp + "." + str(suffix)
                ),
            )
            suffix += 1
        target.rename(archive)
    config.to_yaml_dense(target)
    return target


def reference_data_staging_can_archive_for_reconcile(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> Tuple[bool, str]:
    campaign = Path(campaign_dir)
    training = campaign / "QM_REFERENCE_DATA"
    if not training.is_dir():
        return False, "QM_REFERENCE_DATA is missing"
    tv = ReferenceDataVersioning(training)
    dangling = tv.list_dangling_staging()
    if not dangling:
        return True, "no dangling reference-data staging exists"
    try:
        reference_data_version = int(proposed_state.reference_data_version)
    except (TypeError, ValueError):
        return False, "reference_data_version is not an integer"
    if reference_data_version < 0:
        return False, "reference_data_version is negative"
    try:
        verify_committed_reference_data_version(campaign, reference_data_version)
    except Exception as exc:
        return False, "committed reference-data version is invalid: " + str(exc)[:180]
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
            return False, "refusing to archive symlinked reference-data staging: " + str(path)
        if not path.is_dir():
            return False, "reference-data staging is not a directory: " + str(path)
        _ensure_inside_campaign(campaign, path)
        resolved = path.resolve()
        if training_resolved not in resolved.parents:
            return False, "reference-data staging is outside QM_REFERENCE_DATA: " + str(path)
    return True, "dangling reference-data staging can be archived"


def archive_reference_data_staging_for_reconcile(
    campaign_dir: Union[str, Path],
    proposed_state: CampaignState,
) -> List[str]:
    ok, reason = reference_data_staging_can_archive_for_reconcile(
        campaign_dir,
        proposed_state,
    )
    if not ok:
        raise ValueError(reason)
    campaign = Path(campaign_dir)
    training = campaign / "QM_REFERENCE_DATA"
    tv = ReferenceDataVersioning(training)
    archived: List[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for staging in tv.list_dangling_staging():
        target = _reconcile_archive_target(
            campaign,
            staging.with_name(staging.name + ".before-reconcile-" + stamp),
        )
        suffix = 1
        while target.exists():
            target = _reconcile_archive_target(
                campaign,
                staging.with_name(
                    staging.name
                    + ".before-reconcile-"
                    + stamp
                    + "."
                    + str(suffix)
                ),
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
