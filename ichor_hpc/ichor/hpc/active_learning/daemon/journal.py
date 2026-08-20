"""Locked, segmented NDJSON journal.

The daemon writes one JSON object per line to ``journal.ndjson``. A bounded
cross-process lock serialises full writes and segment rotation; regular files
do not inherit the ``PIPE_BUF`` atomicity guarantee provided for pipes.

Schema is intentionally flexible: every line carries "ts" (ISO-8601 UTC) and
"event" (short string tag); everything else is event-specific payload.
The daemon uses this for replayable per-iteration provenance and for
post-mortem analysis after a campaign halts.

The journal is append-only within each bounded segment. Rotation preserves
recent immutable segments and starts a fresh current file. This module exposes
:func: "append_event" for the writer side and :func: "read_events" for
introspection / CLI / tests.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
import re
import socket
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import portalocker


__all__ = [
    "JOURNAL_LINE_LIMIT_BYTES",
    "EventTooLargeError",
    "JournalCorruptionError",
    "JournalIntegrityDisposition",
    "JOURNAL_EVENT_CONTEXTS",
    "JOURNAL_EVENT_ITERATION_POLICIES",
    "JOURNAL_PHASE_FIRST_EVENTS",
    "KNOWN_EVENT_TYPES",
    "append_event",
    "inspect_journal_integrity",
    "iter_events",
    "read_events",
    "repair_journal_integrity",
    "tail_events",
]


#Documentation only -- the journal is a free-form NDJSON stream and any
#string is a valid event type. KNOWN_EVENT_TYPES enumerates the names the
# daemon and executors currently emit so users and log consumers can build
#dashboards against a stable list.
#
#  campaign_started        -- emitted once when state.json is first written
#  phase_transition        -- {from_phase, to_phase, iteration}
#  sbatch                  -- {phase, job_id, iteration}
#  phase_succeeded         -- after the postprocess of a SLURM-backed phase
#  sacct_error             -- transient sacct failure
#  shutdown_requested      -- daemon exit signal
#  daemon_started          -- on run() entry
#  daemon_stopped          -- on run() exit
#  daemon_interrupted      -- KeyboardInterrupt path
#  state_corrupt           -- state.json parse failure
#  tick_error              -- unhandled exception in a tick
#  tick_exception_halted   -- unhandled exception persisted state as HALTED
#
#  Later (provenance ledger) additions:
#  subspace_built          -- per-iteration count of seeds whose subspace was built
#  reference_data_committed  -- {reference_data_version, n_committed_points} after REFERENCE_COMMIT
#  seed_selected           -- placeholder
#  anti_overlap_flagged    -- placeholder
#  reference_scales_computed -- placeholder
KNOWN_EVENT_TYPES = (
    "campaign_started",
    "campaign_completed",
    "campaign_reopened",
    "scientific_convergence_reached",
    "phase_transition",
    "sbatch",
    "phase_succeeded",
    "sacct_error",
    "sacct_error_timeout",
    "shutdown_requested",
    "daemon_started",
    "daemon_stopped",
    "daemon_interrupted",
    "state_corrupt",
    "tick_error",
    "tick_exception_halted",
    "subspace_built",
    "reference_data_committed",
    "reference_commit_started",
    "reference_commit_move_progress",
    "reference_commit_shard_progress",
    "reference_commit_shards_resolved",
    "reference_commit_cache_complete",
    "reference_commit_published",
    "seed_selection_started",
    "seed_selection_progress",
    "seed_selection_cache",
    "seed_selected",
    "anti_overlap_flagged",
    "reference_scales_computed",
    "failure_action",
    "halt",
    # new additions I:
    "live_postprocess_refused",
    "effective_config_diff",
    "autotune_applied",
    "trajectory_pool_imported",
    "bootstrap_inputs_confirmed",
    "model_bootstrap_staged",
    "model_bootstrap_committed",
    "sacct_empty_timeout",
    # new additions II:
    "phase_succeeded_live",
    "quantum_output_rejected",
    # new additions III:
    "models_committed",
    "ariadne_landing_rejected",
    "ariadne_landing_summary",
    "error_calibration_summary",
    "error_calibration_failed",
    "phase_output_contract_invalid",
    "required_phase_output_missing_after_failure",
    "reconcile_applied",
    "reconcile_transaction_recovered",
    "committed_artifact_settle_retry",
    "ariadne_optional_diagnostics_warning",
    "ariadne_legacy_missing_trajectory_sha256",
    "ariadne_provenance_reconstructed",
    "resolved_phase_resources",
    "scheduler_usage_recorded",
    "scheduler_usage_warning",
    "checkpoint_failed",
    "checkpoint_verified",
    "reconcile_resolved_terminal_intent",
    "user_cancelled_jobs",
    "user_stop_requested",
    "user_stop_boundary_reached",
    "user_stop_control_invalid",
    "user_stop_request_cancelled",
    "user_stop_resumed",
    "sacct_unknown_timeout",
    "sacct_missing_timeout",
    "sacct_empty_but_squeue_active",
    "sacct_rows_missing_but_squeue_active",
    "squeue_liveness_inconclusive",
    "scheduler_uncertain_resumed",
    "transient_phase_retry",
    "transient_retry_ledger_invalid",
    "job_adopt_check_failed",
    "adopted_accounted_job",
    "adopted_inflight_job",
    "expected_tasks_inference_failed",
    "submission_intent_expected_tasks_invalid",
    "submission_intent_update_failed",
    "submission_intent_read_failed",
    "submission_intent_completion_deferred",
    "submission_intent_retired_without_submission",
    "stop_request_completion_deferred",
    "phase_completion_replayed",
    "provenance_index_repaired",
    "provenance_index_repair_failed",
    "postprocess_settle_retry",
    "phase_pre_submit_intent",
    "phase_submitted",
    "queue_lifecycle_update",
    "staging_archived",
    "staging_restored_from_archive",
    "daemon_lease_conflict",
    "daemon_lease_stale_recovered",
    "daemon_lease_cleanup_failed",
    "daemon_lease_heartbeat_failed",
    "daemon_lease_heartbeat_recovered",
    "environment_drift_halted",
    "environment_rebound",
    "environment_generation_advanced",
    "ariadne_seed_provenance_repaired",
    "ariadne_seed_provenance_staged",
    "ariadne_stale_outputs_quarantined",
    "ariadne_publication_archived",
    "ariadne_task_rejected_missing_result",
    "ariadne_task_rejected_malformed_result",
    "ariadne_task_rejected_unusable_result",
    "ariadne_task_rejected_unsafe_landing",
    "ariadne_task_salvaged_from_nonzero_exit",
    "ferebus_candidate_rejected",
    "ferebus_quality_measurement_incomplete",
    "ferebus_candidate_recovery_prepared",
    "ferebus_candidate_recovery_materialised",
    "ferebus_candidate_reprocessed",
    "ferebus_quality_summary",
    "geometry_novelty_scale_precomputed",
    "sampling_protocol_resolved",
    "initial_training_existing_without_bootstrap_handoff",
    "phase_b_novelty_threshold_relaxed",
    "phase_b_geometry_novelty_relaxed",
    "pool_feasibility_checked",
    "quantum_quality_summary",
    "quantum_quality_rejected",
    "seed_posterior_fallback",
    "point_allocation_complete",
    "point_allocation_replacement_prepared",
    "point_allocation_quantum_recorded",
    "aimall_skipped_no_gaussian_acceptances",
    "aimall_quality_revalidated",
    "ariadne_sampling_protocol_replay_failed",
    "legacy_sampling_protocol_repreview",
    "partial_array_recovery_postprocess_only",
    "partial_array_recovery_prepared",
    "active_iteration_finalised",
    "ariadne_task_rejected_invalid_output",
    "dry_run_trajectory_pool_created",
    "daemon_startup_progress",
    "phase_activity_started",
    "phase_activity_progress",
    "phase_activity_completed",
    "phase_activity_failed",
    "scheduler_progress",
    "checkpoint_progress",
)
if len(KNOWN_EVENT_TYPES) != len(set(KNOWN_EVENT_TYPES)):
    raise RuntimeError("journal event catalogue contains duplicate event types")


def _build_event_context_registry() -> Tuple[Dict[str, str], frozenset[str]]:
    """Build the explicit event presentation registry.

    Grouping is declarative rather than name-based: adding an event to
    ``KNOWN_EVENT_TYPES`` requires assigning it exactly one fallback context.
    Phase-first events use their recorded FSM phase when one is available.
    """

    fixed_groups = {
        "CAMPAIGN": (
            "campaign_started",
            "campaign_completed",
            "campaign_reopened",
            "scientific_convergence_reached",
            "phase_transition",
            "active_iteration_finalised",
        ),
        "DAEMON": (
            "daemon_started",
            "daemon_stopped",
            "daemon_interrupted",
            "tick_error",
            "tick_exception_halted",
            "daemon_lease_conflict",
            "daemon_lease_stale_recovered",
            "daemon_lease_cleanup_failed",
            "daemon_lease_heartbeat_failed",
            "daemon_lease_heartbeat_recovered",
            "daemon_startup_progress",
        ),
        "STOP CONTROL": (
            "shutdown_requested",
            "user_cancelled_jobs",
            "user_stop_requested",
            "user_stop_boundary_reached",
            "user_stop_control_invalid",
            "user_stop_request_cancelled",
            "user_stop_resumed",
            "stop_request_completion_deferred",
        ),
        "CONFIG": (
            "effective_config_diff",
            "autotune_applied",
        ),
        "ENVIRONMENT": (
            "environment_drift_halted",
            "environment_rebound",
            "environment_generation_advanced",
        ),
        "STATE": ("state_corrupt",),
        "RECONCILE": (
            "reconcile_applied",
            "reconcile_transaction_recovered",
            "reconcile_resolved_terminal_intent",
            "staging_archived",
            "staging_restored_from_archive",
        ),
        "CHECKPOINT": (
            "checkpoint_failed",
            "checkpoint_verified",
            "checkpoint_progress",
        ),
    }
    phase_groups = {
        "CAMPAIGN": (
            "phase_succeeded",
            "phase_succeeded_live",
            "failure_action",
            "halt",
            "live_postprocess_refused",
            "phase_output_contract_invalid",
            "required_phase_output_missing_after_failure",
            "phase_completion_replayed",
            "phase_activity_started",
            "phase_activity_progress",
            "phase_activity_completed",
            "phase_activity_failed",
        ),
        "SCHEDULER": (
            "sbatch",
            "sacct_error",
            "sacct_error_timeout",
            "sacct_empty_timeout",
            "sacct_unknown_timeout",
            "sacct_missing_timeout",
            "sacct_empty_but_squeue_active",
            "sacct_rows_missing_but_squeue_active",
            "squeue_liveness_inconclusive",
            "scheduler_uncertain_resumed",
            "job_adopt_check_failed",
            "adopted_accounted_job",
            "adopted_inflight_job",
            "expected_tasks_inference_failed",
            "submission_intent_expected_tasks_invalid",
            "submission_intent_update_failed",
            "submission_intent_read_failed",
            "submission_intent_completion_deferred",
            "submission_intent_retired_without_submission",
            "phase_pre_submit_intent",
            "phase_submitted",
            "queue_lifecycle_update",
            "scheduler_usage_recorded",
            "scheduler_usage_warning",
            "scheduler_progress",
        ),
        "BOOTSTRAP": (
            "trajectory_pool_imported",
            "bootstrap_inputs_confirmed",
            "model_bootstrap_staged",
            "model_bootstrap_committed",
            "initial_training_existing_without_bootstrap_handoff",
            "pool_feasibility_checked",
            "dry_run_trajectory_pool_created",
        ),
        "REFERENCE COMMIT": (
            "reference_data_committed",
            "reference_commit_started",
            "reference_commit_move_progress",
            "reference_commit_shard_progress",
            "reference_commit_shards_resolved",
            "reference_commit_cache_complete",
            "reference_commit_published",
        ),
        "SEED SELECT": (
            "seed_selection_started",
            "seed_selection_progress",
            "seed_selection_cache",
            "seed_selected",
            "reference_scales_computed",
            "seed_posterior_fallback",
            "sampling_protocol_resolved",
        ),
        "ARIADNE": (
            "subspace_built",
            "anti_overlap_flagged",
            "ariadne_landing_rejected",
            "ariadne_landing_summary",
            "ariadne_optional_diagnostics_warning",
            "ariadne_legacy_missing_trajectory_sha256",
            "ariadne_provenance_reconstructed",
            "ariadne_seed_provenance_repaired",
            "ariadne_seed_provenance_staged",
            "ariadne_stale_outputs_quarantined",
            "ariadne_publication_archived",
            "ariadne_task_rejected_missing_result",
            "ariadne_task_rejected_malformed_result",
            "ariadne_task_rejected_unusable_result",
            "ariadne_task_rejected_unsafe_landing",
            "ariadne_task_salvaged_from_nonzero_exit",
            "ariadne_sampling_protocol_replay_failed",
            "ariadne_task_rejected_invalid_output",
        ),
        "DIVERSITY": (
            "geometry_novelty_scale_precomputed",
            "phase_b_novelty_threshold_relaxed",
            "phase_b_geometry_novelty_relaxed",
        ),
        "QM": (
            "quantum_output_rejected",
            "quantum_quality_summary",
            "quantum_quality_rejected",
            "aimall_skipped_no_gaussian_acceptances",
            "aimall_quality_revalidated",
            "committed_artifact_settle_retry",
            "postprocess_settle_retry",
        ),
        "POINT ALLOCATION": (
            "point_allocation_complete",
            "point_allocation_replacement_prepared",
            "point_allocation_quantum_recorded",
        ),
        "FEREBUS": (
            "models_committed",
            "ferebus_candidate_rejected",
            "ferebus_quality_measurement_incomplete",
            "ferebus_candidate_recovery_prepared",
            "ferebus_candidate_recovery_materialised",
            "ferebus_candidate_reprocessed",
            "ferebus_quality_summary",
        ),
        "CALIBRATION": (
            "error_calibration_summary",
            "error_calibration_failed",
        ),
        "RESOURCES": ("resolved_phase_resources",),
        "PROVENANCE": (
            "provenance_index_repaired",
            "provenance_index_repair_failed",
        ),
        "RECOVERY": (
            "transient_phase_retry",
            "transient_retry_ledger_invalid",
            "partial_array_recovery_postprocess_only",
            "partial_array_recovery_prepared",
            "legacy_sampling_protocol_repreview",
        ),
    }

    contexts: Dict[str, str] = {}
    phase_first: set[str] = set()
    for context, events in fixed_groups.items():
        for event in events:
            if event in contexts:
                raise RuntimeError("duplicate journal context specification: " + event)
            contexts[event] = context
    for context, events in phase_groups.items():
        for event in events:
            if event in contexts:
                raise RuntimeError("duplicate journal context specification: " + event)
            contexts[event] = context
            phase_first.add(event)
    known = set(KNOWN_EVENT_TYPES)
    specified = set(contexts)
    if known != specified:
        raise RuntimeError(
            "journal context registry mismatch: missing="
            + repr(sorted(known - specified))
            + " extra="
            + repr(sorted(specified - known))
        )
    return contexts, frozenset(phase_first)


JOURNAL_EVENT_CONTEXTS, JOURNAL_PHASE_FIRST_EVENTS = (
    _build_event_context_registry()
)


_STATE_UNAVAILABLE_ITERATION_EVENTS = frozenset(
    {
        "daemon_lease_conflict",
        "daemon_lease_stale_recovered",
        "daemon_lease_cleanup_failed",
        "state_corrupt",
        "user_cancelled_jobs",
    }
)
JOURNAL_EVENT_ITERATION_POLICIES: Dict[str, str] = {
    event: (
        "state_unavailable_allowed"
        if event in _STATE_UNAVAILABLE_ITERATION_EVENTS
        else "required"
    )
    for event in KNOWN_EVENT_TYPES
}
if set(JOURNAL_EVENT_ITERATION_POLICIES) != set(KNOWN_EVENT_TYPES):
    raise RuntimeError("journal iteration-policy registry mismatch")


# Keep individual diagnostic records compact. Cross-process safety comes from
# the bounded journal lock and full-write loop, not from pipe semantics.
JOURNAL_LINE_LIMIT_BYTES = 4000
DEFAULT_JOURNAL_MAX_BYTES = 67_108_864
DEFAULT_JOURNAL_RETAINED_FILES = 8
DEFAULT_JOURNAL_LOCK_TIMEOUT_SECONDS = 30
JOURNAL_APPEND_LOCK_SUFFIX = ".append-lock"


class EventTooLargeError(ValueError):
    """Raised when a serialised event exceeds the diagnostic record limit."""


class JournalCorruptionError(ValueError):
    """Raised whenever a journal reader encounters an invalid record."""

    def __init__(self, path: Path, line_number: int, offset: int, reason: str) -> None:
        self.path = Path(path)
        self.line_number = int(line_number)
        self.offset = int(offset)
        self.reason = str(reason)
        super().__init__(
            str(path)
            + ": journal corruption at line "
            + str(line_number)
            + ", byte "
            + str(offset)
            + ": "
            + str(reason)
        )


@dataclass(frozen=True)
class JournalIntegrityDisposition:
    """Read-only classification of one complete journal stream."""

    disposition: str
    repairable: bool
    journal_path: str
    segment_path: Optional[str] = None
    segment_sha256: Optional[str] = None
    segment_size: Optional[int] = None
    segment_identity: Tuple[int, ...] = ()
    line_number: Optional[int] = None
    offset: Optional[int] = None
    length: Optional[int] = None
    malformed_sha256: Optional[str] = None
    valid_records: int = 0
    retained_records: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disposition": self.disposition,
            "repairable": bool(self.repairable),
            "journal_path": self.journal_path,
            "segment_path": self.segment_path,
            "segment_sha256": self.segment_sha256,
            "segment_size": self.segment_size,
            "segment_identity": list(self.segment_identity),
            "line_number": self.line_number,
            "offset": self.offset,
            "length": self.length,
            "malformed_sha256": self.malformed_sha256,
            "valid_records": int(self.valid_records),
            "retained_records": int(self.retained_records),
            "reason": self.reason,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_advisory_json(value: Any, path: str = "$") -> Tuple[Any, List[str]]:
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        issues: List[str] = []
        for key, item in value.items():
            normalised, found = _normalise_advisory_json(item, path + "." + str(key))
            out[str(key)] = normalised
            issues.extend(found)
        return out, issues
    if isinstance(value, (list, tuple)):
        out_list: List[Any] = []
        issues = []
        for index, item in enumerate(value):
            normalised, found = _normalise_advisory_json(
                item, path + "[" + str(index) + "]"
            )
            out_list.append(normalised)
            issues.extend(found)
        return out_list, issues
    return value, []


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(label + " must be a non-empty ISO-8601 timestamp")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(label + " is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(label + " must include a timezone")
    return parsed.astimezone(timezone.utc)


def _encode_event(event_type: str, payload: Dict[str, Any], ts: Optional[str] = None) -> bytes:
    event = str(event_type)
    if not event or any(ord(character) < 32 for character in event):
        raise ValueError("event type must be a non-empty control-free string")
    timestamp = ts or _now_iso()
    _parse_timestamp(timestamp, "journal event timestamp")
    record = {"ts": timestamp, "event": event}
    for k, v in payload.items():
        if k in ("ts", "event"):
            raise ValueError("payload key " + repr(k) + " is reserved")
        record[k] = v
    record, non_finite = _normalise_advisory_json(record)
    if non_finite:
        record["non_finite_fields"] = non_finite
        record["non_finite_reason"] = "advisory non-finite values were serialised as null"
    encoded = json.dumps(
        record,
        sort_keys=False,
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) + 1 > JOURNAL_LINE_LIMIT_BYTES:
        raise EventTooLargeError(
            "event line is " + str(len(encoded) + 1) + " bytes; "
            "the journal enforces < " + str(JOURNAL_LINE_LIMIT_BYTES) + ". "
            "Split the payload."
        )
    return encoded + b"\n"


def _process_start_identity(pid: int) -> str:
    """Return a same-host process identity that survives PID reuse checks."""
    proc_stat = Path("/proc") / str(int(pid)) / "stat"
    try:
        raw = proc_stat.read_text(encoding="ascii")
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        return fields[19] if closing >= 0 and len(fields) > 19 else ""
    except (OSError, UnicodeError, ValueError):
        return ""


def _same_host_owner_is_dead(owner: Mapping[str, Any]) -> bool:
    if str(owner.get("host") or "") != socket.gethostname():
        return False
    try:
        pid = int(owner.get("pid"))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OverflowError:
        return True
    except (OSError, PermissionError):
        return False
    recorded = str(owner.get("process_start_identity") or "")
    observed = _process_start_identity(pid)
    return bool(recorded and observed and recorded != observed)


def _read_lock_owner(lock_dir: Path) -> Optional[Dict[str, Any]]:
    owner_path = lock_dir / "owner.json"
    try:
        if owner_path.is_symlink():
            return None
        payload = json.loads(owner_path.read_text(encoding="utf-8"), source=owner_path)
    except (OSError, ValueError, UnicodeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _clear_dead_same_host_lock(lock_dir: Path) -> bool:
    """Clear only an authenticated lock owned by a dead local process."""
    try:
        lock_stat = lock_dir.lstat()
    except FileNotFoundError:
        return True
    if not stat.S_ISDIR(lock_stat.st_mode) or stat.S_ISLNK(lock_stat.st_mode):
        return False
    owner = _read_lock_owner(lock_dir)
    if owner is None or not _same_host_owner_is_dead(owner):
        return False
    owner_path = lock_dir / "owner.json"
    retired = lock_dir.with_name(
        "." + lock_dir.name + ".dead-" + uuid.uuid4().hex
    )
    try:
        entries = list(lock_dir.iterdir())
        if entries != [owner_path]:
            return False
        confirmed = _read_lock_owner(lock_dir)
        if confirmed != owner or not _same_host_owner_is_dead(confirmed):
            return False
        os.rename(lock_dir, retired)
        (retired / "owner.json").unlink()
        os.rmdir(retired)
        return True
    except (FileNotFoundError, OSError):
        return False


@contextmanager
def _journal_append_lock(path: Path, timeout_seconds: int) -> Iterator[None]:
    """Acquire a server-atomic mutex suitable for distributed filesystems."""
    from .state import _fsync_parent_dir

    timeout = float(timeout_seconds)
    if timeout < 0:
        raise ValueError("journal lock timeout must be non-negative")
    lock_dir = path.with_name(path.name + JOURNAL_APPEND_LOCK_SUFFIX)
    nonce = uuid.uuid4().hex
    candidate_dir = lock_dir.with_name(
        "." + lock_dir.name + ".candidate-" + nonce
    )
    released_dir = lock_dir.with_name(
        "." + lock_dir.name + ".released-" + nonce
    )
    owner = {
        "schema_version": 1,
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "process_start_identity": _process_start_identity(os.getpid()),
        "nonce": nonce,
        "created_at_iso": _now_iso(),
    }
    encoded = json.dumps(owner, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    os.mkdir(candidate_dir, 0o700)
    candidate_owner = candidate_dir / "owner.json"
    try:
        fd = os.open(
            str(candidate_owner),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            written = 0
            while written < len(encoded):
                progress = os.write(fd, encoded[written:])
                if progress <= 0:
                    raise OSError("journal lock owner write made no progress")
                written += int(progress)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_parent_dir(candidate_owner)
    except Exception:
        candidate_owner.unlink(missing_ok=True)
        try:
            os.rmdir(candidate_dir)
        except OSError:
            pass
        raise

    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                os.rename(candidate_dir, lock_dir)
                _fsync_parent_dir(lock_dir)
                acquired = True
                break
            except OSError as exc:
                if not lock_dir.exists() and not lock_dir.is_symlink():
                    raise OSError(
                        "could not acquire journal append lock: " + str(exc)
                    ) from exc
                _clear_dead_same_host_lock(lock_dir)
                if time.monotonic() >= deadline:
                    current = _read_lock_owner(lock_dir)
                    description = (
                        "unknown owner"
                        if current is None
                        else str(current.get("host") or "?")
                        + ":"
                        + str(current.get("pid") or "?")
                    )
                    raise TimeoutError(
                        "journal append lock remained owned by " + description
                    )
                time.sleep(0.05)
        yield
    finally:
        if not acquired:
            candidate_owner.unlink(missing_ok=True)
            try:
                os.rmdir(candidate_dir)
            except OSError:
                pass
        else:
            current = _read_lock_owner(lock_dir)
            if current is None or str(current.get("nonce") or "") != nonce:
                raise RuntimeError(
                    "journal append lock ownership changed before release"
                )
            os.rename(lock_dir, released_dir)
            _fsync_parent_dir(lock_dir)
            (released_dir / "owner.json").unlink()
            os.rmdir(released_dir)


def _validate_record_body(body: bytes) -> Dict[str, Any]:
    record = json.loads(body.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("journal record must be a JSON object")
    event = record.get("event")
    if not isinstance(event, str) or not event:
        raise ValueError("journal event must be a non-empty string")
    _parse_timestamp(record.get("ts"), "journal event timestamp")
    return record


def _validate_append_target(path: Path) -> int:
    """Reject unsafe paths and any tail that another append would cement."""
    parent_stat = path.parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise ValueError("journal parent is not a regular directory: " + str(path.parent))
    try:
        current_stat = path.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(current_stat.st_mode) or stat.S_ISLNK(current_stat.st_mode):
        raise ValueError("journal is not a regular file: " + str(path))
    size = int(current_stat.st_size)
    if size == 0:
        return 0
    read_size = min(size, JOURNAL_LINE_LIMIT_BYTES)
    with open(path, "rb") as handle:
        handle.seek(size - read_size)
        tail = handle.read(read_size)
    if not tail.endswith(b"\n"):
        raise JournalCorruptionError(
            path,
            0,
            max(0, size - read_size),
            "current journal has an unterminated tail; append refused",
        )
    body_end = len(tail) - 1
    body_start = tail.rfind(b"\n", 0, body_end) + 1
    body = tail[body_start:body_end].strip()
    if not body:
        raise JournalCorruptionError(
            path, 0, size - len(tail) + body_start, "journal ends with an empty record"
        )
    try:
        _validate_record_body(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise JournalCorruptionError(
            path,
            0,
            size - len(tail) + body_start,
            type(exc).__name__ + ": " + str(exc),
        ) from exc
    return size


def append_event(
    journal_path: Union[str, Path],
    event_type: str,
    *,
    ts: Optional[str] = None,
    fsync: bool = False,
    max_bytes: int = DEFAULT_JOURNAL_MAX_BYTES,
    retained_files: int = DEFAULT_JOURNAL_RETAINED_FILES,
    lock_timeout_seconds: int = DEFAULT_JOURNAL_LOCK_TIMEOUT_SECONDS,
    **payload: Any,
) -> str:
    """Append a single NDJSON record to "journal_path".

    Serialises rotation and the complete append under a bounded cross-process
    lock. Returns the encoded "ts" so the caller can echo it into log lines or
    unit tests.

    Setting "fsync=True" forces an fsync after the append; the default
    False matches the design note that journal durability is best-effort --
    the state file is the authoritative checkpoint.
    """
    encoded = _encode_event(event_type, payload, ts=ts)
    p = Path(journal_path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("journal max_bytes must be a positive integer")
    if (
        isinstance(retained_files, bool)
        or not isinstance(retained_files, int)
        or retained_files <= 0
    ):
        raise ValueError("journal retained_files must be a positive integer")
    p.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = p.parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise ValueError("journal parent is not a regular directory: " + str(p.parent))
    lock_path = p.with_name(p.name + ".lock")
    if lock_path.is_symlink():
        raise ValueError("journal advisory lock path is a symlink: " + str(lock_path))
    with _journal_append_lock(p, int(lock_timeout_seconds)):
        # Retain the legacy advisory lock so a stopped-boundary upgrade also
        # serialises against older local processes. The directory mutex is the
        # cross-host authority.
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            timeout=float(lock_timeout_seconds),
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            current_size = _validate_append_target(p)
            if current_size and current_size + len(encoded) > int(max_bytes):
                # Rotation is rare, so validate every retained record before
                # making the current file immutable.
                for _record in iter_events(p):
                    pass
                segment = p.with_name(
                    p.stem
                    + ".segment."
                    + str(time.time_ns()).zfill(20)
                    + "."
                    + str(os.getpid())
                    + "."
                    + uuid.uuid4().hex[:8]
                    + p.suffix
                )
                os.replace(str(p), str(segment))
                from .state import _fsync_parent_dir

                _fsync_parent_dir(p)
                archives = sorted(
                    p.parent.glob(p.stem + ".segment.*" + p.suffix)
                )
                keep_archives = max(0, int(retained_files) - 1)
                for old in archives[: max(0, len(archives) - keep_archives)]:
                    old_stat = old.lstat()
                    if not stat.S_ISREG(old_stat.st_mode) or stat.S_ISLNK(
                        old_stat.st_mode
                    ):
                        raise ValueError(
                            "journal segment is not a regular file: " + str(old)
                        )
                    old.unlink()
                _fsync_parent_dir(p)
            fd = os.open(
                str(p),
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
            try:
                written = 0
                while written < len(encoded):
                    try:
                        progress = os.write(fd, encoded[written:])
                    except InterruptedError:
                        continue
                    if progress <= 0:
                        raise OSError("journal append made no write progress")
                    written += int(progress)
                if fsync:
                    from .state import _fsync_file_descriptor

                    _fsync_file_descriptor(fd)
            finally:
                os.close(fd)
    return json.loads(encoded.decode("utf-8"))["ts"]


def _journal_segments(path: Path) -> List[Path]:
    archives = sorted(path.parent.glob(path.stem + ".segment.*" + path.suffix))
    return archives + ([path] if path.exists() or path.is_symlink() else [])


_OVERLAP_PREFIX_RE = re.compile(r"^[A-Za-z0-9._:-]{4,160}$")
_PROGRESS_SUFFIX_KEYS = (
    "completed",
    "total",
    "unit",
    "running",
    "pending",
    "failed",
    "missing",
    "accepted",
    "rejected",
    "scientific_publication_complete",
    "pending_reason",
    "pending_queue",
    "scheduler_native_state",
    "throughput",
    "scheduler_identity_kind",
)
_PROGRESS_EVENTS = frozenset(
    {
        "phase_activity_started",
        "phase_activity_progress",
        "phase_activity_completed",
        "phase_activity_failed",
        "scheduler_progress",
    }
)


def _segment_identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _progress_overlap_payload(body: bytes) -> Optional[Dict[str, Any]]:
    """Parse only the canonical suffix left by a same-offset overwrite."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text or len(body) > JOURNAL_LINE_LIMIT_BYTES:
        return None
    if any(ord(character) < 32 for character in text):
        return None
    marker = '", "completed":'
    marker_index = text.find(marker)
    if marker_index <= 0:
        return None
    prefix = text[:marker_index]
    if _OVERLAP_PREFIX_RE.fullmatch(prefix) is None:
        return None
    try:
        payload = json.loads(
            ('{"overlap_prefix":"' + text).encode("utf-8").decode("utf-8")
        )
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    keys = list(payload)[1:]
    if keys[:3] != ["completed", "total", "unit"]:
        return None
    positions: List[int] = []
    for key in keys:
        if key not in _PROGRESS_SUFFIX_KEYS:
            return None
        positions.append(_PROGRESS_SUFFIX_KEYS.index(key))
    if positions != sorted(set(positions)):
        return None
    scheduler_kind = payload.get("scheduler_identity_kind")
    if scheduler_kind not in {"slurm", "sge"}:
        return None
    if (
        isinstance(payload.get("completed"), bool)
        or not isinstance(payload.get("completed"), int)
        or isinstance(payload.get("total"), bool)
        or not isinstance(payload.get("total"), int)
        or int(payload["completed"]) < 0
        or int(payload["total"]) <= 0
        or int(payload["completed"]) > int(payload["total"])
        or not isinstance(payload.get("unit"), str)
    ):
        return None
    return dict(payload)


def _overlap_has_neighbour_evidence(
    *,
    payload: Mapping[str, Any],
    body_length: int,
    previous_line_length: int,
    neighbouring_records: Iterable[Tuple[Dict[str, Any], int]],
) -> bool:
    keys = list(payload)[1:]
    overlap_prefix = str(payload.get("overlap_prefix") or "")
    for record, encoded_length in neighbouring_records:
        if str(record.get("event") or "") not in _PROGRESS_EVENTS:
            continue
        record_keys = list(record)
        if "completed" not in record_keys:
            continue
        candidate_keys = record_keys[record_keys.index("completed") :]
        if candidate_keys != keys:
            continue
        completed_index = record_keys.index("completed")
        if not any(
            isinstance(record.get(key), str)
            and str(record.get(key)).endswith(overlap_prefix)
            for key in record_keys[:completed_index]
        ):
            continue
        if any(record.get(key) != payload.get(key) for key in keys):
            continue
        overwritten = int(encoded_length) - int(body_length + 1)
        if abs(overwritten - int(previous_line_length)) <= 2:
            return True
    return False


def _inspect_journal_integrity_once(
    journal_path: Union[str, Path],
) -> JournalIntegrityDisposition:
    """Classify journal damage without changing any campaign file."""
    current = Path(journal_path)
    segments = _journal_segments(current)
    if not segments:
        return JournalIntegrityDisposition(
            disposition="valid",
            repairable=False,
            journal_path=str(current),
            reason="journal is absent",
        )
    valid_records = 0
    issue: Optional[JournalIntegrityDisposition] = None
    for segment in segments:
        try:
            before = segment.lstat()
        except OSError as exc:
            return JournalIntegrityDisposition(
                disposition="unsafe",
                repairable=False,
                journal_path=str(current),
                segment_path=str(segment),
                reason="journal segment is unreadable: " + str(exc),
                valid_records=valid_records,
            )
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            return JournalIntegrityDisposition(
                disposition="unsafe",
                repairable=False,
                journal_path=str(current),
                segment_path=str(segment),
                reason="journal segment is not a regular file",
                valid_records=valid_records,
            )
        raw = segment.read_bytes()
        after = segment.lstat()
        if _segment_identity(before) != _segment_identity(after):
            return JournalIntegrityDisposition(
                disposition="unsafe",
                repairable=False,
                journal_path=str(current),
                segment_path=str(segment),
                reason="journal segment changed during inspection",
                valid_records=valid_records,
            )
        digest = hashlib.sha256(raw).hexdigest()
        lines = raw.splitlines(keepends=True)
        parsed: List[Optional[Dict[str, Any]]] = []
        offsets: List[int] = []
        offset = 0
        bad_indexes: List[int] = []
        for index, line in enumerate(lines):
            offsets.append(offset)
            body = line.strip()
            unterminated_current_tail = bool(
                segment == current
                and index == len(lines) - 1
                and not line.endswith((b"\n", b"\r"))
            )
            if not body:
                parsed.append(None)
                if unterminated_current_tail:
                    bad_indexes.append(index)
                offset += len(line)
                continue
            try:
                record = _validate_record_body(body)
                if unterminated_current_tail:
                    parsed.append(None)
                    bad_indexes.append(index)
                else:
                    parsed.append(record)
                    valid_records += 1
            except (UnicodeDecodeError, ValueError):
                parsed.append(None)
                bad_indexes.append(index)
            offset += len(line)
        if not bad_indexes:
            continue
        if issue is not None or len(bad_indexes) != 1:
            return JournalIntegrityDisposition(
                disposition="unsafe",
                repairable=False,
                journal_path=str(current),
                segment_path=str(segment),
                segment_sha256=digest,
                segment_size=len(raw),
                segment_identity=_segment_identity(after),
                reason="journal has multiple malformed records",
                valid_records=valid_records,
            )
        bad_index = bad_indexes[0]
        line = lines[bad_index]
        body = line.strip()
        common = {
            "journal_path": str(current),
            "segment_path": str(segment),
            "segment_sha256": digest,
            "segment_size": len(raw),
            "segment_identity": _segment_identity(after),
            "line_number": bad_index + 1,
            "offset": offsets[bad_index],
            "length": len(line),
            "malformed_sha256": hashlib.sha256(line).hexdigest(),
            "valid_records": valid_records,
            "retained_records": valid_records,
        }
        if segment == current and bad_index == len(lines) - 1 and not line.endswith(
            (b"\n", b"\r")
        ):
            issue = JournalIntegrityDisposition(
                disposition="recoverable_torn_tail",
                repairable=True,
                reason="one unterminated current-journal tail can be omitted",
                **common,
            )
            continue
        payload = _progress_overlap_payload(body)
        neighbours: List[Tuple[Dict[str, Any], int]] = []
        for candidate_index in range(
            max(0, bad_index - 8), min(len(lines), bad_index + 9)
        ):
            candidate = parsed[candidate_index]
            if candidate is not None:
                neighbours.append((candidate, len(lines[candidate_index])))
        previous_length = len(lines[bad_index - 1]) if bad_index > 0 else 0
        if (
            segment == current
            and line.endswith((b"\n", b"\r"))
            and payload is not None
            and previous_length > 0
            and _overlap_has_neighbour_evidence(
                payload=payload,
                body_length=len(body),
                previous_line_length=previous_length,
                neighbouring_records=neighbours,
            )
        ):
            issue = JournalIntegrityDisposition(
                disposition="recoverable_cross_host_progress_overlap",
                repairable=True,
                reason=(
                    "one bounded progress-event suffix matches a same-offset "
                    "cross-host append overlap"
                ),
                **common,
            )
            continue
        return JournalIntegrityDisposition(
            disposition="unsafe",
            repairable=False,
            reason="malformed journal record is not a recognised append overlap",
            **common,
        )
    if issue is not None:
        return issue
    return JournalIntegrityDisposition(
        disposition="valid",
        repairable=False,
        journal_path=str(current),
        valid_records=valid_records,
        retained_records=valid_records,
        reason="all journal records validate",
    )


def inspect_journal_integrity(
    journal_path: Union[str, Path],
) -> JournalIntegrityDisposition:
    """Classify a journal, retrying bounded races with an active appender."""
    report: Optional[JournalIntegrityDisposition] = None
    for attempt in range(3):
        report = _inspect_journal_integrity_once(journal_path)
        transient = bool(
            report.disposition == "unsafe"
            and (
                report.reason.startswith("journal segment changed during inspection")
                or report.reason.startswith("journal segment is unreadable:")
            )
        )
        if not transient or attempt == 2:
            return report
        time.sleep(0.01)
    assert report is not None
    return report


def _atomic_publish_bytes(path: Path, payload: bytes) -> None:
    from .state import _fsync_parent_dir

    payload_sha256 = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(
        "." + path.name + ".tmp-" + payload_sha256[:16]
    )
    if temporary.exists() or temporary.is_symlink():
        temporary_stat = temporary.lstat()
        if not stat.S_ISREG(temporary_stat.st_mode) or stat.S_ISLNK(
            temporary_stat.st_mode
        ):
            raise ValueError("journal atomic temporary path is unsafe")
        partial = temporary.read_bytes()
        if partial == payload:
            os.replace(temporary, path)
            _fsync_parent_dir(path)
            return
        if not payload.startswith(partial):
            raise ValueError("journal atomic temporary file conflicts")
        temporary.unlink()
        _fsync_parent_dir(temporary)
    fd = os.open(
        str(temporary),
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            progress = os.write(fd, payload[written:])
            if progress <= 0:
                raise OSError("journal repair write made no progress")
            written += int(progress)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_parent_dir(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def repair_journal_integrity(
    journal_path: Union[str, Path],
    expected: Mapping[str, Any],
    *,
    archive_dir: Union[str, Path],
    lock_timeout_seconds: int = DEFAULT_JOURNAL_LOCK_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Archive exact damaged bytes and omit only the classified bad range."""
    current = Path(journal_path)
    expected_disposition = str(expected.get("disposition") or "")
    if expected_disposition not in {
        "recoverable_torn_tail",
        "recoverable_cross_host_progress_overlap",
    } or not bool(expected.get("repairable", False)):
        raise ValueError("journal integrity evidence is not repairable")
    legacy_lock = current.with_name(current.name + ".lock")
    if legacy_lock.is_symlink():
        raise ValueError("journal advisory lock path is a symlink")
    with _journal_append_lock(
        current, int(lock_timeout_seconds)
    ), portalocker.Lock(
        str(legacy_lock),
        mode="a",
        timeout=float(lock_timeout_seconds),
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    ):
        observed = inspect_journal_integrity(current).to_dict()
        for key in (
            "disposition",
            "segment_path",
            "segment_sha256",
            "segment_size",
            "segment_identity",
            "line_number",
            "offset",
            "length",
            "malformed_sha256",
        ):
            if observed.get(key) != expected.get(key):
                raise ValueError("journal changed after repair preview: " + key)
        segment = Path(str(observed["segment_path"]))
        if segment != current:
            raise ValueError("only current-journal repair is supported")
        raw = segment.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(observed["segment_sha256"]):
            raise ValueError("journal changed while repair lock was held")
        offset = int(observed["offset"])
        length = int(observed["length"])
        malformed = raw[offset : offset + length]
        if hashlib.sha256(malformed).hexdigest() != str(
            observed["malformed_sha256"]
        ):
            raise ValueError("journal malformed range changed before repair")

        archive_root = Path(archive_dir)
        archive_root.mkdir(parents=True, exist_ok=True)
        root_stat = archive_root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            raise ValueError("journal quarantine is not a regular directory")
        archive = archive_root / (
            "j-" + str(observed["segment_sha256"])[:16] + ".ndjson"
        )
        if archive.exists() or archive.is_symlink():
            archive_stat = archive.lstat()
            if (
                not stat.S_ISREG(archive_stat.st_mode)
                or stat.S_ISLNK(archive_stat.st_mode)
                or hashlib.sha256(archive.read_bytes()).hexdigest()
                != str(observed["segment_sha256"])
            ):
                raise ValueError("journal quarantine archive conflicts")
        else:
            _atomic_publish_bytes(archive, raw)

        repaired = raw[:offset] + raw[offset + length :]
        # Validate bytes directly without publishing a misleading temporary
        # segment into the journal glob.
        for line in repaired.splitlines(keepends=True):
            body = line.strip()
            if body:
                _validate_record_body(body)
        if repaired and not repaired.endswith(b"\n"):
            raise ValueError("journal repair would leave an unterminated tail")
        _atomic_publish_bytes(current, repaired)
        final = inspect_journal_integrity(current)
        if final.disposition != "valid":
            raise ValueError("journal repair did not produce a valid stream")
        return {
            "changed": True,
            "disposition": expected_disposition,
            "archive_path": str(archive),
            "journal_path": str(current),
            "original_sha256": str(observed["segment_sha256"]),
            "repaired_sha256": hashlib.sha256(repaired).hexdigest(),
            "omitted_offset": offset,
            "omitted_bytes": length,
            "retained_records": int(final.valid_records),
        }


def iter_events(journal_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Yield strict event objects from immutable segments and the current file.

    Any malformed or unterminated record is reported. Recovery classification
    is deliberately separate so ordinary readers never hide telemetry loss.
    """
    p = Path(journal_path)
    segments = _journal_segments(p)
    if not segments:
        return
    for segment in segments:
        if segment.is_symlink() or not segment.is_file():
            raise JournalCorruptionError(segment, 0, 0, "segment is not a regular file")
        raw = segment.read_bytes()
        offset = 0
        lines = raw.splitlines(keepends=True)
        for line_number, line in enumerate(lines, start=1):
            terminated = line.endswith((b"\n", b"\r"))
            body = line.strip()
            if not terminated:
                raise JournalCorruptionError(
                    segment,
                    line_number,
                    offset,
                    "journal record is unterminated",
                )
            if not body:
                offset += len(line)
                continue
            try:
                record = json.loads(body.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("journal record must be a JSON object")
                event = record.get("event")
                if not isinstance(event, str) or not event:
                    raise ValueError("journal event must be a non-empty string")
                _parse_timestamp(record.get("ts"), "journal event timestamp")
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                raise JournalCorruptionError(
                    segment,
                    line_number,
                    offset,
                    type(exc).__name__ + ": " + str(exc),
                ) from exc
            yield record
            offset += len(line)


def tail_events(
    journal_path: Union[str, Path],
    *,
    max_records: int = 4096,
    max_bytes: int = 2 * 1024 * 1024,
) -> List[Dict[str, Any]]:
    """Return a bounded, validated tail without replaying complete segments."""
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
        raise ValueError("journal tail max_records must be a positive integer")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("journal tail max_bytes must be a positive integer")
    current = Path(journal_path)
    remaining = int(max_bytes)
    collected: List[List[Dict[str, Any]]] = []
    for segment in reversed(_journal_segments(current)):
        if remaining <= 0:
            break
        if segment.is_symlink() or not segment.is_file():
            raise JournalCorruptionError(segment, 0, 0, "segment is not a regular file")
        size = int(segment.stat().st_size)
        read_size = min(size, remaining)
        start = size - read_size
        with open(segment, "rb") as handle:
            handle.seek(start)
            raw = handle.read(read_size)
        remaining -= len(raw)
        if start:
            separator = raw.find(b"\n")
            raw = b"" if separator < 0 else raw[separator + 1 :]
        lines = raw.splitlines(keepends=True)
        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            terminated = line.endswith((b"\n", b"\r"))
            body = line.strip()
            if not terminated:
                raise JournalCorruptionError(
                    segment,
                    line_number,
                    start,
                    "journal record is unterminated",
                )
            if not body:
                continue
            try:
                record = json.loads(body.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("journal record must be a JSON object")
                event = record.get("event")
                if not isinstance(event, str) or not event:
                    raise ValueError("journal event must be a non-empty string")
                _parse_timestamp(record.get("ts"), "journal event timestamp")
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                raise JournalCorruptionError(
                    segment,
                    line_number,
                    start,
                    type(exc).__name__ + ": " + str(exc),
                ) from exc
            records.append(record)
        collected.append(records)
        if sum(len(group) for group in collected) >= max_records:
            break
    ordered = [record for group in reversed(collected) for record in group]
    return ordered[-max_records:]


def read_events(
    journal_path: Union[str, Path],
    *,
    since: Optional[str] = None,
    event_type: Optional[Union[str, Iterable[str]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Filtered view over :func: "iter_events".

    "since" is an ISO-8601 lower bound (inclusive); events with "ts" < since
    are dropped. "event_type" may be a single string or an iterable of
    strings; events whose "event" is not in the filter are dropped.
    """
    since_dt = None if since is None else _parse_timestamp(since, "journal --since")
    if event_type is not None and isinstance(event_type, str):
        wanted = {event_type}
    elif event_type is not None:
        wanted = set(event_type)
    else:
        wanted = None
    for record in iter_events(journal_path):
        if wanted is not None and record.get("event") not in wanted:
            continue
        if since_dt is not None and _parse_timestamp(
            record.get("ts"), "journal event timestamp"
        ) < since_dt:
            continue
        yield record
