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
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import portalocker


__all__ = [
    "JOURNAL_LINE_LIMIT_BYTES",
    "EventTooLargeError",
    "JournalCorruptionError",
    "KNOWN_EVENT_TYPES",
    "append_event",
    "iter_events",
    "read_events",
]


#Documentation only -- the journal is a free-form NDJSON stream and any
#string is a valid event type. KNOWN_EVENT_TYPES enumerates the names the
#daemon and executors currently emit so operators / log consumers can build
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
#  reference_data_committed  -- {reference_data_version, n_committed_points} after APPEND
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
    "operator_cancelled_jobs",
    "operator_stop_requested",
    "operator_stop_boundary_reached",
    "operator_stop_control_invalid",
    "operator_stop_request_cancelled",
    "operator_stop_resumed",
    "sacct_unknown_timeout",
    "sacct_missing_timeout",
    "sacct_empty_but_squeue_active",
    "sacct_rows_missing_but_squeue_active",
    "squeue_liveness_inconclusive",
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
    "ariadne_seed_provenance_repaired",
    "ariadne_seed_provenance_staged",
    "ariadne_stale_outputs_quarantined",
    "ariadne_task_rejected_missing_result",
    "ariadne_task_rejected_malformed_result",
    "ariadne_task_rejected_unusable_result",
    "ariadne_task_rejected_unsafe_landing",
    "ariadne_task_salvaged_from_nonzero_exit",
    "ferebus_quality_summary",
    "geometry_novelty_scale_precomputed",
    "sampling_protocol_resolved",
    "initial_training_existing_without_bootstrap_handoff",
    "phase_b_novelty_threshold_relaxed",
    "pool_feasibility_checked",
    "quantum_quality_summary",
    "seed_posterior_fallback",
    "point_allocation_complete",
    "point_allocation_replacement_prepared",
    "point_allocation_quantum_recorded",
    "aimall_skipped_no_gaussian_acceptances",
    "ariadne_sampling_protocol_replay_failed",
    "legacy_sampling_protocol_repreview",
    "partial_array_recovery_postprocess_only",
    "partial_array_recovery_prepared",
    "active_iteration_finalised",
    "ariadne_task_rejected_invalid_output",
    "dry_run_trajectory_pool_created",
)


# Keep individual diagnostic records compact. Cross-process safety comes from
# the bounded journal lock and full-write loop, not from pipe semantics.
JOURNAL_LINE_LIMIT_BYTES = 4000
DEFAULT_JOURNAL_MAX_BYTES = 67_108_864
DEFAULT_JOURNAL_RETAINED_FILES = 8
DEFAULT_JOURNAL_LOCK_TIMEOUT_SECONDS = 30


class EventTooLargeError(ValueError):
    """Raised when a serialised event exceeds the diagnostic record limit."""


class JournalCorruptionError(ValueError):
    """Raised for malformed records other than one torn current tail."""

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
    lock_path = p.with_name(p.name + ".lock")
    with portalocker.Lock(
        str(lock_path),
        mode="a",
        timeout=float(lock_timeout_seconds),
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    ):
        current_size = p.stat().st_size if p.is_file() else 0
        if current_size and current_size + len(encoded) > int(max_bytes):
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
                if old.is_symlink() or not old.is_file():
                    raise ValueError("journal segment is not a regular file: " + str(old))
                old.unlink()
            _fsync_parent_dir(p)
        fd = os.open(
            str(p),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
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
    return archives + ([path] if path.is_file() else [])


def iter_events(journal_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Yield strict event objects from immutable segments and the current file.

    Only an unterminated final line in the current file is treated as a torn
    best-effort append. Interior or archived corruption is reported.
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
                is_torn_current_tail = (
                    segment == p
                    and line_number == len(lines)
                    and not terminated
                )
                if is_torn_current_tail:
                    return
                raise JournalCorruptionError(
                    segment,
                    line_number,
                    offset,
                    type(exc).__name__ + ": " + str(exc),
                ) from exc
            yield record
            offset += len(line)


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
