"""Append-only NDJSON journal.

The daemon writes one JSON object per line to "journal.ndjson". Each entry
is opened O_APPEND and is constrained to under PIPE_BUF bytes (4 KiB on
a Linux cluster, conservatively used as the safe limit on all platforms). POSIX
guarantees that a single write(2) of less than PIPE_BUF on an O_APPEND
file descriptor is atomic with respect to concurrent appenders, which is
the property we rely on for crash-and-concurrent-process safety.

Schema is intentionally flexible: every line carries "ts" (ISO-8601 UTC) and
"event" (short string tag); everything else is event-specific payload.
The daemon uses this for replayable per-iteration provenance and for
post-mortem analysis after a campaign halts.

The journal is append-only; on rotation (operator-driven, rare), the old
file is renamed and a fresh empty one is created. This module exposes
:func: "append_event" for the writer side and :func: "read_events" for
introspection / CLI / tests.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Union


__all__ = [
    "JOURNAL_LINE_LIMIT_BYTES",
    "EventTooLargeError",
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
#  training_set_committed  -- {training_set_version, n_committed_points} after APPEND
#  seed_selected           -- placeholder
#  anti_overlap_flagged    -- placeholder
#  reference_scales_computed -- placeholder
#  preset_loaded           -- placeholder
KNOWN_EVENT_TYPES = (
    "campaign_started",
    "phase_transition",
    "sbatch",
    "phase_succeeded",
    "sacct_error",
    "shutdown_requested",
    "daemon_started",
    "daemon_stopped",
    "daemon_interrupted",
    "state_corrupt",
    "tick_error",
    "tick_exception_halted",
    "subspace_built",
    "training_set_committed",
    "seed_selected",
    "anti_overlap_flagged",
    "reference_scales_computed",
    "preset_loaded",
    "failure_action",
    "halt",
    # new additions I:
    "live_postprocess_refused",
    "effective_config_diff",
    "autotune_applied",
    "trajectory_pool_filtered",
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
    "resolved_phase_resources",
    "reconcile_resolved_terminal_intent",
    "operator_cancelled_jobs",
    "sacct_unknown_timeout",
    "sacct_missing_timeout",
    "sacct_empty_but_squeue_active",
    "sacct_rows_missing_but_squeue_active",
    "squeue_liveness_inconclusive",
    "transient_phase_retry",
    "transient_retry_ledger_invalid",
    "job_adopt_check_failed",
    "adopted_inflight_job",
    "submission_intent_update_failed",
    "submission_intent_read_failed",
    "provenance_index_repaired",
    "provenance_index_repair_failed",
    "postprocess_settle_retry",
    "phase_pre_submit_intent",
    "phase_submitted",
    "staging_archived",
    "daemon_lease_conflict",
    "daemon_lease_stale_recovered",
    "daemon_lease_cleanup_failed",
    "ariadne_seed_provenance_repaired",
)


#Linux PIPE_BUF is 4096; POSIX requires write(2) of <= PIPE_BUF to be atomic
#on O_APPEND files. We enforce strictly less than that to leave room for the
#trailing newline and any UTF-8 multibyte overhead.
JOURNAL_LINE_LIMIT_BYTES = 4000


class EventTooLargeError(ValueError):
    """Raised when a serialised event would exceed PIPE_BUF and so cannot be
    appended atomically. Callers must split the payload."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_event(event_type: str, payload: Dict[str, Any], ts: Optional[str] = None) -> bytes:
    record = {"ts": ts or _now_iso(), "event": str(event_type)}
    for k, v in payload.items():
        if k in ("ts", "event"):
            raise ValueError("payload key " + repr(k) + " is reserved")
        record[k] = v
    encoded = json.dumps(record, sort_keys=False, default=str).encode("utf-8")
    if len(encoded) + 1 > JOURNAL_LINE_LIMIT_BYTES:
        raise EventTooLargeError(
            "event line is " + str(len(encoded) + 1) + " bytes; "
            "the journal enforces < " + str(JOURNAL_LINE_LIMIT_BYTES) + " for "
            "atomic concurrent appends. Split the payload."
        )
    return encoded + b"\n"


def append_event(
    journal_path: Union[str, Path],
    event_type: str,
    *,
    ts: Optional[str] = None,
    fsync: bool = False,
    **payload: Any,
) -> str:
    """Append a single NDJSON record to "journal_path".

    Opens the file with O_APPEND so concurrent processes do not interleave
    bytes within a write. Returns the encoded "ts" so the caller can echo it
    into log lines or unit tests.

    Setting "fsync=True" forces an fsync after the append; the default
    False matches the design note that journal durability is best-effort --
    the state file is the authoritative checkpoint.
    """
    encoded = _encode_event(event_type, payload, ts=ts)
    p = Path(journal_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        str(p),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.write(fd, encoded)
        if fsync:
            try:
                os.fsync(fd)
            except (OSError, NotImplementedError):
                pass
    finally:
        os.close(fd)
    return json.loads(encoded.decode("utf-8"))["ts"]


def iter_events(journal_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Yield each well-formed event from the journal in file order.

    Malformed lines (typically incomplete trailing writes on a crashed
    process) are silently skipped; the daemon's reconcile path inspects
    "iter_events" for analysis but treats malformed lines as informational
    rather than authoritative.
    """
    p = Path(journal_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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
    if event_type is not None and isinstance(event_type, str):
        wanted = {event_type}
    elif event_type is not None:
        wanted = set(event_type)
    else:
        wanted = None
    for record in iter_events(journal_path):
        if wanted is not None and record.get("event") not in wanted:
            continue
        if since is not None and str(record.get("ts", "")) < since:
            continue
        yield record
