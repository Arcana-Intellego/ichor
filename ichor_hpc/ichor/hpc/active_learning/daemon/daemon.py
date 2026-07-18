"""Long-running campaign daemon (login-node, single-process).

This module wires everything into one state machine. The
production deployment is a single Python process on a configured CSF login node that:

    1. Acquires an exclusive flock on .DATA/ACTIVE_LEARNING/daemon.lock.
       Refuses to start (with a clear message) if the lock is held by
       another PID.
    2. Reads or initialises state.json (the source of truth, see :mod: .state).
    3. Enters its main loop. On each tick:
         a. Inspect current phase and pending sacct JobID.
         b. Either submit the phase, poll the JobID, or run an inline action.
         c. On terminal sacct outcome: postprocess via the PhaseExecutor;
            either scrub failed tasks (failure_threshold_fraction allows it)
            or halt.
         d. Persist new state atomically (tempfile + os.replace + fsync).
         e. Sleep for config.poll_interval_seconds.
    4. Honours SIGTERM / SIGINT: finishes the current tick, marks
       shutdown_requested, writes state, releases the lock, exits 0.

Every "real" external system (sacct, sleep, time, PhaseExecutor) is injected
through the constructor so the daemon is fully testable without touching
SLURM, sleeping, or running real submissions.

The real PhaseExecutors ships :class: MockPhaseExecutor
which simulates everything in <1 ms per tick and exercises the full
state-machine progression.
"""
from __future__ import annotations

import os
import copy
import hashlib
import inspect
import secrets
from ..strict_json import strict_json as json
import signal
import socket
import sys
import time
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..config import CampaignConfig
from ..layout import trained_models_dir
from ..submit.sacct_poll import JobObservation, aggregate_states, poll_job
from .journal import append_event
from .phase_executor import (
    BackendSubmissionError,
    FailureAction,
    INLINE_PHASES,
    MockPhaseExecutor,
    PhaseExecutor,
    PhaseResult,
    SBATCH_PHASES,
)
from .reconcile import stateful_campaign_artifacts
from .state import (
    CampaignPhase,
    CampaignState,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    atomic_write_json,
    fresh_campaign_state,
    make_lifecycle_context,
    read_state,
    write_state,
)
from .filesystem import operational_data_dir
from . import submission_intent as _submission_intent
from .lease import (
    LeaseHeartbeatError,
    evaluate_lease_liveness,
    validate_lease_heartbeat,
)


__all__ = [
    "Daemon",
    "DaemonAlreadyRunningError",
    "TickStatus",
    "DEFAULT_DATA_SUBDIR",
    "PHASE_ORDER",
    "next_phase",
]


DEFAULT_DATA_SUBDIR = Path(".DATA") / "ACTIVE_LEARNING"
DAEMON_LOCK_FILENAME = "daemon.lock"
DAEMON_PID_FILENAME = "daemon.pid"
DAEMON_LEASE_DIRNAME = "daemon.lease.d"
DAEMON_HEARTBEAT_FILENAME = "heartbeat.json"
TRANSIENT_RETRY_LEDGER_FILENAME = "transient_retries.json"
JOURNAL_FILENAME = "journal.ndjson"
LAST_EXCEPTION_FILENAME = "LAST_EXCEPTION.json"

PHASE_ORDER: Tuple[CampaignPhase, ...] = (
    CampaignPhase.INIT,
    CampaignPhase.PHASE_A_DIVERSITY,
    CampaignPhase.INITIAL_GAUSSIAN,
    CampaignPhase.INITIAL_AIMALL,
    CampaignPhase.INITIAL_ALLOCATION_CHECK,
    CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
    CampaignPhase.REFERENCE_COMMIT,
    CampaignPhase.INITIAL_FEREBUS,
    CampaignPhase.SEED_SELECT,
    CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.PHASE_B_DIVERSITY,
    CampaignPhase.SPLIT,
    CampaignPhase.GAUSSIAN,
    CampaignPhase.AIMALL,
    CampaignPhase.ALLOCATION_CHECK,
    CampaignPhase.REPLACEMENT_GAUSSIAN,
    CampaignPhase.REPLACEMENT_AIMALL,
    CampaignPhase.FEREBUS,
    CampaignPhase.STOP_CHECK,
)


class DaemonAlreadyRunningError(RuntimeError):
    """Raised when the flock cannot be acquired because another daemon
    holds it."""


class DaemonLeaseOwnershipError(RuntimeError):
    """Raised when this process can no longer prove lease ownership."""


class TickStatus(str):
    SUBMITTED = "SUBMITTED"
    POLLING = "POLLING"
    ADVANCED = "ADVANCED"
    SCRUBBED = "SCRUBBED"
    HALTED = "HALTED"
    RETRYING = "RETRYING"
    TERMINAL = "TERMINAL"
    SHUTDOWN = "SHUTDOWN"


def _journal_metadata_payload(payload: Any) -> Dict[str, Any]:
    data = dict(payload) if isinstance(payload, dict) else {}
    for key in (
        "phase",
        "job_id",
        "iteration",
        "expected_tasks",
        "submitted_at_iso",
        "event",
        "ts",
    ):
        data.pop(key, None)
    return data


TRANSIENT_RETRY_STATUSES = frozenset({
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "REVOKED",
})

_STRICT_FAILURE_REQUIRES_POSTPROCESS = frozenset(SBATCH_PHASES)

_ALLOWED_PHASE_OVERRIDES = {
    CampaignPhase.INITIAL_ALLOCATION_CHECK: frozenset({
        CampaignPhase.REFERENCE_COMMIT,
        CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
    }),
    CampaignPhase.INITIAL_REPLACEMENT_AIMALL: frozenset({
        CampaignPhase.INITIAL_ALLOCATION_CHECK,
    }),
    CampaignPhase.ALLOCATION_CHECK: frozenset({
        CampaignPhase.REFERENCE_COMMIT,
        CampaignPhase.REPLACEMENT_GAUSSIAN,
    }),
    CampaignPhase.REPLACEMENT_AIMALL: frozenset({
        CampaignPhase.ALLOCATION_CHECK,
    }),
}


def next_phase(current: CampaignPhase, iteration: int, max_iterations: int) -> Tuple[CampaignPhase, int]:
    """Return (next_phase, next_iteration).

    Bootstrap phases use iteration 0. INITIAL_FEREBUS advances to active
    SEED_SELECT iteration 1. STOP_CHECK loops to the next one-based active
    iteration until max_iterations active cycles have completed.
    """
    if current is CampaignPhase.INITIAL_FEREBUS:
        if int(iteration) != 0:
            raise ValueError("INITIAL_FEREBUS requires bootstrap iteration 0")
        return CampaignPhase.SEED_SELECT, 1
    if current is CampaignPhase.STOP_CHECK:
        if int(iteration) < 1:
            raise ValueError("STOP_CHECK requires active iteration >= 1")
        next_iter = iteration + 1
        if iteration >= max_iterations:
            return CampaignPhase.DONE, iteration
        return CampaignPhase.SEED_SELECT, next_iter
    if current is CampaignPhase.INITIAL_ALLOCATION_CHECK:
        return CampaignPhase.REFERENCE_COMMIT, iteration
    if current is CampaignPhase.ALLOCATION_CHECK:
        return CampaignPhase.REFERENCE_COMMIT, iteration
    if current is CampaignPhase.REFERENCE_COMMIT:
        if int(iteration) == 0:
            return CampaignPhase.INITIAL_FEREBUS, iteration
        return CampaignPhase.FEREBUS, iteration
    if current is CampaignPhase.INITIAL_REPLACEMENT_AIMALL:
        return CampaignPhase.INITIAL_ALLOCATION_CHECK, iteration
    if current is CampaignPhase.REPLACEMENT_AIMALL:
        return CampaignPhase.ALLOCATION_CHECK, iteration
    try:
        idx = PHASE_ORDER.index(current)
    except ValueError as exc:
        raise ValueError("unexpected current phase: " + str(current)) from exc
    if idx + 1 >= len(PHASE_ORDER):
        return CampaignPhase.DONE, iteration
    return PHASE_ORDER[idx + 1], iteration


def _halt_reason_code(reason: str) -> str:
    text = str(reason)
    lower = text.lower()
    if "mandatory custom bootstrap" in lower:
        return "mandatory_custom_bootstrap_failed"
    if "point-allocation reserve exhausted" in lower:
        return "replacement_reserve_exhausted"
    prefix = text.split(":", 1)[0].strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in prefix).strip("_")
    return cleaned[:80] or "daemon_halt"


def _halt_recovery_action(reason_code: str) -> str:
    if reason_code == "mandatory_custom_bootstrap_failed":
        return (
            "inspect the failed supplied bootstrap quantum output; correct the "
            "user input or start a new campaign because mandatory custom "
            "geometries cannot be replaced"
        )
    if reason_code == "replacement_reserve_exhausted":
        return (
            "the immutable allocation cannot be repaired in place; start a new "
            "campaign with a larger pre-QM candidate reserve or a smaller required batch"
        )
    if reason_code == "ferebus_quality_failed":
        return (
            "inspect FEREBUS_QUALITY.json; adjust only justified quality "
            "thresholds and reconcile, or use reconcile --retrain-ferebus --apply"
        )
    return "inspect the recorded reason and run reconcile before restarting"


@dataclass
class Daemon:
    """Login-node campaign daemon.

    Attributes
    ----------
    campaign_dir
        Root directory of the campaign (<scratch>/<system>/campaign).
    config
        Validated CampaignConfig (typically loaded from campaign.yaml).
    executor
        PhaseExecutor implementation; defaults to MockPhaseExecutor for safety
        if not provided.
    sacct_poller
        Callable taking a JobID and returning a list of JobObservation. Defaults
        to ichor.hpc.active_learning.submit.sacct_poll.poll_job.
    sleep_fn
        Callable taking seconds (float) used between ticks. Tests pass a stub
        that records intervals instead of actually sleeping.
    """

    campaign_dir: Path
    config: CampaignConfig
    executor: PhaseExecutor = field(default_factory=MockPhaseExecutor)
    sacct_poller: Callable[..., List[JobObservation]] = poll_job
    sleep_fn: Callable[[float], None] = time.sleep
    # live-mode only: given (state, phase) returns the JobID of an already-running job for that exact
    # phase+iteration, or None. lets a restart/reconcile adopt a job a crash orphaned instead of
    # double-submitting (A24/A25). None (mock/dry) -> the check is skipped and we submit as before.
    job_finder: Optional[Callable[..., Optional[str]]] = None
    # live-mode only: given (state, phase, active_intent), return accounting
    # evidence for an expected job name. This closes the PRE_SUBMIT/no-JobID
    # crash window where sbatch accepted the job before mark_submitted wrote
    # the allocation id.
    job_name_accounting_finder: Optional[Callable[..., Any]] = None
    # live-mode only: given a Slurm JobID, returns a small object with active,
    # inconclusive, rows and error attributes. Used to distinguish genuinely
    # missing sacct array rows from throttled jobs that are still visible in
    # squeue.
    job_liveness_checker: Optional[Callable[[str], Any]] = None
    # Live-mode only advisory collector. It is injected by the CLI so mock and
    # dry-run daemons never contact Slurm for accounting telemetry.
    resource_usage_collector: Optional[Callable[..., Dict[str, Any]]] = None
    scheduler_identity_kind: Optional[str] = None

    #internal flags; not part of the public dataclass surface.
    _shutdown_requested: bool = field(default=False, init=False, repr=False)
    _lock_held: Optional[Any] = field(default=None, init=False, repr=False)
    _provenance_index_repair_attempted: bool = field(default=False, init=False, repr=False)
    _lease_owner_token: Optional[str] = field(default=None, init=False, repr=False)
    _lease_heartbeat_thread: Optional[threading.Thread] = field(
        default=None, init=False, repr=False
    )
    _lease_heartbeat_stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _lease_io_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _lease_state_snapshot: Optional[Dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    _lease_heartbeat_failures: int = field(default=0, init=False, repr=False)
    _lease_failure_message: Optional[str] = field(default=None, init=False, repr=False)
    _last_environment_binding: Optional[Dict[str, Any]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.campaign_dir = Path(self.campaign_dir)
        if self.scheduler_identity_kind is None:
            self.scheduler_identity_kind = str(
                getattr(self.executor, "scheduler_identity_kind", "synthetic")
            )
        if self.scheduler_identity_kind not in {"synthetic", "slurm"}:
            raise ValueError("scheduler_identity_kind must be synthetic or slurm")

    # --- path helpers ---------------------------------------------------

    def data_dir(self) -> Path:
        return operational_data_dir(self.campaign_dir)

    def state_path(self) -> Path:
        return self.data_dir() / DEFAULT_STATE_FILENAME

    def lock_path(self) -> Path:
        return self.data_dir() / DAEMON_LOCK_FILENAME

    def pid_path(self) -> Path:
        return self.data_dir() / DAEMON_PID_FILENAME

    def lease_path(self) -> Path:
        return self.data_dir() / DAEMON_LEASE_DIRNAME

    def heartbeat_path(self) -> Path:
        return self.lease_path() / DAEMON_HEARTBEAT_FILENAME

    def transient_retry_ledger_path(self) -> Path:
        return self.data_dir() / TRANSIENT_RETRY_LEDGER_FILENAME

    def journal_path(self) -> Path:
        return self.data_dir() / JOURNAL_FILENAME

    def last_exception_path(self) -> Path:
        return self.data_dir() / LAST_EXCEPTION_FILENAME

    # --- lock + lifecycle ----------------------------------------------

    @contextmanager
    def _acquire_lock(self):
        """Context manager that holds an exclusive non-blocking flock on
        daemon.lock for the lifetime of the daemon."""
        import portalocker

        self.data_dir().mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path()
        #portalocker.Lock accepts file paths or file handles; we want
        #fail_when_locked so a second daemon refuses to start.
        lock = portalocker.Lock(
            str(lock_file),
            mode="a+",
            timeout=0,
            fail_when_locked=True,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        )
        try:
            handle = lock.acquire()
        except (portalocker.LockException, portalocker.AlreadyLocked) as exc:
            raise DaemonAlreadyRunningError(
                "daemon.lock already held by another process; refuse to start"
            ) from exc
        self._lock_held = lock
        try:
            yield handle
        finally:
            try:
                lock.release()
            except Exception:
                pass
            self._lock_held = None

    def _read_heartbeat(self) -> Optional[Dict[str, Any]]:
        path = self.heartbeat_path()
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return validate_lease_heartbeat(payload)
        except (OSError, ValueError):
            return None

    def _lease_age_seconds(self) -> float:
        heartbeat = self._read_heartbeat()
        if heartbeat is not None:
            liveness = evaluate_lease_liveness(
                heartbeat,
                stale_seconds=int(self.config.runtime.lease_stale_seconds),
                clock_skew_tolerance_seconds=int(
                    self.config.runtime.clock_skew_tolerance_seconds
                ),
            )
            if liveness.age_seconds is not None:
                return max(0.0, float(liveness.age_seconds))
        try:
            return max(0.0, time.time() - self.lease_path().stat().st_mtime)
        except OSError:
            return 0.0

    @contextmanager
    def _acquire_lease(self):
        self.data_dir().mkdir(parents=True, exist_ok=True)
        lease = self.lease_path()
        stale_after = int(getattr(self.config.runtime, "lease_stale_seconds", 900))
        skew = int(getattr(self.config.runtime, "clock_skew_tolerance_seconds", 60))
        try:
            os.mkdir(str(lease))
        except FileExistsError:
            raw_heartbeat: Any = None
            heartbeat_error: Optional[str] = None
            try:
                raw_heartbeat = json.loads(
                    self.heartbeat_path().read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                heartbeat_error = type(exc).__name__ + ": " + str(exc)
            liveness = evaluate_lease_liveness(
                raw_heartbeat,
                stale_seconds=stale_after,
                clock_skew_tolerance_seconds=skew,
            )
            heartbeat = raw_heartbeat if isinstance(raw_heartbeat, dict) else {}
            age = (
                self._lease_age_seconds()
                if liveness.age_seconds is None
                else max(0.0, float(liveness.age_seconds))
            )
            if liveness.disposition in {"fresh", "clock_skew"} or (
                liveness.disposition == "invalid" and age < stale_after
            ):
                self._journal(
                    "daemon_lease_conflict",
                    host=str(heartbeat.get("host", "?")),
                    pid=str(heartbeat.get("pid", "?")),
                    phase=str(heartbeat.get("phase", "?")),
                    iteration=str(heartbeat.get("iteration", "?")),
                    age_seconds=float(age),
                    stale_after_seconds=float(stale_after),
                    lease_disposition=liveness.disposition,
                    lease_error=str(liveness.error or heartbeat_error or "")[:180],
                )
                raise DaemonAlreadyRunningError(
                    "daemon lease is active or inconclusive; refuse to start "
                    + "(host="
                    + str(heartbeat.get("host", "?"))
                    + ", pid="
                    + str(heartbeat.get("pid", "?"))
                    + ", phase="
                    + str(heartbeat.get("phase", "?"))
                    + ", iteration="
                    + str(heartbeat.get("iteration", "?"))
                    + ")"
                )
            stale = lease.with_name(
                lease.name + ".stale." + str(int(time.time())) + "." + str(os.getpid())
            )
            try:
                lease.rename(stale)
            except OSError as exc:
                raise DaemonAlreadyRunningError(
                    "daemon lease looked stale but could not be recovered: " + str(exc)
                ) from exc
            self._journal(
                "daemon_lease_stale_recovered",
                stale_lease=str(stale),
                previous_host=str(heartbeat.get("host", "?")),
                previous_pid=str(heartbeat.get("pid", "?")),
                previous_phase=str(heartbeat.get("phase", "?")),
                previous_iteration=str(heartbeat.get("iteration", "?")),
                age_seconds=float(age),
                stale_after_seconds=float(stale_after),
            )
            os.mkdir(str(lease))
        self._lease_owner_token = secrets.token_hex(16)
        self._lease_heartbeat_stop.clear()
        self._lease_heartbeat_failures = 0
        self._lease_failure_message = None
        self._write_lease_heartbeat(initial=True)
        self._start_lease_heartbeat_thread()
        try:
            yield
        finally:
            self._stop_lease_heartbeat_thread()
            token = self._lease_owner_token
            try:
                with self._lease_io_lock:
                    current = self._read_heartbeat()
                    validate_lease_heartbeat(
                        current,
                        expected_owner_token=token,
                    )
                    self.heartbeat_path().unlink()
                    os.rmdir(str(lease))
            except (OSError, ValueError) as exc:
                self._journal(
                    "daemon_lease_cleanup_failed",
                    path=str(lease),
                    operation="compare_and_remove_owned_lease",
                    error=type(exc).__name__ + ": " + str(exc)[:160],
                )
            finally:
                self._lease_owner_token = None
                self._lease_state_snapshot = None

    def _record_lease_heartbeat_failure(self, exc: Exception) -> None:
        self._lease_heartbeat_failures += 1
        self._lease_failure_message = type(exc).__name__ + ": " + str(exc)
        self._journal(
            "daemon_lease_heartbeat_failed",
            failure_count=int(self._lease_heartbeat_failures),
            failure_max=int(self.config.runtime.lease_heartbeat_failure_max),
            error=self._lease_failure_message[:180],
        )

    def _write_lease_heartbeat(
        self,
        state: Optional[CampaignState] = None,
        *,
        initial: bool = False,
    ) -> None:
        lease = self.lease_path()
        token = self._lease_owner_token
        if not lease.is_dir() or token is None:
            return
        if state is not None:
            self._lease_state_snapshot = {
                "phase": state.phase.value,
                "iteration": int(state.iteration),
                "campaign_uid": str(getattr(state, "campaign_uid", "")),
            }
        payload: Dict[str, Any] = {
            "schema_version": 2,
            "owner_token": token,
            "time": time.time(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        if self._lease_state_snapshot is not None:
            payload.update(dict(self._lease_state_snapshot))
        try:
            with self._lease_io_lock:
                if not initial:
                    current = self._read_heartbeat()
                    validate_lease_heartbeat(
                        current,
                        expected_owner_token=token,
                    )
                atomic_write_json(self.heartbeat_path(), payload)
            if self._lease_heartbeat_failures:
                self._journal(
                    "daemon_lease_heartbeat_recovered",
                    prior_failure_count=int(self._lease_heartbeat_failures),
                )
            self._lease_heartbeat_failures = 0
            self._lease_failure_message = None
        except Exception as exc:
            self._record_lease_heartbeat_failure(exc)
            if isinstance(exc, LeaseHeartbeatError):
                self._lease_heartbeat_failures = int(
                    self.config.runtime.lease_heartbeat_failure_max
                )
            self._assert_lease_healthy()

    def _lease_heartbeat_worker(self) -> None:
        interval = float(self.config.runtime.lease_heartbeat_seconds)
        while not self._lease_heartbeat_stop.wait(interval):
            try:
                self._write_lease_heartbeat()
            except DaemonLeaseOwnershipError:
                return

    def _start_lease_heartbeat_thread(self) -> None:
        thread = threading.Thread(
            target=self._lease_heartbeat_worker,
            name="ichor-daemon-lease-heartbeat",
            daemon=True,
        )
        self._lease_heartbeat_thread = thread
        thread.start()

    def _stop_lease_heartbeat_thread(self) -> None:
        self._lease_heartbeat_stop.set()
        thread = self._lease_heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, float(self.config.runtime.lease_heartbeat_seconds) + 1.0))
        self._lease_heartbeat_thread = None

    def _assert_lease_healthy(self) -> None:
        if self._lease_heartbeat_failures >= int(
            self.config.runtime.lease_heartbeat_failure_max
        ):
            raise DaemonLeaseOwnershipError(
                "daemon lease heartbeat failed repeatedly; scientific progression "
                "has stopped: " + str(self._lease_failure_message or "unknown failure")
            )

    def _write_pid(self) -> None:
        self.pid_path().write_text(str(os.getpid()) + "\n", encoding="utf-8")

    def _remove_pid(self) -> None:
        try:
            self.pid_path().unlink()
        except OSError:
            pass

    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._on_signal)
            signal.signal(signal.SIGTERM, self._on_signal)
        except (ValueError, AttributeError):
            # Off the main thread (e.g. tests) -- skip silently.
            pass

    def _on_signal(self, signum, frame) -> None:
        self._shutdown_requested = True

    # --- state IO -------------------------------------------------------

    def _read_or_initialise_state(self) -> CampaignState:
        from ..layout import reject_legacy_campaign_layout

        reject_legacy_campaign_layout(self.campaign_dir)
        #ensure the data dir exists before any state IO. run() also does thi
        #via _acquire_lock(), but tick() can be called directly in tests.
        self.data_dir().mkdir(parents=True, exist_ok=True)
        sp = self.state_path()
        if not sp.exists():
            artefacts = stateful_campaign_artifacts(self.campaign_dir)
            if artefacts:
                raise StateSchemaError(
                    "state.json is missing but this campaign is not empty; "
                    "refusing fresh initialisation because that could "
                    "overwrite provenance. Run `ichor-al-daemon reconcile "
                    "--campaign-dir "
                    + str(self.campaign_dir)
                    + "`. Stateful artefacts: "
                    + ", ".join(artefacts[:8])
                )
            state = fresh_campaign_state(
                max_iterations=self.config.campaign.max_iterations
            )
            from .config_lock import ensure_config_lock

            ensure_config_lock(
                self.campaign_dir,
                self.config,
                campaign_uid=str(state.campaign_uid),
            )
            write_state(sp, state)
            self._journal(
                "campaign_started",
                campaign_uid=state.campaign_uid,
                max_iterations=state.max_iterations,
            )
            self._repair_provenance_index_best_effort()
            return state
        state = read_state(sp)
        self._repair_provenance_index_best_effort()
        return state

    def _repair_provenance_index_best_effort(self) -> None:
        if self._provenance_index_repair_attempted:
            return
        self._provenance_index_repair_attempted = True
        try:
            from ..versioning.provenance import repair_index_from_committed_pointdirs

            added = repair_index_from_committed_pointdirs(
                self.campaign_dir,
                self.campaign_dir / "QM_REFERENCE_DATA",
            )
            if added:
                self._journal("provenance_index_repaired", records_added=int(added))
        except Exception as exc:
            self._journal(
                "provenance_index_repair_failed",
                error=type(exc).__name__ + ": " + str(exc)[:180],
            )

    def _journal(self, event_type: str, **payload: Any) -> None:
        try:
            append_event(
                self.journal_path(),
                event_type,
                max_bytes=int(self.config.runtime.journal_max_bytes),
                retained_files=int(self.config.runtime.journal_retained_files),
                lock_timeout_seconds=int(
                    self.config.runtime.ledger_lock_timeout_seconds
                ),
                **payload,
            )
        except Exception:
            # Journal writes are best-effort; never let logging crash the daemon.
            pass

    def _write_last_exception(self, exc: Exception) -> None:
        from datetime import datetime, timezone
        import traceback

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        try:
            state = read_state(self.state_path())
            payload["phase"] = state.phase.value
            payload["iteration"] = int(state.iteration)
            payload["pending_jobs"] = dict(state.pending_jobs)
        except Exception:
            pass
        try:
            atomic_write_json(self.last_exception_path(), payload)
        except Exception:
            pass

    # --- state-machine step --------------------------------------------

    def tick(self) -> str:
        """Run one state-machine step. Returns a TickStatus string.

        Idempotent on the state file: kill -9 between any two ticks (or any
        time before the final write_state inside this method) leaves the
        on-disk state at the previous snapshot, so the next start re-runs
        the same step.
        """
        self._assert_lease_healthy()
        state = self._read_or_initialise_state()
        self._write_lease_heartbeat(state)

        if state.shutdown_requested:
            try:
                request = self._read_stop_control(state)
                if request is not None and str(request.get("status")) != "completed":
                    from .stop_control import complete_stop_request

                    complete_stop_request(
                        self.campaign_dir,
                        str(request.get("request_id")),
                        reason="shutdown_state_recovered",
                        completion_receipt=state.last_completion_receipt,
                    )
            except Exception as exc:
                self._journal(
                    "user_stop_control_invalid",
                    phase=state.phase.value,
                    iteration=int(state.iteration),
                    error=type(exc).__name__ + ": " + str(exc)[:180],
                )
            return TickStatus.SHUTDOWN
        if state.is_terminal:
            receipt_status = self._recover_phase_completion(state)
            if receipt_status is not None:
                return receipt_status
            self._handle_stop_control_before_phase(state)
            return TickStatus.TERMINAL

        receipt_status = self._recover_phase_completion(state)
        if receipt_status is not None:
            return receipt_status

        stop_status = self._handle_stop_control_before_phase(state)
        if stop_status is not None:
            return stop_status

        phase = state.phase
        pending = state.pending_jobs.get(phase.value)

        if pending is None:
            return self._on_phase_entry(state, phase)
        return self._on_pending(state, phase, pending)

    def _read_stop_control(self, state: CampaignState) -> Optional[Dict[str, Any]]:
        from .stop_control import read_stop_request

        return read_stop_request(
            self.campaign_dir,
            expected_campaign_uid=str(state.campaign_uid),
        )

    def _phase_has_started(
        self,
        state: CampaignState,
        request: Mapping[str, Any],
    ) -> bool:
        if bool(request.get("phase_started_at_request", False)):
            return True
        phase_name = str(request.get("target_phase") or "")
        if state.pending_jobs.get(phase_name):
            return True
        try:
            intent = _submission_intent.load_intent(
                self.campaign_dir,
                phase_name,
                int(request.get("target_iteration", -1)),
                expected_campaign_uid=str(state.campaign_uid),
            )
        except Exception:
            intent = None
        if isinstance(intent, dict):
            try:
                return int(intent.get("replacement_round", 0)) == int(
                    request.get("target_replacement_round", 0)
                )
            except (TypeError, ValueError):
                return True
        return False

    def _matching_stop_boundary_receipt(
        self,
        state: CampaignState,
        request: Mapping[str, Any],
    ) -> Optional[Dict[str, str]]:
        from .completion_receipts import (
            read_completion_receipt,
            receipt_dir,
            receipt_reference,
            validate_completion_reference,
        )

        mode = str(request.get("mode") or "")
        if mode == "after_phase":
            expected_phase = str(request.get("target_phase") or "")
            expected_iteration = int(request.get("target_iteration", -1))
            expected_round: Optional[int] = int(
                request.get("target_replacement_round", 0)
            )
        elif mode == "after_iteration":
            expected_iteration = int(request.get("target_iteration", -1))
            expected_phase = (
                CampaignPhase.INITIAL_FEREBUS.value
                if expected_iteration == 0
                else CampaignPhase.STOP_CHECK.value
            )
            expected_round = None
        else:
            return None
        matches: List[Tuple[str, Path]] = []
        root = receipt_dir(self.campaign_dir)
        if not root.is_dir():
            return None
        for path in root.glob("*.json"):
            try:
                payload = read_completion_receipt(path)
                if str(payload.get("campaign_uid")) != str(state.campaign_uid):
                    continue
                if str(payload.get("phase")) != expected_phase:
                    continue
                if int(payload.get("iteration", -1)) != expected_iteration:
                    continue
                if expected_round is not None and int(
                    payload.get("replacement_round", -1)
                ) != expected_round:
                    continue
                reference = receipt_reference(self.campaign_dir, path)
                validate_completion_reference(
                    self.campaign_dir,
                    reference,
                    expected_campaign_uid=str(state.campaign_uid),
                )
                matches.append((str(payload.get("created_at_iso") or ""), path))
            except Exception:
                continue
        if not matches:
            return None
        return receipt_reference(
            self.campaign_dir,
            max(matches, key=lambda item: item[0])[1],
        )

    def _apply_cancelled_jobs_from_request(
        self,
        state: CampaignState,
        request: Mapping[str, Any],
    ) -> None:
        summary = request.get("cancellation_summary")
        if not isinstance(summary, Mapping):
            return
        cancelled_ids = {
            str(item.get("job_id"))
            for item in (summary.get("cancelled") or [])
            if isinstance(item, Mapping) and str(item.get("job_id") or "")
        }
        if not cancelled_ids:
            return
        intent_keys = {
            (str(key.get("phase")), int(key.get("iteration")))
            for item in (summary.get("cancelled") or [])
            if isinstance(item, Mapping)
            for key in (item.get("intent_keys") or [])
            if isinstance(key, Mapping)
            and key.get("phase") is not None
            and key.get("iteration") is not None
        }
        for phase_name, iteration in sorted(intent_keys):
            _submission_intent.mark_failed(
                self.campaign_dir,
                phase_name,
                iteration,
                "user_cancelled_via_stop",
            )
        for phase_name, job_id in list(state.pending_jobs.items()):
            if job_id is not None and str(job_id) in cancelled_ids:
                state.pending_jobs[phase_name] = None
                self._clear_sacct_streaks(state, str(job_id))

    def _latch_stop_request(
        self,
        state: CampaignState,
        request: Mapping[str, Any],
        *,
        reason: str,
        completion_receipt: Optional[Mapping[str, Any]] = None,
    ) -> str:
        from .stop_control import complete_stop_request, describe_stop_request

        self._apply_cancelled_jobs_from_request(state, request)
        state.shutdown_requested = True
        state.lifecycle_context = make_lifecycle_context(
            disposition="stopped",
            reason_code="user_stop_request",
            message=describe_stop_request(request, completed=True),
            from_phase=state.phase,
            iteration=int(state.iteration),
            source="daemon_stop_control",
            scheduler_uncertain=any(bool(value) for value in state.pending_jobs.values()),
            recovery_action="use resume to continue from the recorded phase",
            details={
                "request_id": str(request.get("request_id")),
                "mode": str(request.get("mode")),
                "target_phase": request.get("target_phase"),
                "target_iteration": request.get("target_iteration"),
                "target_replacement_round": request.get("target_replacement_round"),
            },
        )
        self._persist(state)
        if str(request.get("status")) == "completed":
            completed = dict(request)
        else:
            completed = complete_stop_request(
                self.campaign_dir,
                str(request.get("request_id")),
                reason=str(reason),
                completion_receipt=completion_receipt,
            )
        self._journal(
            "user_stop_boundary_reached",
            request_id=str(request.get("request_id")),
            mode=str(request.get("mode")),
            phase=state.phase.value,
            iteration=int(state.iteration),
            reason=str(reason),
            target_phase=request.get("target_phase"),
            target_iteration=request.get("target_iteration"),
            target_replacement_round=request.get("target_replacement_round"),
            completion_receipt=(
                dict(completion_receipt)
                if isinstance(completion_receipt, Mapping)
                else None
            ),
            control_updated=bool(completed is not None),
        )
        self._journal(
            "shutdown_requested",
            from_phase=state.phase.value,
            iteration=int(state.iteration),
            stop_mode=str(request.get("mode")),
            request_id=str(request.get("request_id")),
        )
        return TickStatus.SHUTDOWN

    def _handle_stop_control_before_phase(
        self,
        state: CampaignState,
    ) -> Optional[str]:
        try:
            request = self._read_stop_control(state)
        except Exception as exc:
            self._journal(
                "user_stop_control_invalid",
                phase=state.phase.value,
                iteration=int(state.iteration),
                error=type(exc).__name__ + ": " + str(exc)[:180],
            )
            if state.is_terminal:
                return None
            return self._halt_scheduler_uncertain(
                state,
                state.phase,
                "stop_control_invalid: " + type(exc).__name__ + ": " + str(exc)[:180],
            )
        if request is None:
            return None
        mode = str(request.get("mode"))
        if state.is_terminal:
            from .stop_control import complete_stop_request

            complete_stop_request(
                self.campaign_dir,
                str(request.get("request_id")),
                reason="campaign_terminal",
                completion_receipt=state.last_completion_receipt,
            )
            return None
        if mode == "immediate" and str(request.get("status")) == "cancelling":
            # The CLI records this state before calling scancel and then adds
            # the cancellation summary. Waiting here prevents the daemon from
            # racing ahead and persisting stale pending-job metadata.
            return TickStatus.POLLING
        if mode == "immediate" or str(request.get("status")) == "completed":
            return self._latch_stop_request(
                state,
                request,
                reason="immediate",
                completion_receipt=request.get("completion_receipt"),
            )
        if mode == "after_phase":
            target_key = (
                str(request.get("target_phase")),
                int(request.get("target_iteration", -1)),
                int(request.get("target_replacement_round", -1)),
            )
            current_key = (
                state.phase.value,
                int(state.iteration),
                int(state.replacement_round),
            )
            if current_key != target_key:
                boundary_receipt = self._matching_stop_boundary_receipt(
                    state,
                    request,
                )
                if boundary_receipt is not None:
                    return self._latch_stop_request(
                        state,
                        request,
                        reason="boundary_already_completed",
                        completion_receipt=boundary_receipt,
                    )
                return self._halt_scheduler_uncertain(
                    state,
                    state.phase,
                    "after_phase_target_passed_without_receipt: target="
                    + repr(target_key)
                    + " current="
                    + repr(current_key),
                )
            if not self._phase_has_started(state, request):
                return self._latch_stop_request(
                    state,
                    request,
                    reason="phase_not_started",
                )
            return None
        target_iteration = int(request.get("target_iteration", -1))
        if int(state.iteration) > target_iteration:
            boundary_receipt = self._matching_stop_boundary_receipt(
                state,
                request,
            )
            if boundary_receipt is not None:
                return self._latch_stop_request(
                    state,
                    request,
                    reason="boundary_already_completed",
                    completion_receipt=boundary_receipt,
                )
            return self._halt_scheduler_uncertain(
                state,
                state.phase,
                "after_iteration_target_passed_without_receipt: target="
                + str(target_iteration)
                + " current="
                + str(int(state.iteration)),
            )
        return None

    def _recover_phase_completion(self, state: CampaignState) -> Optional[str]:
        """Validate the latest receipt or replay one written before state advance."""
        from .completion_receipts import (
            CompletionReceiptError,
            replayable_completion_receipts,
            validate_completion_reference,
        )
        reference = getattr(state, "last_completion_receipt", None)
        if isinstance(reference, dict):
            try:
                payload = validate_completion_reference(
                    self.campaign_dir,
                    reference,
                    expected_campaign_uid=str(state.campaign_uid),
                )
            except Exception as exc:
                return self._halt(
                    state,
                    state.phase,
                    "completion_receipt_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:180],
                )
            intent = _submission_intent.load_intent(
                self.campaign_dir,
                str(payload["phase"]),
                int(payload["iteration"]),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if intent is not None and str(intent.get("status") or "") not in {
                "COMPLETED",
                "FAILED",
                "SUPERSEDED",
            }:
                try:
                    _submission_intent.mark_completed(
                        self.campaign_dir,
                        str(payload["phase"]),
                        int(payload["iteration"]),
                        completion_receipt=dict(reference),
                    )
                except Exception as exc:
                    self._journal(
                        "submission_intent_completion_deferred",
                        phase=str(payload.get("phase") or ""),
                        iteration=int(payload.get("iteration", 0)),
                        error=type(exc).__name__ + ": " + str(exc)[:180],
                        completion_receipt=dict(reference),
                    )

        try:
            matches = replayable_completion_receipts(
                self.campaign_dir,
                state,
            )
        except CompletionReceiptError as exc:
            return self._halt(
                state,
                state.phase,
                "completion_receipt_recovery_failed: " + str(exc)[:200],
            )
        if not matches:
            return None
        if len(matches) != 1:
            return self._halt(
                state,
                state.phase,
                "completion_receipt_recovery_ambiguous: found "
                + str(len(matches))
                + " receipts for the same authoritative state",
            )
        match = matches[0]
        payload = dict(match["payload"])
        recovered = CampaignState.from_dict(dict(payload["state_after"]))
        recovered.last_completion_receipt = dict(match["reference"])
        self._persist(recovered)
        self._complete_intent_after_advance(
            str(payload["phase"]),
            int(payload["iteration"]),
            recovered.last_completion_receipt,
        )
        self._journal(
            "phase_completion_replayed",
            phase=str(payload["phase"]),
            iteration=int(payload["iteration"]),
            to_phase=recovered.phase.value,
            to_iteration=int(recovered.iteration),
            completion_receipt=dict(recovered.last_completion_receipt),
        )
        return TickStatus.ADVANCED

    def _verify_committed_artifacts_if_enabled(
        self,
        state: CampaignState,
        phase: CampaignPhase,
    ) -> Optional[str]:
        if not bool(getattr(self.executor, "strict_committed_artifact_verification", False)):
            return None
        attempts = max(
            1,
            int(getattr(self.config.runtime, "postprocess_settle_attempts", 3)),
        )
        settle_seconds = max(
            0,
            int(getattr(self.config.runtime, "postprocess_settle_seconds", 10)),
        )
        for attempt in range(attempts):
            try:
                from .artifact_contracts import verify_state_referenced_artifacts
                from .artifact_snapshot import build_committed_artifact_snapshot

                snapshot = build_committed_artifact_snapshot(
                    self.campaign_dir,
                    verification_level="metadata",
                )
                verify_state_referenced_artifacts(
                    self.campaign_dir,
                    state,
                    strict_models=True,
                    verification="metadata",
                    snapshot=snapshot,
                )
                return None
            except Exception as exc:
                reason = (
                    "committed_artifact_contract_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:180]
                )
                if attempt + 1 < attempts and self._looks_like_file_settle(reason):
                    self._journal(
                        "committed_artifact_settle_retry",
                        phase=phase.value,
                        iteration=int(state.iteration),
                        attempt=int(attempt + 1),
                        reason=reason[:180],
                    )
                    self._write_lease_heartbeat(state)
                    if settle_seconds:
                        self.sleep_fn(float(settle_seconds))
                    continue
                return self._halt(state, phase, reason)
        return None

    def _verify_environment_boundary(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        *,
        boundary: str,
    ) -> Optional[str]:
        """Fail closed if an initialised campaign's execution bytes drifted."""
        from ..execution_identity import (
            ExecutionIdentityError,
            assert_environment_unchanged,
            execution_identity_path,
        )

        self._last_environment_binding = None
        identity_path = execution_identity_path(self.campaign_dir)
        if not identity_path.exists() and not identity_path.is_symlink():
            # Direct unit-level daemon construction remains supported.  The
            # public start path always creates the identity before execution.
            return None
        try:
            status = assert_environment_unchanged(
                self.campaign_dir,
                campaign_uid=str(state.campaign_uid),
                config=self.config,
            )
        except (ExecutionIdentityError, OSError, ValueError) as exc:
            return self._halt_environment_drift(
                state,
                phase,
                "environment_drift_before_"
                + str(boundary)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)[:200],
            )
        self._last_environment_binding = {
            "generation": int(status["generation"]),
            "generation_digest_sha256": str(
                status["generation_digest_sha256"]
            ),
        }
        return None

    def _verify_intent_environment_binding(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Require submitted work to remain bound to its exact generation."""
        from ..execution_identity import execution_identity_path

        if phase.value not in SBATCH_PHASES:
            return None
        identity_path = execution_identity_path(self.campaign_dir)
        if not identity_path.exists() and not identity_path.is_symlink():
            return None
        try:
            bound_intent = intent or _submission_intent.load_intent(
                self.campaign_dir,
                phase.value,
                int(state.iteration),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if self._last_environment_binding is None:
                raise ValueError("active environment binding is unavailable")
            intent_generation = bound_intent.get("environment_generation")
            intent_digest = bound_intent.get(
                "environment_generation_digest_sha256"
            )
            if intent_generation is None or intent_digest is None:
                raise ValueError("submission intent has no environment binding")
            if int(intent_generation) != int(
                self._last_environment_binding["generation"]
            ) or str(intent_digest) != str(
                self._last_environment_binding["generation_digest_sha256"]
            ):
                raise ValueError(
                    "submission intent environment generation does not match "
                    "the active generation"
                )
        except (OSError, TypeError, ValueError) as exc:
            return self._halt_environment_drift(
                state,
                phase,
                "submission_intent_environment_binding_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:200],
            )
        return None

    def _strict_artifact_checks_enabled(self) -> bool:
        return bool(getattr(self.executor, "strict_committed_artifact_verification", False))

    def _checkpoint_before_seed_selection(
        self,
        state: CampaignState,
    ) -> Optional[str]:
        """Publish the completed iteration before launching new sampling."""
        if state.phase is not CampaignPhase.SEED_SELECT:
            return None
        retention = self.config.retention
        destination = retention.checkpoint_destination
        if destination is None:
            if bool(retention.checkpoint_required):
                return self._halt(
                    state,
                    state.phase,
                    "required_checkpoint_destination_missing",
                )
            return None
        completed_iteration = int(state.iteration) - 1
        if completed_iteration < 0:
            return None
        frequency = int(retention.checkpoint_every_iterations)
        if completed_iteration != 0 and completed_iteration % frequency != 0:
            return None
        try:
            from .checkpoints import create_checkpoint

            result = create_checkpoint(
                self.campaign_dir,
                str(destination),
                iteration=completed_iteration,
                verify_after_write=bool(
                    retention.checkpoint_verify_after_write
                ),
                allow_active_lease=True,
            )
        except Exception as exc:
            reason = (
                "checkpoint_failed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:220]
            )
            self._journal(
                "checkpoint_failed",
                phase=state.phase.value,
                iteration=int(state.iteration),
                completed_iteration=completed_iteration,
                required=bool(retention.checkpoint_required),
                error=reason,
            )
            if bool(retention.checkpoint_required):
                return self._halt(state, state.phase, reason)
            return None
        self._journal(
            "checkpoint_verified",
            phase=state.phase.value,
            iteration=int(state.iteration),
            completed_iteration=completed_iteration,
            checkpoint=str(result.get("checkpoint") or ""),
            manifest_sha256=str(result.get("manifest_sha256") or ""),
        )
        return None

    def _on_phase_entry(self, state: CampaignState, phase: CampaignPhase) -> str:
        """Called once when entering a phase with no pending JobID."""
        phase_name = phase.value
        environment_status = self._verify_environment_boundary(
            state,
            phase,
            boundary="submission",
        )
        if environment_status is not None:
            return environment_status
        verify_status = self._verify_committed_artifacts_if_enabled(state, phase)
        if verify_status is not None:
            return verify_status
        checkpoint_status = self._checkpoint_before_seed_selection(state)
        if checkpoint_status is not None:
            return checkpoint_status
        active_intent = None
        if phase_name in SBATCH_PHASES:
            try:
                active_intent = _submission_intent.load_active_intent(
                    self.campaign_dir, phase_name, int(state.iteration),
                    expected_campaign_uid=str(state.campaign_uid),
                )
            except Exception as exc:
                return self._halt(
                    state,
                    phase,
                    "submission_intent_invalid: "
                    + type(exc).__name__ + ": " + str(exc)[:160],
                )
        if active_intent is not None:
            intent_environment_status = self._verify_intent_environment_binding(
                state,
                phase,
                active_intent,
            )
            if intent_environment_status is not None:
                return intent_environment_status
        # before submitting, check whether a job for THIS phase+iteration is already running on the
        # cluster. a crash in the submit->persist window just below, or a reconcile that cleared
        # pending_jobs, can leave a real job running that state.json has forgotten -- resubmitting
        # would then race a duplicate into the same staging dirs. if we find it, adopt + poll it
        # instead. only for sbatch phases (inline phases have no job) and only when a finder is wired
        # (live mode); mock/dry leave it None and submit as before. (A24/A25)
        if self.job_finder is not None and phase_name in SBATCH_PHASES:
            lookup_inconclusive = False
            lookup_rows = []
            try:
                lookup = self.job_finder(state, phase)
                existing = getattr(lookup, "job_id", lookup)
                lookup_inconclusive = bool(getattr(lookup, "inconclusive", False))
                lookup_rows = list(getattr(lookup, "rows", []) or [])
            except Exception as exc:
                self._journal(
                    "job_adopt_check_failed", phase=phase_name, error=str(exc)[:200],
                )
                if active_intent is not None:
                    return self._halt_scheduler_uncertain(
                        state,
                        phase,
                        "active_submission_adoption_check_failed: "
                        + type(exc).__name__ + ": " + str(exc)[:160],
                    )
                existing = None
                lookup_inconclusive = True
            if existing:
                existing_job_id = str(existing)
                expected_tasks = self._expected_tasks_for_adoption(
                    active_intent,
                    state,
                    phase,
                )
                if expected_tasks is None and self._strict_artifact_checks_enabled():
                    return self._halt_scheduler_uncertain(
                        state,
                        phase,
                        "active_submission_expected_tasks_unavailable: "
                        + phase_name
                        + "@"
                        + str(int(state.iteration)),
                    )
                state.pending_jobs[phase_name] = existing_job_id
                try:
                    _submission_intent.mark_adopted(
                        self.campaign_dir,
                        phase_name,
                        int(state.iteration),
                        existing_job_id,
                        expected_tasks=expected_tasks,
                    )
                except Exception as exc:
                    self._journal(
                        "submission_intent_update_failed",
                        phase=phase_name,
                        iteration=int(state.iteration),
                        error=str(exc)[:200],
                    )
                self._persist(state)
                self._journal(
                    "adopted_inflight_job", phase=phase_name,
                    job_id=existing_job_id, iteration=state.iteration,
                    n_matching_sacct_rows=len(lookup_rows),
                    matching_sacct_rows_sample=lookup_rows[:8],
                    matching_sacct_rows_truncated=bool(len(lookup_rows) > 8),
                    expected_tasks=expected_tasks,
                )
                return TickStatus.SUBMITTED
            if active_intent is not None and lookup_inconclusive:
                return self._halt_scheduler_uncertain(
                    state,
                    phase,
                    "active_submission_adoption_inconclusive: "
                    "sacct lookup failed or was inconclusive; refusing to supersede active intent",
                )
        if active_intent is not None and phase_name in SBATCH_PHASES:
            active_job_id = active_intent.get("job_id")
            if active_job_id is None:
                recovered = self._recover_pre_submit_intent_without_job_id(
                    state,
                    phase,
                    active_intent,
                )
                if recovered is not None:
                    return recovered
            if active_job_id is not None and self.job_liveness_checker is not None:
                liveness = self._check_job_liveness(str(active_job_id))
                if liveness is not None and bool(getattr(liveness, "active", False)):
                    expected_tasks = self._expected_tasks_for_adoption(
                        active_intent,
                        state,
                        phase,
                    )
                    if expected_tasks is None and self._strict_artifact_checks_enabled():
                        return self._halt_scheduler_uncertain(
                            state,
                            phase,
                            "active_submission_expected_tasks_unavailable: "
                            + phase_name
                            + "@"
                            + str(int(state.iteration)),
                        )
                    state.pending_jobs[phase_name] = str(active_job_id)
                    try:
                        _submission_intent.mark_adopted(
                            self.campaign_dir,
                            phase_name,
                            int(state.iteration),
                            str(active_job_id),
                            expected_tasks=expected_tasks,
                        )
                    except Exception as exc:
                        self._journal(
                            "submission_intent_update_failed",
                            phase=phase_name,
                            iteration=int(state.iteration),
                            error=str(exc)[:200],
                        )
                    self._persist(state)
                    self._journal(
                        "adopted_inflight_job",
                        phase=phase_name,
                        job_id=str(active_job_id),
                        iteration=state.iteration,
                        n_matching_sacct_rows=0,
                        matching_sacct_rows_sample=[],
                        matching_sacct_rows_truncated=False,
                        expected_tasks=expected_tasks,
                        squeue_rows_sample=self._queue_rows_sample(liveness),
                        squeue_state_counts=self._queue_state_counts(liveness),
                    )
                    return TickStatus.SUBMITTED
                if liveness is not None and bool(getattr(liveness, "inconclusive", False)):
                    return self._halt_scheduler_uncertain(
                        state,
                        phase,
                        "active_submission_liveness_blocks_resubmit: "
                        + str(active_job_id)
                    )
                accounted = self._adopt_accounted_intent_job(
                    state,
                    phase,
                    active_intent,
                    str(active_job_id),
                )
                if accounted is not None:
                    return accounted
            if active_job_id is not None and self.job_liveness_checker is None:
                accounted = self._adopt_accounted_intent_job(
                    state,
                    phase,
                    active_intent,
                    str(active_job_id),
                )
                if accounted is not None:
                    return accounted
                return self._halt_scheduler_uncertain(
                    state,
                    phase,
                    "active_submission_liveness_unavailable: "
                    + str(active_job_id)
                    + "; refusing to supersede without scheduler evidence",
                )
            try:
                _submission_intent.mark_superseded(
                    self.campaign_dir,
                    phase_name,
                    int(state.iteration),
                    "no_inflight_job_found_before_resubmit",
                )
            except Exception as exc:
                return self._halt(
                    state,
                    phase,
                    "submission_intent_supersede_failed: "
                    + type(exc).__name__ + ": " + str(exc)[:160],
                )
        intent_written = False
        if phase_name in SBATCH_PHASES:
            try:
                planned_expected_tasks = self._infer_expected_tasks_from_artifacts(
                    state,
                    phase,
                )
                _submission_intent.write_pre_submit_intent(
                    self.campaign_dir,
                    campaign_uid=str(getattr(state, "campaign_uid", "")),
                    phase_name=phase_name,
                    iteration=int(state.iteration),
                    replacement_round=int(
                        getattr(state, "replacement_round", 0)
                    ),
                    expected_tasks=planned_expected_tasks,
                    decision_contract=self._submission_decision_contract(),
                    scheduler_identity_kind=self.scheduler_identity_kind,
                    environment_generation=(
                        None
                        if self._last_environment_binding is None
                        else int(self._last_environment_binding["generation"])
                    ),
                    environment_generation_digest_sha256=(
                        None
                        if self._last_environment_binding is None
                        else str(
                            self._last_environment_binding[
                                "generation_digest_sha256"
                            ]
                        )
                    ),
                )
                intent_written = True
            except Exception as exc:
                return self._halt(
                    state,
                    phase,
                    "submission_intent_write_failed: "
                    + type(exc).__name__ + ": " + str(exc)[:160],
                )
        try:
            result = self.executor.submit_or_run(state, phase)
            result.validate(stage="submit", phase_name=phase_name)
        except BackendSubmissionError as exc:
            #a backend submission (sbatch) failed outright. halt cleanly so an
            # user can look, instead of letting it bubble up and take the
            #whole daemon down mid-campaign.
            if intent_written:
                try:
                    _submission_intent.mark_failed(
                        self.campaign_dir, phase_name, int(state.iteration), str(exc)[:200],
                    )
                except Exception:
                    pass
            return self._halt(
                state, phase, "backend_submission_failed: " + str(exc)[:200]
            )
        except Exception as exc:
            if intent_written:
                try:
                    _submission_intent.mark_failed(
                        self.campaign_dir,
                        phase_name,
                        int(state.iteration),
                        type(exc).__name__ + ": " + str(exc)[:180],
                    )
                except Exception:
                    pass
            return self._halt(
                state,
                phase,
                "phase_entry_exception: "
                + type(exc).__name__ + ": " + str(exc)[:180],
            )
        if result.submitted_job_id and not result.is_complete:
            if phase_name in SBATCH_PHASES:
                submitted_intent = None
                try:
                    submitted_intent = _submission_intent.mark_submitted(
                        self.campaign_dir,
                        phase_name,
                        int(state.iteration),
                        result.submitted_job_id,
                        expected_tasks=result.expected_tasks,
                        submission_metadata=getattr(
                            result,
                            "submission_metadata",
                            None,
                        ),
                    )
                except Exception as exc:
                    self._journal(
                        "submission_intent_update_failed",
                        phase=phase_name,
                        iteration=int(state.iteration),
                        error=str(exc)[:200],
                    )
            state.pending_jobs[phase_name] = result.submitted_job_id
            self._apply_state_updates(state, result.state_updates)
            #persist first, journal second. journal is best-effort;
            #state is authoritative. If _persist raises (disk full / Lustre
            #EIO), the journal entry would otherwise lie about a sbatch that
            #never made it into state.json.
            self._persist(state)
            self._journal(
                "sbatch", phase=phase_name, job_id=result.submitted_job_id,
                iteration=state.iteration,
                expected_tasks=result.expected_tasks,
                **_journal_metadata_payload(
                    getattr(result, "submission_metadata", {}) or {}
                ),
                submitted_at_iso=(
                    submitted_intent.get("submitted_at_iso")
                    if isinstance(submitted_intent, dict)
                    else None
                ),
            )
            return TickStatus.SUBMITTED
        if result.is_complete:
            # Inline phase, postprocess-only recovery, or an executor that
            # decided no submission was needed.  This path must honour the
            # same failure contract as ordinary terminal-job postprocessing.
            if result.failure_reason:
                if intent_written and phase_name in SBATCH_PHASES:
                    try:
                        _submission_intent.mark_failed(
                            self.campaign_dir,
                            phase_name,
                            int(state.iteration),
                            str(result.failure_reason)[:200],
                        )
                    except Exception:
                        pass
                return self._halt(state, phase, str(result.failure_reason))
            completed_iteration = int(state.iteration)
            advanced = self._advance(
                state,
                phase,
                result.state_updates,
                next_phase_override=result.next_phase_override,
            )
            if not advanced:
                return TickStatus.HALTED
            if intent_written and phase_name in SBATCH_PHASES:
                self._complete_intent_after_advance(
                    phase_name,
                    completed_iteration,
                    state.last_completion_receipt,
                )
            return TickStatus.ADVANCED
        #defensive: executor returned neither a JobID nor completion.
        raise RuntimeError(
            "executor returned no submitted_job_id and is_complete=False for "
            + phase.value
        )

    def _poll_sacct(
        self,
        job_id: str,
        *,
        expected_task_count: Optional[int] = None,
    ) -> Sequence[JobObservation]:
        """Poll sacct without masking a TypeError raised inside the poller."""
        try:
            signature = inspect.signature(self.sacct_poller)
        except (TypeError, ValueError):
            accepts_expected = True
            accepts_timeout = True
        else:
            has_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            accepts_expected = (
                "expected_task_count" in signature.parameters
                or has_kwargs
            )
            accepts_timeout = (
                "timeout_seconds" in signature.parameters
                or has_kwargs
            )
        kwargs: Dict[str, Any] = {}
        if accepts_expected:
            kwargs["expected_task_count"] = expected_task_count
        if accepts_timeout:
            kwargs["timeout_seconds"] = int(
                self.config.runtime.scheduler_command_timeout_seconds
            )
        return self.sacct_poller(str(job_id), **kwargs)

    def _adopt_accounted_intent_job(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        active_intent: Dict[str, Any],
        job_id: str,
    ) -> Optional[str]:
        phase_name = phase.value
        expected_tasks = self._expected_tasks_for_adoption(
            active_intent,
            state,
            phase,
        )
        if expected_tasks is None and self._strict_artifact_checks_enabled():
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "active_submission_expected_tasks_unavailable: "
                + phase_name
                + "@"
                + str(int(state.iteration)),
            )
        try:
            observations = self._poll_sacct(
                str(job_id),
                expected_task_count=expected_tasks,
            )
        except RuntimeError as exc:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "active_submission_accounting_lookup_failed: "
                + str(job_id)
                + ": "
                + str(exc)[:160],
            )
        summary = aggregate_states(
            str(job_id),
            observations,
            expected_task_count=expected_tasks,
            submission_kind=str(active_intent.get("submission_kind")),
            strict_parent_job_id=(self.scheduler_identity_kind == "slurm"),
        )
        if not observations or int(getattr(summary, "n_tasks", 0)) <= 0:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "active_submission_accounting_inconclusive: "
                + str(job_id)
                + " has no conclusive sacct rows; refusing to supersede active intent",
            )
        if int(getattr(summary, "n_unknown", 0)) > 0:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "active_submission_accounting_unknown: "
                + str(job_id)
                + " has UNKNOWN sacct rows; refusing to supersede active intent",
            )
        state.pending_jobs[phase_name] = str(job_id)
        try:
            _submission_intent.mark_adopted(
                self.campaign_dir,
                phase_name,
                int(state.iteration),
                str(job_id),
                expected_tasks=expected_tasks,
            )
        except Exception as exc:
            self._journal(
                "submission_intent_update_failed",
                phase=phase_name,
                iteration=int(state.iteration),
                error=str(exc)[:200],
            )
        self._persist(state)
        self._journal(
            "adopted_accounted_job",
            phase=phase_name,
            job_id=str(job_id),
            iteration=state.iteration,
            expected_tasks=expected_tasks,
            n_observed=int(getattr(summary, "n_observed", 0)),
            n_expected=getattr(summary, "n_expected", None),
            n_missing=int(getattr(summary, "n_missing", 0)),
            n_pending_or_running=int(getattr(summary, "n_pending_or_running", 0)),
            n_completed=int(getattr(summary, "n_completed", 0)),
            n_failed=int(getattr(summary, "n_failed", 0)),
            terminal=bool(getattr(summary, "is_terminal", False)),
        )
        return TickStatus.SUBMITTED

    def _recover_pre_submit_intent_without_job_id(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        active_intent: Dict[str, Any],
    ) -> Optional[TickStatus]:
        phase_name = phase.value
        expected_name = str(active_intent.get("expected_job_name") or "")
        if not expected_name:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "pre_submit_intent_missing_expected_job_name: "
                + phase_name
                + "@"
                + str(int(state.iteration)),
            )
        if self.job_name_accounting_finder is None:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "pre_submit_no_job_id_accounting_unavailable: "
                + expected_name
                + "; refusing to supersede active intent",
            )
        try:
            lookup = self.job_name_accounting_finder(state, phase, active_intent)
        except Exception as exc:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "pre_submit_no_job_id_accounting_lookup_failed: "
                + expected_name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)[:160],
            )
        if lookup is None:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "pre_submit_no_job_id_accounting_inconclusive: "
                + expected_name
                + " produced no lookup result; refusing to resubmit",
            )
        if bool(getattr(lookup, "inconclusive", False)):
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "pre_submit_no_job_id_accounting_inconclusive: "
                + expected_name
                + ": "
                + str(getattr(lookup, "error", "") or "inconclusive"),
            )
        job_id = getattr(lookup, "job_id", None)
        if not job_id:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "pre_submit_no_job_id_no_accounted_job: "
                + expected_name
                + "; refusing to supersede without user review",
            )
        expected_tasks = self._expected_tasks_for_adoption(
            active_intent,
            state,
            phase,
        )
        if expected_tasks is None and self._strict_artifact_checks_enabled():
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "active_submission_expected_tasks_unavailable: "
                + phase_name
                + "@"
                + str(int(state.iteration)),
            )
        recovered_job_id = str(job_id)
        state.pending_jobs[phase_name] = recovered_job_id
        try:
            _submission_intent.mark_adopted(
                self.campaign_dir,
                phase_name,
                int(state.iteration),
                recovered_job_id,
                expected_tasks=expected_tasks,
            )
        except Exception as exc:
            self._journal(
                "submission_intent_update_failed",
                phase=phase_name,
                iteration=int(state.iteration),
                error=str(exc)[:200],
            )
        self._persist(state)
        rows = list(getattr(lookup, "rows", []) or [])
        self._journal(
            "pre_submit_intent_terminal_adopted"
            if bool(getattr(lookup, "terminal", False))
            else "pre_submit_intent_recovered_by_job_name",
            phase=phase_name,
            job_id=recovered_job_id,
            iteration=state.iteration,
            expected_job_name=expected_name,
            expected_tasks=expected_tasks,
            terminal=bool(getattr(lookup, "terminal", False)),
            successful=bool(getattr(lookup, "successful", False)),
            failed=bool(getattr(lookup, "failed", False)),
            n_matching_sacct_rows=len(rows),
            matching_sacct_rows_sample=rows[:8],
            matching_sacct_rows_truncated=bool(len(rows) > 8),
        )
        return TickStatus.SUBMITTED

    def _on_pending(self, state: CampaignState, phase: CampaignPhase, job_id: str) -> str:
        """Called while a SLURM job for `phase` is in flight."""
        expected_tasks = self._expected_tasks_for_pending(state, phase, job_id)
        if expected_tasks is None and self._strict_artifact_checks_enabled():
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "expected_task_count_unavailable_for_active_job: "
                + phase.value
                + " job_id="
                + str(job_id),
            )
        try:
            observations = self._poll_sacct(
                job_id,
                expected_task_count=expected_tasks,
            )
        except RuntimeError as exc:
            error_key = str(job_id) + ":ERROR"
            current = int(state.sacct_empty_streak.get(error_key, 0)) + 1
            state.sacct_empty_streak[error_key] = current
            self._persist(state)
            liveness = self._check_job_liveness(job_id)
            max_errors = int(
                getattr(self.config.runtime, "poll_sacct_error_max_ticks", 10)
            )
            self._journal(
                "sacct_error", phase=phase.value, job_id=job_id,
                error=str(exc)[:200],
                streak=int(current),
                max_ticks=int(max_errors),
                squeue_active=(
                    None if liveness is None else bool(getattr(liveness, "active", False))
                ),
                squeue_inconclusive=(
                    None if liveness is None else bool(getattr(liveness, "inconclusive", False))
                ),
            )
            if current >= max_errors:
                self._journal(
                    "sacct_error_timeout",
                    phase=phase.value,
                    iteration=int(state.iteration),
                    job_id=str(job_id),
                    streak=int(current),
                    max_ticks=int(max_errors),
                )
                return self._halt_scheduler_uncertain(
                    state,
                    phase,
                    "sacct_error_timeout: "
                    + str(current)
                    + "/"
                    + str(max_errors)
                    + " accounting calls failed for job_id="
                    + str(job_id),
                )
            return TickStatus.POLLING
        error_key = str(job_id) + ":ERROR"
        if error_key in state.sacct_empty_streak:
            state.sacct_empty_streak.pop(error_key, None)
            self._persist(state)
        summary = aggregate_states(
            job_id,
            observations,
            expected_task_count=expected_tasks,
            submission_kind=_submission_intent.submission_kind_for_phase(
                phase.value
            ),
            strict_parent_job_id=(self.scheduler_identity_kind == "slurm"),
        )
        if observations:
            first_status = getattr(observations[0].status, "value", observations[0].status)
            self._record_queue_lifecycle(
                state,
                phase,
                job_id,
                "first_sacct",
                status=str(first_status),
                n_expected=getattr(summary, "n_expected", None),
                n_observed=int(getattr(summary, "n_observed", 0)),
                n_missing=int(getattr(summary, "n_missing", 0)),
            )
        #Detect empty sacct response BEFORE checking is_terminal --
        #an empty observations list is the "accounting aged out" signature
        #we want to escalate after a streak. is_terminal is False for n=0
        #so without this gate the streak path is unreachable.
        if not observations or summary.n_tasks == 0:
            current = state.sacct_empty_streak.get(job_id, 0) + 1
            state.sacct_empty_streak[job_id] = current
            self._persist(state)
            liveness = self._check_job_liveness(job_id)
            if self._liveness_blocks_accounting_timeout(liveness):
                self._journal_sparse_accounting_liveness(
                    phase=phase,
                    job_id=job_id,
                    streak=current,
                    liveness=liveness,
                    kind="empty",
                    summary=summary,
                    iteration=state.iteration,
                )
                return TickStatus.POLLING
            if liveness is not None and bool(getattr(liveness, "inconclusive", False)):
                return self._handle_squeue_inconclusive_liveness(
                    state,
                    phase,
                    job_id,
                    kind="empty",
                    liveness=liveness,
                    summary=summary,
                    accounting_streak=current,
                )
            max_ticks = int(
                getattr(self.config.runtime, "poll_sacct_empty_max_ticks", 10)
            )
            if max_ticks > 0 and current >= max_ticks:
                self._journal(
                    "sacct_empty_timeout",
                    phase=phase.value, job_id=job_id,
                    streak=int(current), max_ticks=int(max_ticks),
                    squeue_active=(
                        None if liveness is None
                        else bool(getattr(liveness, "active", False))
                    ),
                    squeue_inconclusive=(
                        None if liveness is None
                        else bool(getattr(liveness, "inconclusive", False))
                    ),
                    iteration=state.iteration,
                )
                return self._halt_scheduler_uncertain(
                    state,
                    phase,
                    "sacct_empty_timeout: no conclusive accounting evidence for "
                    + str(job_id),
                )
            return TickStatus.POLLING

        #Non-empty response - clear the streak.
        if job_id in state.sacct_empty_streak:
            state.sacct_empty_streak.pop(job_id, None)
            self._persist(state)

        unknown_key = job_id + ":UNKNOWN"
        if int(getattr(summary, "n_unknown", 0)) > 0:
            current = state.sacct_empty_streak.get(unknown_key, 0) + 1
            state.sacct_empty_streak[unknown_key] = current
            self._persist(state)
            max_unknown = int(
                getattr(self.config.runtime, "poll_sacct_unknown_max_ticks", 3)
            )
            if max_unknown > 0 and current >= max_unknown:
                liveness = self._check_job_liveness(job_id)
                self._journal(
                    "sacct_unknown_timeout",
                    phase=phase.value,
                    job_id=job_id,
                    streak=int(current),
                    max_ticks=int(max_unknown),
                    iteration=state.iteration,
                    squeue_active=(
                        None if liveness is None
                        else bool(getattr(liveness, "active", False))
                    ),
                    squeue_inconclusive=(
                        None if liveness is None
                        else bool(getattr(liveness, "inconclusive", False))
                    ),
                )
                return self._halt_scheduler_uncertain(
                    state,
                    phase,
                    "sacct_unknown_timeout: "
                    + str(current)
                    + "/"
                    + str(max_unknown)
                    + " ticks contained UNKNOWN scheduler states",
                )
            return TickStatus.POLLING
        if unknown_key in state.sacct_empty_streak:
            state.sacct_empty_streak.pop(unknown_key, None)
            self._persist(state)

        missing_key = job_id + ":MISSING"
        if int(getattr(summary, "n_missing", 0)) > 0:
            current = state.sacct_empty_streak.get(missing_key, 0) + 1
            state.sacct_empty_streak[missing_key] = current
            self._persist(state)
            liveness = self._check_job_liveness(job_id)
            if self._liveness_blocks_accounting_timeout(liveness):
                self._journal_sparse_accounting_liveness(
                    phase=phase,
                    job_id=job_id,
                    streak=current,
                    liveness=liveness,
                    kind="missing",
                    summary=summary,
                    iteration=state.iteration,
                )
                return TickStatus.POLLING
            if liveness is not None and bool(getattr(liveness, "inconclusive", False)):
                return self._handle_squeue_inconclusive_liveness(
                    state,
                    phase,
                    job_id,
                    kind="missing",
                    liveness=liveness,
                    summary=summary,
                    accounting_streak=current,
                )
            max_missing = int(
                getattr(self.config.runtime, "poll_sacct_missing_max_ticks", 3)
            )
            if max_missing > 0 and current >= max_missing:
                self._journal(
                    "sacct_missing_timeout",
                    phase=phase.value,
                    job_id=job_id,
                    streak=int(current),
                    max_ticks=int(max_missing),
                    n_expected=getattr(summary, "n_expected", None),
                    n_observed=int(getattr(summary, "n_observed", 0)),
                    n_missing=int(getattr(summary, "n_missing", 0)),
                    squeue_active=(
                        None if liveness is None
                        else bool(getattr(liveness, "active", False))
                    ),
                    squeue_inconclusive=(
                        None if liveness is None
                        else bool(getattr(liveness, "inconclusive", False))
                    ),
                    iteration=state.iteration,
                )
                return self._halt_scheduler_uncertain(
                    state,
                    phase,
                    "sacct_missing_timeout: "
                    + str(current)
                    + "/"
                    + str(max_missing)
                    + " ticks missed "
                    + str(int(getattr(summary, "n_missing", 0)))
                    + " expected Slurm array task rows",
                )
            return TickStatus.POLLING
        if missing_key in state.sacct_empty_streak:
            state.sacct_empty_streak.pop(missing_key, None)
            self._persist(state)

        if not summary.is_terminal:
            return TickStatus.POLLING

        self._record_queue_lifecycle(
            state,
            phase,
            job_id,
            "terminal",
            status=(
                "COMPLETED" if bool(summary.is_fully_successful) else "FAILED"
            ),
            n_expected=getattr(summary, "n_expected", None),
            n_observed=int(getattr(summary, "n_observed", 0)),
            n_missing=int(getattr(summary, "n_missing", 0)),
        )
        self._collect_terminal_resource_usage(state, phase, job_id)

        #Job has reached terminal state(s). Decide between postprocess and
        #failure handling based on the success ratio.
        if phase == CampaignPhase.ARIADNE_ARRAY:
            return self._postprocess(state, phase, observations, summary)

        if summary.is_fully_successful:
            return self._postprocess(state, phase, observations, summary)

        success_ratio = summary.n_completed / summary.n_tasks
        try:
            failure_threshold_fraction = (
                _submission_intent.snapshotted_failure_threshold_fraction(
                    self.campaign_dir,
                    phase.value,
                    int(state.iteration),
                    expected_campaign_uid=str(state.campaign_uid),
                )
            )
        except Exception as exc:
            return self._halt(
                state,
                phase,
                "submission_decision_contract_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180],
            )
        failure_threshold = 1.0 - failure_threshold_fraction

        if success_ratio >= failure_threshold:
            return self._postprocess(state, phase, observations, summary)
        return self._handle_failure(state, phase, observations, summary)

    def _submission_decision_contract(self) -> Dict[str, Any]:
        """Snapshot decision controls that must not change in flight."""
        from .config_lock import canonical_config, config_fingerprint

        return {
            "failure_threshold_fraction": float(
                self.config.runtime.failure_threshold_fraction
            ),
            "config_sha256": config_fingerprint(canonical_config(self.config)),
        }

    def _collect_terminal_resource_usage(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        job_id: str,
    ) -> None:
        if not bool(getattr(self.config.resources, "scheduler_usage_telemetry", True)):
            return
        collector = self.resource_usage_collector
        if collector is None:
            return
        try:
            intent = _submission_intent.load_intent(
                self.campaign_dir,
                phase.value,
                int(state.iteration),
                expected_campaign_uid=str(state.campaign_uid),
            )
            if not isinstance(intent, dict):
                raise ValueError("submission intent is unavailable")
            if str(intent.get("job_id") or "") != str(job_id):
                raise ValueError("submission intent JobID does not match terminal job")
            collector_kwargs: Dict[str, Any] = {
                "intent": intent,
                "history_limit": int(
                    getattr(
                        self.config.resources,
                        "scheduler_usage_history_limit",
                        5000,
                    )
                ),
            }
            try:
                collector_signature = inspect.signature(collector)
            except (TypeError, ValueError):
                accepts_timeout = True
            else:
                accepts_timeout = (
                    "timeout_seconds" in collector_signature.parameters
                    or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in collector_signature.parameters.values()
                    )
                )
            if accepts_timeout:
                collector_kwargs["timeout_seconds"] = int(
                    self.config.runtime.scheduler_command_timeout_seconds
                )
            summary = collector(self.campaign_dir, **collector_kwargs)
            self._journal(
                "scheduler_usage_recorded",
                phase=phase.value,
                iteration=int(state.iteration),
                job_id=str(job_id),
                attempt_id=str(intent.get("attempt_id") or ""),
                n_rows=int(summary.get("n_rows", 0)),
                p95_rss_mib=summary.get("p95_rss_mib"),
                p95_elapsed_seconds=summary.get("p95_elapsed_seconds"),
            )
        except Exception as exc:
            self._journal(
                "scheduler_usage_warning",
                phase=phase.value,
                iteration=int(state.iteration),
                job_id=str(job_id),
                error=type(exc).__name__ + ": " + str(exc)[:300],
            )

    def _expected_tasks_for_pending(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        job_id: str,
    ) -> Optional[int]:
        if phase.value not in SBATCH_PHASES:
            return None
        try:
            data = _submission_intent.load_intent(
                self.campaign_dir,
                phase.value,
                int(state.iteration),
                expected_campaign_uid=str(state.campaign_uid),
            )
        except Exception as exc:
            self._journal(
                "submission_intent_read_failed",
                phase=phase.value,
                iteration=int(state.iteration),
                error=str(exc)[:200],
            )
            return None
        if not isinstance(data, dict):
            return self._infer_expected_tasks_from_artifacts(state, phase)
        recorded_job = data.get("job_id")
        if recorded_job is not None and str(recorded_job) != str(job_id):
            return None
        raw = data.get("expected_tasks")
        if raw is None:
            return self._infer_expected_tasks_from_artifacts(state, phase)
        try:
            expected = int(raw)
        except (TypeError, ValueError):
            self._journal(
                "submission_intent_expected_tasks_invalid",
                phase=phase.value,
                iteration=int(state.iteration),
                value=repr(raw),
            )
            return self._infer_expected_tasks_from_artifacts(state, phase)
        return expected if expected > 0 else self._infer_expected_tasks_from_artifacts(state, phase)

    def _expected_tasks_for_adoption(
        self,
        active_intent: Optional[Dict[str, Any]],
        state: CampaignState,
        phase: CampaignPhase,
    ) -> Optional[int]:
        if isinstance(active_intent, dict):
            raw = active_intent.get("expected_tasks")
            if raw is not None:
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    value = 0
                if value > 0:
                    return value
        return self._infer_expected_tasks_from_artifacts(state, phase)

    def _infer_expected_tasks_from_artifacts(
        self,
        state: CampaignState,
        phase: CampaignPhase,
    ) -> Optional[int]:
        from .scheduler_contracts import infer_expected_tasks_from_artifacts

        phase_name = phase.value

        def record_error(exc: Exception) -> None:
            self._journal(
                "expected_tasks_inference_failed",
                phase=phase_name,
                iteration=int(state.iteration),
                error=type(exc).__name__ + ": " + str(exc)[:160],
            )

        return infer_expected_tasks_from_artifacts(
            self.campaign_dir,
            phase=phase,
            iteration=int(state.iteration),
            replacement_round=int(getattr(state, "replacement_round", 0)),
            on_error=record_error,
        )

    @staticmethod
    def _count_nonempty_lines(path: Path) -> Optional[int]:
        if not Path(path).is_file():
            return None
        count = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count if count > 0 else None

    def _check_job_liveness(self, job_id: str) -> Optional[Any]:
        if self.job_liveness_checker is None:
            return None
        try:
            return self.job_liveness_checker(str(job_id))
        except Exception as exc:
            return SimpleNamespace(
                active=False,
                inconclusive=True,
                rows=[],
                error=type(exc).__name__ + ": " + str(exc),
            )

    @staticmethod
    def _queue_rows_sample(liveness: Any, limit: int = 5) -> List[Dict[str, str]]:
        rows = getattr(liveness, "rows", []) or []
        out: List[Dict[str, str]] = []
        for row in list(rows)[: max(0, int(limit))]:
            try:
                job, state = row
            except Exception:
                out.append({"job_id": str(row), "state": ""})
                continue
            out.append({"job_id": str(job), "state": str(state)})
        return out

    @staticmethod
    def _queue_state_counts(liveness: Any) -> Dict[str, int]:
        rows = getattr(liveness, "rows", []) or []
        counts: Dict[str, int] = {}
        for row in list(rows):
            try:
                _job, state = row
            except Exception:
                state = getattr(row, "state", "")
            key = str(state or "").strip().upper()
            if not key:
                continue
            counts[key] = int(counts.get(key, 0)) + 1
        return counts

    def _record_queue_lifecycle(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        job_id: str,
        event: str,
        *,
        status: Optional[str] = None,
        n_expected: Optional[int] = None,
        n_observed: Optional[int] = None,
        n_missing: Optional[int] = None,
        rows_sample: Optional[Any] = None,
    ) -> None:
        try:
            update = _submission_intent.record_queue_lifecycle(
                self.campaign_dir,
                phase.value,
                int(state.iteration),
                event,
                job_id=str(job_id),
                status=status,
                n_expected=n_expected,
                n_observed=n_observed,
                n_missing=n_missing,
                rows_sample=rows_sample,
            )
        except Exception as exc:
            self._journal(
                "submission_intent_update_failed",
                phase=phase.value,
                iteration=int(state.iteration),
                error="queue_lifecycle_" + str(event) + ": " + str(exc)[:180],
            )
            return
        changed = update.get("changed_keys") if isinstance(update, dict) else None
        if isinstance(changed, list) and changed:
            self._journal(
                "queue_lifecycle_update",
                phase=phase.value,
                iteration=int(state.iteration),
                job_id=str(job_id),
                queue_event=str(event),
                status=status,
                changed_keys=[str(k) for k in changed[:8]],
                n_expected=n_expected,
                n_observed=n_observed,
                n_missing=n_missing,
            )

    def _postprocess(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        observations: Sequence[JobObservation],
        summary,
    ) -> str:
        environment_status = self._verify_environment_boundary(
            state,
            phase,
            boundary="postprocess",
        )
        if environment_status is not None:
            return environment_status
        intent_environment_status = self._verify_intent_environment_binding(
            state,
            phase,
        )
        if intent_environment_status is not None:
            return intent_environment_status
        attempts = max(1, int(getattr(self.config.runtime, "postprocess_settle_attempts", 3)))
        settle_seconds = max(0, int(getattr(self.config.runtime, "postprocess_settle_seconds", 10)))
        result = None
        self._record_queue_lifecycle(
            state,
            phase,
            str(summary.parent_job_id),
            "postprocess_started",
            status="started",
            n_expected=getattr(summary, "n_expected", None),
            n_observed=int(getattr(summary, "n_observed", 0)),
            n_missing=int(getattr(summary, "n_missing", 0)),
        )
        for attempt in range(attempts):
            try:
                result = self.executor.postprocess(state, phase, observations)
                result.validate(stage="postprocess", phase_name=phase.value)
            except Exception as exc:
                reason = "postprocess_exception: " + type(exc).__name__ + ": " + str(exc)[:180]
                if attempt + 1 < attempts and self._looks_like_file_settle(reason):
                    self._journal(
                        "postprocess_settle_retry",
                        phase=phase.value,
                        iteration=state.iteration,
                        attempt=int(attempt + 1),
                        reason=reason,
                    )
                    self._write_lease_heartbeat(state)
                    if settle_seconds:
                        self.sleep_fn(float(settle_seconds))
                    continue
                self._record_queue_lifecycle(
                    state,
                    phase,
                    str(summary.parent_job_id),
                    "postprocess_finished",
                    status="failed",
                    n_expected=getattr(summary, "n_expected", None),
                    n_observed=int(getattr(summary, "n_observed", 0)),
                    n_missing=int(getattr(summary, "n_missing", 0)),
                )
                return self._halt(state, phase, reason)
            if (
                result.failure_reason
                and attempt + 1 < attempts
                and self._looks_like_file_settle(str(result.failure_reason))
            ):
                self._journal(
                    "postprocess_settle_retry",
                    phase=phase.value,
                    iteration=state.iteration,
                    attempt=int(attempt + 1),
                    reason=str(result.failure_reason)[:180],
                )
                self._write_lease_heartbeat(state)
                if settle_seconds:
                    self.sleep_fn(float(settle_seconds))
                continue
            break
        if result is None:
            self._record_queue_lifecycle(
                state,
                phase,
                str(summary.parent_job_id),
                "postprocess_finished",
                status="failed",
                n_expected=getattr(summary, "n_expected", None),
                n_observed=int(getattr(summary, "n_observed", 0)),
                n_missing=int(getattr(summary, "n_missing", 0)),
            )
            return self._halt(state, phase, "postprocess_failed_without_result")
        if result.failure_reason:
            self._record_queue_lifecycle(
                state,
                phase,
                str(summary.parent_job_id),
                "postprocess_finished",
                status="failed",
                n_expected=getattr(summary, "n_expected", None),
                n_observed=int(getattr(summary, "n_observed", 0)),
                n_missing=int(getattr(summary, "n_missing", 0)),
            )
            return self._halt(state, phase, result.failure_reason)
        self._clear_sacct_streaks(state, str(summary.parent_job_id))
        contract_error = self._transition_output_contract_error(
            state,
            phase,
            result.state_updates,
        )
        if contract_error is not None:
            self._record_queue_lifecycle(
                state,
                phase,
                str(summary.parent_job_id),
                "postprocess_finished",
                status="failed",
                n_expected=getattr(summary, "n_expected", None),
                n_observed=int(getattr(summary, "n_observed", 0)),
                n_missing=int(getattr(summary, "n_missing", 0)),
            )
            state.pending_jobs[phase.value] = None
            self._journal(
                "phase_output_contract_invalid",
                phase=phase.value,
                iteration=state.iteration,
                reason=contract_error,
            )
            return self._halt(
                state,
                phase,
                "phase_output_contract_invalid: " + contract_error,
            )
        completed_iteration = int(state.iteration)
        if phase.value in SBATCH_PHASES:
            self._record_queue_lifecycle(
                state,
                phase,
                str(summary.parent_job_id),
                "postprocess_finished",
                status="succeeded",
                n_expected=getattr(summary, "n_expected", None),
                n_observed=int(getattr(summary, "n_observed", 0)),
                n_missing=int(getattr(summary, "n_missing", 0)),
            )
        # Clear the pending job in the prospective state and advance it
        # before making the submission intent inactive.  The write-ahead
        # completion receipt makes either crash window replayable.
        state.pending_jobs[phase.value] = None
        advanced = self._advance(
            state,
            phase,
            result.state_updates,
            next_phase_override=result.next_phase_override,
            job_id=str(summary.parent_job_id),
            expected_tasks=getattr(summary, "n_expected", None),
        )
        if not advanced:
            return TickStatus.HALTED
        if phase.value in SBATCH_PHASES:
            self._complete_intent_after_advance(
                phase.value,
                completed_iteration,
                state.last_completion_receipt,
            )
        self._journal(
            "phase_succeeded", phase=phase.value, iteration=completed_iteration,
            n_completed=summary.n_completed, n_failed=summary.n_failed,
            n_tasks=summary.n_tasks,
            completion_receipt=(
                dict(state.last_completion_receipt)
                if isinstance(state.last_completion_receipt, dict)
                else None
            ),
        )
        return TickStatus.ADVANCED

    def _logical_failure_task_ids(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        dense_indices: Sequence[int],
    ) -> List[int]:
        """Translate scheduler array indexes through the immutable attempt map."""
        intent = _submission_intent.load_intent(
            self.campaign_dir,
            phase.value,
            int(state.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
        metadata = intent.get("submission_metadata") if isinstance(intent, dict) else None
        bundle_raw = metadata.get("script_bundle") if isinstance(metadata, dict) else None
        if not isinstance(bundle_raw, str) or not bundle_raw:
            return [int(value) for value in dense_indices]
        map_path = Path(bundle_raw) / "array_task_map.json"
        if not map_path.is_file():
            return [int(value) for value in dense_indices]
        from .script_bundles import read_array_task_map

        mapping = list(read_array_task_map(map_path))
        translated = []
        for dense in dense_indices:
            index = int(dense)
            if index < 0 or index >= len(mapping):
                raise ValueError(
                    "scheduler failure index is outside the immutable array map"
                )
            translated.append(int(mapping[index]))
        return translated

    def _handle_failure(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        observations: Sequence[JobObservation],
        summary,
    ) -> str:
        if self._should_retry_transient_failure(state, phase, observations):
            return self._retry_transient_phase(state, phase, observations, summary)

        action = self.executor.handle_failure(state, phase, observations)
        dense_failure_indices = list(summary.failure_indices)
        logical_failure_task_ids = self._logical_failure_task_ids(
            state,
            phase,
            dense_failure_indices,
        )
        self._journal(
            "failure_action",
            phase=phase.value,
            iteration=state.iteration,
            action=str(action.value if hasattr(action, "value") else action),
            dense_failure_indices=dense_failure_indices,
            logical_failure_task_ids=logical_failure_task_ids,
            n_tasks=summary.n_tasks,
            n_completed=summary.n_completed,
            n_failed=summary.n_failed,
        )
        if action == FailureAction.HALT:
            self._clear_sacct_streaks(state, str(summary.parent_job_id))
            return self._halt(
                state, phase,
                "too_many_failures: " + str(summary.n_failed) + "/" + str(summary.n_tasks),
            )
        if self._strict_artifact_checks_enabled() and phase in (
            CampaignPhase.INITIAL_FEREBUS,
            CampaignPhase.FEREBUS,
        ):
            self._clear_sacct_streaks(state, str(summary.parent_job_id))
            state.pending_jobs[phase.value] = None
            contract_error = self._transition_output_contract_error(state, phase, {})
            if contract_error is not None:
                action_value = str(action.value if hasattr(action, "value") else action)
                self._journal(
                    "required_phase_output_missing_after_failure",
                    phase=phase.value,
                    iteration=state.iteration,
                    action=action_value,
                    n_tasks=summary.n_tasks,
                    n_completed=summary.n_completed,
                    n_failed=summary.n_failed,
                    reason=contract_error,
                )
                return self._halt(
                    state,
                    phase,
                    "required_phase_output_missing_after_failure: " + contract_error,
                )
        if (
            self._strict_artifact_checks_enabled()
            and phase.value in _STRICT_FAILURE_REQUIRES_POSTPROCESS
        ):
            self._clear_sacct_streaks(state, str(summary.parent_job_id))
            return self._halt(
                state,
                phase,
                "phase_failed_requires_postprocess: "
                + phase.value
                + " cannot scrub_and_continue after scheduler failure",
            )
        #SCRUB_AND_CONTINUE: clear pending, advance regardless of partial loss.
        if phase.value in SBATCH_PHASES:
            try:
                _submission_intent.mark_failed(
                    self.campaign_dir,
                    phase.value,
                    int(state.iteration),
                    "scrub_and_continue_after_failures",
                )
            except Exception:
                pass
        self._clear_sacct_streaks(state, str(summary.parent_job_id))
        state.pending_jobs[phase.value] = None
        contract_error = self._transition_output_contract_error(state, phase, {})
        if contract_error is not None:
            action_value = str(action.value if hasattr(action, "value") else action)
            self._journal(
                "required_phase_output_missing_after_failure",
                phase=phase.value,
                iteration=state.iteration,
                action=action_value,
                n_tasks=summary.n_tasks,
                n_completed=summary.n_completed,
                n_failed=summary.n_failed,
                reason=contract_error,
            )
            return self._halt(
                state,
                phase,
                "required_phase_output_missing_after_failure: " + contract_error,
            )
        return (
            TickStatus.SCRUBBED
            if self._advance(
                state,
                phase,
                {},
                job_id=str(summary.parent_job_id),
                expected_tasks=getattr(summary, "n_expected", None),
            )
            else TickStatus.HALTED
        )

    def _clear_sacct_streaks(self, state: CampaignState, job_id: str) -> None:
        state.sacct_empty_streak.pop(str(job_id), None)
        state.sacct_empty_streak.pop(str(job_id) + ":UNKNOWN", None)
        state.sacct_empty_streak.pop(str(job_id) + ":MISSING", None)
        state.sacct_empty_streak.pop(str(job_id) + ":SQUEUE_INCONCLUSIVE:empty", None)
        state.sacct_empty_streak.pop(str(job_id) + ":SQUEUE_INCONCLUSIVE:missing", None)
        state.sacct_empty_streak.pop(str(job_id) + ":ERROR", None)

    def _complete_intent_after_advance(
        self,
        phase_name: str,
        iteration: int,
        completion_receipt: Optional[Dict[str, Any]],
    ) -> None:
        try:
            _submission_intent.mark_completed(
                self.campaign_dir,
                phase_name,
                int(iteration),
                completion_receipt=(
                    dict(completion_receipt)
                    if isinstance(completion_receipt, dict)
                    else None
                ),
            )
        except Exception as exc:
            self._journal(
                "submission_intent_completion_deferred",
                phase=phase_name,
                iteration=int(iteration),
                error=type(exc).__name__ + ": " + str(exc)[:180],
                completion_receipt=(
                    dict(completion_receipt)
                    if isinstance(completion_receipt, dict)
                    else None
                ),
            )

    def _liveness_blocks_accounting_timeout(self, liveness: Optional[Any]) -> bool:
        return liveness is not None and bool(getattr(liveness, "active", False))

    def _handle_squeue_inconclusive_liveness(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        job_id: str,
        *,
        kind: str,
        liveness: Any,
        summary: Any,
        accounting_streak: int,
    ) -> str:
        key = str(job_id) + ":SQUEUE_INCONCLUSIVE:" + str(kind)
        current = int(state.sacct_empty_streak.get(key, 0)) + 1
        state.sacct_empty_streak[key] = current
        self._persist(state)
        self._journal_sparse_accounting_liveness(
            phase=phase,
            job_id=job_id,
            streak=accounting_streak,
            liveness=liveness,
            kind=kind,
            summary=summary,
            iteration=state.iteration,
        )
        max_ticks = int(
            getattr(self.config.runtime, "poll_squeue_inconclusive_max_ticks", 10)
        )
        if max_ticks > 0 and current >= max_ticks:
            return self._halt_scheduler_uncertain(
                state,
                phase,
                "squeue_inconclusive_timeout: "
                + str(current)
                + "/"
                + str(max_ticks)
                + " ticks could not prove Slurm liveness for job_id="
                + str(job_id),
            )
        return TickStatus.POLLING

    def _journal_sparse_accounting_liveness(
        self,
        *,
        phase: CampaignPhase,
        job_id: str,
        streak: int,
        liveness: Any,
        kind: str,
        summary: Any,
        iteration: int,
    ) -> None:
        payload = {
            "phase": phase.value,
            "job_id": job_id,
            "n_expected": getattr(summary, "n_expected", None),
            "n_observed": int(getattr(summary, "n_observed", 0)),
            "n_completed": int(getattr(summary, "n_completed", 0)),
            "n_failed": int(getattr(summary, "n_failed", 0)),
            "n_missing": int(getattr(summary, "n_missing", 0)),
            "streak": int(streak),
            "accounting_kind": str(kind),
            "iteration": int(iteration),
        }
        if bool(getattr(liveness, "active", False)):
            event = (
                "sacct_empty_but_squeue_active"
                if str(kind) == "empty"
                else "sacct_rows_missing_but_squeue_active"
            )
            payload["squeue_rows_sample"] = self._queue_rows_sample(liveness)
            payload["squeue_state_counts"] = self._queue_state_counts(liveness)
            try:
                first_state = (
                    payload["squeue_rows_sample"][0].get("state")
                    if payload["squeue_rows_sample"]
                    else "active"
                )
                _submission_intent.record_queue_lifecycle(
                    self.campaign_dir,
                    phase.value,
                    int(iteration),
                    "first_squeue",
                    job_id=str(job_id),
                    status=str(first_state),
                    n_expected=getattr(summary, "n_expected", None),
                    n_observed=int(getattr(summary, "n_observed", 0)),
                    n_missing=int(getattr(summary, "n_missing", 0)),
                    rows_sample=payload["squeue_rows_sample"],
                )
            except Exception as exc:
                payload["queue_lifecycle_warning"] = str(exc)[:160]
            self._journal(event, **payload)
            return
        payload["error"] = str(getattr(liveness, "error", "") or "")[:200]
        self._journal("squeue_liveness_inconclusive", **payload)

    def _looks_like_file_settle(self, reason: str) -> bool:
        text = str(reason).lower()
        markers = (
            "missing",
            "unreadable",
            "not found",
            "no such file",
            "manifest",
            "expected_model_missing",
            "ferebus_model_root_missing",
        )
        return any(marker in text for marker in markers)

    def _load_transient_retry_ledger(self) -> Dict[str, Any]:
        path = self.transient_retry_ledger_path()
        if not path.is_file():
            return {"schema_version": 2, "attempts": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                "transient retry ledger is unreadable: " + str(path)
            ) from exc
        if (
            not isinstance(data, dict)
            or isinstance(data.get("schema_version"), bool)
            or data.get("schema_version") != 2
        ):
            raise ValueError("transient retry ledger has an unsupported schema")
        attempts = data.get("attempts")
        if not isinstance(attempts, dict):
            raise ValueError("transient retry ledger attempts must be an object")
        for key, value in attempts.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "transient retry ledger attempt is not an integer for " + str(key)
                )
            count = value
            if count < 0:
                raise ValueError(
                    "transient retry ledger attempt is negative for " + str(key)
                )
        return {"schema_version": 2, "attempts": attempts}

    def _retry_key(self, state: CampaignState, phase: CampaignPhase) -> str:
        intent = _submission_intent.load_active_intent(
            self.campaign_dir,
            phase.value,
            int(state.iteration),
            expected_campaign_uid=str(state.campaign_uid),
        )
        if not isinstance(intent, dict):
            raise ValueError("transient retry requires an active submission intent")
        metadata = intent.get("submission_metadata")
        task_set_digest = (
            str(metadata.get("logical_task_set_sha256") or "")
            if isinstance(metadata, dict)
            else ""
        )
        if not task_set_digest:
            expected = intent.get("expected_tasks")
            if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
                raise ValueError("transient retry intent has no exact task cardinality")
            task_set_digest = hashlib.sha256(
                ("range:" + str(expected)).encode("ascii")
            ).hexdigest()
        return (
            phase.value
            + "@"
            + str(int(state.iteration))
            + "@round="
            + str(int(state.replacement_round))
            + "@tasks="
            + task_set_digest
        )

    def _should_retry_transient_failure(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        observations: Sequence[JobObservation],
    ) -> bool:
        max_retry = int(getattr(self.config.runtime, "transient_phase_retry_max", 1))
        if max_retry <= 0 or phase.value not in SBATCH_PHASES:
            return False
        failure_statuses = [
            str(obs.status.value if hasattr(obs.status, "value") else obs.status)
            for obs in observations
            if bool(getattr(obs, "is_failure", False))
        ]
        if not failure_statuses:
            return False
        if any(status not in TRANSIENT_RETRY_STATUSES for status in failure_statuses):
            return False
        try:
            ledger = self._load_transient_retry_ledger()
        except Exception as exc:
            self._journal(
                "transient_retry_ledger_invalid",
                phase=phase.value,
                iteration=int(state.iteration),
                reason=type(exc).__name__ + ": " + str(exc)[:180],
            )
            return False
        attempts = ledger.get("attempts", {})
        try:
            key = self._retry_key(state, phase)
        except Exception as exc:
            self._journal(
                "transient_retry_ledger_invalid",
                phase=phase.value,
                iteration=int(state.iteration),
                reason=type(exc).__name__ + ": " + str(exc)[:180],
            )
            return False
        return int(attempts.get(key, 0)) < max_retry

    def _retry_transient_phase(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        observations: Sequence[JobObservation],
        summary,
    ) -> str:
        if self._strict_artifact_checks_enabled() and phase in (
            CampaignPhase.INITIAL_FEREBUS,
            CampaignPhase.FEREBUS,
        ):
            training_raw = getattr(state, "reference_data_version", -1)
            try:
                reference_data_version = int(training_raw)
            except (TypeError, ValueError):
                reference_data_version = -1
            if reference_data_version >= 0:
                from ..versioning.trained_models import TrainedModelVersioning
                from ..layout import trained_models_dir

                candidate = TrainedModelVersioning(
                    trained_models_dir(self.campaign_dir)
                ).iteration_path(reference_data_version)
                if candidate.is_dir():
                    return self._halt(
                        state,
                        phase,
                        "transient_retry_found_committed_model_output: "
                        + str(candidate)
                        + "; run reconcile before retrying",
                    )
        ledger = self._load_transient_retry_ledger()
        attempts = dict(ledger.get("attempts", {}))
        key = self._retry_key(state, phase)
        attempt = int(attempts.get(key, 0)) + 1
        attempts[key] = attempt
        atomic_write_json(
            self.transient_retry_ledger_path(),
            {"schema_version": 2, "attempts": attempts},
        )
        try:
            _submission_intent.mark_failed(
                self.campaign_dir,
                phase.value,
                int(state.iteration),
                "transient_scheduler_retry",
            )
        except Exception:
            pass
        self._clear_sacct_streaks(state, str(summary.parent_job_id))
        state.pending_jobs[phase.value] = None
        self._persist(state)
        statuses = [
            str(obs.status.value if hasattr(obs.status, "value") else obs.status)
            for obs in observations
        ]
        dense_failure_indices = list(summary.failure_indices)
        logical_failure_task_ids = self._logical_failure_task_ids(
            state,
            phase,
            dense_failure_indices,
        )
        self._journal(
            "transient_phase_retry",
            phase=phase.value,
            iteration=state.iteration,
            attempt=int(attempt),
            statuses=statuses,
            dense_failure_indices=dense_failure_indices,
            logical_failure_task_ids=logical_failure_task_ids,
        )
        return TickStatus.RETRYING

    def _halt(self, state: CampaignState, phase: CampaignPhase, reason: str) -> str:
        if phase.value in SBATCH_PHASES:
            try:
                _submission_intent.mark_failed(
                    self.campaign_dir, phase.value, int(state.iteration), reason,
                )
            except Exception:
                pass
            state.pending_jobs[phase.value] = None
        reason_code = _halt_reason_code(reason)
        state.lifecycle_context = make_lifecycle_context(
            disposition="halted",
            reason_code=reason_code,
            message=str(reason),
            from_phase=phase,
            iteration=int(state.iteration),
            source="daemon",
            recovery_action=_halt_recovery_action(reason_code),
        )
        state.phase = CampaignPhase.HALTED
        self._persist(state)
        self._journal(
            "halt",
            from_phase=phase.value,
            reason=reason,
            reason_code=reason_code,
            recovery_action=_halt_recovery_action(reason_code),
            iteration=state.iteration,
        )
        return TickStatus.HALTED

    def _halt_scheduler_uncertain(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        reason: str,
    ) -> str:
        reason_code = _halt_reason_code(reason)
        pending_job = state.pending_jobs.get(phase.value)
        state.lifecycle_context = make_lifecycle_context(
            disposition="halted",
            reason_code=reason_code,
            message=str(reason),
            from_phase=phase,
            iteration=int(state.iteration),
            source="daemon",
            job_id=None if pending_job is None else str(pending_job),
            scheduler_uncertain=True,
            recovery_action=(
                "inspect sacct and squeue for the preserved job, then reconcile; "
                "do not resubmit until job liveness is conclusive"
            ),
        )
        state.phase = CampaignPhase.HALTED
        self._persist(state)
        self._journal(
            "halt",
            from_phase=phase.value,
            reason=reason,
            reason_code=reason_code,
            scheduler_uncertain=True,
            preserves_pending_jobs=True,
            preserves_submission_intent=True,
            iteration=state.iteration,
        )
        return TickStatus.HALTED

    def _halt_environment_drift(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        reason: str,
    ) -> str:
        """Halt without changing scheduler ownership or submission evidence."""
        pending_job = state.pending_jobs.get(phase.value)
        state.lifecycle_context = make_lifecycle_context(
            disposition="halted",
            reason_code="environment_drift",
            message=str(reason),
            from_phase=phase,
            iteration=int(state.iteration),
            source="daemon_environment_guard",
            job_id=None if pending_job is None else str(pending_job),
            scheduler_uncertain=bool(pending_job),
            recovery_action=(
                "inspect environment-status, preserve scheduler evidence, then "
                "reconcile to an idle boundary before rebind-environment"
            ),
        )
        state.phase = CampaignPhase.HALTED
        self._persist(state)
        self._journal(
            "environment_drift_halted",
            from_phase=phase.value,
            iteration=int(state.iteration),
            reason=str(reason),
            job_id=None if pending_job is None else str(pending_job),
            preserves_pending_jobs=True,
            preserves_submission_intent=True,
        )
        return TickStatus.HALTED

    def _halt_after_tick_exception(self, exc: Exception) -> bool:
        """Move the campaign to HALTED after an unexpected tick exception.

        This is deliberately not routed through ``_halt``. A generic daemon
        exception is not evidence that a submitted Slurm job failed, so pending
        jobs and active submission intents must remain intact for user
        cancellation or reconciliation.
        """
        try:
            state = read_state(self.state_path())
        except Exception:
            return False
        prior_phase = state.phase
        state.lifecycle_context = make_lifecycle_context(
            disposition="halted",
            reason_code="tick_exception",
            message=type(exc).__name__ + ": " + str(exc)[:200],
            from_phase=prior_phase,
            iteration=int(state.iteration),
            source="daemon",
            scheduler_uncertain=any(bool(value) for value in state.pending_jobs.values()),
            recovery_action=(
                "inspect LAST_EXCEPTION.json and any preserved Slurm jobs, then reconcile"
            ),
            details={"exception_type": type(exc).__name__},
        )
        state.phase = CampaignPhase.HALTED
        try:
            self._persist(state)
        except Exception:
            return False
        self._journal(
            "tick_exception_halted",
            from_phase=prior_phase.value,
            iteration=int(state.iteration),
            error=type(exc).__name__ + ": " + str(exc)[:200],
        )
        return True

    def _transition_output_contract_error(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        state_updates: Dict[str, Any],
    ) -> Optional[str]:
        """Return a reason if advancing would violate a producer handoff.

        This guard is intentionally driven by the same strict-artifact flag
        used by live executors. Mock/dry executors can still exercise the FSM
        without fabricating full FEREBUS model directories, while live runs
        cannot advance past a required model-producing phase unless the
        committed model contract is actually satisfied.
        """
        if not bool(getattr(self.executor, "strict_committed_artifact_verification", False)):
            return None
        if phase not in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
            return None

        if "models_version" not in state_updates:
            return (
                "required fresh FEREBUS models_version missing from state_updates for "
                + phase.value
            )
        raw_version = state_updates.get("models_version")
        try:
            models_version = int(raw_version)
        except (TypeError, ValueError):
            return (
                "required FEREBUS model version is not an integer for "
                + phase.value
                + ": "
                + repr(raw_version)
            )
        if models_version < 0:
            return (
                "required FEREBUS model version is negative for "
                + phase.value
                + ": "
                + str(models_version)
            )
        raw_training = state_updates.get(
            "reference_data_version",
            getattr(state, "reference_data_version", -1),
        )
        try:
            reference_data_version = int(raw_training)
        except (TypeError, ValueError):
            return (
                "required FEREBUS reference-data version is not an integer for "
                + phase.value
                + ": "
                + repr(raw_training)
            )
        if reference_data_version < 0:
            return (
                "required FEREBUS reference-data version is negative for "
                + phase.value
                + ": "
                + str(reference_data_version)
            )
        if models_version != reference_data_version:
            return (
                "FEREBUS model/reference-data version skew after "
                + phase.value
                + ": models_version="
                + str(models_version)
                + ", reference_data_version="
                + str(reference_data_version)
            )
        expected_version = 0 if phase is CampaignPhase.INITIAL_FEREBUS else int(state.iteration)
        if models_version != expected_version:
            return (
                "FEREBUS committed the wrong daemon iteration version after "
                + phase.value
                + ": expected="
                + str(expected_version)
                + ", observed="
                + str(models_version)
            )

        from ..versioning.trained_models import TrainedModelVersioning
        from ..layout import trained_models_dir

        expected_path = TrainedModelVersioning(
            trained_models_dir(self.campaign_dir)
        ).iteration_path(models_version)
        try:
            from .artifact_contracts import (
                verify_committed_model_version,
                verify_committed_reference_data_version,
            )
            from .artifact_snapshot import build_committed_artifact_snapshot

            snapshot = build_committed_artifact_snapshot(
                self.campaign_dir,
                verification_level="metadata",
            )
            verify_committed_reference_data_version(
                self.campaign_dir,
                reference_data_version,
                verification="metadata",
                snapshot=snapshot,
            )
            verify_committed_model_version(
                self.campaign_dir,
                models_version,
                verification="metadata",
                snapshot=snapshot,
            )
        except Exception as exc:
            return (
                "required FEREBUS model commit missing or invalid after "
                + phase.value
                + " (models_version="
                + str(models_version)
                + ", expected_path="
                + str(expected_path)
                + "): "
                + type(exc).__name__
                + ": "
                + str(exc)[:220]
            )
        return None

    def _advance(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        state_updates: Dict[str, Any],
        *,
        next_phase_override: Optional[str] = None,
        job_id: Optional[str] = None,
        expected_tasks: Optional[int] = None,
    ) -> bool:
        completion_reason = state_updates.get("campaign_completion_reason")
        applied_updates = {
            key: value
            for key, value in state_updates.items()
            if key != "campaign_completion_reason"
        }
        if completion_reason is not None:
            if phase is not CampaignPhase.STOP_CHECK:
                self._journal(
                    "phase_output_contract_invalid",
                    phase=phase.value,
                    iteration=int(state.iteration),
                    reason="campaign_completion_reason is valid only for STOP_CHECK",
                )
                self._halt(
                    state,
                    phase,
                    "phase_output_contract_invalid: campaign_completion_reason "
                    "is valid only for STOP_CHECK",
                )
                return False
            before = self._authoritative_state_before_transition(state)
            after = copy.deepcopy(state)
            self._apply_state_updates(after, applied_updates)
            after.phase = CampaignPhase.DONE
            after.lifecycle_context = make_lifecycle_context(
                disposition="completed",
                reason_code="scientific_convergence",
                message="scientific convergence criterion reached: "
                + str(completion_reason),
                from_phase=phase,
                iteration=int(state.iteration),
                source="daemon",
                recovery_action=(
                    "campaign is complete; use resume --reopen-converged only "
                    "after deliberately increasing max_iterations"
                ),
                details={"criterion": str(completion_reason)},
            )
            self._persist_transition_with_receipt(
                state,
                before,
                after,
                phase,
                dict(state_updates),
                next_phase=CampaignPhase.DONE,
                next_iteration=int(state.iteration),
                job_id=job_id,
                expected_tasks=expected_tasks,
            )
            self._journal(
                "campaign_completed",
                from_phase=phase.value,
                iteration=int(state.iteration),
                reason_code="scientific_convergence",
                criterion=str(completion_reason),
                completion_receipt=dict(state.last_completion_receipt or {}),
            )
            return True

        #NB:STOP_CHECK ghost-iteration fix. When _inline_stop_check
        #returns {"shutdown_requested": True}, the daemon must NOT also
        #advance phase + iteration -- otherwise state.json snapshots show
        #"iteration N+1 / SEED_SELECT, shutdown_requested=True" which is
        #confusing and produces a stale "in-flight" appearance on restart.
        #Persist the flag + journal a shutdown_requested event; next tick
        #exits cleanly via state.shutdown_requested.
        if bool(state_updates.get("shutdown_requested", False)):
            before = self._authoritative_state_before_transition(state)
            after = copy.deepcopy(state)
            self._apply_state_updates(after, applied_updates)
            after.lifecycle_context = make_lifecycle_context(
                disposition="stopped",
                reason_code="executor_stop_request",
                message="executor requested an orderly campaign stop",
                from_phase=phase,
                iteration=int(state.iteration),
                source="daemon",
                recovery_action="use resume to clear the user stop request",
            )
            self._persist_transition_with_receipt(
                state,
                before,
                after,
                phase,
                state_updates,
                next_phase=phase,
                next_iteration=int(state.iteration),
                job_id=job_id,
                expected_tasks=expected_tasks,
            )
            self._journal(
                "shutdown_requested", from_phase=phase.value,
                iteration=int(state.iteration),
            )
            return True

        contract_error = self._transition_output_contract_error(
            state,
            phase,
            applied_updates,
        )
        if contract_error is not None:
            self._journal(
                "phase_output_contract_invalid",
                phase=phase.value,
                iteration=state.iteration,
                reason=contract_error,
            )
            self._halt(
                state,
                phase,
                "phase_output_contract_invalid: " + contract_error,
            )
            return False

        new_phase, new_iter = next_phase(phase, state.iteration, state.max_iterations)
        if next_phase_override is not None:
            try:
                requested_phase = CampaignPhase(str(next_phase_override))
            except ValueError as exc:
                self._halt(
                    state,
                    phase,
                    "invalid next-phase override: " + repr(next_phase_override),
                )
                return False
            if requested_phase not in _ALLOWED_PHASE_OVERRIDES.get(phase, frozenset()):
                self._halt(
                    state,
                    phase,
                    "disallowed next-phase override "
                    + phase.value + " -> " + requested_phase.value,
                )
                return False
            new_phase = requested_phase
        prior_iter = state.iteration
        before = self._authoritative_state_before_transition(state)
        after = copy.deepcopy(state)
        after.phase = new_phase
        after.iteration = new_iter
        self._apply_state_updates(after, applied_updates)
        if new_phase is CampaignPhase.DONE:
            after.lifecycle_context = make_lifecycle_context(
                disposition="completed",
                reason_code="max_iterations_reached",
                message="configured maximum active-learning iterations reached",
                from_phase=phase,
                iteration=int(state.iteration),
                source="daemon",
                recovery_action=(
                    "campaign is complete; increase campaign.max_iterations and "
                    "use resume --reopen-converged only after reviewing campaign quality"
                ),
            )
        self._persist_transition_with_receipt(
            state,
            before,
            after,
            phase,
            state_updates,
            next_phase=new_phase,
            next_iteration=int(new_iter),
            job_id=job_id,
            expected_tasks=expected_tasks,
        )
        self._journal(
            "phase_transition", from_phase=phase.value, to_phase=new_phase.value,
            iteration=new_iter, prior_iteration=prior_iter,
            completion_receipt=(
                dict(state.last_completion_receipt)
                if isinstance(state.last_completion_receipt, dict)
                else None
            ),
        )
        if new_phase is CampaignPhase.DONE:
            self._journal(
                "campaign_completed",
                from_phase=phase.value,
                iteration=int(new_iter),
                reason_code="max_iterations_reached",
                completion_receipt=dict(state.last_completion_receipt or {}),
            )
        return True

    def _authoritative_state_before_transition(self, state: CampaignState) -> CampaignState:
        try:
            before = read_state(self.state_path())
        except FileNotFoundError:
            before = copy.deepcopy(state)
        if before.phase is not state.phase or int(before.iteration) != int(state.iteration):
            raise RuntimeError(
                "state changed while preparing phase completion: on_disk="
                + before.phase.value
                + "@"
                + str(int(before.iteration))
                + " in_memory="
                + state.phase.value
                + "@"
                + str(int(state.iteration))
            )
        return before

    def _phase_completion_evidence_paths(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        state_updates: Dict[str, Any],
    ) -> List[Path]:
        from ..handoff_manifests import (
            ariadne_results_path,
            phase_a_sample_manifest_path,
            phase_b_selection_path,
            seeds_picked_path,
        )
        from ..layout import (
            active_allocation_dir,
            active_iteration_dir,
            bootstrap_selection_dir,
            trained_models_dir as canonical_trained_models_dir,
        )

        campaign = self.campaign_dir
        iteration = int(state.iteration)
        paths: List[Path] = []
        if phase is CampaignPhase.PHASE_A_DIVERSITY:
            paths.append(phase_a_sample_manifest_path(bootstrap_selection_dir(campaign)))
        elif phase is CampaignPhase.SEED_SELECT:
            paths.append(seeds_picked_path(active_iteration_dir(campaign, iteration)))
        elif phase is CampaignPhase.ARIADNE_ARRAY:
            from ..handoff_manifests import ariadne_batch_decision_path

            iter_dir = active_iteration_dir(campaign, iteration)
            paths.extend(
                [
                    ariadne_results_path(iter_dir),
                    ariadne_batch_decision_path(iter_dir),
                ]
            )
        elif phase is CampaignPhase.PHASE_B_DIVERSITY:
            paths.append(phase_b_selection_path(active_iteration_dir(campaign, iteration)))
        elif phase is CampaignPhase.SPLIT:
            paths.append(
                active_allocation_dir(active_iteration_dir(campaign, iteration))
                / "SPLIT_RECEIPT.json"
            )
        elif phase in (
            CampaignPhase.INITIAL_ALLOCATION_CHECK,
            CampaignPhase.ALLOCATION_CHECK,
            CampaignPhase.INITIAL_REPLACEMENT_AIMALL,
            CampaignPhase.REPLACEMENT_AIMALL,
        ):
            from ..point_allocation import point_allocation_path

            context = "bootstrap" if phase.value.startswith("INITIAL_") else "active"
            paths.append(
                point_allocation_path(
                    campaign,
                    context=context,
                    iteration=0 if context == "bootstrap" else iteration,
                )
            )
        elif phase in (
            CampaignPhase.INITIAL_GAUSSIAN,
            CampaignPhase.INITIAL_AIMALL,
            CampaignPhase.INITIAL_REPLACEMENT_GAUSSIAN,
            CampaignPhase.GAUSSIAN,
            CampaignPhase.AIMALL,
            CampaignPhase.REPLACEMENT_GAUSSIAN,
        ):
            from .input_staging import quantum_acceptance_manifest_path
            from ..replacement_sampling import replacement_round_dir

            if "REPLACEMENT" in phase.value:
                context = "bootstrap" if phase.value.startswith("INITIAL_") else "active"
                staging = replacement_round_dir(
                    campaign,
                    context=context,
                    iteration=0 if context == "bootstrap" else iteration,
                    replacement_round=int(getattr(state, "replacement_round", 0)),
                )
            else:
                from ..layout import staging_phase_dir

                staging = staging_phase_dir(campaign, phase.value, iteration)
            paths.append(
                quantum_acceptance_manifest_path(staging, phase_name=phase.value)
            )
            if "AIMALL" in phase.value:
                paths.append(staging / "quantum_quality.json")
        elif phase is CampaignPhase.REFERENCE_COMMIT:
            from ..versioning.reference_data import reference_data_version_path
            from ..versioning.reference_data import ReferenceDataVersioning
            from ..layout import qm_reference_data_dir

            version = int(state_updates.get("reference_data_version", state.reference_data_version))
            if version >= 0:
                paths.append(
                    reference_data_version_path(
                        ReferenceDataVersioning(qm_reference_data_dir(campaign)).iteration_path(version)
                    )
                )
        elif phase in (CampaignPhase.INITIAL_FEREBUS, CampaignPhase.FEREBUS):
            from ..versioning.trained_models import trained_model_set_path
            from ..versioning.trained_models import TrainedModelVersioning

            version = int(state_updates.get("models_version", state.models_version))
            if version >= 0:
                paths.append(
                    trained_model_set_path(
                        TrainedModelVersioning(canonical_trained_models_dir(campaign)).iteration_path(version)
                    )
                )
        elif phase is CampaignPhase.STOP_CHECK:
            from ..versioning.sampling_iterations import active_iteration_manifest_path

            paths.append(active_iteration_manifest_path(campaign, iteration))
        return paths

    def _stop_request_satisfied_by_transition(
        self,
        request: Mapping[str, Any],
        *,
        before: CampaignState,
        after: CampaignState,
        phase: CampaignPhase,
    ) -> Optional[str]:
        if after.phase in {CampaignPhase.DONE, CampaignPhase.HALTED}:
            return "campaign_terminal"
        mode = str(request.get("mode") or "")
        if mode == "immediate":
            return "immediate_after_current_tick"
        if mode == "after_phase":
            if (
                phase.value == str(request.get("target_phase") or "")
                and int(before.iteration) == int(request.get("target_iteration", -1))
                and int(before.replacement_round)
                == int(request.get("target_replacement_round", -1))
            ):
                return "phase_completed"
            return None
        if mode != "after_iteration":
            return None
        target = int(request.get("target_iteration", -1))
        if target == 0:
            if (
                phase is CampaignPhase.INITIAL_FEREBUS
                and int(before.iteration) == 0
            ):
                return "bootstrap_iteration_completed"
            return None
        if phase is CampaignPhase.STOP_CHECK and int(before.iteration) == target:
            return "active_iteration_completed"
        return None

    def _prepare_stop_control_for_transition(
        self,
        *,
        before: CampaignState,
        after: CampaignState,
        phase: CampaignPhase,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        request = self._read_stop_control(before)
        if request is None:
            return None, None
        if (
            str(request.get("mode")) == "immediate"
            and str(request.get("status")) == "cancelling"
        ):
            return None, None
        reason = self._stop_request_satisfied_by_transition(
            request,
            before=before,
            after=after,
            phase=phase,
        )
        if reason is None:
            return None, None
        if after.phase not in {CampaignPhase.DONE, CampaignPhase.HALTED}:
            from .stop_control import describe_stop_request

            self._apply_cancelled_jobs_from_request(after, request)
            after.shutdown_requested = True
            after.lifecycle_context = make_lifecycle_context(
                disposition="stopped",
                reason_code="user_stop_boundary_reached",
                message=describe_stop_request(request, completed=True),
                from_phase=phase,
                iteration=int(after.iteration),
                source="daemon_stop_control",
                scheduler_uncertain=any(
                    bool(value) for value in after.pending_jobs.values()
                ),
                recovery_action="use resume to continue from the recorded phase",
                details={
                    "request_id": str(request.get("request_id")),
                    "mode": str(request.get("mode")),
                    "target_phase": request.get("target_phase"),
                    "target_iteration": request.get("target_iteration"),
                    "target_replacement_round": request.get(
                        "target_replacement_round"
                    ),
                },
            )
        return request, reason

    def _persist_transition_with_receipt(
        self,
        destination: CampaignState,
        before: CampaignState,
        after: CampaignState,
        phase: CampaignPhase,
        state_updates: Dict[str, Any],
        *,
        next_phase: CampaignPhase,
        next_iteration: int,
        job_id: Optional[str],
        expected_tasks: Optional[int],
    ) -> None:
        from .completion_receipts import (
            evidence_records,
            receipt_reference,
            write_completion_receipt,
        )
        from .config_lock import canonical_config, config_fingerprint

        intent = _submission_intent.load_intent(
            self.campaign_dir,
            phase.value,
            int(before.iteration),
            expected_campaign_uid=str(before.campaign_uid),
        )
        evidence_paths = self._phase_completion_evidence_paths(
            before,
            phase,
            state_updates,
        )
        stop_request, stop_reason = self._prepare_stop_control_for_transition(
            before=before,
            after=after,
            phase=phase,
        )
        try:
            after = CampaignState.from_dict(after.to_dict())
        except Exception as exc:
            raise ValueError(
                "prospective campaign state is invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        if not bool(
            getattr(
                self.executor,
                "strict_completion_receipt_evidence",
                getattr(
                    self.executor,
                    "strict_committed_artifact_verification",
                    False,
                ),
            )
        ):
            # The pure FSM mock deliberately creates no handoff artefacts.  Keep
            # its receipts useful for transition tests without weakening the
            # fail-closed evidence contract used by live execution.
            evidence_paths = [path for path in evidence_paths if path.exists()]
        evidence = evidence_records(self.campaign_dir, evidence_paths)
        receipt_config_sha = config_fingerprint(canonical_config(self.config))
        if intent is not None:
            decision_contract = intent.get("decision_contract")
            if isinstance(decision_contract, Mapping):
                snapshotted = decision_contract.get("config_sha256")
                if isinstance(snapshotted, str) and snapshotted:
                    receipt_config_sha = snapshotted
        receipt_path = write_completion_receipt(
            self.campaign_dir,
            campaign_uid=str(before.campaign_uid),
            phase=phase.value,
            iteration=int(before.iteration),
            replacement_round=int(getattr(before, "replacement_round", 0)),
            config_sha256=receipt_config_sha,
            state_before=before,
            state_after=after,
            next_phase=next_phase.value,
            next_iteration=int(next_iteration),
            state_updates=state_updates,
            evidence=evidence,
            job_id=job_id or (str(intent.get("job_id")) if intent and intent.get("job_id") else None),
            expected_tasks=(
                expected_tasks
                if expected_tasks is not None
                else (int(intent["expected_tasks"]) if intent and intent.get("expected_tasks") is not None else None)
            ),
            submission_identity=(
                str(intent.get("submission_identity"))
                if intent and intent.get("submission_identity")
                else None
            ),
        )
        after.last_completion_receipt = receipt_reference(self.campaign_dir, receipt_path)
        self._persist(after)
        destination.__dict__.clear()
        destination.__dict__.update(copy.deepcopy(after.__dict__))
        if stop_request is not None and stop_reason is not None:
            from .stop_control import complete_stop_request

            try:
                complete_stop_request(
                    self.campaign_dir,
                    str(stop_request.get("request_id")),
                    reason=str(stop_reason),
                    completion_receipt=after.last_completion_receipt,
                )
            except Exception as exc:
                self._journal(
                    "stop_request_completion_deferred",
                    request_id=str(stop_request.get("request_id")),
                    phase=phase.value,
                    iteration=int(before.iteration),
                    error=type(exc).__name__ + ": " + str(exc)[:180],
                    completion_receipt=dict(after.last_completion_receipt or {}),
                )
            else:
                self._journal(
                    "user_stop_boundary_reached",
                    request_id=str(stop_request.get("request_id")),
                    mode=str(stop_request.get("mode")),
                    phase=phase.value,
                    iteration=int(before.iteration),
                    resulting_phase=after.phase.value,
                    resulting_iteration=int(after.iteration),
                    reason=str(stop_reason),
                    target_phase=stop_request.get("target_phase"),
                    target_iteration=stop_request.get("target_iteration"),
                    target_replacement_round=stop_request.get(
                        "target_replacement_round"
                    ),
                    completion_receipt=dict(after.last_completion_receipt or {}),
                )

    def _apply_state_updates(self, state: CampaignState, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed = {
            "reference_data_version", "validation_set_version", "models_version",
            "replacement_round",
            "last_acquisition_alpha0", "stop_streak", "max_iterations",
            #STOP_CHECK extensions:
            "alpha_history", "shutdown_requested",
            #anti-overlap diagnostic + sacct stale-job streak.
            "last_n_anti_overlap_flagged", "sacct_empty_streak",
            #reference_scales/_iteration are deliberately NOT here: the
            #login-node executor sets them straight on the state during
            #SEED_SELECT and they ride through _advance/_persist. an executor
            #handing them back via state_updates would be a wiring mistake, so
            #rejecting them is intentional.
        }
        for k, v in updates.items():
            if k not in allowed:
                # tighten the contract. silent drops were a future-
                # bug magnet -- a new executor field added later would
                # just vanish without anyone noticing. raising loudly
                # forces every new wiring to be deliberate.
                raise ValueError(
                    "unexpected state update key " + repr(k)
                    + "; permitted keys: " + repr(sorted(allowed))
                )
            setattr(state, k, v)

    def _persist(self, state: CampaignState) -> None:
        if not bool(getattr(state, "shutdown_requested", False)):
            try:
                on_disk = read_state(self.state_path())
            except (FileNotFoundError, StateSchemaError, json.JSONDecodeError):
                on_disk = None
            if on_disk is not None and bool(on_disk.shutdown_requested):
                state.shutdown_requested = True
        write_state(self.state_path(), state)

    # --- public lifecycle ----------------------------------------------

    def request_shutdown(self) -> None:
        """Set the in-process shutdown flag and persist it to disk.

        Callers outside the daemon process must use the CLI stop-control
        manifest rather than rewriting ``state.json``. This in-process method
        is reserved for signal handlers and tests.
        """
        self._shutdown_requested = True
        try:
            state = read_state(self.state_path())
            state.shutdown_requested = True
            state.lifecycle_context = make_lifecycle_context(
                disposition="stopped",
                reason_code="user_stop_request",
                message="user requested an orderly daemon stop",
                from_phase=state.phase,
                iteration=int(state.iteration),
                source="daemon_api",
                recovery_action="use resume to continue from the recorded phase",
            )
            self._persist(state)
        except (FileNotFoundError, StateSchemaError, json.JSONDecodeError):
            pass

    def run(
        self,
        *,
        max_ticks: Optional[int] = None,
        catch_keyboard_interrupt: bool = True,
        readiness_callback: Optional[Callable[[], None]] = None,
    ) -> int:
        """Main loop. Returns an integer exit code suitable for sys.exit.

        Acquires the lock, writes the PID file, installs signal handlers,
        then runs ticks until terminal state, shutdown, or "max_ticks".

        Tests pass "max_ticks" to bound the loop deterministically and
        "catch_keyboard_interrupt=False" to surface KeyboardInterrupt
        instead of returning 130.
        """
        try:
            with self._acquire_lock():
                with self._acquire_lease():
                    self._write_pid()
                    self._install_signal_handlers()
                    # Initialise state before writing daemon_started. A
                    # pre-state journal entry would otherwise make a brand-new
                    # campaign look like a recovery case on the first tick.
                    try:
                        self._read_or_initialise_state()
                    except (StateSchemaError, json.JSONDecodeError) as exc:
                        self._journal("state_corrupt", error=str(exc)[:200])
                        print(
                            "state.json failed validation: " + str(exc)
                            + "\nRun `ichor-al-daemon reconcile --campaign-dir "
                            + str(self.campaign_dir) + "` for manual recovery.",
                            file=sys.stderr,
                        )
                        return 2
                    self._journal("daemon_started", pid=os.getpid())
                    if readiness_callback is not None:
                        readiness_callback()
                    try:
                        return self._run_loop(max_ticks=max_ticks)
                    except KeyboardInterrupt:
                        if not catch_keyboard_interrupt:
                            raise
                        self._journal("daemon_interrupted")
                        return 130
                    finally:
                        self._journal("daemon_stopped", pid=os.getpid())
                        self._remove_pid()
        except DaemonAlreadyRunningError as exc:
            print(str(exc), file=sys.stderr)
            return 11

    def _run_loop(self, *, max_ticks: Optional[int]) -> int:
        idle_streak = 0
        ticks = 0
        while True:
            if max_ticks is not None and ticks >= max_ticks:
                return 0
            ticks += 1

            if self._shutdown_requested:
                #persist the flag and exit cleanly.
                try:
                    state = read_state(self.state_path())
                    state.shutdown_requested = True
                    state.lifecycle_context = make_lifecycle_context(
                        disposition="stopped",
                        reason_code="signal_stop_request",
                        message="daemon received SIGINT or SIGTERM",
                        from_phase=state.phase,
                        iteration=int(state.iteration),
                        source="signal",
                        recovery_action="use resume to continue from the recorded phase",
                    )
                    self._persist(state)
                except (FileNotFoundError, StateSchemaError):
                    pass
                self._journal("shutdown_requested")
                return 0

            try:
                status = self.tick()
                self._assert_lease_healthy()
            except (StateSchemaError, json.JSONDecodeError) as exc:
                self._journal("state_corrupt", error=str(exc)[:200])
                print(
                    "state.json failed validation: " + str(exc)
                    + "\nRun `ichor-al-daemon reconcile --campaign-dir "
                    + str(self.campaign_dir) + "` for manual recovery.",
                    file=sys.stderr,
                )
                return 2
            except Exception as exc:
                self._journal("tick_error", error=type(exc).__name__ + ": " + str(exc)[:200])
                self._write_last_exception(exc)
                if bool(getattr(self.config.runtime, "halt_on_tick_exception", True)):
                    if self._halt_after_tick_exception(exc):
                        return 21
                raise

            if status == TickStatus.HALTED:
                return 20
            if status == TickStatus.TERMINAL:
                try:
                    terminal_state = read_state(self.state_path())
                except Exception:
                    return 22
                return 20 if terminal_state.phase is CampaignPhase.HALTED else 0
            if status == TickStatus.SHUTDOWN:
                return 0
            try:
                latest_state = read_state(self.state_path())
            except Exception:
                latest_state = None
            if latest_state is not None and bool(latest_state.shutdown_requested):
                return 0
            if status == TickStatus.POLLING:
                idle_streak += 1
            else:
                idle_streak = 0
            poll = (
                self.config.runtime.poll_interval_idle_seconds
                if idle_streak >= 3
                else self.config.runtime.poll_interval_seconds
            )
            try:
                state = read_state(self.state_path())
                self._write_lease_heartbeat(state)
            except Exception:
                self._write_lease_heartbeat()
            self.sleep_fn(float(poll))
