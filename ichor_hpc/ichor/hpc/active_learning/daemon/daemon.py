"""Long-running campaign daemon (login-node, single-process).

This module wires everything into one state machine. The
production deployment is a single Python process on a CSF4 login node that:

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
import json
import signal
import socket
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..config import CampaignConfig
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
from .state import (
    CampaignPhase,
    CampaignState,
    DEFAULT_STATE_FILENAME,
    StateSchemaError,
    atomic_write_json,
    fresh_campaign_state,
    read_state,
    write_state,
)
from . import submission_intent as _submission_intent


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

PHASE_ORDER: Tuple[CampaignPhase, ...] = (
    CampaignPhase.INIT,
    CampaignPhase.PHASE_A_POLUS,
    CampaignPhase.INITIAL_GAUSSIAN,
    CampaignPhase.INITIAL_AIMALL,
    CampaignPhase.INITIAL_FEREBUS,
    CampaignPhase.SEED_SELECT,
    CampaignPhase.ARIADNE_ARRAY,
    CampaignPhase.PHASE_B_POLUS,
    CampaignPhase.SPLIT,
    CampaignPhase.GAUSSIAN,
    CampaignPhase.AIMALL,
    CampaignPhase.APPEND,
    CampaignPhase.FEREBUS,
    CampaignPhase.STOP_CHECK,
)


class DaemonAlreadyRunningError(RuntimeError):
    """Raised when the flock cannot be acquired because another daemon
    holds it."""


class TickStatus(str):
    SUBMITTED = "SUBMITTED"
    POLLING = "POLLING"
    ADVANCED = "ADVANCED"
    SCRUBBED = "SCRUBBED"
    HALTED = "HALTED"
    RETRYING = "RETRYING"
    TERMINAL = "TERMINAL"
    SHUTDOWN = "SHUTDOWN"


TRANSIENT_RETRY_STATUSES = frozenset({
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "REVOKED",
})


def next_phase(current: CampaignPhase, iteration: int, max_iterations: int) -> Tuple[CampaignPhase, int]:
    """Return (next_phase, next_iteration).

    INIT -> PHASE_A_POLUS, then forward through PHASE_ORDER. STOP_CHECK loops
    back to SEED_SELECT with iteration+1, unless that exceeds max_iterations
    in which case the next phase is DONE.
    """
    if current is CampaignPhase.STOP_CHECK:
        next_iter = iteration + 1
        if next_iter >= max_iterations:
            return CampaignPhase.DONE, iteration
        return CampaignPhase.SEED_SELECT, next_iter
    try:
        idx = PHASE_ORDER.index(current)
    except ValueError as exc:
        raise ValueError("unexpected current phase: " + str(current)) from exc
    if idx + 1 >= len(PHASE_ORDER):
        return CampaignPhase.DONE, iteration
    return PHASE_ORDER[idx + 1], iteration


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
    # live-mode only: given a Slurm JobID, returns a small object with active,
    # inconclusive, rows and error attributes. Used to distinguish genuinely
    # missing sacct array rows from throttled jobs that are still visible in
    # squeue.
    job_liveness_checker: Optional[Callable[[str], Any]] = None

    #internal flags; not part of the public dataclass surface.
    _shutdown_requested: bool = field(default=False, init=False, repr=False)
    _lock_held: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.campaign_dir = Path(self.campaign_dir)

    # --- path helpers ---------------------------------------------------

    def data_dir(self) -> Path:
        return self.campaign_dir / DEFAULT_DATA_SUBDIR

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
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _lease_age_seconds(self) -> float:
        heartbeat = self._read_heartbeat()
        if heartbeat is not None:
            try:
                return max(0.0, time.time() - float(heartbeat.get("time", 0.0)))
            except Exception:
                pass
        try:
            return max(0.0, time.time() - self.lease_path().stat().st_mtime)
        except OSError:
            return 0.0

    @contextmanager
    def _acquire_lease(self):
        self.data_dir().mkdir(parents=True, exist_ok=True)
        lease = self.lease_path()
        stale_after = float(getattr(self.config.runtime, "lease_stale_seconds", 900))
        try:
            os.mkdir(str(lease))
        except FileExistsError:
            age = self._lease_age_seconds()
            heartbeat = self._read_heartbeat() or {}
            if age < stale_after:
                raise DaemonAlreadyRunningError(
                    "daemon lease is fresh; refuse to start "
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
            os.mkdir(str(lease))
        self._write_lease_heartbeat()
        try:
            yield
        finally:
            try:
                self.heartbeat_path().unlink()
            except OSError:
                pass
            try:
                os.rmdir(str(lease))
            except OSError:
                pass

    def _write_lease_heartbeat(self, state: Optional[CampaignState] = None) -> None:
        lease = self.lease_path()
        if not lease.is_dir():
            return
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "time": time.time(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        if state is not None:
            payload.update({
                "phase": state.phase.value,
                "iteration": int(state.iteration),
                "campaign_uid": str(getattr(state, "campaign_uid", "")),
            })
        try:
            atomic_write_json(self.heartbeat_path(), payload)
        except Exception:
            pass

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
        #ensure the data dir exists before any state IO. run() also does thi
        #via _acquire_lock(), but tick() can be called directly in tests.
        self.data_dir().mkdir(parents=True, exist_ok=True)
        sp = self.state_path()
        if not sp.exists():
            state = fresh_campaign_state(max_iterations=self.config.max_iterations)
            write_state(sp, state)
            self._journal(
                "campaign_started",
                campaign_uid=state.campaign_uid,
                max_iterations=state.max_iterations,
            )
            return state
        return read_state(sp)

    def _journal(self, event_type: str, **payload: Any) -> None:
        try:
            append_event(self.journal_path(), event_type, **payload)
        except Exception:
            # Journal writes are best-effort; never let logging crash the daemon.
            pass

    # --- state-machine step --------------------------------------------

    def tick(self) -> str:
        """Run one state-machine step. Returns a TickStatus string.

        Idempotent on the state file: kill -9 between any two ticks (or any
        time before the final write_state inside this method) leaves the
        on-disk state at the previous snapshot, so the next start re-runs
        the same step.
        """
        state = self._read_or_initialise_state()
        self._write_lease_heartbeat(state)

        if state.shutdown_requested:
            return TickStatus.SHUTDOWN
        if state.is_terminal:
            return TickStatus.TERMINAL

        phase = state.phase
        pending = state.pending_jobs.get(phase.value)

        if pending is None:
            return self._on_phase_entry(state, phase)
        return self._on_pending(state, phase, pending)

    def _verify_committed_artifacts_if_enabled(
        self,
        state: CampaignState,
        phase: CampaignPhase,
    ) -> Optional[str]:
        if not bool(getattr(self.executor, "strict_committed_artifact_verification", False)):
            return None
        try:
            from .artifact_contracts import verify_state_referenced_artifacts
            verify_state_referenced_artifacts(
                self.campaign_dir,
                state,
                strict_models=True,
            )
        except Exception as exc:
            return self._halt(
                state,
                phase,
                "committed_artifact_contract_invalid: "
                + type(exc).__name__
                + ": "
                + str(exc)[:180],
            )
        return None

    def _on_phase_entry(self, state: CampaignState, phase: CampaignPhase) -> str:
        """Called once when entering a phase with no pending JobID."""
        phase_name = phase.value
        verify_status = self._verify_committed_artifacts_if_enabled(state, phase)
        if verify_status is not None:
            return verify_status
        active_intent = None
        if phase_name in SBATCH_PHASES:
            try:
                active_intent = _submission_intent.load_active_intent(
                    self.campaign_dir, phase_name, int(state.iteration),
                )
            except Exception as exc:
                return self._halt(
                    state,
                    phase,
                    "submission_intent_invalid: "
                    + type(exc).__name__ + ": " + str(exc)[:160],
                )
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
                    return self._halt(
                        state,
                        phase,
                        "active_submission_adoption_check_failed: "
                        + type(exc).__name__ + ": " + str(exc)[:160],
                    )
                existing = None
                lookup_inconclusive = True
            if existing:
                existing_job_id = str(existing)
                state.pending_jobs[phase_name] = existing_job_id
                try:
                    _submission_intent.mark_adopted(
                        self.campaign_dir,
                        phase_name,
                        int(state.iteration),
                        existing_job_id,
                        expected_tasks=(
                            None if active_intent is None
                            else active_intent.get("expected_tasks")
                        ),
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
                )
                return TickStatus.SUBMITTED
            if active_intent is not None and lookup_inconclusive:
                return self._halt(
                    state,
                    phase,
                    "active_submission_adoption_inconclusive: "
                    "sacct lookup failed or was inconclusive; refusing to supersede active intent",
                )
        if active_intent is not None and phase_name in SBATCH_PHASES:
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
                _submission_intent.write_pre_submit_intent(
                    self.campaign_dir,
                    campaign_uid=str(getattr(state, "campaign_uid", "")),
                    phase_name=phase_name,
                    iteration=int(state.iteration),
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
        except BackendSubmissionError as exc:
            #a backend submission (sbatch) failed outright. halt cleanly so an
            #operator can look, instead of letting it bubble up and take the
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
                try:
                    _submission_intent.mark_submitted(
                        self.campaign_dir,
                        phase_name,
                        int(state.iteration),
                        result.submitted_job_id,
                        expected_tasks=result.expected_tasks,
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
            )
            return TickStatus.SUBMITTED
        if result.is_complete:
            #inline phase, or executor decided no submission needed.
            return (
                TickStatus.ADVANCED
                if self._advance(state, phase, result.state_updates)
                else TickStatus.HALTED
            )
        #defensive: executor returned neither a JobID nor completion.
        raise RuntimeError(
            "executor returned no submitted_job_id and is_complete=False for "
            + phase.value
        )

    def _on_pending(self, state: CampaignState, phase: CampaignPhase, job_id: str) -> str:
        """Called while a SLURM job for `phase` is in flight."""
        try:
            observations = self.sacct_poller(job_id)
        except RuntimeError as exc:
            # sacct hiccup -- log and keep polling next tick. Don't escalate
            #immediately because transient sacct failures are common.
            self._journal(
                "sacct_error", phase=phase.value, job_id=job_id,
                error=str(exc)[:200],
            )
            return TickStatus.POLLING

        expected_tasks = self._expected_tasks_for_pending(state, phase, job_id)
        summary = aggregate_states(
            job_id,
            observations,
            expected_task_count=expected_tasks,
        )
        #Detect empty sacct response BEFORE checking is_terminal --
        #an empty observations list is the "accounting aged out" signature
        #we want to escalate after a streak. is_terminal is False for n=0
        #so without this gate the streak path is unreachable.
        if not observations or summary.n_tasks == 0:
            current = state.sacct_empty_streak.get(job_id, 0) + 1
            state.sacct_empty_streak[job_id] = current
            self._persist(state)
            max_ticks = int(getattr(self.config, "poll_sacct_empty_max_ticks", 10))
            if max_ticks > 0 and current >= max_ticks:
                self._journal(
                    "sacct_empty_timeout",
                    phase=phase.value, job_id=job_id,
                    streak=int(current), max_ticks=int(max_ticks),
                    iteration=state.iteration,
                )
                #treat as a full failure of every task in the array. Route
                #through the standard failure handler so executor policy
                #(SCRUB_AND_CONTINUE vs HALT) still applies.
                return self._handle_failure(state, phase, observations, summary)
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
                self._journal(
                    "sacct_unknown_timeout",
                    phase=phase.value,
                    job_id=job_id,
                    streak=int(current),
                    max_ticks=int(max_unknown),
                    iteration=state.iteration,
                )
                return self._halt(
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
            liveness = self._check_job_liveness(job_id)
            if liveness is not None and (
                bool(getattr(liveness, "active", False))
                or bool(getattr(liveness, "inconclusive", False))
            ):
                if missing_key in state.sacct_empty_streak:
                    state.sacct_empty_streak.pop(missing_key, None)
                    self._persist(state)
                if bool(getattr(liveness, "active", False)):
                    self._journal(
                        "sacct_rows_missing_but_squeue_active",
                        phase=phase.value,
                        job_id=job_id,
                        n_expected=getattr(summary, "n_expected", None),
                        n_observed=int(getattr(summary, "n_observed", 0)),
                        n_missing=int(getattr(summary, "n_missing", 0)),
                        squeue_rows_sample=self._queue_rows_sample(liveness),
                        iteration=state.iteration,
                    )
                else:
                    self._journal(
                        "squeue_liveness_inconclusive",
                        phase=phase.value,
                        job_id=job_id,
                        n_expected=getattr(summary, "n_expected", None),
                        n_observed=int(getattr(summary, "n_observed", 0)),
                        n_missing=int(getattr(summary, "n_missing", 0)),
                        error=str(getattr(liveness, "error", "") or "")[:200],
                        iteration=state.iteration,
                    )
                return TickStatus.POLLING
            current = state.sacct_empty_streak.get(missing_key, 0) + 1
            state.sacct_empty_streak[missing_key] = current
            self._persist(state)
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
                    iteration=state.iteration,
                )
                return self._halt(
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

        #Job has reached terminal state(s). Decide between postprocess and
        #failure handling based on the success ratio.

        success_ratio = summary.n_completed / summary.n_tasks
        failure_threshold = 1.0 - self.config.failure_threshold_fraction

        if summary.is_fully_successful or success_ratio >= failure_threshold:
            return self._postprocess(state, phase, observations, summary)
        return self._handle_failure(state, phase, observations, summary)

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
            return None
        recorded_job = data.get("job_id")
        if recorded_job is not None and str(recorded_job) != str(job_id):
            return None
        raw = data.get("expected_tasks")
        if raw is None:
            return None
        try:
            expected = int(raw)
        except (TypeError, ValueError):
            self._journal(
                "submission_intent_expected_tasks_invalid",
                phase=phase.value,
                iteration=int(state.iteration),
                value=repr(raw),
            )
            return None
        return expected if expected > 0 else None

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

    def _postprocess(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        observations: Sequence[JobObservation],
        summary,
    ) -> str:
        attempts = max(1, int(getattr(self.config.runtime, "postprocess_settle_attempts", 3)))
        settle_seconds = max(0, int(getattr(self.config.runtime, "postprocess_settle_seconds", 10)))
        result = None
        for attempt in range(attempts):
            try:
                result = self.executor.postprocess(state, phase, observations)
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
            return self._halt(state, phase, "postprocess_failed_without_result")
        if result.failure_reason:
            return self._halt(state, phase, result.failure_reason)
        self._clear_sacct_streaks(state, str(summary.parent_job_id))
        contract_error = self._transition_output_contract_error(
            state,
            phase,
            result.state_updates,
        )
        if contract_error is not None:
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
            try:
                _submission_intent.mark_completed(
                    self.campaign_dir, phase.value, completed_iteration,
                )
            except Exception as exc:
                self._journal(
                    "submission_intent_update_failed",
                    phase=phase.value,
                    iteration=completed_iteration,
                    error=str(exc)[:200],
                )
        #clear the pending job and advance.
        state.pending_jobs[phase.value] = None
        self._journal(
            "phase_succeeded", phase=phase.value, iteration=state.iteration,
            n_completed=summary.n_completed, n_failed=summary.n_failed,
            n_tasks=summary.n_tasks,
        )
        return (
            TickStatus.ADVANCED
            if self._advance(state, phase, result.state_updates)
            else TickStatus.HALTED
        )

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
        self._journal(
            "failure_action",
            phase=phase.value,
            iteration=state.iteration,
            action=str(action.value if hasattr(action, "value") else action),
            failure_indices=list(summary.failure_indices),
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
        return TickStatus.SCRUBBED if self._advance(state, phase, {}) else TickStatus.HALTED

    def _clear_sacct_streaks(self, state: CampaignState, job_id: str) -> None:
        state.sacct_empty_streak.pop(str(job_id), None)
        state.sacct_empty_streak.pop(str(job_id) + ":UNKNOWN", None)
        state.sacct_empty_streak.pop(str(job_id) + ":MISSING", None)

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
            return {"schema_version": 1, "attempts": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": 1, "attempts": {}}
        if not isinstance(data, dict) or int(data.get("schema_version", -1)) != 1:
            return {"schema_version": 1, "attempts": {}}
        attempts = data.get("attempts")
        if not isinstance(attempts, dict):
            attempts = {}
        return {"schema_version": 1, "attempts": attempts}

    def _retry_key(self, phase: CampaignPhase, iteration: int) -> str:
        return phase.value + "@" + str(int(iteration))

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
        ledger = self._load_transient_retry_ledger()
        attempts = ledger.get("attempts", {})
        key = self._retry_key(phase, int(state.iteration))
        return int(attempts.get(key, 0)) < max_retry

    def _retry_transient_phase(
        self,
        state: CampaignState,
        phase: CampaignPhase,
        observations: Sequence[JobObservation],
        summary,
    ) -> str:
        ledger = self._load_transient_retry_ledger()
        attempts = dict(ledger.get("attempts", {}))
        key = self._retry_key(phase, int(state.iteration))
        attempt = int(attempts.get(key, 0)) + 1
        attempts[key] = attempt
        atomic_write_json(
            self.transient_retry_ledger_path(),
            {"schema_version": 1, "attempts": attempts},
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
        self._journal(
            "transient_phase_retry",
            phase=phase.value,
            iteration=state.iteration,
            attempt=int(attempt),
            statuses=statuses,
            failure_indices=list(summary.failure_indices),
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
        state.phase = CampaignPhase.HALTED
        self._journal("halt", from_phase=phase.value, reason=reason,
                      iteration=state.iteration)
        self._persist(state)
        return TickStatus.HALTED

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

        raw_version = state_updates.get("models_version", getattr(state, "models_version", -1))
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

        expected_path = (
            self.campaign_dir
            / "6_TRAINED_MODELS"
            / ("iteration-" + str(models_version).zfill(4))
        )
        try:
            from .artifact_contracts import verify_committed_model_version
            verify_committed_model_version(self.campaign_dir, models_version)
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

    def _advance(self, state: CampaignState, phase: CampaignPhase, state_updates: Dict[str, Any]) -> bool:
        #NB:STOP_CHECK ghost-iteration fix. When _inline_stop_check
        #returns {"shutdown_requested": True}, the daemon must NOT also
        #advance phase + iteration -- otherwise state.json snapshots show
        #"iteration N+1 / SEED_SELECT, shutdown_requested=True" which is
        #confusing and produces a stale "in-flight" appearance on restart.
        #Persist the flag + journal a shutdown_requested event; next tick
        #exits cleanly via state.shutdown_requested.
        if bool(state_updates.get("shutdown_requested", False)):
            self._apply_state_updates(state, state_updates)
            self._persist(state)
            self._journal(
                "shutdown_requested", from_phase=phase.value,
                iteration=int(state.iteration),
            )
            return True

        contract_error = self._transition_output_contract_error(
            state,
            phase,
            state_updates,
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
        prior_iter = state.iteration
        state.phase = new_phase
        state.iteration = new_iter
        self._apply_state_updates(state, state_updates)
        #NB:persist before journal -- state.json is authoritative; the
        # journal is best-effort observability. A persist failure here would
        #otherwise leave the journal claiming a transition that did not
        # actually happen on disk, triggering duplicate work on restart.
        self._persist(state)
        self._journal(
            "phase_transition", from_phase=phase.value, to_phase=new_phase.value,
            iteration=new_iter, prior_iteration=prior_iter,
        )
        return True

    def _apply_state_updates(self, state: CampaignState, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed = {
            "training_set_version", "validation_set_version", "models_version",
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
            except (FileNotFoundError, StateSchemaError):
                on_disk = None
            if on_disk is not None and bool(on_disk.shutdown_requested):
                state.shutdown_requested = True
        write_state(self.state_path(), state)

    # --- public lifecycle ----------------------------------------------

    def request_shutdown(self) -> None:
        """Set the in-process shutdown flag and persist it to disk.

        Callers from outside the daemon process should write
        shutdown_requested = true directly into state.json via
        :func: "write_state" so the running daemon picks it up on its next
        tick. This in-process method is for signal handlers and tests.
        """
        self._shutdown_requested = True
        try:
            state = read_state(self.state_path())
            state.shutdown_requested = True
            self._persist(state)
        except (FileNotFoundError, StateSchemaError):
            pass

    def run(
        self,
        *,
        max_ticks: Optional[int] = None,
        catch_keyboard_interrupt: bool = True,
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
                    self._journal("daemon_started", pid=os.getpid())
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
                    self._persist(state)
                except (FileNotFoundError, StateSchemaError):
                    pass
                self._journal("shutdown_requested")
                return 0

            try:
                status = self.tick()
            except StateSchemaError as exc:
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
                raise

            if status in (TickStatus.TERMINAL, TickStatus.HALTED, TickStatus.SHUTDOWN):
                return 0
            if status == TickStatus.POLLING:
                idle_streak += 1
            else:
                idle_streak = 0
            poll = (
                self.config.poll_interval_idle_seconds
                if idle_streak >= 3
                else self.config.poll_interval_seconds
            )
            try:
                state = read_state(self.state_path())
                self._write_lease_heartbeat(state)
            except Exception:
                self._write_lease_heartbeat()
            self.sleep_fn(float(poll))
