"""SLURM job-state polling via SLURM's sacct.

The daemon submits child jobs (FEREBUS, Gaussian array, AIMAll array, ARIADNE
array) with "sbatch --parsable" (see :mod: .pyferebus_wrap) and tracks them
by JobID. This module parses "sacct" output into a structured
:class:"JobStatus" so the state machine can branch on terminal outcomes.

Typical command:

    sacct -j <id> --format=JobID,State,ExitCode,Elapsed -X -P -n

  - '-X' excludes ".batch" / ".extern" sub-steps (we want job-level state).
  - '-P' uses pipe-delimited output (no decorative padding).
  - '-n' omits the header row.

For a single job the parsed state is straightforward. For an array job, sacct
returns one row per task plus a parent summary row. :func: 'aggregate_states'
collapses the array into a single :class: 'ArrayJobSummary' that the daemon
uses to decide between "scrub one task" vs "abort the iteration".
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .slurm_contracts import (
    parse_job_row_id,
    parse_squeue_job_id,
    run_scheduler_command,
    validate_parent_job_id,
)


__all__ = [
    "JobStatus",
    "JobObservation",
    "ArrayJobSummary",
    "JobQueueLookup",
    "TERMINAL_STATES",
    "NON_TERMINAL_STATES",
    "SUCCESS_STATES",
    "FAILURE_STATES",
    "parse_sacct_output",
    "aggregate_states",
    "poll_job",
    "find_active_job_by_id_detailed",
    "JobNameLookup",
    "find_active_job_by_name_detailed",
    "JobNameAccountingLookup",
    "find_accounted_job_by_name_detailed",
    "find_running_job_by_name_detailed",
    "find_running_job_by_name",
]


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CONFIGURING = "CONFIGURING"
    COMPLETING = "COMPLETING"
    REQUEUED = "REQUEUED"
    RESIZING = "RESIZING"
    STAGE_OUT = "STAGE_OUT"
    SUSPENDED = "SUSPENDED"
    EXPEDITING = "EXPEDITING"
    POWER_UP_NODE = "POWER_UP_NODE"
    REQUEUE_FED = "REQUEUE_FED"
    REQUEUE_HOLD = "REQUEUE_HOLD"
    RESV_DEL_HOLD = "RESV_DEL_HOLD"
    SIGNALING = "SIGNALING"
    UPDATE_DB = "UPDATE_DB"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    CANCELLED = "CANCELLED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    PREEMPTED = "PREEMPTED"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    REVOKED = "REVOKED"
    SPECIAL_EXIT = "SPECIAL_EXIT"
    STOPPED = "STOPPED"
    LAUNCH_FAILED = "LAUNCH_FAILED"
    RECONFIG_FAIL = "RECONFIG_FAIL"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_sacct(cls, raw: str) -> "JobStatus":
        #sacct often emits "CANCELLED+" or "CANCELLED by 12345".
        token = raw.strip().split()[0].upper() if raw and raw.strip() else ""
        if token == "CANCELLED+":
            token = "CANCELLED"
        elif token.endswith("+"):
            return cls.UNKNOWN
        try:
            return cls(token)
        except ValueError:
            return cls.UNKNOWN


NON_TERMINAL_STATES = frozenset({
    JobStatus.PENDING,
    JobStatus.RUNNING,
    JobStatus.CONFIGURING,
    JobStatus.COMPLETING,
    JobStatus.REQUEUED,
    JobStatus.RESIZING,
    JobStatus.STAGE_OUT,
    JobStatus.SUSPENDED,
    JobStatus.EXPEDITING,
    JobStatus.POWER_UP_NODE,
    JobStatus.REQUEUE_FED,
    JobStatus.REQUEUE_HOLD,
    JobStatus.RESV_DEL_HOLD,
    JobStatus.SIGNALING,
    JobStatus.UPDATE_DB,
    JobStatus.SPECIAL_EXIT,
    JobStatus.STOPPED,
})

TERMINAL_STATES = frozenset({
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.TIMEOUT,
    JobStatus.NODE_FAIL,
    JobStatus.CANCELLED,
    JobStatus.OUT_OF_MEMORY,
    JobStatus.PREEMPTED,
    JobStatus.BOOT_FAIL,
    JobStatus.DEADLINE,
    JobStatus.REVOKED,
    JobStatus.LAUNCH_FAILED,
    JobStatus.RECONFIG_FAIL,
})

SUCCESS_STATES = frozenset({JobStatus.COMPLETED})

FAILURE_STATES = TERMINAL_STATES - SUCCESS_STATES





@dataclass(frozen=True)
class JobObservation:
    """One sacct row, normalised."""

    job_id: str
    status: JobStatus
    exit_code: Optional[Tuple[int, int]]  #(returncode, signal)
    elapsed_seconds: Optional[int]
    raw_status: Optional[str] = None
    parse_error: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES

    @property
    def is_success(self) -> bool:
        return self.status in SUCCESS_STATES and self.exit_code == (0, 0)

    @property
    def is_failure(self) -> bool:
        return self.status in FAILURE_STATES or (
            self.status in TERMINAL_STATES and not self.is_success
        )


@dataclass(frozen=True)
class ArrayJobSummary:
    """Aggregated view of an array job's task observations."""

    parent_job_id: str
    observations: List[JobObservation]
    n_tasks: int
    n_completed: int
    n_failed: int
    n_pending_or_running: int
    n_unknown: int = 0
    n_expected: Optional[int] = None
    n_observed: int = 0
    n_missing: int = 0
    failure_indices: List[int] = field(default_factory=list)
    conflicting_task_indices: List[int] = field(default_factory=list)
    out_of_range_task_indices: List[int] = field(default_factory=list)
    malformed_job_ids: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return (
            self.n_pending_or_running == 0
            and self.n_unknown == 0
            and not self.conflicting_task_indices
            and not self.out_of_range_task_indices
            and self.n_tasks > 0
        )

    @property
    def is_fully_successful(self) -> bool:
        return self.is_terminal and self.n_failed == 0 and self.n_completed == self.n_tasks


@dataclass(frozen=True)
class JobQueueLookup:
    """Best-effort squeue liveness check for an existing Slurm job id."""

    active: bool
    inconclusive: bool = False
    rows: List[Tuple[str, str]] = field(default_factory=list)
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.active)


def _parse_elapsed(text: str) -> Optional[int]:
    """Convert sacct Elapsed string (HH:MM:SS or DD-HH:MM:SS) to seconds."""
    text = text.strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        days_str, rest = text.split("-", 1)
        try:
            days = int(days_str)
        except ValueError:
            return None
    else:
        rest = text
    parts = rest.split(":")
    try:
        if len(parts) == 3:
            h, m, s = (int(p) for p in parts)
        elif len(parts) == 2:
            h = 0
            m, s = (int(p) for p in parts)
        elif len(parts) == 1:
            h = m = 0
            s = int(parts[0])
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def _parse_exit_code(text: str) -> Optional[Tuple[int, int]]:
    text = text.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        return None
    ret, sig = int(parts[0]), int(parts[1])
    if ret > 255 or sig > 255:
        return None
    return (ret, sig)


def parse_sacct_output(stdout: str) -> List[JobObservation]:
    """Parse pipe-delimited ('-P') sacct output into a list of JobObservation.

    Expects the four columns: JobID|State|ExitCode|Elapsed (in this order).
    Rows whose JobID is empty are skipped silently. Unknown JobStatus values
    map to JobStatus.UNKNOWN rather than raising, so the caller can decide
    how to handle truly weird output.
    """
    observations: List[JobObservation] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 4:
            continue
        job_id, state, exit_code, elapsed = parts[0].strip(), parts[1], parts[2], parts[3]
        if not job_id:
            continue
        status = JobStatus.from_sacct(state)
        parsed_exit = _parse_exit_code(exit_code)
        parse_errors: List[str] = []
        try:
            parse_job_row_id(job_id)
        except ValueError as exc:
            parse_errors.append(str(exc))
        if status in TERMINAL_STATES and parsed_exit is None:
            parse_errors.append("terminal Slurm row has malformed ExitCode")
        if status is JobStatus.UNKNOWN:
            parse_errors.append("unrecognised or truncated Slurm state " + repr(state.strip()))
        if parse_errors:
            status = JobStatus.UNKNOWN
        observations.append(JobObservation(
            job_id=job_id,
            status=status,
            exit_code=parsed_exit,
            elapsed_seconds=_parse_elapsed(elapsed),
            raw_status=state.strip(),
            parse_error="; ".join(parse_errors) or None,
        ))
    return observations


def aggregate_states(
    parent_job_id: str,
    observations: Sequence[JobObservation],
    expected_task_count: Optional[int] = None,
    *,
    submission_kind: Optional[str] = None,
    strict_parent_job_id: bool = True,
) -> ArrayJobSummary:
    """Collapse per-task observations into one summary.

    Task rows have job_id like '<parent>_<task_index>'. We pick out tasks
    belonging to 'parent_job_id' and count terminal vs non-terminal states.
    """
    if expected_task_count is not None and (
        isinstance(expected_task_count, bool)
        or not isinstance(expected_task_count, int)
        or expected_task_count <= 0
    ):
        raise ValueError("expected_task_count must be an exact positive integer")
    expected = expected_task_count
    if strict_parent_job_id:
        parent = validate_parent_job_id(parent_job_id)
    else:
        parent = str(parent_job_id)
        if not parent:
            raise ValueError("synthetic parent job identity must be non-empty")
    prefix = parent + "_"
    parent_rows = [o for o in observations if o.job_id == parent]
    saw_array_row = any(o.job_id.startswith(prefix) for o in observations)
    grouped: Dict[int, List[JobObservation]] = {}
    out_of_range: List[int] = []
    malformed_job_ids: List[str] = []
    for observation in observations:
        try:
            if strict_parent_job_id:
                row_parent, task_index, step = parse_job_row_id(observation.job_id)
            else:
                row_id = str(observation.job_id)
                if row_id == parent:
                    row_parent, task_index, step = parent, None, None
                elif row_id.startswith(prefix) and row_id[len(prefix):].isdigit():
                    row_parent = parent
                    task_index = int(row_id[len(prefix):])
                    step = None
                else:
                    raise ValueError("synthetic accounting row identity mismatch")
        except ValueError:
            if str(observation.job_id).startswith(parent):
                malformed_job_ids.append(str(observation.job_id))
            continue
        if row_parent != parent or step is not None or task_index is None:
            continue
        task_id = int(task_index)
        if expected is not None and not 0 <= task_id < expected:
            out_of_range.append(task_id)
            continue
        grouped.setdefault(task_id, []).append(observation)

    collapsed: List[JobObservation] = []
    conflicts: List[int] = []
    for task_id in sorted(grouped):
        rows = grouped[task_id]
        outcomes = {(row.status, row.exit_code) for row in rows}
        if len(outcomes) != 1:
            conflicts.append(task_id)
            collapsed.append(
                JobObservation(
                    job_id=prefix + str(task_id),
                    status=JobStatus.UNKNOWN,
                    exit_code=None,
                    elapsed_seconds=max(
                        (int(row.elapsed_seconds) for row in rows if row.elapsed_seconds is not None),
                        default=None,
                    ),
                )
            )
            continue
        exemplar = rows[0]
        collapsed.append(
            JobObservation(
                job_id=prefix + str(task_id),
                status=exemplar.status,
                exit_code=exemplar.exit_code,
                elapsed_seconds=max(
                    (int(row.elapsed_seconds) for row in rows if row.elapsed_seconds is not None),
                    default=None,
                ),
            )
        )

    # A non-array job is represented by its parent allocation row. For an
    # array, parent summaries are never allowed to stand in for logical tasks.
    if submission_kind not in {None, "scalar", "array"}:
        raise ValueError("submission_kind must be scalar, array, or null")
    synthetic_single_parent = bool(
        not strict_parent_job_id
        and submission_kind == "array"
        and expected in {None, 1}
        and parent_rows
        and not grouped
    )
    use_task_rows = (
        (
            submission_kind == "array"
            and not synthetic_single_parent
        )
        or bool(grouped)
        or saw_array_row
        or bool(expected is not None and expected > 1)
    )
    task_obs = collapsed if use_task_rows else list(parent_rows[:1])
    observed_count = len(collapsed) if use_task_rows else len(task_obs)
    missing = 0
    if expected is not None and expected > observed_count:
        missing = expected - observed_count
    n_completed = sum(1 for o in task_obs if o.is_success)
    n_failed = sum(1 for o in task_obs if o.is_failure)
    unique_out_of_range = sorted(set(out_of_range))
    n_unknown = (
        sum(1 for o in task_obs if o.status == JobStatus.UNKNOWN)
        + len(unique_out_of_range)
        + len(malformed_job_ids)
    )
    n_pending = sum(1 for o in task_obs if not o.is_terminal) + missing
    failure_indices: List[int] = []
    for o in task_obs:
        if o.is_failure and "_" in o.job_id:
            suffix = o.job_id.rsplit("_", 1)[1]
            try:
                failure_indices.append(int(suffix))
            except ValueError:
                continue
    return ArrayJobSummary(
        parent_job_id=str(parent_job_id),
        observations=list(task_obs),
        n_tasks=(expected if expected is not None else observed_count),
        n_completed=n_completed,
        n_failed=n_failed,
        n_pending_or_running=n_pending,
        n_unknown=n_unknown,
        n_expected=expected,
        n_observed=observed_count,
        n_missing=missing,
        failure_indices=sorted(failure_indices),
        conflicting_task_indices=sorted(conflicts),
        out_of_range_task_indices=unique_out_of_range,
        malformed_job_ids=sorted(set(malformed_job_ids)),
    )



def poll_job(
    job_id: str,
    *,
    sacct_runner: Optional[Callable[..., Any]] = None,
    extra_args: Sequence[str] = (),
    timeout_seconds: int = 60,
) -> List[JobObservation]:
    """Run 'sacct -j <id> ...' and return the parsed observations.

    'sacct_runner' defaults to 'subprocess.run'; tests pass a stub that
    returns a pre-baked CompletedProcess-like object. Raises RuntimeError
    on non-zero sacct exit so the daemon's polling loop can back off.
    """
    if sacct_runner is None:
        sacct_runner = subprocess.run
    cmd = [
        "sacct",
        "-j", str(job_id),
        "--format=JobIDRaw,State%40,ExitCode,ElapsedRaw",
        "--array", "-X", "-P", "-n",
    ] + list(extra_args)
    try:
        completed = run_scheduler_command(
            sacct_runner,
            cmd,
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("sacct command timed out") from exc
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        stderr = getattr(completed, "stderr", "") or ""
        raise RuntimeError(
            "sacct exited with code " + str(return_code) + ": " + repr(stderr)
        )
    stdout = getattr(completed, "stdout", "") or ""
    return parse_sacct_output(stdout)


def _squeue_invalid_job_id(stderr: str) -> bool:
    text = str(stderr or "").lower()
    return "invalid job id specified" in text or "invalid job id" in text


def _base_allocation_id(job_id: str) -> str:
    return str(job_id).split("_", 1)[0].split(".", 1)[0]


def find_active_job_by_id_detailed(
    job_id: str,
    *,
    squeue_runner: Optional[Callable[..., Any]] = None,
    timeout_seconds: int = 60,
) -> JobQueueLookup:
    """Return whether ``squeue`` still shows a Slurm job or array as active.

    ``sacct`` can lag behind throttled array jobs on CSF3/CSF4: pending array
    elements may still be visible in ``squeue`` while their task rows are not
    yet present in accounting.  The daemon uses this as a liveness guard so
    sparse accounting rows do not falsely kill a valid campaign.
    """
    if squeue_runner is None:
        squeue_runner = subprocess.run
    cmd = [
        "squeue",
        "-j", str(job_id),
        "--noheader",
        "--format=%i|%T",
    ]
    try:
        requested_parent = validate_parent_job_id(job_id)
        completed = run_scheduler_command(
            squeue_runner,
            cmd,
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return JobQueueLookup(
            active=False,
            inconclusive=True,
            error=type(exc).__name__ + ": " + str(exc),
        )
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        stderr = getattr(completed, "stderr", "") or ""
        if _squeue_invalid_job_id(stderr):
            return JobQueueLookup(active=False, inconclusive=False, rows=[])
        return JobQueueLookup(
            active=False,
            inconclusive=True,
            error="squeue exited with code " + str(return_code) + ": " + repr(stderr),
        )
    stdout = getattr(completed, "stdout", "") or ""
    rows: List[Tuple[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            return JobQueueLookup(active=False, inconclusive=True, rows=rows, error="malformed squeue row")
        jid = parts[0].strip()
        if not jid:
            continue
        try:
            row_parent = parse_squeue_job_id(jid)
        except ValueError as exc:
            return JobQueueLookup(active=False, inconclusive=True, rows=rows, error=str(exc))
        if row_parent != requested_parent:
            return JobQueueLookup(
                active=False,
                inconclusive=True,
                rows=rows + [(jid, parts[1].strip())],
                error="squeue returned a foreign JobID for " + requested_parent,
            )
        state = parts[1].strip() if len(parts) > 1 else ""
        rows.append((jid, state))
    return JobQueueLookup(active=bool(rows), rows=rows)


@dataclass(frozen=True)
class JobNameLookup:
    job_id: Optional[str]
    inconclusive: bool = False
    rows: List[Tuple[str, str]] = field(default_factory=list)
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.job_id)


@dataclass(frozen=True)
class JobNameAccountingLookup:
    """Accounting lookup for one expected Slurm job name.

    This is stricter than the normal running-job lookup. It may return a
    terminal allocation as well as an active one, so daemon crash recovery can
    avoid resubmitting a job whose JobID was never persisted.
    """

    job_id: Optional[str]
    terminal: bool = False
    successful: bool = False
    failed: bool = False
    inconclusive: bool = False
    rows: List[Tuple[str, str]] = field(default_factory=list)
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.job_id)


def find_active_job_by_name_detailed(
    name: str,
    *,
    squeue_runner: Optional[Callable[..., Any]] = None,
    timeout_seconds: int = 60,
) -> JobNameLookup:
    """Find an active Slurm job by name using ``squeue``.

    This is a liveness fallback for the short window where ``sbatch`` has
    accepted a job, but ``sacct`` has not yet made the job visible by name.
    """
    if squeue_runner is None:
        squeue_runner = subprocess.run
    cmd = [
        "squeue",
        "--name", str(name),
        "--noheader",
        "--format=%i|%T|%j",
    ]
    try:
        completed = run_scheduler_command(
            squeue_runner,
            cmd,
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return JobNameLookup(
            None,
            inconclusive=True,
            error=type(exc).__name__ + ": " + str(exc),
        )
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        stderr = getattr(completed, "stderr", "") or ""
        return JobNameLookup(
            None,
            inconclusive=True,
            error="squeue exited with code " + str(return_code) + ": " + repr(stderr),
        )
    rows: List[Tuple[str, str]] = []
    active_ids = set()
    for line in (getattr(completed, "stdout", "") or "").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        job_id = parts[0].strip() if len(parts) > 0 else ""
        state = parts[1].strip() if len(parts) > 1 else ""
        job_name = parts[2].strip() if len(parts) > 2 else ""
        if not job_id:
            continue
        try:
            base_id = parse_squeue_job_id(job_id)
        except ValueError as exc:
            return JobNameLookup(None, inconclusive=True, rows=rows, error=str(exc))
        if job_name and job_name != str(name):
            continue
        rows.append((job_id, state))
        active_ids.add(base_id)
    if not active_ids:
        return JobNameLookup(None, inconclusive=False, rows=rows)
    if len(active_ids) > 1:
        return JobNameLookup(
            None,
            inconclusive=True,
            rows=rows,
            error="multiple active jobs share expected name: "
            + repr(sorted(active_ids)),
        )
    return JobNameLookup(sorted(active_ids)[0], inconclusive=False, rows=rows)


def find_accounted_job_by_name_detailed(
    name: str,
    *,
    expected_task_count: Optional[int] = None,
    sacct_runner: Optional[Callable[..., Any]] = None,
    squeue_runner: Optional[Callable[..., Any]] = None,
    use_squeue_fallback: bool = False,
    submission_kind: Optional[str] = None,
    timeout_seconds: int = 60,
) -> JobNameAccountingLookup:
    """Return active or terminal accounting evidence for one expected job name."""
    if sacct_runner is None:
        sacct_runner = subprocess.run
    cmd = [
        "sacct", "--name", str(name),
        "--format=JobIDRaw,State%40,ExitCode,ElapsedRaw", "--array", "-X", "-P", "-n",
    ]
    try:
        completed = run_scheduler_command(
            sacct_runner,
            cmd,
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            error=type(exc).__name__ + ": " + str(exc),
        )
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        stderr = getattr(completed, "stderr", "") or ""
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            error="sacct exited with code " + str(return_code) + ": " + repr(stderr),
        )
    observations = parse_sacct_output(getattr(completed, "stdout", "") or "")
    rows = [(str(o.job_id), str(o.status.value)) for o in observations]
    base_ids = sorted({_base_allocation_id(o.job_id) for o in observations})
    if not base_ids:
        if use_squeue_fallback:
            active = find_active_job_by_name_detailed(
                name,
                squeue_runner=squeue_runner,
                timeout_seconds=int(timeout_seconds),
            )
            if active.job_id:
                return JobNameAccountingLookup(
                    active.job_id,
                    terminal=False,
                    inconclusive=False,
                    rows=list(active.rows),
                )
            if active.inconclusive:
                return JobNameAccountingLookup(
                    None,
                    inconclusive=True,
                    rows=list(active.rows),
                    error=active.error,
                )
        return JobNameAccountingLookup(None, rows=rows)
    if len(base_ids) > 1:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            rows=rows,
            error="multiple jobs share expected name: " + repr(base_ids),
        )
    job_id = base_ids[0]
    summary = aggregate_states(
        job_id,
        observations,
        expected_task_count=expected_task_count,
        submission_kind=submission_kind,
    )
    if use_squeue_fallback and (
        expected_task_count is None or int(summary.n_missing) > 0
    ):
        active = find_active_job_by_name_detailed(
            name,
            squeue_runner=squeue_runner,
            timeout_seconds=int(timeout_seconds),
        )
        if active.job_id:
            return JobNameAccountingLookup(
                active.job_id,
                terminal=False,
                inconclusive=False,
                rows=list(active.rows),
            )
        if active.inconclusive:
            return JobNameAccountingLookup(
                None,
                inconclusive=True,
                rows=rows + list(active.rows),
                error=active.error,
            )
    if int(summary.n_unknown) > 0:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            rows=rows,
            error="sacct returned UNKNOWN rows for expected job name",
        )
    if int(summary.n_tasks) <= 0 or int(summary.n_observed) <= 0:
        return JobNameAccountingLookup(
            None,
            inconclusive=True,
            rows=rows,
            error="sacct returned no usable rows for expected job name",
        )
    if int(summary.n_pending_or_running) > 0:
        return JobNameAccountingLookup(
            job_id,
            terminal=False,
            successful=False,
            failed=False,
            rows=rows,
        )
    if bool(summary.is_fully_successful):
        return JobNameAccountingLookup(
            job_id,
            terminal=True,
            successful=True,
            failed=False,
            rows=rows,
        )
    if bool(summary.is_terminal):
        return JobNameAccountingLookup(
            job_id,
            terminal=True,
            successful=False,
            failed=True,
            rows=rows,
        )
    return JobNameAccountingLookup(
        None,
        inconclusive=True,
        rows=rows,
        error="sacct rows are not conclusively active or terminal",
    )


def find_running_job_by_name_detailed(
    name: str,
    *,
    sacct_runner: Optional[Callable[..., Any]] = None,
    squeue_runner: Optional[Callable[..., Any]] = None,
    use_squeue_fallback: bool = False,
    timeout_seconds: int = 60,
) -> JobNameLookup:
    """Look for a still-running (or queued) SLURM job with this --job-name and return its JobID,
    or None.

    used on phase entry / after a reconcile to spot a job a crash orphaned, so the daemon can adopt
    and poll it rather than submit a duplicate that would race it into the same staging dirs
    (A24/A25). only NON-terminal states count -- a COMPLETED/FAILED job of the same name from an
    earlier run must not be adopted. returns the base allocation id (array task suffix dropped) so
    the caller polls the whole array. best-effort: any sacct hiccup just returns None and the caller
    falls back to submitting, which is the normal path.
    """
    if sacct_runner is None:
        sacct_runner = subprocess.run
    cmd = [
        "sacct", "--name", str(name),
        "--format=JobIDRaw,State%40", "--array", "-X", "-P", "-n",
    ]
    try:
        completed = run_scheduler_command(
            sacct_runner,
            cmd,
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return JobNameLookup(None, inconclusive=True, error=type(exc).__name__ + ": " + str(exc))
    if int(getattr(completed, "returncode", 1)) != 0:
        stderr = getattr(completed, "stderr", "") or ""
        return JobNameLookup(
            None,
            inconclusive=True,
            error="sacct exited with code "
            + str(int(getattr(completed, "returncode", 1)))
            + ": "
            + repr(stderr),
        )
    stdout = getattr(completed, "stdout", "") or ""
    non_terminal = set()
    rows: List[Tuple[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        if not job_id:
            continue
        status = JobStatus.from_sacct(parts[1])
        rows.append((job_id, status.value))
        if status in NON_TERMINAL_STATES:
            # 123_4 -> 123 (and 123.batch -> 123): adopt the whole allocation, not a sub-step.
            base = _base_allocation_id(job_id)
            non_terminal.add(base)
    if not non_terminal:
        if use_squeue_fallback:
            fallback = find_active_job_by_name_detailed(
                name,
                squeue_runner=squeue_runner,
                timeout_seconds=int(timeout_seconds),
            )
            if fallback.job_id or fallback.inconclusive:
                return fallback
        return JobNameLookup(None, inconclusive=False, rows=rows)
    if len(non_terminal) > 1:
        return JobNameLookup(
            None,
            inconclusive=True,
            rows=rows,
            error="multiple active jobs share expected name: "
            + repr(sorted(non_terminal)),
        )
    return JobNameLookup(sorted(non_terminal)[0], inconclusive=False, rows=rows)


def find_running_job_by_name(
    name: str,
    *,
    sacct_runner: Optional[Callable[..., Any]] = None,
    squeue_runner: Optional[Callable[..., Any]] = None,
    use_squeue_fallback: bool = False,
    timeout_seconds: int = 60,
) -> Optional[str]:
    return find_running_job_by_name_detailed(
        name,
        sacct_runner=sacct_runner,
        squeue_runner=squeue_runner,
        use_squeue_fallback=use_squeue_fallback,
        timeout_seconds=int(timeout_seconds),
    ).job_id
