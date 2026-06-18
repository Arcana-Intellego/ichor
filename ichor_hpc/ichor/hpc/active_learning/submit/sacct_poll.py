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
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_sacct(cls, raw: str) -> "JobStatus":
        #sacct often emits "CANCELLED+" or "CANCELLED by 12345".
        token = raw.strip().split()[0].rstrip("+").upper() if raw and raw.strip() else ""
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
    JobStatus.SPECIAL_EXIT,
    JobStatus.STOPPED,
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

    @property
    def is_terminal(self) -> bool:
        return self.n_pending_or_running == 0 and self.n_tasks > 0

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
    try:
        ret = int(parts[0])
        sig = int(parts[1]) if len(parts) > 1 else 0
        return (ret, sig)
    except ValueError:
        return None


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
        if len(parts) < 4:
            continue
        job_id, state, exit_code, elapsed = parts[0].strip(), parts[1], parts[2], parts[3]
        if not job_id:
            continue
        observations.append(JobObservation(
            job_id=job_id,
            status=JobStatus.from_sacct(state),
            exit_code=_parse_exit_code(exit_code),
            elapsed_seconds=_parse_elapsed(elapsed),
        ))
    return observations


def aggregate_states(
    parent_job_id: str,
    observations: Sequence[JobObservation],
    expected_task_count: Optional[int] = None,
) -> ArrayJobSummary:
    """Collapse per-task observations into one summary.

    Task rows have job_id like '<parent>_<task_index>'. We pick out tasks
    belonging to 'parent_job_id' and count terminal vs non-terminal states.
    """
    prefix = str(parent_job_id) + "_"
    task_obs = [
        o for o in observations
        if o.job_id == str(parent_job_id) or o.job_id.startswith(prefix)
    ]
    #drop the parent summary row when we have explicit task rows.
    has_tasks = any(o.job_id.startswith(prefix) for o in task_obs)
    if has_tasks:
        task_obs = [o for o in task_obs if o.job_id.startswith(prefix)]
    observed_count = len(task_obs)
    expected = None if expected_task_count is None else max(0, int(expected_task_count))
    missing = 0
    if expected is not None and expected > observed_count:
        missing = expected - observed_count
    n_completed = sum(1 for o in task_obs if o.is_success)
    n_failed = sum(1 for o in task_obs if o.is_failure)
    n_unknown = sum(1 for o in task_obs if o.status == JobStatus.UNKNOWN)
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
    )



def poll_job(
    job_id: str,
    *,
    sacct_runner: Optional[Callable[..., Any]] = None,
    extra_args: Sequence[str] = (),
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
        "--format=JobID,State,ExitCode,Elapsed",
        "-X", "-P", "-n",
    ] + list(extra_args)
    completed = sacct_runner(cmd, check=False, capture_output=True, text=True)
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


def find_active_job_by_id_detailed(
    job_id: str,
    *,
    squeue_runner: Optional[Callable[..., Any]] = None,
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
        completed = squeue_runner(cmd, check=False, capture_output=True, text=True)
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
        jid = parts[0].strip()
        if not jid:
            continue
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


def find_running_job_by_name_detailed(
    name: str,
    *,
    sacct_runner: Optional[Callable[..., Any]] = None,
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
        "--format=JobID,State", "-X", "-P", "-n",
    ]
    try:
        completed = sacct_runner(cmd, check=False, capture_output=True, text=True)
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
            base = job_id.split("_", 1)[0].split(".", 1)[0]
            non_terminal.add(base)
    if not non_terminal:
        return JobNameLookup(None, inconclusive=False, rows=rows)
    # lowest id == earliest submission; adopt that one if somehow several share the name.
    return JobNameLookup(sorted(non_terminal)[0], inconclusive=False, rows=rows)


def find_running_job_by_name(
    name: str,
    *,
    sacct_runner: Optional[Callable[..., Any]] = None,
) -> Optional[str]:
    return find_running_job_by_name_detailed(
        name,
        sacct_runner=sacct_runner,
    ).job_id
