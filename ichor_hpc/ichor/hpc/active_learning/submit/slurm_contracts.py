"""Strict Slurm identity and subprocess contracts used by active learning."""
from __future__ import annotations

import inspect
import re
import subprocess
from typing import Any, Callable, Optional, Tuple


_PARENT_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
_TASK_JOB_ID_RE = re.compile(r"^([1-9][0-9]*)_([0-9]+)$")
_STEP_JOB_ID_RE = re.compile(r"^([1-9][0-9]*)\.(batch|extern)$")
_SQUEUE_ARRAY_RANGE_RE = re.compile(
    r"^([1-9][0-9]*)_\[[0-9,-]+(?:%[1-9][0-9]*)?\]$"
)
_CLUSTER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_parent_job_id(value: Any) -> str:
    job_id = str(value)
    if not _PARENT_JOB_ID_RE.fullmatch(job_id):
        raise ValueError("Slurm parent JobID must be a positive decimal integer")
    return job_id


def parse_job_row_id(value: Any) -> Tuple[str, Optional[int], Optional[str]]:
    """Return ``(parent, task_index, step)`` for one canonical Slurm row ID."""
    job_id = str(value).strip()
    if _PARENT_JOB_ID_RE.fullmatch(job_id):
        return job_id, None, None
    task = _TASK_JOB_ID_RE.fullmatch(job_id)
    if task is not None:
        return task.group(1), int(task.group(2)), None
    step = _STEP_JOB_ID_RE.fullmatch(job_id)
    if step is not None:
        return step.group(1), None, step.group(2)
    raise ValueError("malformed or grouped Slurm JobID row: " + repr(job_id))


def parse_squeue_job_id(value: Any) -> str:
    """Return the parent allocation for one canonical ``squeue %i`` value."""
    job_id = str(value).strip()
    grouped = _SQUEUE_ARRAY_RANGE_RE.fullmatch(job_id)
    if grouped is not None:
        return grouped.group(1)
    parent, _task, _step = parse_job_row_id(job_id)
    return parent


def parse_sbatch_parsable_output(stdout: Any) -> Tuple[str, Optional[str]]:
    """Parse exactly one ``sbatch --parsable`` result line."""
    text = str(stdout or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "sbatch --parsable must emit exactly one non-empty line; got "
            + str(len(lines))
        )
    parts = lines[0].split(";")
    if len(parts) not in {1, 2}:
        raise ValueError("sbatch --parsable returned too many fields")
    job_id = validate_parent_job_id(parts[0].strip())
    cluster: Optional[str] = None
    if len(parts) == 2:
        cluster = parts[1].strip()
        if not cluster or not _CLUSTER_RE.fullmatch(cluster):
            raise ValueError("sbatch --parsable returned an invalid cluster name")
    return job_id, cluster


def _runner_accepts_timeout(runner: Callable[..., Any]) -> bool:
    if runner is subprocess.run:
        return True
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True
    return "timeout" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def run_scheduler_command(
    runner: Callable[..., Any],
    command: Any,
    *,
    timeout_seconds: int,
    **kwargs: Any,
) -> Any:
    """Run one scheduler client with a validated finite timeout.

    Legacy test seams with an explicit signature may omit ``timeout``. Real
    scheduler calls and flexible injected runners always receive it.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("scheduler command timeout must be a positive integer")
    if _runner_accepts_timeout(runner):
        kwargs["timeout"] = int(timeout_seconds)
    return runner(command, **kwargs)


__all__ = [
    "parse_job_row_id",
    "parse_squeue_job_id",
    "parse_sbatch_parsable_output",
    "run_scheduler_command",
    "validate_parent_job_id",
]
