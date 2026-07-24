"""Scheduler-neutral daemon operations.

The Slurm backend deliberately delegates to the established implementation.
The SGE backend normalises native one-based arrays into ICHOR's zero-based
logical task contract.
"""
from __future__ import annotations

import getpass
import inspect
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from . import sacct_poll
from . import sge
from .slurm_contracts import (
    parse_sbatch_parsable_output,
    run_scheduler_command,
    validate_parent_job_id,
)


@dataclass(frozen=True)
class SchedulerSubmission:
    job_id: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SchedulerUsageEvidence:
    command: tuple[str, ...]
    stdout: str
    records: Optional[Sequence[dict[str, str]]] = None


def _call_with_supported_kwargs(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Preserve established injected scheduler-call signatures."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(*args, **kwargs)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return function(*args, **supported)


class SchedulerBackend:
    identity_kind = ""
    display_name = ""
    submit_command = ""
    accounting_command = ""
    queue_command = ""
    cancel_command = ""

    def validate_job_id(self, value: Any) -> str:
        raise NotImplementedError

    def submit(
        self,
        script: Any,
        *,
        binding_sha256: str,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
        cwd: Optional[str] = None,
    ) -> SchedulerSubmission:
        raise NotImplementedError

    def poll_job(
        self,
        job_id: str,
        *,
        accounting_runner: Optional[Callable[..., Any]] = None,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
        cancellation_requested: bool = False,
    ):
        raise NotImplementedError

    def find_active_job_by_id(
        self,
        job_id: str,
        *,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
    ):
        raise NotImplementedError

    def find_running_job_by_name(
        self,
        name: str,
        *,
        accounting_runner: Optional[Callable[..., Any]] = None,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
    ):
        raise NotImplementedError

    def find_accounted_job_by_name(
        self,
        name: str,
        *,
        expected_task_count: Optional[int],
        submission_kind: Optional[str],
        accounting_runner: Optional[Callable[..., Any]] = None,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
    ):
        raise NotImplementedError

    def cancellation_lookup(
        self,
        job_id: str,
        *,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> dict:
        raise NotImplementedError

    def collect_usage_evidence(
        self,
        job_id: str,
        *,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> SchedulerUsageEvidence:
        raise NotImplementedError

    def cancel(
        self,
        job_id: str,
        *,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> tuple[bool, str]:
        canonical = self.validate_job_id(job_id)
        try:
            completed = run_scheduler_command(
                runner,
                [self.cancel_command, canonical],
                timeout_seconds=int(timeout_seconds),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            return False, type(exc).__name__ + ": " + str(exc)
        if int(getattr(completed, "returncode", 1)) == 0:
            return True, ""
        stderr = getattr(completed, "stderr", "") or ""
        stdout = getattr(completed, "stdout", "") or ""
        return False, stderr.strip() or stdout.strip() or self.cancel_command + " failed"


class SlurmScheduler(SchedulerBackend):
    identity_kind = "slurm"
    display_name = "Slurm"
    submit_command = "sbatch"
    accounting_command = "sacct"
    queue_command = "squeue"
    cancel_command = "scancel"

    def validate_job_id(self, value: Any) -> str:
        return validate_parent_job_id(value)

    def submit(
        self,
        script: Any,
        *,
        binding_sha256: str,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
        cwd: Optional[str] = None,
    ) -> SchedulerSubmission:
        completed = run_scheduler_command(
            runner,
            [
                "sbatch",
                "--parsable",
                "--export=ALL,ICHOR_SCRIPT_BINDING_SHA256=" + str(binding_sha256),
                str(script),
            ],
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        if int(getattr(completed, "returncode", 1)) != 0:
            raise RuntimeError(
                "sbatch exited with code "
                + str(int(getattr(completed, "returncode", 1)))
                + ". stdout: "
                + repr(stdout)
                + " stderr: "
                + repr(stderr)
            )
        job_id, _cluster = parse_sbatch_parsable_output(stdout)
        return SchedulerSubmission(job_id=job_id, stdout=stdout, stderr=stderr)

    def poll_job(
        self,
        job_id: str,
        *,
        accounting_runner: Optional[Callable[..., Any]] = None,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
        cancellation_requested: bool = False,
    ):
        del queue_runner, cancellation_requested
        return _call_with_supported_kwargs(
            sacct_poll.poll_job,
            job_id,
            sacct_runner=accounting_runner,
            timeout_seconds=int(timeout_seconds),
        )

    def find_active_job_by_id(
        self,
        job_id: str,
        *,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
    ):
        return _call_with_supported_kwargs(
            sacct_poll.find_active_job_by_id_detailed,
            job_id,
            squeue_runner=queue_runner,
            timeout_seconds=int(timeout_seconds),
        )

    def find_running_job_by_name(
        self,
        name: str,
        *,
        accounting_runner: Optional[Callable[..., Any]] = None,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
    ):
        return _call_with_supported_kwargs(
            sacct_poll.find_running_job_by_name_detailed,
            name,
            sacct_runner=accounting_runner,
            squeue_runner=queue_runner,
            use_squeue_fallback=True,
            timeout_seconds=int(timeout_seconds),
        )

    def find_accounted_job_by_name(
        self,
        name: str,
        *,
        expected_task_count: Optional[int],
        submission_kind: Optional[str],
        accounting_runner: Optional[Callable[..., Any]] = None,
        queue_runner: Optional[Callable[..., Any]] = None,
        timeout_seconds: int = 60,
    ):
        return _call_with_supported_kwargs(
            sacct_poll.find_accounted_job_by_name_detailed,
            name,
            expected_task_count=expected_task_count,
            sacct_runner=accounting_runner,
            squeue_runner=queue_runner,
            use_squeue_fallback=True,
            submission_kind=submission_kind,
            timeout_seconds=int(timeout_seconds),
        )

    def cancellation_lookup(
        self,
        job_id: str,
        *,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> dict:
        parent = self.validate_job_id(job_id)
        try:
            completed = run_scheduler_command(
                queue_runner,
                [
                    "squeue",
                    "-j",
                    parent,
                    "--noheader",
                    "--format=%i|%T|%j|%u",
                ],
                timeout_seconds=int(timeout_seconds),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            return {
                "active": False,
                "inconclusive": True,
                "rows": [],
                "error": type(exc).__name__ + ": " + str(exc),
            }
        if int(getattr(completed, "returncode", 1)) != 0:
            stderr = getattr(completed, "stderr", "") or ""
            if sacct_poll._squeue_invalid_job_id(stderr):
                return {
                    "active": False,
                    "inconclusive": False,
                    "rows": [],
                    "error": None,
                }
            return {
                "active": False,
                "inconclusive": True,
                "rows": [],
                "error": "squeue exited with code "
                + str(int(getattr(completed, "returncode", 1)))
                + ": "
                + repr(stderr),
            }
        rows = []
        for line in (getattr(completed, "stdout", "") or "").splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) != 4:
                return {
                    "active": False,
                    "inconclusive": True,
                    "rows": rows,
                    "error": "malformed squeue cancellation row",
                }
            row_parent = sacct_poll.parse_squeue_job_id(parts[0].strip())
            if row_parent != parent:
                return {
                    "active": False,
                    "inconclusive": True,
                    "rows": rows,
                    "error": "squeue returned a foreign JobID during cancellation",
                }
            rows.append(
                {
                    "job_id": parts[0].strip(),
                    "state": parts[1].strip(),
                    "job_name": parts[2].strip(),
                    "owner": parts[3].strip(),
                }
            )
        return {
            "active": bool(rows),
            "inconclusive": False,
            "rows": rows,
            "error": None,
        }

    def collect_usage_evidence(
        self,
        job_id: str,
        *,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> SchedulerUsageEvidence:
        parent = self.validate_job_id(job_id)
        command = (
            "sacct",
            "-n",
            "-P",
            "--array",
            "-j",
            parent,
            "--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,MaxVMSize,TotalCPU",
        )
        completed = run_scheduler_command(
            runner,
            list(command),
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
        )
        if int(getattr(completed, "returncode", 1)) != 0:
            raise RuntimeError(
                "sacct telemetry failed: "
                + str(getattr(completed, "stderr", "") or "")
            )
        return SchedulerUsageEvidence(
            command=command,
            stdout=str(getattr(completed, "stdout", "") or ""),
        )


class SgeScheduler(SchedulerBackend):
    identity_kind = "sge"
    display_name = "Sun Grid Engine"
    submit_command = "qsub"
    accounting_command = "qacct"
    queue_command = "qstat"
    cancel_command = "qdel"

    def validate_job_id(self, value: Any) -> str:
        return sge.validate_sge_parent_job_id(value)

    def submit(
        self,
        script: Any,
        *,
        binding_sha256: str,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
        cwd: Optional[str] = None,
    ) -> SchedulerSubmission:
        completed = run_scheduler_command(
            runner,
            [
                "qsub",
                "-terse",
                "-v",
                "ICHOR_SCRIPT_BINDING_SHA256=" + str(binding_sha256),
                str(script),
            ],
            timeout_seconds=int(timeout_seconds),
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        if int(getattr(completed, "returncode", 1)) != 0:
            raise RuntimeError(
                "qsub exited with code "
                + str(int(getattr(completed, "returncode", 1)))
                + ". stdout: "
                + repr(stdout)
                + " stderr: "
                + repr(stderr)
            )
        return SchedulerSubmission(
            job_id=sge.parse_qsub_terse_output(stdout),
            stdout=stdout,
            stderr=stderr,
        )

    def poll_job(
        self,
        job_id: str,
        *,
        accounting_runner: Callable[..., Any] = subprocess.run,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
        cancellation_requested: bool = False,
    ):
        return sge.poll_job(
            job_id,
            qacct_runner=accounting_runner,
            qstat_runner=queue_runner,
            timeout_seconds=int(timeout_seconds),
            cancellation_requested=bool(cancellation_requested),
        )

    def find_active_job_by_id(
        self,
        job_id: str,
        *,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ):
        return sge.find_active_job_by_id_detailed(
            job_id,
            qstat_runner=queue_runner,
            timeout_seconds=int(timeout_seconds),
        )

    def find_running_job_by_name(
        self,
        name: str,
        *,
        accounting_runner: Callable[..., Any] = subprocess.run,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ):
        del accounting_runner
        return sge.find_running_job_by_name_detailed(
            name,
            qstat_runner=queue_runner,
            timeout_seconds=int(timeout_seconds),
        )

    def find_accounted_job_by_name(
        self,
        name: str,
        *,
        expected_task_count: Optional[int],
        submission_kind: Optional[str],
        accounting_runner: Callable[..., Any] = subprocess.run,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ):
        return sge.find_accounted_job_by_name_detailed(
            name,
            expected_task_count=expected_task_count,
            qacct_runner=accounting_runner,
            qstat_runner=queue_runner,
            use_qstat_fallback=True,
            submission_kind=submission_kind,
            timeout_seconds=int(timeout_seconds),
        )

    def cancellation_lookup(
        self,
        job_id: str,
        *,
        queue_runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> dict:
        try:
            parent = self.validate_job_id(job_id)
            rows = [
                row
                for row in sge.query_qstat(
                    runner=queue_runner,
                    timeout_seconds=int(timeout_seconds),
                )
                if row.parent_job_id == parent
            ]
        except Exception as exc:
            return {
                "active": False,
                "inconclusive": True,
                "rows": [],
                "error": type(exc).__name__ + ": " + str(exc),
            }
        return {
            "active": bool(rows),
            "inconclusive": False,
            "rows": [
                {
                    "job_id": row.logical_job_id,
                    "state": row.state,
                    "job_name": row.job_name,
                    "owner": row.owner,
                }
                for row in rows
            ],
            "error": None,
        }

    def collect_usage_evidence(
        self,
        job_id: str,
        *,
        runner: Callable[..., Any] = subprocess.run,
        timeout_seconds: int = 60,
    ) -> SchedulerUsageEvidence:
        parent = self.validate_job_id(job_id)
        records = tuple(
            sge.query_qacct(
                parent,
                runner=runner,
                timeout_seconds=int(timeout_seconds),
            )
        )
        return SchedulerUsageEvidence(
            command=("qacct", "-j", parent),
            stdout="",
            records=records,
        )


_BACKENDS = {
    "slurm": SlurmScheduler(),
    "sge": SgeScheduler(),
}


def get_scheduler_backend(kind: Any) -> SchedulerBackend:
    value = str(kind or "").strip().lower()
    try:
        return _BACKENDS[value]
    except KeyError as exc:
        raise ValueError("unsupported scheduler: " + repr(value)) from exc


def supported_scheduler_kinds() -> Sequence[str]:
    return tuple(sorted(_BACKENDS))


def current_scheduler_user() -> str:
    value = str(getpass.getuser() or "").strip()
    if not value:
        raise ValueError("could not determine the current scheduler user")
    return value


__all__ = [
    "SchedulerBackend",
    "SchedulerSubmission",
    "SchedulerUsageEvidence",
    "SgeScheduler",
    "SlurmScheduler",
    "get_scheduler_backend",
    "current_scheduler_user",
    "supported_scheduler_kinds",
]
