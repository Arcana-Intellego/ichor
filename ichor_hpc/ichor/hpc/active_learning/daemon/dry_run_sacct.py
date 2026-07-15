"""Dry-run sacct poller.

When the daemon is started with ``--mode dry_run``, no real Slurm jobs are
submitted. The DryRunPhaseExecutor manufactures JobIDs of the form
"DRYRUN-<PHASE>-<iteration>"; this poller recognises that prefix and
returns a synthetic COMPLETED observation immediately, so the daemon's
poll loop exercises the full sacct -> postprocess flow without contacting
the queue.

Any non-synthetic JobID is rejected. Dry-run mode must never contact Slurm,
including when state has been copied from a live campaign by mistake.
"""
from __future__ import annotations

from typing import List

from ..submit.sacct_poll import JobObservation, JobStatus


__all__ = ["DRYRUN_PREFIX", "DryRunSacctPoller"]


DRYRUN_PREFIX = "DRYRUN-"


class DryRunSacctPoller:
    """Callable used as "Daemon.sacct_poller" in dry-run mode.

    Parameters
    ----------
    elapsed_seconds
        Elapsed time reported on the synthetic COMPLETED observation.
        Defaults to 1; tests pass 0 to keep things tidy.
    """

    def __init__(
        self,
        *,
        elapsed_seconds: int = 1,
    ) -> None:
        self._elapsed = int(elapsed_seconds)
        self.invocations: List[str] = []

    def __call__(self, job_id, **kwargs) -> List[JobObservation]:
        job_id_str = str(job_id)
        self.invocations.append(job_id_str)
        if not job_id_str.startswith(DRYRUN_PREFIX):
            raise ValueError(
                "dry-run accounting refuses non-synthetic Slurm JobID "
                + repr(job_id_str)
            )
        raw_expected = kwargs.get("expected_task_count")
        try:
            expected = int(raw_expected) if raw_expected is not None else 1
        except (TypeError, ValueError) as exc:
            raise ValueError("expected_task_count must be an integer") from exc
        if expected <= 0:
            raise ValueError("expected_task_count must be > 0 in dry-run mode")
        return [
            JobObservation(
                job_id=(
                    job_id_str
                    if expected == 1
                    else job_id_str + "_" + str(task_index)
                ),
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=self._elapsed,
            )
            for task_index in range(expected)
        ]
