"""Dry-run sacct poller.

When the daemon is started with ``--mode dry_run``, no real Slurm jobs are
submitted. The DryRunPhaseExecutor manufactures JobIDs of the form
"DRYRUN-<PHASE>-<iteration>"; this poller recognises that prefix and
returns a synthetic COMPLETED observation immediately, so the daemon's
poll loop exercises the full sacct -> postprocess flow without contacting
the queue.

Real-cluster JobIDs (the daemon does still call sbatch for legitimately
non-dry phases) are passed through to a fallback poller, which defaults to
the real :func:"ichor.hpc.active_learning.submit.sacct_poll.poll_job".
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from ..submit.sacct_poll import JobObservation, JobStatus, poll_job


__all__ = ["DRYRUN_PREFIX", "DryRunSacctPoller"]


DRYRUN_PREFIX = "DRYRUN-"


class DryRunSacctPoller:
    """Callable used as "Daemon.sacct_poller" in dry-run mode.

    Parameters
    ----------
    elapsed_seconds
        Elapsed time reported on the synthetic COMPLETED observation.
        Defaults to 1; tests pass 0 to keep things tidy.
    fallback_poller
        Used for any JobID that does not start with "DRYRUN-" so the
        dry-run mode can coexist with a hybrid campaign that submits some
        real jobs (e.g. a real FEREBUS while everything else is stubbed).
        Defaults to :func:"poll_job"; tests override with a stub.
    """

    def __init__(
        self,
        *,
        elapsed_seconds: int = 1,
        fallback_poller: Optional[Callable[..., Sequence[JobObservation]]] = None,
    ) -> None:
        self._elapsed = int(elapsed_seconds)
        self._fallback = fallback_poller or poll_job
        self.invocations: List[str] = []

    def __call__(self, job_id, **kwargs) -> List[JobObservation]:
        job_id_str = str(job_id)
        self.invocations.append(job_id_str)
        if job_id_str.startswith(DRYRUN_PREFIX):
            raw_expected = kwargs.get("expected_task_count")
            try:
                expected = int(raw_expected) if raw_expected is not None else 1
            except (TypeError, ValueError) as exc:
                raise ValueError("expected_task_count must be an integer") from exc
            if expected <= 0:
                expected = 1
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
        observations = self._fallback(job_id_str, **kwargs)
        return list(observations) if not isinstance(observations, list) else observations
