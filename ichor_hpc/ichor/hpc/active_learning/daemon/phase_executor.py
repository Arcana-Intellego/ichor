"""PhaseExecutor protocol + mock implementation.

The daemon delegates the per-phase work (ICHOR diversity, Gaussian / AIMAll
arrays, FEREBUS training, ARIADNE adversarial attack, atomic appends etc.)
to a PhaseExecutor. The protocol has three operations:

    submit_or_run(state, phase) -> PhaseResult
        decide what to do at the start of a phase. For SLURM-backed phases
        this submits the job and returns the JobID. For inline phases
        (SEED_SELECT, SPLIT, APPEND, STOP_CHECK in production) this runs
        the work synchronously and returns is_complete=True immediately.

    postprocess(state, phase, observations) -> PhaseResult
        Called once the SLURM job has reached a terminal sacct state. The
        executor parses the output, updates manifests, etc., and returns a
        PhaseResult with state_updates to advance the campaign state.

    handle_failure(state, phase, observations) -> FailureAction
        Called when a SLURM job terminates unsuccessfully. Returns a
        FailureAction enum: SCRUB_AND_CONTINUE (default for Gaussian/AIMAll
        per the plan), HALT (catastrophic), or RETRY (per-job retry, future).

MockPhaseExecutor is the default: each call records itself in a log and
returns immediate completion with empty state_updates. This lets the daemon
state machine progression be tested end-to-end without invoking any of the
heavy external backend machinery;
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence


__all__ = [
    "PhaseResult",
    "FailureAction",
    "PhaseExecutor",
    "MockPhaseExecutor",
    "BackendSubmissionError",
    "INLINE_PHASES",
    "SBATCH_PHASES",
]


class BackendSubmissionError(RuntimeError):
    """raised when a backend submission (sbatch) fails outright. the daemon
    treats this as a halt-worthy backend problem and stops the campaign
    cleanly so it can be resumed, rather than letting the error crash it."""


#phases the production daemon runs inline (no SLURM job). The mock executor
#treats every phase as inline, so this set is only consulted by real
#executors.
INLINE_PHASES = frozenset({
    "INIT", "INITIAL_ALLOCATION_CHECK", "ALLOCATION_CHECK", "SEED_SELECT",
    "SPLIT", "APPEND", "STOP_CHECK", "DONE", "HALTED",
})
SBATCH_PHASES = frozenset({
    "PHASE_A_DIVERSITY", "INITIAL_GAUSSIAN", "INITIAL_AIMALL", "INITIAL_FEREBUS",
    "INITIAL_REPLACEMENT_GAUSSIAN", "INITIAL_REPLACEMENT_AIMALL",
    "ARIADNE_ARRAY", "PHASE_B_DIVERSITY", "GAUSSIAN", "AIMALL",
    "REPLACEMENT_GAUSSIAN", "REPLACEMENT_AIMALL", "FEREBUS",
})


class FailureAction(str, Enum):
    SCRUB_AND_CONTINUE = "SCRUB_AND_CONTINUE"
    HALT = "HALT"
    RETRY = "RETRY"


@dataclass
class PhaseResult:
    """Outcome of one PhaseExecutor call.

    is_complete=False with a submitted_job_id means the daemon should put
    the JobID into state.pending_jobs[phase] and start polling sacct.

    is_complete=True with no job ID means the work was done inline.
    state_updates are merged into the CampaignState at the end of the tick.
    """

    is_complete: bool = False
    submitted_job_id: Optional[str] = None
    expected_tasks: Optional[int] = None
    state_updates: Dict[str, Any] = field(default_factory=dict)
    journal_events: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None
    submission_metadata: Dict[str, Any] = field(default_factory=dict)
    next_phase_override: Optional[str] = None

    def validate(self, *, stage: str, phase_name: str) -> "PhaseResult":
        """Validate one mutually exclusive executor outcome."""
        if stage not in {"submit", "postprocess"}:
            raise ValueError("PhaseResult stage must be submit or postprocess")
        if not isinstance(self.is_complete, bool):
            raise ValueError("PhaseResult.is_complete must be a boolean")
        if self.submitted_job_id is not None and (
            not isinstance(self.submitted_job_id, str) or not self.submitted_job_id
        ):
            raise ValueError("PhaseResult.submitted_job_id must be a non-empty string")
        if self.is_complete and self.submitted_job_id is not None:
            raise ValueError("completed PhaseResult cannot also submit a job")
        if not self.is_complete and self.submitted_job_id is None:
            raise ValueError("incomplete PhaseResult must contain a submitted job ID")
        if not self.is_complete and self.failure_reason is not None:
            raise ValueError("incomplete PhaseResult cannot contain failure_reason")
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str) or not self.failure_reason.strip()
        ):
            raise ValueError("PhaseResult.failure_reason must be a non-empty string")
        if self.expected_tasks is not None and (
            not isinstance(self.expected_tasks, int)
            or isinstance(self.expected_tasks, bool)
            or self.expected_tasks <= 0
        ):
            raise ValueError("PhaseResult.expected_tasks must be a positive integer")
        if self.submitted_job_id is None and self.expected_tasks is not None:
            raise ValueError("PhaseResult.expected_tasks requires a submitted job")
        if not isinstance(self.state_updates, dict):
            raise ValueError("PhaseResult.state_updates must be an object")
        if not self.is_complete and self.state_updates:
            raise ValueError("submitted PhaseResult cannot contain state updates")
        if not isinstance(self.submission_metadata, dict):
            raise ValueError("PhaseResult.submission_metadata must be an object")
        if self.submitted_job_id is None and self.submission_metadata:
            raise ValueError("PhaseResult submission metadata requires a submitted job")
        if not isinstance(self.journal_events, list) or any(
            not isinstance(event, dict) for event in self.journal_events
        ):
            raise ValueError("PhaseResult.journal_events must be a list of objects")
        if self.next_phase_override is not None:
            if not self.is_complete:
                raise ValueError("next_phase_override requires a completed PhaseResult")
            from .state import CampaignPhase

            try:
                CampaignPhase(self.next_phase_override)
            except (TypeError, ValueError) as exc:
                raise ValueError("PhaseResult next_phase_override is unknown") from exc
        if stage == "postprocess" and self.submitted_job_id is not None:
            raise ValueError("postprocess PhaseResult cannot submit another job")
        if stage == "postprocess" and not self.is_complete:
            raise ValueError("postprocess PhaseResult must be complete")
        completion_reason = self.state_updates.get("campaign_completion_reason")
        if completion_reason is not None:
            if str(phase_name) != "STOP_CHECK":
                raise ValueError(
                    "campaign_completion_reason is valid only for STOP_CHECK"
                )
            if not isinstance(completion_reason, str) or not completion_reason.strip():
                raise ValueError("campaign_completion_reason must be a non-empty string")
        return self


class PhaseExecutor(Protocol):
    """Structural type for daemon phase executors."""

    def submit_or_run(self, state, phase) -> PhaseResult: ...
    def postprocess(self, state, phase, observations: Sequence[Any]) -> PhaseResult: ...
    def handle_failure(self, state, phase, observations: Sequence[Any]) -> FailureAction: ...


@dataclass
class _RecordedCall:
    """Audit trail entry for the mock executor."""

    operation: str
    phase: str
    iteration: int
    extra: Dict[str, Any] = field(default_factory=dict)


class MockPhaseExecutor:
    """Stub executor that records calls and returns immediate completion.

    Optional knobs:
        fail_on_phases: iterable of phase names; submit_or_run for these
            returns is_complete=False with a fake JobID, and postprocess
            then returns failure_reason set so the daemon exercises its
            failure path.
        fake_job_ids: iterable yielding the JobIDs returned by
            submit_or_run when a phase is configured to "run as SLURM";
            defaults to monotonic "MOCK-<n>".
    """

    def __init__(
        self,
        *,
        treat_as_sbatch: Optional[Sequence[str]] = None,
        fail_on_phases: Optional[Sequence[str]] = None,
        fail_action: FailureAction = FailureAction.SCRUB_AND_CONTINUE,
    ) -> None:
        self._treat_as_sbatch = frozenset(treat_as_sbatch or ())
        self._fail_on_phases = frozenset(fail_on_phases or ())
        self._fail_action = fail_action
        self.calls: List[_RecordedCall] = []
        self._counter = 0

    def submit_or_run(self, state, phase) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        iteration = getattr(state, "iteration", -1)
        self.calls.append(_RecordedCall("submit_or_run", phase_name, iteration))
        if phase_name in self._treat_as_sbatch:
            self._counter += 1
            return PhaseResult(
                is_complete=False,
                submitted_job_id="MOCK-" + str(self._counter),
            )
        # Inline phases complete immediately.
        return PhaseResult(is_complete=True)

    def postprocess(self, state, phase, observations: Sequence[Any]) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        iteration = getattr(state, "iteration", -1)
        self.calls.append(_RecordedCall("postprocess", phase_name, iteration))
        if phase_name in self._fail_on_phases:
            return PhaseResult(
                is_complete=False,
                failure_reason="mock failure injected for " + phase_name,
            )
        return PhaseResult(is_complete=True)

    def handle_failure(self, state, phase, observations: Sequence[Any]) -> FailureAction:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        iteration = getattr(state, "iteration", -1)
        self.calls.append(_RecordedCall(
            "handle_failure", phase_name, iteration,
            extra={"action": self._fail_action.value},
        ))
        return self._fail_action

    def operations(self) -> List[str]:
        """Compact list of (operation, phase) pairs for assertions."""
        return [c.operation + ":" + c.phase for c in self.calls]
