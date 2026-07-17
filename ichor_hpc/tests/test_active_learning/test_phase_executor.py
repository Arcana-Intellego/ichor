"""Tests for ichor.hpc.active_learning.daemon.phase_executor."""
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.daemon.phase_executor import (
    FailureAction,
    INLINE_PHASES,
    MockPhaseExecutor,
    PhaseResult,
    SBATCH_PHASES,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase


def _state(iteration=0):
    return SimpleNamespace(iteration=iteration)


def test_inline_and_sbatch_sets_disjoint():
    assert INLINE_PHASES.isdisjoint(SBATCH_PHASES)


def test_phase_result_defaults():
    r = PhaseResult()
    assert r.is_complete is False
    assert r.submitted_job_id is None
    assert r.state_updates == {}
    assert r.journal_events == []
    assert r.failure_reason is None


def test_phase_result_rejects_completion_and_submission_together():
    result = PhaseResult(is_complete=True, submitted_job_id="123")

    with pytest.raises(ValueError, match="cannot also submit"):
        result.validate(stage="submit", phase_name="GAUSSIAN")


def test_phase_result_rejects_fractional_expected_tasks():
    result = PhaseResult(
        is_complete=False,
        submitted_job_id="123",
        expected_tasks=1.5,
    )

    with pytest.raises(ValueError, match="positive integer"):
        result.validate(stage="submit", phase_name="GAUSSIAN")


def test_phase_result_restricts_convergence_to_stop_check():
    result = PhaseResult(
        is_complete=True,
        state_updates={"campaign_completion_reason": "converged"},
    )

    with pytest.raises(ValueError, match="only for STOP_CHECK"):
        result.validate(stage="submit", phase_name="REFERENCE_COMMIT")


def test_mock_inline_phase_completes_immediately():
    e = MockPhaseExecutor()
    r = e.submit_or_run(_state(), CampaignPhase.REFERENCE_COMMIT)
    assert r.is_complete is True
    assert r.submitted_job_id is None
    assert e.operations() == ["submit_or_run:REFERENCE_COMMIT"]


def test_mock_sbatch_phase_returns_pending_jobid():
    e = MockPhaseExecutor(treat_as_sbatch={"GAUSSIAN", "AIMALL"})
    r1 = e.submit_or_run(_state(), CampaignPhase.GAUSSIAN)
    r2 = e.submit_or_run(_state(), CampaignPhase.AIMALL)
    assert r1.is_complete is False
    assert r1.submitted_job_id == "MOCK-1"
    assert r2.submitted_job_id == "MOCK-2"


def test_mock_postprocess_default_complete():
    e = MockPhaseExecutor()
    r = e.postprocess(_state(), CampaignPhase.FEREBUS, observations=[])
    assert r.is_complete is True
    assert r.failure_reason is None


def test_mock_postprocess_failure_injection():
    e = MockPhaseExecutor(fail_on_phases={"GAUSSIAN"})
    r = e.postprocess(_state(), CampaignPhase.GAUSSIAN, observations=[])
    assert r.is_complete is False
    assert r.failure_reason and "GAUSSIAN" in r.failure_reason


def test_mock_handle_failure_returns_configured_action():
    e = MockPhaseExecutor(fail_action=FailureAction.HALT)
    action = e.handle_failure(_state(), CampaignPhase.AIMALL, observations=[])
    assert action == FailureAction.HALT


def test_mock_operations_log_in_call_order():
    e = MockPhaseExecutor(treat_as_sbatch={"FEREBUS"})
    e.submit_or_run(_state(), CampaignPhase.SEED_SELECT)
    e.submit_or_run(_state(), CampaignPhase.FEREBUS)
    e.postprocess(_state(), CampaignPhase.FEREBUS, observations=[])
    assert e.operations() == [
        "submit_or_run:SEED_SELECT",
        "submit_or_run:FEREBUS",
        "postprocess:FEREBUS",
    ]
