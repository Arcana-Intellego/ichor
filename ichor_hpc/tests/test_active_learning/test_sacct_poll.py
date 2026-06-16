"""Tests for ichor.hpc.active_learning.submit.sacct_poll."""
from dataclasses import dataclass, field
from typing import List

import pytest

from ichor.hpc.active_learning.submit.sacct_poll import (
    ArrayJobSummary,
    FAILURE_STATES,
    JobObservation,
    JobStatus,
    SUCCESS_STATES,
    TERMINAL_STATES,
    aggregate_states,
    find_active_job_by_id_detailed,
    parse_sacct_output,
    poll_job,
)


@dataclass
class _StubResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _StubRunner:
    result: _StubResult
    calls: List[List[str]] = field(default_factory=list)

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        return self.result


def test_jobstatus_from_sacct_known_states():
    for s in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL",
              "CANCELLED", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
        assert JobStatus.from_sacct(s) is getattr(JobStatus, s)


def test_jobstatus_from_sacct_strips_plus_and_extra_text():
    assert JobStatus.from_sacct("CANCELLED+") is JobStatus.CANCELLED
    assert JobStatus.from_sacct("CANCELLED by 1234") is JobStatus.CANCELLED


def test_jobstatus_unknown_falls_back():
    assert JobStatus.from_sacct("WEIRD_NEW_STATE") is JobStatus.UNKNOWN
    assert JobStatus.from_sacct("") is JobStatus.UNKNOWN


def test_terminal_state_sets_are_consistent():
    assert SUCCESS_STATES <= TERMINAL_STATES
    assert FAILURE_STATES <= TERMINAL_STATES
    assert SUCCESS_STATES.isdisjoint(FAILURE_STATES)
    assert TERMINAL_STATES == SUCCESS_STATES | FAILURE_STATES
    assert JobStatus.RUNNING not in TERMINAL_STATES
    assert JobStatus.PENDING not in TERMINAL_STATES
    assert JobStatus.UNKNOWN not in TERMINAL_STATES


def test_parse_sacct_single_completed():
    out = "12345|COMPLETED|0:0|00:01:30\n"
    obs = parse_sacct_output(out)
    assert len(obs) == 1
    o = obs[0]
    assert o.job_id == "12345"
    assert o.status is JobStatus.COMPLETED
    assert o.exit_code == (0, 0)
    assert o.elapsed_seconds == 90
    assert o.is_terminal
    assert o.is_success


def test_parse_sacct_failed_with_exit_code():
    out = "12345|FAILED|2:0|00:05:00\n"
    obs = parse_sacct_output(out)
    assert obs[0].status is JobStatus.FAILED
    assert obs[0].exit_code == (2, 0)
    assert obs[0].is_failure


def test_completed_with_nonzero_or_malformed_exit_code_is_failure():
    out = "12345|COMPLETED|1:0|00:01:00\n12346|COMPLETED|not-an-exit|00:01:00\n"
    obs = parse_sacct_output(out)
    assert obs[0].is_failure
    assert obs[1].is_failure
    assert not obs[0].is_success
    assert not obs[1].is_success


def test_parse_sacct_handles_dd_hh_mm_ss_elapsed():
    out = "12345|TIMEOUT|0:0|2-04:00:00\n"
    obs = parse_sacct_output(out)
    assert obs[0].elapsed_seconds == 2 * 86400 + 4 * 3600


def test_parse_sacct_handles_running():
    out = "12345|RUNNING|0:0|00:10:00\n"
    obs = parse_sacct_output(out)
    assert obs[0].status is JobStatus.RUNNING
    assert not obs[0].is_terminal


def test_parse_sacct_handles_cancelled_plus():
    out = "12345|CANCELLED+|0:15|00:00:30\n"
    obs = parse_sacct_output(out)
    assert obs[0].status is JobStatus.CANCELLED
    assert obs[0].is_failure


def test_parse_sacct_skips_short_or_empty_rows():
    out = "\n12345|COMPLETED|0:0|00:00:30\nbroken row\n"
    obs = parse_sacct_output(out)
    # Only the well-formed row is kept.
    assert len(obs) == 1
    assert obs[0].job_id == "12345"


def test_aggregate_states_array_all_success():
    obs = [
        JobObservation("777_0", JobStatus.COMPLETED, (0, 0), 90),
        JobObservation("777_1", JobStatus.COMPLETED, (0, 0), 91),
        JobObservation("777_2", JobStatus.COMPLETED, (0, 0), 92),
    ]
    summary = aggregate_states("777", obs)
    assert summary.n_tasks == 3
    assert summary.n_completed == 3
    assert summary.n_failed == 0
    assert summary.n_pending_or_running == 0
    assert summary.failure_indices == []
    assert summary.is_terminal
    assert summary.is_fully_successful


def test_aggregate_states_array_mixed():
    obs = [
        JobObservation("777_0", JobStatus.COMPLETED, (0, 0), 90),
        JobObservation("777_1", JobStatus.NODE_FAIL, None, None),
        JobObservation("777_2", JobStatus.RUNNING, None, None),
        JobObservation("777_3", JobStatus.FAILED, (1, 0), 30),
    ]
    summary = aggregate_states("777", obs)
    assert summary.n_completed == 1
    assert summary.n_failed == 2
    assert summary.n_pending_or_running == 1
    assert summary.failure_indices == [1, 3]
    assert not summary.is_terminal
    assert not summary.is_fully_successful


def test_aggregate_drops_parent_row_when_task_rows_present():
    obs = [
        JobObservation("777",   JobStatus.COMPLETED, None, None),  # parent summary
        JobObservation("777_0", JobStatus.COMPLETED, (0, 0), 90),
        JobObservation("777_1", JobStatus.COMPLETED, (0, 0), 91),
    ]
    summary = aggregate_states("777", obs)
    assert summary.n_tasks == 2


def test_aggregate_single_non_array_job():
    obs = [JobObservation("99", JobStatus.COMPLETED, (0, 0), 100)]
    summary = aggregate_states("99", obs)
    assert summary.n_tasks == 1
    assert summary.is_fully_successful


def test_aggregate_parent_only_failed_is_terminal_failure_but_unknown_is_inconclusive():
    failed = aggregate_states("99", [JobObservation("99", JobStatus.FAILED, (1, 0), 100)])
    assert failed.is_terminal
    assert failed.n_failed == 1
    unknown = aggregate_states("99", [JobObservation("99", JobStatus.UNKNOWN, None, None)])
    assert not unknown.is_terminal
    assert unknown.n_failed == 0
    assert unknown.n_unknown == 1
    assert unknown.n_pending_or_running == 1


def test_expected_task_count_marks_missing_array_rows_pending():
    obs = [
        JobObservation("777_0", JobStatus.COMPLETED, (0, 0), 90),
        JobObservation("777_1", JobStatus.COMPLETED, (0, 0), 91),
    ]
    summary = aggregate_states("777", obs, expected_task_count=5)
    assert summary.n_tasks == 5
    assert summary.n_observed == 2
    assert summary.n_missing == 3
    assert summary.n_completed == 2
    assert summary.n_pending_or_running == 3
    assert not summary.is_terminal


def test_poll_job_invokes_sacct_with_correct_flags():
    runner = _StubRunner(result=_StubResult(stdout="42|COMPLETED|0:0|00:00:01\n"))
    obs = poll_job("42", sacct_runner=runner)
    assert obs[0].job_id == "42"
    assert obs[0].status is JobStatus.COMPLETED
    # Single sacct call with the expected flag set.
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call[0] == "sacct"
    assert "-j" in call and "42" in call
    assert "--format=JobID,State,ExitCode,Elapsed" in call
    assert "-X" in call and "-P" in call and "-n" in call


def test_poll_job_raises_on_sacct_nonzero():
    runner = _StubRunner(result=_StubResult(returncode=1, stderr="sacct: invalid job id"))
    with pytest.raises(RuntimeError, match="sacct exited with code 1"):
        poll_job("42", sacct_runner=runner)


def test_poll_job_passes_through_extra_args():
    runner = _StubRunner(result=_StubResult(stdout=""))
    poll_job("42", sacct_runner=runner, extra_args=["--starttime", "2024-01-01"])
    assert runner.calls[0][-2:] == ["--starttime", "2024-01-01"]


def test_find_active_job_by_id_uses_squeue_rows():
    runner = _StubRunner(
        result=_StubResult(stdout="16153025_[6-9%2]|PENDING\n16153025_4|RUNNING\n")
    )
    lookup = find_active_job_by_id_detailed("16153025", squeue_runner=runner)
    assert lookup.active
    assert not lookup.inconclusive
    assert lookup.rows == [
        ("16153025_[6-9%2]", "PENDING"),
        ("16153025_4", "RUNNING"),
    ]
    call = runner.calls[0]
    assert call[:3] == ["squeue", "-j", "16153025"]
    assert "--noheader" in call


def test_find_active_job_by_id_empty_squeue_is_conclusive_inactive():
    runner = _StubRunner(result=_StubResult(stdout=""))
    lookup = find_active_job_by_id_detailed("16153025", squeue_runner=runner)
    assert not lookup.active
    assert not lookup.inconclusive
    assert lookup.rows == []


def test_find_active_job_by_id_squeue_error_is_inconclusive():
    runner = _StubRunner(result=_StubResult(returncode=1, stderr="slurmctld busy"))
    lookup = find_active_job_by_id_detailed("16153025", squeue_runner=runner)
    assert not lookup.active
    assert lookup.inconclusive
    assert "squeue exited with code 1" in str(lookup.error)
