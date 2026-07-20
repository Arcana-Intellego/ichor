"""Tests for ichor.hpc.active_learning.daemon.daemon."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import (
    DAEMON_LOCK_FILENAME,
    Daemon,
    DaemonAlreadyRunningError,
    PHASE_ORDER,
    TickStatus,
    next_phase,
)
from ichor.hpc.active_learning.daemon.journal import iter_events
from ichor.hpc.active_learning.daemon.phase_executor import (
    FailureAction,
    MockPhaseExecutor,
    PhaseResult,
    PostprocessRetryDisposition,
)
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    StateSchemaError,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.submit.sacct_poll import JobObservation, JobStatus


_SBATCH_PHASES = (
    "PHASE_A_DIVERSITY", "INITIAL_GAUSSIAN", "INITIAL_AIMALL", "INITIAL_FEREBUS",
    "ARIADNE_ARRAY", "PHASE_B_DIVERSITY", "GAUSSIAN", "AIMALL", "FEREBUS",
)


def _completed_poll(job_id, **kw):
    return [JobObservation(job_id=job_id, status=JobStatus.COMPLETED, exit_code=(0, 0), elapsed_seconds=1)]


def _failed_poll(job_id, **kw):
    return [JobObservation(job_id=job_id, status=JobStatus.FAILED, exit_code=(1, 0), elapsed_seconds=1)]


def _node_fail_poll(job_id, **kw):
    return [JobObservation(job_id=job_id, status=JobStatus.NODE_FAIL, exit_code=(1, 0), elapsed_seconds=1)]


def _timeout_poll(job_id, **kw):
    return [JobObservation(job_id=job_id, status=JobStatus.TIMEOUT, exit_code=(0, 0), elapsed_seconds=1)]


def _running_poll(job_id, **kw):
    return [JobObservation(job_id=job_id, status=JobStatus.RUNNING, exit_code=None, elapsed_seconds=None)]


def _failed_ferebus_array_poll(job_id, **kw):
    return [
        JobObservation(
            job_id=str(job_id) + "_" + str(i),
            status=JobStatus.FAILED,
            exit_code=(1, 0),
            elapsed_seconds=1,
        )
        for i in range(12)
    ]


def _half_successful_array_poll(job_id, **kw):
    return [
        JobObservation(
            job_id=str(job_id) + "_0",
            status=JobStatus.COMPLETED,
            exit_code=(0, 0),
            elapsed_seconds=1,
        ),
        JobObservation(
            job_id=str(job_id) + "_1",
            status=JobStatus.FAILED,
            exit_code=(1, 0),
            elapsed_seconds=1,
        ),
    ]


class _StrictMockExecutor(MockPhaseExecutor):
    strict_committed_artifact_verification = True


def _make_daemon(
    tmp_path,
    *,
    max_iterations=1,
    executor=None,
    sacct=None,
    job_liveness_checker=None,
    job_name_accounting_finder=None,
) -> Daemon:
    cfg = CampaignConfig(max_iterations=max_iterations)
    executor = executor or MockPhaseExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    sacct = sacct or _completed_poll
    daemon = Daemon(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        executor=executor,
        sacct_poller=sacct,
        sleep_fn=lambda s: None,
        job_liveness_checker=job_liveness_checker,
        job_name_accounting_finder=job_name_accounting_finder,
    )
    daemon.data_dir().mkdir(parents=True, exist_ok=True)
    return daemon


def test_run_handles_malformed_state_json_without_traceback(tmp_path):
    d = _make_daemon(tmp_path)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    d.state_path().write_text("{bad json", encoding="utf-8")
    startup = []

    rc = d.run(
        max_ticks=1,
        startup_callback=lambda state, stage, failure: startup.append(
            (state, stage, failure)
        ),
    )

    assert rc == 2
    assert startup[0][0:2] == ("failed", "state_validation")
    assert not d.pid_path().exists()
    events = list(iter_events(d.data_dir() / "journal.ndjson"))
    assert any(e.get("event") == "state_corrupt" for e in events)


def test_run_acknowledges_ownership_before_environment_transition(
    tmp_path,
    monkeypatch,
):
    d = _make_daemon(tmp_path)
    write_state(d.state_path(), fresh_campaign_state(max_iterations=1))
    startup = []
    startup_repairs = []

    def repair_unsubmitted_intents(state):
        startup_repairs.append(state.phase.value)

    def prepare_environment(state):
        assert startup_repairs == [state.phase.value]
        assert startup[-1][0:2] == (
            "ownership_acquired",
            "environment_transition",
        )

    monkeypatch.setattr(
        d,
        "_repair_completed_unsubmitted_intents",
        repair_unsubmitted_intents,
    )
    monkeypatch.setattr(d, "_prepare_environment_generation", prepare_environment)

    rc = d.run(
        max_ticks=0,
        startup_callback=lambda state, stage, failure: startup.append(
            (state, stage, failure)
        ),
    )

    assert rc == 0
    assert [item[0] for item in startup] == [
        "ownership_acquired",
        "ready",
        "stopped",
    ]
    assert not d.pid_path().exists()


def test_run_environment_transition_failure_is_reported_and_removes_pid(
    tmp_path,
    monkeypatch,
):
    d = _make_daemon(tmp_path)
    write_state(d.state_path(), fresh_campaign_state(max_iterations=1))
    startup = []

    def fail_environment(state):
        raise RuntimeError("environment probe failed")

    monkeypatch.setattr(d, "_prepare_environment_generation", fail_environment)

    rc = d.run(
        max_ticks=0,
        startup_callback=lambda state, stage, failure: startup.append(
            (state, stage, failure)
        ),
    )

    assert rc == 13
    assert [item[0] for item in startup] == ["ownership_acquired", "failed"]
    assert startup[-1][1] == "environment_transition"
    assert "environment probe failed" in startup[-1][2]
    assert not d.pid_path().exists()


def test_pre_submit_intent_without_job_id_recovers_by_accounting_name(tmp_path):
    def lookup(state, phase, intent):
        assert phase is CampaignPhase.PHASE_A_DIVERSITY
        assert intent["status"] == "PRE_SUBMIT"
        return SimpleNamespace(
            job_id="777",
            terminal=True,
            successful=True,
            failed=False,
            inconclusive=False,
            rows=[("777", "COMPLETED")],
            reason="terminal_success",
        )

    d = _make_daemon(tmp_path, job_name_accounting_finder=lookup)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    write_state(d.state_path(), state)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
    )

    status = d.tick()

    assert status is TickStatus.SUBMITTED
    recovered = read_state(d.state_path())
    assert recovered.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] == "777"
    intent = submission_intent.load_intent(
        d.campaign_dir,
        CampaignPhase.PHASE_A_DIVERSITY.value,
        0,
    )
    assert intent["status"] == "ADOPTED"
    assert intent["job_id"] == "777"


def test_queue_lifecycle_timestamps_are_recorded_for_successful_job(tmp_path):
    d = _make_daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = "999"
    write_state(d.state_path(), state)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
    )
    submission_intent.mark_submitted(
        d.campaign_dir,
        CampaignPhase.PHASE_A_DIVERSITY.value,
        0,
        "999",
        expected_tasks=1,
    )

    assert d.tick() == TickStatus.ADVANCED

    intent = submission_intent.load_intent(
        d.campaign_dir,
        CampaignPhase.PHASE_A_DIVERSITY.value,
        0,
    )
    lifecycle = intent["queue_lifecycle"]
    assert lifecycle["submitted_at_iso"]
    assert lifecycle["first_sacct_at_iso"]
    assert lifecycle["first_sacct_status"] == "COMPLETED"
    assert lifecycle["terminal_at_iso"]
    assert lifecycle["terminal_status"] == "COMPLETED"
    assert lifecycle["postprocess_started_at_iso"]
    assert lifecycle["postprocess_finished_at_iso"]
    assert lifecycle["postprocess_seconds"] >= 0.0
    assert lifecycle["completed_at_iso"]
    events = list(iter_events(d.journal_path()))
    lifecycle_events = [
        e for e in events if e.get("event") == "queue_lifecycle_update"
    ]
    assert any(
        e.get("queue_event") == "first_sacct"
        and e.get("status") == "COMPLETED"
        and "first_sacct_at_iso" in e.get("changed_keys", [])
        for e in lifecycle_events
    )


def test_pre_submit_intent_without_accounted_job_halts_instead_of_resubmitting(tmp_path):
    def lookup(state, phase, intent):
        return SimpleNamespace(
            job_id=None,
            terminal=False,
            successful=False,
            failed=False,
            inconclusive=False,
            rows=[],
            reason="not_found",
        )

    executor = MockPhaseExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor, job_name_accounting_finder=lookup)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    write_state(d.state_path(), state)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.PHASE_A_DIVERSITY.value,
        iteration=0,
    )

    status = d.tick()

    assert status is TickStatus.HALTED
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.HALTED
    assert not [c for c in executor.calls if c.operation == "submit_or_run"]
    events = list(iter_events(d.data_dir() / "journal.ndjson"))
    assert any(
        e.get("event") == "halt"
        and "pre_submit_no_job_id_no_accounted_job" in str(e.get("reason"))
        for e in events
    )


def test_next_phase_progression_through_first_iteration():
    cur = CampaignPhase.INIT
    iteration = 0
    seen = [cur]
    while cur is not CampaignPhase.STOP_CHECK:
        cur, iteration = next_phase(cur, iteration, 3)
        seen.append(cur)
    # All 14 phases visited
    assert seen[0] is CampaignPhase.INIT
    assert seen[-1] is CampaignPhase.STOP_CHECK
    assert iteration == 1


def test_next_phase_stop_check_loops_when_iteration_below_max():
    nxt, ni = next_phase(CampaignPhase.STOP_CHECK, iteration=1, max_iterations=3)
    assert nxt is CampaignPhase.SEED_SELECT
    assert ni == 2


def test_next_phase_stop_check_terminates_when_iteration_reaches_max():
    nxt, ni = next_phase(CampaignPhase.STOP_CHECK, iteration=3, max_iterations=3)
    assert nxt is CampaignPhase.DONE
    assert ni == 3


def test_tick_initialises_state_on_first_call(tmp_path):
    d = _make_daemon(tmp_path)
    assert not d.state_path().exists()
    status = d.tick()
    # First tick performs INIT -> PHASE_A_DIVERSITY advance (INIT is inline).
    assert status == TickStatus.ADVANCED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.PHASE_A_DIVERSITY


def test_tick_refuses_fresh_state_when_campaign_has_committed_artifacts(tmp_path):
    d = _make_daemon(tmp_path)
    committed = d.campaign_dir / "QM_REFERENCE_DATA" / "iteration-000000"
    committed.mkdir(parents=True)
    (committed / "marker.txt").write_text("training\n", encoding="utf-8")

    with pytest.raises(StateSchemaError, match="state.json is missing"):
        d.tick()

    assert not d.state_path().exists()


def test_tick_submits_sbatch_phase_then_polls(tmp_path):
    d = _make_daemon(tmp_path)
    d.tick()  # INIT -> PHASE_A_DIVERSITY
    # Now phase is PHASE_A_DIVERSITY which is SLURM-backed.
    status = d.tick()
    assert status == TickStatus.SUBMITTED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.PHASE_A_DIVERSITY
    assert state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] == "MOCK-1"
    # Next tick: poll returns COMPLETED -> advance.
    status = d.tick()
    assert status == TickStatus.ADVANCED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.INITIAL_GAUSSIAN
    assert state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] is None


def test_sbatch_journal_metadata_reserved_keys_do_not_crash_or_override(tmp_path):
    class MetadataExecutor(MockPhaseExecutor):
        def __init__(self):
            super().__init__(treat_as_sbatch=set(_SBATCH_PHASES))

        def submit_or_run(self, state, phase):
            if phase is CampaignPhase.PHASE_A_DIVERSITY:
                return PhaseResult(
                    is_complete=False,
                    submitted_job_id="123",
                    expected_tasks=1,
                    submission_metadata={
                        "phase": "BAD",
                        "job_id": "BAD",
                        "iteration": 999,
                        "expected_tasks": 999,
                        "submitted_at_iso": "BAD",
                        "event": "BAD",
                        "ts": "BAD",
                        "array_recovery": {"n_retry": 1},
                    },
                )
            return super().submit_or_run(state, phase)

    d = _make_daemon(tmp_path, executor=MetadataExecutor())
    d.tick()  # INIT -> PHASE_A_DIVERSITY

    status = d.tick()

    assert status == TickStatus.SUBMITTED
    events = list(iter_events(d.journal_path()))
    sbatch = [event for event in events if event["event"] == "sbatch"][-1]
    assert sbatch["phase"] == CampaignPhase.PHASE_A_DIVERSITY.value
    assert sbatch["job_id"] == "123"
    assert sbatch["iteration"] == 0
    assert sbatch["expected_tasks"] == 1
    assert sbatch["array_recovery"] == {"n_retry": 1}


def test_phase_entry_exception_halts_and_persists_state(tmp_path):
    class RaisingExecutor(MockPhaseExecutor):
        def submit_or_run(self, state, phase):
            raise RuntimeError("simulated inline failure")

    d = _make_daemon(tmp_path, executor=RaisingExecutor())
    status = d.tick()

    assert status == TickStatus.HALTED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.HALTED
    events = list(iter_events(d.journal_path()))
    halt_events = [e for e in events if e.get("event") == "halt"]
    assert halt_events
    assert "phase_entry_exception: RuntimeError" in halt_events[-1]["reason"]


def test_tick_polling_keeps_state_when_job_running(tmp_path):
    d = _make_daemon(tmp_path, sacct=_running_poll)
    d.tick()  # INIT -> PHASE_A_DIVERSITY
    d.tick()  # submit PHASE_A_DIVERSITY
    status = d.tick()  # poll -> RUNNING -> POLLING
    assert status == TickStatus.POLLING
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.PHASE_A_DIVERSITY
    assert state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] == "MOCK-1"


def _install_pending_array_state(d: Daemon, *, phase=CampaignPhase.INITIAL_AIMALL) -> None:
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = phase
    state.pending_jobs[phase.value] = "777"
    write_state(d.state_path(), state)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=phase.value,
        iteration=0,
        expected_tasks=5,
    )
    submission_intent.mark_submitted(
        d.campaign_dir,
        phase.value,
        0,
        "777",
        expected_tasks=5,
    )


def _partial_array_poll(job_id, **kw):
    return [
        JobObservation(
            job_id=str(job_id) + "_0",
            status=JobStatus.COMPLETED,
            exit_code=(0, 0),
            elapsed_seconds=1,
        ),
        JobObservation(
            job_id=str(job_id) + "_1",
            status=JobStatus.COMPLETED,
            exit_code=(0, 0),
            elapsed_seconds=1,
        ),
    ]


def test_phase_entry_adopts_active_intent_job_id_when_squeue_active(tmp_path):
    campaign = tmp_path / "campaign"
    cfg = CampaignConfig(max_iterations=1)
    d = Daemon(
        campaign_dir=campaign,
        config=cfg,
        executor=MockPhaseExecutor(treat_as_sbatch=set(_SBATCH_PHASES)),
        job_finder=lambda state, phase: SimpleNamespace(
            job_id=None,
            inconclusive=False,
            rows=[],
        ),
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=True,
            inconclusive=False,
            rows=[(str(job_id) + "_[0-4%2]", "PENDING")],
            error=None,
        ),
    )
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.INITIAL_GAUSSIAN
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_GAUSSIAN.value,
        iteration=0,
        expected_tasks=5,
    )
    submission_intent.mark_submitted(
        campaign,
        CampaignPhase.INITIAL_GAUSSIAN.value,
        0,
        "888",
        expected_tasks=5,
    )

    assert d._on_phase_entry(state, state.phase) == TickStatus.SUBMITTED

    persisted = read_state(d.state_path())
    assert persisted.phase is CampaignPhase.INITIAL_GAUSSIAN
    assert persisted.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "888"
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_GAUSSIAN.value,
        0,
    )
    assert intent["status"] == "ADOPTED"
    assert intent["job_id"] == "888"


def test_missing_sacct_rows_keep_polling_when_squeue_still_active(tmp_path):
    d = _make_daemon(
        tmp_path,
        sacct=_partial_array_poll,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=True,
            inconclusive=False,
            rows=[
                (str(job_id) + "_2", "RUNNING"),
                (str(job_id) + "_[3-4]", "PENDING"),
            ],
            error=None,
        ),
    )
    d.config.runtime.poll_sacct_missing_max_ticks = 2
    _install_pending_array_state(d)

    assert d.tick() == TickStatus.POLLING
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.INITIAL_AIMALL
    assert state.pending_jobs[CampaignPhase.INITIAL_AIMALL.value] == "777"
    assert state.sacct_empty_streak.get("777:MISSING") == 1
    events = list(iter_events(d.journal_path()))
    assert any(e.get("event") == "sacct_rows_missing_but_squeue_active" for e in events)

    assert d.tick() == TickStatus.POLLING
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.INITIAL_AIMALL
    assert halted.pending_jobs[CampaignPhase.INITIAL_AIMALL.value] == "777"
    assert halted.sacct_empty_streak.get("777:MISSING") == 2
    intent = submission_intent.load_intent(d.campaign_dir, CampaignPhase.INITIAL_AIMALL.value, 0)
    assert intent["status"] == "SUBMITTED"
    events = list(iter_events(d.journal_path()))
    assert not any(e.get("event") == "sacct_missing_timeout" for e in events)


def test_missing_sacct_rows_keep_polling_when_squeue_inconclusive(tmp_path):
    d = _make_daemon(
        tmp_path,
        sacct=_partial_array_poll,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=False,
            inconclusive=True,
            rows=[],
            error="squeue transient failure",
        ),
    )
    d.config.runtime.poll_sacct_missing_max_ticks = 2
    _install_pending_array_state(d)

    assert d.tick() == TickStatus.POLLING
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.INITIAL_AIMALL
    assert state.sacct_empty_streak.get("777:MISSING") == 1
    events = list(iter_events(d.journal_path()))
    assert any(e.get("event") == "squeue_liveness_inconclusive" for e in events)

    assert d.tick() == TickStatus.POLLING
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.INITIAL_AIMALL
    assert halted.pending_jobs[CampaignPhase.INITIAL_AIMALL.value] == "777"
    assert halted.sacct_empty_streak.get("777:MISSING") == 2
    intent = submission_intent.load_intent(d.campaign_dir, CampaignPhase.INITIAL_AIMALL.value, 0)
    assert intent["status"] == "SUBMITTED"
    events = list(iter_events(d.journal_path()))
    assert not any(e.get("event") == "sacct_missing_timeout" for e in events)


def test_missing_sacct_rows_halt_when_squeue_confirms_job_gone(tmp_path):
    d = _make_daemon(
        tmp_path,
        sacct=_partial_array_poll,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=False,
            inconclusive=False,
            rows=[],
            error=None,
        ),
    )
    d.config.runtime.poll_sacct_missing_max_ticks = 1
    _install_pending_array_state(d)

    assert d.tick() == TickStatus.HALTED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.HALTED
    events = list(iter_events(d.journal_path()))
    assert any(e.get("event") == "sacct_missing_timeout" for e in events)


def test_tick_scrubs_when_failure_below_threshold(tmp_path):
    # Default threshold is 0.5 -> tolerate up to 50% failures.
    # With one task and one failure, success_ratio = 0 < (1 - 0.5) = 0.5, so HALT.
    executor = MockPhaseExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor, sacct=_failed_poll)
    d.tick()  # INIT -> PHASE_A_DIVERSITY
    d.tick()  # submit PHASE_A_DIVERSITY
    status = d.tick()  # poll -> FAILED -> handle_failure (scrub by default)
    # SCRUB_AND_CONTINUE is the mock default action; full failure -> halt
    # because all failed and threshold not met. Actually with 1 failure / 1
    # task, success_ratio = 0 < 1 - 0.5 = 0.5; we go through _handle_failure
    # which returns SCRUB_AND_CONTINUE -> daemon advances.
    assert status == TickStatus.SCRUBBED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.INITIAL_GAUSSIAN  # advanced past failed phase


def test_inflight_failure_threshold_uses_submission_snapshot(tmp_path):
    class TwoTaskExecutor(MockPhaseExecutor):
        def submit_or_run(self, state, phase):
            result = super().submit_or_run(state, phase)
            if result.submitted_job_id is not None:
                result.expected_tasks = 2
            return result

    executor = TwoTaskExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(
        tmp_path,
        executor=executor,
        sacct=_half_successful_array_poll,
    )
    d.config.runtime.failure_threshold_fraction = 0.5
    assert d.tick() == TickStatus.ADVANCED
    assert d.tick() == TickStatus.SUBMITTED

    # An iteration-future edit must not reinterpret an already-submitted batch.
    d.config.runtime.failure_threshold_fraction = 0.0
    status = d.tick()

    assert status == TickStatus.ADVANCED
    assert "postprocess:PHASE_A_DIVERSITY" in executor.operations()
    assert "handle_failure:PHASE_A_DIVERSITY" not in executor.operations()


def test_tick_halts_when_executor_handles_failure_with_halt(tmp_path):
    executor = MockPhaseExecutor(
        treat_as_sbatch=set(_SBATCH_PHASES),
        fail_action=FailureAction.HALT,
    )
    d = _make_daemon(tmp_path, executor=executor, sacct=_failed_poll)
    d.tick()  # advance to PHASE_A_DIVERSITY
    d.tick()  # submit PHASE_A_DIVERSITY
    status = d.tick()  # poll fails, executor halts
    assert status == TickStatus.HALTED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.HALTED


def test_strict_sbatch_producer_failure_halts_instead_of_scrub(tmp_path):
    executor = _StrictMockExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor, sacct=_failed_poll)
    d.tick()  # INIT -> PHASE_A_DIVERSITY
    d.tick()  # submit PHASE_A_DIVERSITY

    status = d.tick()

    assert status == TickStatus.HALTED
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.HALTED
    events = list(iter_events(d.journal_path()))
    halt = [e for e in events if e.get("event") == "halt"][-1]
    assert "phase_failed_requires_postprocess" in halt["reason"]


def test_tick_terminal_state_returns_terminal(tmp_path):
    d = _make_daemon(tmp_path)
    # Force the state directly into DONE.
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.DONE
    write_state(d.state_path(), state)
    assert d.tick() == TickStatus.TERMINAL


def test_tick_shutdown_flag_short_circuits(tmp_path):
    d = _make_daemon(tmp_path)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=3)
    state.shutdown_requested = True
    write_state(d.state_path(), state)
    assert d.tick() == TickStatus.SHUTDOWN


def test_run_through_complete_campaign(tmp_path):
    """End-to-end: daemon should drive a mock campaign through 2 iterations
    and exit in DONE. ALL phases (inline + sbatch) get visited per iteration."""
    sleeps: List[float] = []
    d = _make_daemon(
        tmp_path,
        max_iterations=2,
        executor=MockPhaseExecutor(treat_as_sbatch=set(_SBATCH_PHASES)),
        sacct=_completed_poll,
    )
    d.sleep_fn = lambda s: sleeps.append(s)
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE
    # Two iterations means each per-iter sbatch phase ran twice (5 phases x 2 iters)
    # plus the initial 4 sbatch phases (PHASE_A_DIVERSITY + INITIAL_*).
    events = list(iter_events(d.journal_path()))
    sbatch_events = [e for e in events if e["event"] == "sbatch"]
    expected_sbatch = 4 + 2 * 5      # 4 initial + 5 per-iter * 2 iters
    assert len(sbatch_events) == expected_sbatch


def test_lock_blocks_second_daemon_in_same_process(tmp_path):
    d1 = _make_daemon(tmp_path)
    d2 = _make_daemon(tmp_path)
    with d1._acquire_lock():
        with pytest.raises(DaemonAlreadyRunningError):
            with d2._acquire_lock():
                pass


def test_fresh_lease_blocks_second_daemon(tmp_path):
    d1 = _make_daemon(tmp_path)
    d2 = _make_daemon(tmp_path)

    with d1._acquire_lease():
        assert d1.heartbeat_path().is_file()
        with pytest.raises(DaemonAlreadyRunningError, match="lease is active or inconclusive"):
            with d2._acquire_lease():
                pass

    assert not d1.lease_path().exists()
    events = list(iter_events(d1.journal_path()))
    conflicts = [e for e in events if e["event"] == "daemon_lease_conflict"]
    assert conflicts
    assert conflicts[-1]["phase"] == "?"


def test_stale_lease_is_recovered(tmp_path):
    d = _make_daemon(tmp_path)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    d.lease_path().mkdir()
    d.heartbeat_path().write_text(
        json.dumps({
            "schema_version": 2,
            "owner_token": "a" * 32,
            "time": time.time() - 3600.0,
            "pid": 123,
            "host": "old-host",
            "phase": "AIMALL",
            "iteration": 4,
        }),
        encoding="utf-8",
    )
    d.config.runtime.lease_stale_seconds = 1

    with d._acquire_lease():
        assert d.heartbeat_path().is_file()

    stale_dirs = list(d.data_dir().glob("daemon.lease.d.stale.*"))
    assert stale_dirs
    events = list(iter_events(d.journal_path()))
    recovered = [e for e in events if e["event"] == "daemon_lease_stale_recovered"]
    assert recovered
    assert recovered[-1]["previous_host"] == "old-host"
    assert recovered[-1]["previous_phase"] == "AIMALL"


def test_stale_lease_owner_cannot_remove_successor_lease(tmp_path):
    d = _make_daemon(tmp_path)
    d.config.runtime.lease_heartbeat_seconds = 300

    with d._acquire_lease():
        successor_token = "b" * 32
        payload = json.loads(d.heartbeat_path().read_text(encoding="utf-8"))
        payload["owner_token"] = successor_token
        d.heartbeat_path().write_text(json.dumps(payload), encoding="utf-8")

    assert d.lease_path().is_dir()
    successor = json.loads(d.heartbeat_path().read_text(encoding="utf-8"))
    assert successor["owner_token"] == successor_token
    events = list(iter_events(d.journal_path()))
    assert any(event["event"] == "daemon_lease_cleanup_failed" for event in events)


def test_request_shutdown_writes_flag_to_state(tmp_path):
    d = _make_daemon(tmp_path)
    d.tick()  # initialises state.json
    d.request_shutdown()
    state = read_state(d.state_path())
    assert state.shutdown_requested is True


def test_postprocess_settle_retries_initially_missing_artifacts(tmp_path):
    class SettlingExecutor(MockPhaseExecutor):
        def __init__(self):
            super().__init__(treat_as_sbatch=set(_SBATCH_PHASES))
            self.calls = 0

        def postprocess(self, state, phase, observations):
            self.calls += 1
            if self.calls == 1:
                return PhaseResult(
                    is_complete=True,
                    failure_reason="expected_model_missing",
                    retry_disposition=(
                        PostprocessRetryDisposition.FILESYSTEM_SETTLE
                    ),
                )
            return PhaseResult(is_complete=True)

    sleeps = []
    executor = SettlingExecutor()
    d = _make_daemon(tmp_path, executor=executor, sacct=_completed_poll)
    d.config.runtime.postprocess_settle_attempts = 2
    d.config.runtime.postprocess_settle_seconds = 5
    d.sleep_fn = lambda seconds: sleeps.append(seconds)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = "MOCK-1"
    write_state(d.state_path(), state)

    status = d.tick()

    assert status == TickStatus.ADVANCED
    assert executor.calls == 2
    assert sleeps == [5.0]
    events = list(iter_events(d.journal_path()))
    assert any(e.get("event") == "postprocess_settle_retry" for e in events)


def test_postprocess_failure_text_does_not_implicitly_retry(tmp_path):
    class TerminalExecutor(MockPhaseExecutor):
        def __init__(self):
            super().__init__(treat_as_sbatch=set(_SBATCH_PHASES))
            self.calls = 0

        def postprocess(self, state, phase, observations):
            self.calls += 1
            return PhaseResult(
                is_complete=True,
                failure_reason="quality_measurement_missing_but_terminal",
            )

    executor = TerminalExecutor()
    d = _make_daemon(tmp_path, executor=executor, sacct=_completed_poll)
    d.config.runtime.postprocess_settle_attempts = 3
    d.config.runtime.postprocess_settle_seconds = 0
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = "MOCK-1"
    write_state(d.state_path(), state)

    status = d.tick()

    assert status == TickStatus.HALTED
    assert executor.calls == 1
    assert not any(
        event.get("event") == "postprocess_settle_retry"
        for event in iter_events(d.journal_path())
    )


def test_phase_entry_complete_failure_halts_instead_of_advancing(tmp_path):
    class FailedInlineExecutor(MockPhaseExecutor):
        def submit_or_run(self, state, phase):
            return PhaseResult(
                is_complete=True,
                failure_reason="postprocess_only_validation_failed",
            )

    d = _make_daemon(tmp_path, executor=FailedInlineExecutor())
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    write_state(d.state_path(), state)

    status = d.tick()

    assert status == TickStatus.HALTED
    recovered = read_state(d.state_path())
    assert recovered.phase is CampaignPhase.HALTED
    events = list(iter_events(d.journal_path()))
    assert events[-1]["event"] == "halt"
    assert events[-1]["reason"] == "postprocess_only_validation_failed"


def test_existing_jobless_ariadne_postprocess_intent_skips_scheduler_lookup(
    tmp_path,
    monkeypatch,
):
    class LocalPostprocessExecutor(MockPhaseExecutor):
        def __init__(self):
            super().__init__(treat_as_sbatch=set(_SBATCH_PHASES))
            self.calls = 0

        def submit_or_run(self, state, phase):
            self.calls += 1
            return PhaseResult(is_complete=True)

    executor = LocalPostprocessExecutor()

    def unexpected_scheduler_lookup(*_args, **_kwargs):
        raise AssertionError("scheduler lookup attempted for local postprocessing")

    d = _make_daemon(
        tmp_path,
        executor=executor,
        job_name_accounting_finder=unexpected_scheduler_lookup,
    )
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    source = {
        "decision_contract": {
            "failure_threshold_fraction": 0.25,
            "config_sha256": "c" * 64,
        },
        "source_sha256": "d" * 64,
    }
    active_intent = {
        "status": "PRE_SUBMIT",
        "job_id": None,
        "postprocess_source": source,
        "environment_generation": 1,
        "environment_generation_digest_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        d,
        "_ariadne_postprocess_source_if_complete",
        lambda _state: source,
    )
    monkeypatch.setattr(
        submission_intent,
        "load_active_intent",
        lambda *_args, **_kwargs: active_intent,
    )
    monkeypatch.setattr(d, "_verify_environment_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        d,
        "_verify_committed_artifacts_if_enabled",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(d, "_checkpoint_before_seed_selection", lambda *_args: None)
    monkeypatch.setattr(d, "_verify_intent_environment_binding", lambda *_args: None)
    monkeypatch.setattr(d, "_advance", lambda *_args, **_kwargs: True)
    completed = []
    monkeypatch.setattr(
        d,
        "_complete_intent_after_advance",
        lambda *args, **kwargs: completed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        submission_intent,
        "write_pre_submit_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replacement postprocess intent was written")
        ),
    )

    status = d._on_phase_entry(state, CampaignPhase.ARIADNE_ARRAY)

    assert status == TickStatus.ADVANCED
    assert executor.calls == 1
    assert len(completed) == 1


def test_ariadne_postprocess_intent_copies_original_decision_contract(
    tmp_path,
    monkeypatch,
):
    class LocalPostprocessExecutor(MockPhaseExecutor):
        def submit_or_run(self, state, phase):
            return PhaseResult(is_complete=True)

    d = _make_daemon(tmp_path, executor=LocalPostprocessExecutor())
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.ARIADNE_ARRAY
    state.iteration = 1
    d._last_environment_binding = {
        "generation": 7,
        "generation_digest_sha256": "7" * 64,
    }
    source = {
        "decision_contract": {
            "failure_threshold_fraction": 0.125,
            "config_sha256": "6" * 64,
        },
        "source_sha256": "5" * 64,
    }
    monkeypatch.setattr(
        d,
        "_ariadne_postprocess_source_if_complete",
        lambda _state: source,
    )
    monkeypatch.setattr(
        submission_intent,
        "load_active_intent",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(d, "_verify_environment_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        d,
        "_verify_committed_artifacts_if_enabled",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(d, "_checkpoint_before_seed_selection", lambda *_args: None)
    monkeypatch.setattr(d, "_infer_expected_tasks_from_artifacts", lambda *_args: 3)
    captured = []
    monkeypatch.setattr(
        submission_intent,
        "write_pre_submit_intent",
        lambda *_args, **kwargs: captured.append(kwargs) or {},
    )
    monkeypatch.setattr(d, "_advance", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(d, "_complete_intent_after_advance", lambda *_args, **_kwargs: None)

    status = d._on_phase_entry(state, CampaignPhase.ARIADNE_ARRAY)

    assert status == TickStatus.ADVANCED
    assert len(captured) == 1
    assert captured[0]["decision_contract"] == source["decision_contract"]
    assert captured[0]["postprocess_source"] == source
    assert captured[0]["environment_generation"] == 7


def test_scheduler_free_completion_retires_pre_submit_intent(tmp_path):
    d = _make_daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=1)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
        expected_tasks=6,
    )
    reference = {
        "path": ".DATA/ACTIVE_LEARNING/phase_completions/" + "a" * 64 + ".json",
        "receipt_id": "a" * 64,
        "sha256": "b" * 64,
    }

    d._complete_intent_after_advance(
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        reference,
        completed_without_submission=True,
    )
    d._complete_intent_after_advance(
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        reference,
        completed_without_submission=True,
    )

    retired = submission_intent.load_intent(
        d.campaign_dir,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert retired is not None
    assert retired["status"] == "SUPERSEDED"
    assert retired["completion_receipt"] == reference
    events = list(iter_events(d.journal_path()))
    assert sum(
        event.get("event") == "submission_intent_retired_without_submission"
        for event in events
    ) == 1
    assert not any(
        event.get("event") == "submission_intent_completion_deferred"
        for event in events
    )


def test_startup_repairs_historical_jobless_intent_from_completion_receipt(
    tmp_path,
):
    from ichor.hpc.active_learning.daemon.completion_receipts import (
        receipt_reference,
        write_completion_receipt,
    )

    d = _make_daemon(tmp_path)
    before = fresh_campaign_state(max_iterations=2)
    before.phase = CampaignPhase.INITIAL_FEREBUS
    before.reference_data_version = 0
    before.validation_set_version = 0
    intent = submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=before.campaign_uid,
        phase_name=before.phase.value,
        iteration=0,
        expected_tasks=6,
    )
    after = CampaignState.from_dict(before.to_dict())
    after.phase = CampaignPhase.SEED_SELECT
    after.iteration = 1
    after.models_version = 0
    receipt_path = write_completion_receipt(
        d.campaign_dir,
        campaign_uid=before.campaign_uid,
        phase=before.phase.value,
        iteration=0,
        replacement_round=0,
        config_sha256="c" * 64,
        state_before=before,
        state_after=after,
        next_phase=after.phase.value,
        next_iteration=after.iteration,
        state_updates={"models_version": 0},
        evidence=[],
        job_id=None,
        expected_tasks=6,
        submission_identity=str(intent["submission_identity"]),
    )
    after.last_completion_receipt = receipt_reference(
        d.campaign_dir,
        receipt_path,
    )

    assert d._recover_phase_completion(after) is None

    retired = submission_intent.load_intent(
        d.campaign_dir,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert retired is not None
    assert retired["status"] == "SUPERSEDED"
    assert retired["completion_receipt"] == after.last_completion_receipt
    assert not any(
        event.get("event") == "submission_intent_completion_deferred"
        for event in iter_events(d.journal_path())
    )


def test_scientific_convergence_transitions_to_done_with_durable_context(tmp_path):
    d = _make_daemon(tmp_path)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=10)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 3
    state.reference_data_version = 3
    state.models_version = 3
    write_state(d.state_path(), state)

    advanced = d._advance(
        state,
        CampaignPhase.STOP_CHECK,
        {"campaign_completion_reason": "alpha0_streak"},
    )

    assert advanced is True
    completed = read_state(d.state_path())
    assert completed.phase is CampaignPhase.DONE
    assert completed.shutdown_requested is False
    assert completed.lifecycle_context["disposition"] == "completed"
    assert completed.lifecycle_context["reason_code"] == "scientific_convergence"
    assert completed.lifecycle_context["details"]["criterion"] == "alpha0_streak"
    assert completed.last_completion_receipt is not None


def test_max_iterations_transitions_to_done_with_distinct_reason(tmp_path):
    d = _make_daemon(tmp_path)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=2)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 2
    state.reference_data_version = 2
    state.models_version = 2
    write_state(d.state_path(), state)

    assert d._advance(state, CampaignPhase.STOP_CHECK, {}) is True

    completed = read_state(d.state_path())
    assert completed.phase is CampaignPhase.DONE
    assert completed.iteration == 2
    assert completed.lifecycle_context["reason_code"] == "max_iterations_reached"
    assert completed.last_completion_receipt is not None


def test_run_loop_returns_nonzero_for_preexisting_halted_state(tmp_path):
    d = _make_daemon(tmp_path)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    write_state(d.state_path(), state)

    assert d._run_loop(max_ticks=1) == 20


def test_generic_tick_exception_writes_last_exception_sidecar_and_halts(tmp_path):
    d = _make_daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] = "12345"
    write_state(d.state_path(), state)

    def raise_tick():
        raise RuntimeError("simulated tick failure")

    d.tick = raise_tick

    assert d._run_loop(max_ticks=1) == 21

    payload = json.loads(d.last_exception_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["exception_type"] == "RuntimeError"
    assert "simulated tick failure" in payload["message"]
    assert payload["phase"] == CampaignPhase.PHASE_A_DIVERSITY.value
    assert payload["iteration"] == 0
    assert "traceback" in payload
    recovered = read_state(d.state_path())
    assert recovered.phase is CampaignPhase.HALTED
    assert recovered.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] == "12345"
    events = list(iter_events(d.journal_path()))
    assert any(e.get("event") == "tick_error" for e in events)
    assert any(e.get("event") == "tick_exception_halted" for e in events)


def test_generic_tick_exception_can_re_raise_when_halt_disabled(tmp_path):
    d = _make_daemon(tmp_path)
    d.config.runtime.halt_on_tick_exception = False
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.PHASE_A_DIVERSITY
    write_state(d.state_path(), state)

    def raise_tick():
        raise RuntimeError("simulated tick failure")

    d.tick = raise_tick

    with pytest.raises(RuntimeError, match="simulated tick failure"):
        d._run_loop(max_ticks=1)

    assert read_state(d.state_path()).phase is CampaignPhase.PHASE_A_DIVERSITY


def test_transient_scheduler_failure_retries_once(tmp_path):
    d = _make_daemon(tmp_path, sacct=_node_fail_poll)
    d.tick()
    d.tick()

    status = d.tick()

    assert status == TickStatus.RETRYING
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.PHASE_A_DIVERSITY
    assert state.pending_jobs[CampaignPhase.PHASE_A_DIVERSITY.value] is None
    payload = json.loads(d.transient_retry_ledger_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert len(payload["attempts"]) == 1
    key, count = next(iter(payload["attempts"].items()))
    assert key.startswith("PHASE_A_DIVERSITY@0@round=0@tasks=")
    assert count == 1


def test_timeout_failure_does_not_transient_retry_by_default(tmp_path):
    d = _make_daemon(tmp_path, sacct=_timeout_poll)
    d.tick()
    d.tick()

    status = d.tick()

    assert status == TickStatus.SCRUBBED
    assert not d.transient_retry_ledger_path().exists()
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.INITIAL_GAUSSIAN


def test_required_ferebus_output_missing_after_failure_halts_at_producer(tmp_path):
    executor = _StrictMockExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor, sacct=_failed_ferebus_array_poll)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.INITIAL_FEREBUS
    state.pending_jobs[CampaignPhase.INITIAL_FEREBUS.value] = "16177329"
    write_state(d.state_path(), state)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
        expected_tasks=12,
        decision_contract=d._submission_decision_contract(),
    )
    submission_intent.mark_submitted(
        d.campaign_dir,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        "16177329",
        expected_tasks=12,
    )

    status = d.tick()

    assert status == TickStatus.HALTED
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.HALTED
    events = list(iter_events(d.journal_path()))
    assert any(
        e.get("event") == "required_phase_output_missing_after_failure"
        and e.get("phase") == "INITIAL_FEREBUS"
        for e in events
    )
    assert not any(
        e.get("event") == "phase_transition"
        and e.get("from_phase") == "INITIAL_FEREBUS"
        and e.get("to_phase") == "SEED_SELECT"
        for e in events
    )
    halt = [e for e in events if e.get("event") == "halt"][-1]
    assert halt["from_phase"] == "INITIAL_FEREBUS"
    assert "required_phase_output_missing_after_failure" in halt["reason"]


def test_required_ferebus_output_missing_after_success_halts_at_producer(tmp_path):
    executor = _StrictMockExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor, sacct=_completed_poll)
    d.data_dir().mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.INITIAL_FEREBUS
    state.pending_jobs[CampaignPhase.INITIAL_FEREBUS.value] = "16177329"
    write_state(d.state_path(), state)
    submission_intent.write_pre_submit_intent(
        d.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
        expected_tasks=1,
        decision_contract=d._submission_decision_contract(),
    )
    submission_intent.mark_submitted(
        d.campaign_dir,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        "16177329",
        expected_tasks=1,
    )

    status = d.tick()

    assert status == TickStatus.HALTED
    halted = read_state(d.state_path())
    assert halted.phase is CampaignPhase.HALTED
    events = list(iter_events(d.journal_path()))
    assert any(
        e.get("event") == "phase_output_contract_invalid"
        and e.get("phase") == "INITIAL_FEREBUS"
        for e in events
    )
    assert not any(e.get("event") == "phase_succeeded" for e in events)
    assert not any(
        e.get("event") == "phase_transition"
        and e.get("from_phase") == "INITIAL_FEREBUS"
        for e in events
    )


def test_ferebus_transition_requires_fresh_model_update(tmp_path):
    executor = _StrictMockExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.FEREBUS
    state.reference_data_version = 1
    state.models_version = 0

    reason = d._transition_output_contract_error(state, CampaignPhase.FEREBUS, {})

    assert reason is not None
    assert "fresh FEREBUS models_version missing" in reason


def test_ferebus_transition_rejects_training_model_skew(tmp_path):
    executor = _StrictMockExecutor(treat_as_sbatch=set(_SBATCH_PHASES))
    d = _make_daemon(tmp_path, executor=executor)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.FEREBUS
    state.reference_data_version = 2
    state.models_version = 1

    reason = d._transition_output_contract_error(
        state,
        CampaignPhase.FEREBUS,
        {"models_version": 1},
    )

    assert reason is not None
    assert "model/reference-data version skew" in reason


def test_journal_records_phase_transitions(tmp_path):
    d = _make_daemon(tmp_path)
    d.run(max_ticks=12, catch_keyboard_interrupt=False)
    events = list(iter_events(d.journal_path()))
    transitions = [e for e in events if e["event"] == "phase_transition"]
    assert transitions, "expected at least one phase_transition event"
    assert transitions[0]["from_phase"] == "INIT"
    assert transitions[0]["to_phase"] == "PHASE_A_DIVERSITY"


# --- M15 F3: _apply_state_updates strict contract ----------------------


def test_apply_state_updates_rejects_unknown_keys():
    """The state-update contract must reject keys not in the allowed set,
    rather than silently dropping them (which was a future-bug magnet)."""
    import pytest
    from pathlib import Path
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
    from ichor.hpc.active_learning.daemon.state import CampaignState

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cd = Path(d)
        (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
        daemon = Daemon(
            campaign_dir=cd, config=CampaignConfig(),
            executor=MockPhaseExecutor(), sleep_fn=lambda s: None,
        )
        st = CampaignState()
        with pytest.raises(ValueError, match="unexpected state update key"):
            daemon._apply_state_updates(st, {"banana_count": 99})


def test_apply_state_updates_accepts_M15_new_keys():
    from pathlib import Path
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
    from ichor.hpc.active_learning.daemon.state import CampaignState

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cd = Path(d)
        (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
        daemon = Daemon(
            campaign_dir=cd, config=CampaignConfig(),
            executor=MockPhaseExecutor(), sleep_fn=lambda s: None,
        )
        st = CampaignState()
        daemon._apply_state_updates(st, {
            "last_n_anti_overlap_flagged": 4,
            "sacct_empty_streak": {"1": 2},
        })
        assert st.last_n_anti_overlap_flagged == 4
        assert st.sacct_empty_streak == {"1": 2}


# --- M15 F13: STOP_CHECK ghost-iteration fix ---------------------------


def test_stop_check_shutdown_freezes_phase_and_iteration():
    """When _inline_stop_check returns shutdown_requested=True, the daemon
    must NOT advance phase or iteration. The on-disk state should reflect
    the iteration that actually completed, not the would-be next one.
    Verified by directly invoking _advance with the shutdown payload."""
    import tempfile
    from pathlib import Path
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, CampaignState,
    )

    with tempfile.TemporaryDirectory() as d:
        cd = Path(d)
        (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
        daemon = Daemon(
            campaign_dir=cd, config=CampaignConfig(),
            executor=MockPhaseExecutor(), sleep_fn=lambda s: None,
        )
        state = CampaignState(
            iteration=6, max_iterations=50,
            phase=CampaignPhase.STOP_CHECK,
        )
        daemon._advance(
            state, CampaignPhase.STOP_CHECK,
            {"shutdown_requested": True, "alpha_history": [0.01, 0.005, 0.002]},
        )
        # Phase and iteration MUST be unchanged.
        assert state.phase == CampaignPhase.STOP_CHECK
        assert state.iteration == 6
        # But shutdown_requested IS set.
        assert state.shutdown_requested is True
        # And alpha_history was applied.
        assert state.alpha_history == [0.01, 0.005, 0.002]


def test_stop_check_shutdown_journals_shutdown_requested():
    """The fix path must emit a shutdown_requested journal event so the
    operator-visible audit trail captures the trigger."""
    import json
    import tempfile
    from pathlib import Path
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, CampaignState,
    )

    with tempfile.TemporaryDirectory() as d:
        cd = Path(d)
        (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
        daemon = Daemon(
            campaign_dir=cd, config=CampaignConfig(),
            executor=MockPhaseExecutor(), sleep_fn=lambda s: None,
        )
        state = CampaignState(
            iteration=8, phase=CampaignPhase.STOP_CHECK,
        )
        daemon._advance(
            state, CampaignPhase.STOP_CHECK,
            {"shutdown_requested": True},
        )
        journal_path = cd / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
        events = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        shutdowns = [e for e in events if e.get("event") == "shutdown_requested"]
        assert shutdowns
        assert shutdowns[-1]["from_phase"] == "STOP_CHECK"
        assert shutdowns[-1]["iteration"] == 8


def test_advance_without_shutdown_still_increments_phase():
    """The fix must NOT break the normal advance path -- only the
    shutdown-flag case is short-circuited."""
    import tempfile
    from pathlib import Path
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, CampaignState,
    )

    with tempfile.TemporaryDirectory() as d:
        cd = Path(d)
        (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
        daemon = Daemon(
            campaign_dir=cd, config=CampaignConfig(),
            executor=MockPhaseExecutor(), sleep_fn=lambda s: None,
        )
        state = CampaignState(
            iteration=3, phase=CampaignPhase.STOP_CHECK,
        )
        daemon._advance(state, CampaignPhase.STOP_CHECK, {})
        # Normal STOP_CHECK -> SEED_SELECT advance.
        assert state.phase == CampaignPhase.SEED_SELECT
        assert state.iteration == 4
