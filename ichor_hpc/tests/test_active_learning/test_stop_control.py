"""Tests for receipt-backed daemon stop control."""
import json

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.phase_executor import MockPhaseExecutor
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.stop_control import (
    StopControlError,
    build_stop_request,
    install_stop_request,
    read_stop_request,
    stop_request_history_dir,
    stop_request_path,
    update_stop_request,
)
from ichor.hpc.active_learning.submit.sacct_poll import JobObservation, JobStatus


def _completed_poll(job_id, **kwargs):
    del kwargs
    return [
        JobObservation(
            job_id=str(job_id),
            status=JobStatus.COMPLETED,
            exit_code=(0, 0),
            elapsed_seconds=1,
        )
    ]


def _daemon(tmp_path, *, max_iterations=3, executor=None):
    daemon = Daemon(
        campaign_dir=tmp_path / "campaign",
        config=CampaignConfig(max_iterations=max_iterations),
        executor=executor or MockPhaseExecutor(),
        sacct_poller=_completed_poll,
        sleep_fn=lambda seconds: None,
    )
    daemon.data_dir().mkdir(parents=True, exist_ok=True)
    return daemon


def test_stop_request_is_idempotent_and_immediate_supersedes_drain(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.AIMALL
    state.iteration = 2
    request = build_stop_request(
        state,
        mode="after_phase",
        phase_started=True,
    )

    created, disposition = install_stop_request(daemon.campaign_dir, request)
    repeated, repeated_disposition = install_stop_request(
        daemon.campaign_dir,
        build_stop_request(
            state,
            mode="after_phase",
            phase_started=True,
        ),
    )

    assert disposition == "created"
    assert repeated_disposition == "existing"
    assert repeated["request_id"] == created["request_id"]
    with pytest.raises(StopControlError, match="already active"):
        install_stop_request(
            daemon.campaign_dir,
            build_stop_request(state, mode="after_iteration"),
        )

    immediate, immediate_disposition = install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="immediate"),
    )

    assert immediate_disposition == "created"
    assert read_stop_request(daemon.campaign_dir)["request_id"] == immediate["request_id"]
    assert (stop_request_history_dir(daemon.campaign_dir) / (created["request_id"] + ".json")).is_file()


def test_malformed_stop_request_fails_closed(tmp_path):
    daemon = _daemon(tmp_path)
    path = stop_request_path(daemon.campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(StopControlError, match="request_id"):
        read_stop_request(daemon.campaign_dir)


def test_malformed_cancellation_summary_fails_closed(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    request = build_stop_request(state, mode="immediate")
    request["cancellation_summary"] = {
        "cancelled": [{"job_id": "123", "intent_keys": "not-a-list"}],
        "skipped": [],
        "failed": [],
    }

    with pytest.raises(StopControlError, match="intent_keys"):
        install_stop_request(daemon.campaign_dir, request)


def test_immediate_stop_is_latched_by_daemon_not_cli(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    write_state(daemon.state_path(), state)
    request, _ = install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="immediate"),
    )

    assert daemon.tick() == TickStatus.SHUTDOWN

    stopped = read_state(daemon.state_path())
    control = read_stop_request(daemon.campaign_dir)
    assert stopped.shutdown_requested is True
    assert stopped.phase is CampaignPhase.INIT
    assert control["request_id"] == request["request_id"]
    assert control["status"] == "completed"
    assert control["completion_reason"] == "immediate"


def test_after_phase_stops_before_an_unstarted_phase(tmp_path):
    class RefusingExecutor(MockPhaseExecutor):
        def submit_or_run(self, state, phase):
            raise AssertionError("an unstarted phase must not be entered")

    daemon = _daemon(tmp_path, executor=RefusingExecutor())
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    write_state(daemon.state_path(), state)
    install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="after_phase", phase_started=False),
    )

    assert daemon.tick() == TickStatus.SHUTDOWN
    stopped = read_state(daemon.state_path())
    assert stopped.phase is CampaignPhase.SEED_SELECT
    assert stopped.iteration == 1
    assert stopped.shutdown_requested is True
    assert read_stop_request(daemon.campaign_dir)["completion_reason"] == "phase_not_started"


def test_after_phase_finishes_started_phase_and_binds_completion_receipt(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.PHASE_A_POLUS
    state.pending_jobs[CampaignPhase.PHASE_A_POLUS.value] = "101"
    write_state(daemon.state_path(), state)
    install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="after_phase", phase_started=True),
    )

    assert daemon.tick() == TickStatus.ADVANCED

    stopped = read_state(daemon.state_path())
    control = read_stop_request(daemon.campaign_dir)
    assert stopped.phase is not CampaignPhase.PHASE_A_POLUS
    assert stopped.shutdown_requested is True
    assert stopped.last_completion_receipt is not None
    assert control["status"] == "completed"
    assert control["completion_reason"] == "phase_completed"
    assert control["completion_receipt"] == stopped.last_completion_receipt


def test_historical_same_phase_receipt_does_not_satisfy_new_drain(tmp_path):
    daemon = _daemon(tmp_path)
    original = fresh_campaign_state(max_iterations=3)
    original.phase = CampaignPhase.SEED_SELECT
    original.iteration = 1
    original.reference_data_version = 0
    original.models_version = 0
    write_state(daemon.state_path(), original)
    assert daemon._advance(original, CampaignPhase.SEED_SELECT, {}) is True
    historical_receipt = read_state(daemon.state_path()).last_completion_receipt

    rerun = fresh_campaign_state(
        max_iterations=3,
        campaign_uid=original.campaign_uid,
    )
    rerun.phase = CampaignPhase.SEED_SELECT
    rerun.iteration = 1
    rerun.reference_data_version = 0
    rerun.models_version = 0
    write_state(daemon.state_path(), rerun)
    install_stop_request(
        daemon.campaign_dir,
        build_stop_request(rerun, mode="after_phase", phase_started=True),
    )

    assert daemon.tick() == TickStatus.ADVANCED

    stopped = read_state(daemon.state_path())
    assert stopped.phase is not CampaignPhase.SEED_SELECT
    assert stopped.shutdown_requested is True
    assert stopped.last_completion_receipt != historical_receipt


def test_after_iteration_stops_on_stop_check_receipt(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.STOP_CHECK
    state.iteration = 1
    state.reference_data_version = 1
    state.models_version = 1
    write_state(daemon.state_path(), state)
    install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="after_iteration"),
    )

    assert daemon._advance(state, CampaignPhase.STOP_CHECK, {}) is True

    stopped = read_state(daemon.state_path())
    control = read_stop_request(daemon.campaign_dir)
    assert stopped.phase is CampaignPhase.SEED_SELECT
    assert stopped.iteration == 2
    assert stopped.shutdown_requested is True
    assert control["completion_reason"] == "active_iteration_completed"
    assert control["completion_receipt"] == stopped.last_completion_receipt


def test_crossed_iteration_boundary_is_recovered_from_existing_receipt(tmp_path):
    daemon = _daemon(tmp_path)
    observed = fresh_campaign_state(max_iterations=3)
    observed.phase = CampaignPhase.STOP_CHECK
    observed.iteration = 1
    observed.reference_data_version = 1
    observed.models_version = 1
    request = build_stop_request(observed, mode="after_iteration")
    write_state(daemon.state_path(), observed)
    assert daemon._advance(observed, CampaignPhase.STOP_CHECK, {}) is True
    install_stop_request(daemon.campaign_dir, request)

    assert daemon.tick() == TickStatus.SHUTDOWN

    stopped = read_state(daemon.state_path())
    control = read_stop_request(daemon.campaign_dir)
    assert stopped.phase is CampaignPhase.SEED_SELECT
    assert stopped.iteration == 2
    assert stopped.shutdown_requested is True
    assert control["completion_reason"] == "boundary_already_completed"
    assert control["completion_receipt"] is not None


def test_crossed_iteration_boundary_without_receipt_halts_fail_closed(tmp_path):
    daemon = _daemon(tmp_path)
    observed = fresh_campaign_state(max_iterations=3)
    observed.phase = CampaignPhase.STOP_CHECK
    observed.iteration = 1
    request = build_stop_request(observed, mode="after_iteration")
    current = fresh_campaign_state(
        max_iterations=3,
        campaign_uid=observed.campaign_uid,
    )
    current.phase = CampaignPhase.SEED_SELECT
    current.iteration = 2
    write_state(daemon.state_path(), current)
    install_stop_request(daemon.campaign_dir, request)

    assert daemon.tick() == TickStatus.HALTED
    halted = read_state(daemon.state_path())
    assert halted.phase is CampaignPhase.HALTED
    assert "target_passed_without_receipt" in halted.lifecycle_context["message"]


def test_cancelling_status_waits_for_cli_cancellation_summary(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.INITIAL_GAUSSIAN
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "202"
    write_state(daemon.state_path(), state)
    install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="immediate", cancel_jobs=True),
    )

    assert daemon.tick() == TickStatus.POLLING
    waiting = read_state(daemon.state_path())
    assert waiting.shutdown_requested is False
    assert waiting.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "202"
    with pytest.raises(StopControlError, match="cancellation request is already"):
        install_stop_request(
            daemon.campaign_dir,
            build_stop_request(state, mode="immediate", cancel_jobs=False),
        )


def test_cancelling_request_does_not_race_phase_transition(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.SEED_SELECT
    state.iteration = 1
    state.reference_data_version = 0
    state.models_version = 0
    write_state(daemon.state_path(), state)
    install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="immediate", cancel_jobs=True),
    )

    assert daemon._advance(state, CampaignPhase.SEED_SELECT, {}) is True

    advanced = read_state(daemon.state_path())
    control = read_stop_request(daemon.campaign_dir)
    assert advanced.phase is not CampaignPhase.SEED_SELECT
    assert advanced.shutdown_requested is False
    assert control["status"] == "cancelling"


def test_cancelled_job_summary_clears_exact_state_and_intent(tmp_path):
    daemon = _daemon(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.INITIAL_FEREBUS
    state.pending_jobs[CampaignPhase.INITIAL_FEREBUS.value] = "303"
    write_state(daemon.state_path(), state)
    submission_intent.write_pre_submit_intent(
        daemon.campaign_dir,
        campaign_uid=state.campaign_uid,
        phase_name=state.phase.value,
        iteration=0,
    )
    submission_intent.mark_submitted(
        daemon.campaign_dir,
        state.phase.value,
        0,
        "303",
    )
    request, _ = install_stop_request(
        daemon.campaign_dir,
        build_stop_request(state, mode="immediate", cancel_jobs=True),
    )
    update_stop_request(
        daemon.campaign_dir,
        request["request_id"],
        status="requested",
        cancellation_summary={
            "cancelled": [
                {
                    "job_id": "303",
                    "phases": [state.phase.value],
                    "intent_keys": [{"phase": state.phase.value, "iteration": 0}],
                }
            ],
            "skipped": [],
            "failed": [],
        },
    )

    assert daemon.tick() == TickStatus.SHUTDOWN

    stopped = read_state(daemon.state_path())
    intent = submission_intent.load_intent(
        daemon.campaign_dir,
        state.phase.value,
        0,
    )
    assert stopped.pending_jobs[state.phase.value] is None
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "operator_cancelled_via_stop"
