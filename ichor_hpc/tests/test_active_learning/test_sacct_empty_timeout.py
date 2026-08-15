"""M15 F6 tests: sacct empty-result streak escalation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import Daemon, TickStatus
from ichor.hpc.active_learning.daemon.phase_executor import (
    MockPhaseExecutor,
    PhaseResult,
)
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    CampaignState,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.submit.sacct_poll import (
    JobObservation,
    JobStatus,
)


def _setup_daemon_with_pending_job(
    tmp_path,
    empty_poller,
    max_ticks=3,
    job_liveness_checker=None,
):
    """Daemon with state preloaded with a pending SBATCH job."""
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = CampaignState(
        iteration=2,
        phase=CampaignPhase.GAUSSIAN,
        pending_jobs={"GAUSSIAN": "99999"},
    )
    state_path = cd / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    write_state(state_path, state)
    submission_intent.write_pre_submit_intent(
        cd,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.GAUSSIAN.value,
        iteration=2,
        expected_tasks=1,
    )
    submission_intent.mark_submitted(
        cd,
        CampaignPhase.GAUSSIAN.value,
        2,
        "99999",
        expected_tasks=1,
    )
    cfg = CampaignConfig()
    cfg.poll_sacct_empty_max_ticks = max_ticks
    daemon = Daemon(
        campaign_dir=cd, config=cfg,
        executor=MockPhaseExecutor(), sacct_poller=empty_poller,
        sleep_fn=lambda s: None,
        job_liveness_checker=job_liveness_checker,
    )
    return daemon, state_path


def test_default_poll_sacct_empty_max_ticks_is_ten():
    assert CampaignConfig().poll_sacct_empty_max_ticks == 10
    assert CampaignConfig().runtime.poll_sacct_error_max_ticks == 10
    assert CampaignConfig().runtime.poll_sacct_missing_max_ticks == 12


def test_repeated_sacct_command_errors_wait_while_scheduler_job_is_active(tmp_path):
    def broken_poller(job_id):
        raise RuntimeError("sacct unavailable")

    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        broken_poller,
        max_ticks=10,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=True,
            inconclusive=False,
            rows=[(str(job_id) + "_0", "RUNNING")],
            error=None,
        ),
    )
    daemon.config.runtime.poll_sacct_error_max_ticks = 2

    for _ in range(5):
        assert daemon.tick() == TickStatus.POLLING

    state = read_state(state_path)
    assert state.phase is CampaignPhase.GAUSSIAN
    assert state.pending_jobs[CampaignPhase.GAUSSIAN.value] == "99999"
    assert "99999:ERROR" not in state.sacct_empty_streak
    events = [
        json.loads(line)
        for line in daemon.journal_path().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not any(
        event.get("event") == "sacct_error_timeout" for event in events
    )


def test_repeated_sacct_command_errors_halt_after_job_is_retired(tmp_path):
    def broken_poller(job_id):
        raise RuntimeError("sacct unavailable")

    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        broken_poller,
        max_ticks=10,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=False,
            inconclusive=False,
            rows=[],
            error=None,
        ),
    )
    daemon.config.runtime.poll_sacct_error_max_ticks = 2

    assert daemon.tick() == TickStatus.POLLING
    assert daemon.tick() == TickStatus.HALTED
    assert read_state(state_path).phase is CampaignPhase.HALTED


def test_empty_sacct_increments_streak(tmp_path):
    """Each tick where sacct returns [] must increment the streak."""
    empty_poller = lambda job_id: []  # no observations at all
    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path, empty_poller, max_ticks=5,
    )
    for expected in range(1, 4):
        status = daemon.tick()
        # Tick must keep polling (streak < max_ticks).
        assert status == TickStatus.POLLING
        from ichor.hpc.active_learning.daemon.state import read_state
        st = read_state(state_path)
        assert st.sacct_empty_streak.get("99999") == expected


def test_streak_escalates_to_failure_at_max(tmp_path):
    """Once streak reaches max_ticks the daemon must escalate."""
    empty_poller = lambda job_id: []
    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path, empty_poller, max_ticks=3,
    )
    # Drive ticks 1, 2 -> still polling. Tick 3 -> escalation.
    daemon.tick()
    daemon.tick()
    final_status = daemon.tick()
    # SCRUB_AND_CONTINUE is MockPhaseExecutor's default failure policy.
    assert final_status in (TickStatus.SCRUBBED, TickStatus.HALTED)
    # And the journal must contain sacct_empty_timeout.
    journal_path = tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    events = [
        json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    timeouts = [e for e in events if e.get("event") == "sacct_empty_timeout"]
    assert timeouts, "sacct_empty_timeout event missing"
    assert timeouts[-1]["job_id"] == "99999"
    assert timeouts[-1]["streak"] == 3


def test_empty_sacct_keeps_polling_when_squeue_active(tmp_path):
    empty_poller = lambda job_id: []
    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        empty_poller,
        max_ticks=2,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=True,
            inconclusive=False,
            rows=[(str(job_id) + "_[0-2%1]", "PENDING")],
            error=None,
        ),
    )

    assert daemon.tick() == TickStatus.POLLING
    assert daemon.tick() == TickStatus.POLLING

    st = read_state(state_path)
    assert st.phase is CampaignPhase.GAUSSIAN
    assert st.pending_jobs[CampaignPhase.GAUSSIAN.value] == "99999"
    assert "99999" not in st.sacct_empty_streak
    events = [
        json.loads(line) for line in (tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e.get("event") == "sacct_empty_but_squeue_active" for e in events)
    assert not any(e.get("event") == "sacct_empty_timeout" for e in events)


def test_empty_sacct_keeps_polling_when_squeue_inconclusive(tmp_path):
    empty_poller = lambda job_id: []
    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        empty_poller,
        max_ticks=2,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=False,
            inconclusive=True,
            rows=[],
            error="squeue unavailable",
        ),
    )

    assert daemon.tick() == TickStatus.POLLING
    assert daemon.tick() == TickStatus.POLLING

    st = read_state(state_path)
    assert st.phase is CampaignPhase.GAUSSIAN
    assert st.pending_jobs[CampaignPhase.GAUSSIAN.value] == "99999"
    assert "99999" not in st.sacct_empty_streak
    assert st.sacct_empty_streak["99999:SQUEUE_INCONCLUSIVE:empty"] == 2
    events = [
        json.loads(line) for line in (tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e.get("event") == "squeue_liveness_inconclusive" for e in events)
    assert not any(e.get("event") == "sacct_empty_timeout" for e in events)


def test_non_empty_response_resets_streak(tmp_path):
    """A non-empty terminal sacct response must clear the streak."""
    calls = {"n": 0}

    def variable_poller(job_id):
        calls["n"] += 1
        if calls["n"] < 3:
            return []  # empty for the first two ticks
        # Then a clean completion
        return [JobObservation(
            job_id=job_id, status=JobStatus.COMPLETED,
            exit_code=(0, 0), elapsed_seconds=10,
        )]

    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path, variable_poller, max_ticks=10,
    )
    daemon.tick()  # streak 1
    daemon.tick()  # streak 2
    daemon.tick()  # non-empty -> streak cleared, postprocess runs
    from ichor.hpc.active_learning.daemon.state import read_state
    st = read_state(state_path)
    # MockPhaseExecutor's postprocess advances; pending_jobs cleared.
    assert st.sacct_empty_streak.get("99999", 0) == 0


def test_max_ticks_zero_disables_escalation(tmp_path):
    """Setting poll_sacct_empty_max_ticks=0 must keep polling forever
    (operator opt-out)."""
    empty_poller = lambda job_id: []
    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path, empty_poller, max_ticks=0,
    )
    for _ in range(15):
        status = daemon.tick()
        assert status == TickStatus.POLLING


def test_missing_expected_array_rows_halt_after_bounded_ticks(tmp_path):
    def short_array_poller(job_id):
        return [
            JobObservation(
                job_id=str(job_id) + "_0",
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=10,
            )
        ]

    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        short_array_poller,
        max_ticks=10,
    )
    daemon.config.runtime.poll_sacct_missing_max_ticks = 2
    submission_intent.mark_submitted(
        tmp_path / "campaign",
        "GAUSSIAN",
        2,
        "99999",
        expected_tasks=3,
    )

    assert daemon.tick() == TickStatus.POLLING
    final_status = daemon.tick()

    assert final_status == TickStatus.HALTED
    state = read_state(state_path)
    assert state.phase is CampaignPhase.HALTED
    journal_path = tmp_path / "campaign" / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    events = [
        json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [e for e in events if e.get("event") == "sacct_missing_timeout"]
    assert missing
    assert missing[-1]["n_expected"] == 3
    assert missing[-1]["n_observed"] == 1
    assert missing[-1]["n_missing"] == 2


def test_unknown_accounting_waits_while_exact_job_remains_active(tmp_path):
    def unknown_poller(job_id):
        return [
            JobObservation(
                job_id=str(job_id),
                status=JobStatus.UNKNOWN,
                exit_code=None,
                elapsed_seconds=None,
            )
        ]

    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        unknown_poller,
        max_ticks=10,
        job_liveness_checker=lambda job_id: SimpleNamespace(
            active=True,
            inconclusive=False,
            rows=[(str(job_id), "RUNNING")],
            error=None,
        ),
    )
    daemon.config.runtime.poll_sacct_unknown_max_ticks = 3

    for _ in range(6):
        assert daemon.tick() == TickStatus.POLLING
    state = read_state(state_path)
    assert state.phase is CampaignPhase.GAUSSIAN
    assert "99999:UNKNOWN" not in state.sacct_empty_streak


def test_unknown_grace_starts_only_after_exact_job_retires(tmp_path):
    liveness_calls = {"count": 0}

    def unknown_poller(job_id):
        return [
            JobObservation(
                job_id=str(job_id),
                status=JobStatus.UNKNOWN,
                exit_code=None,
                elapsed_seconds=None,
            )
        ]

    def liveness(job_id):
        liveness_calls["count"] += 1
        active = liveness_calls["count"] <= 3
        return SimpleNamespace(
            active=active,
            inconclusive=False,
            rows=[(str(job_id), "RUNNING")] if active else [],
            error=None,
        )

    daemon, state_path = _setup_daemon_with_pending_job(
        tmp_path,
        unknown_poller,
        max_ticks=10,
        job_liveness_checker=liveness,
    )
    daemon.config.runtime.poll_sacct_unknown_max_ticks = 2

    for _ in range(3):
        assert daemon.tick() == TickStatus.POLLING
    assert "99999:UNKNOWN" not in read_state(state_path).sacct_empty_streak
    assert daemon.tick() == TickStatus.POLLING
    assert read_state(state_path).sacct_empty_streak["99999:UNKNOWN"] == 1
    assert daemon.tick() == TickStatus.HALTED
