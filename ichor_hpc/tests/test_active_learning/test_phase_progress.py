import json

import ichor.hpc.active_learning.cli as cli

from ichor.hpc.active_learning.daemon import phase_progress
from ichor.hpc.active_learning.daemon.phase_progress import (
    PhaseProgressReporter,
    format_progress_stage,
    newest_matching_progress,
    progress_path,
    read_phase_progress_records,
)


def test_ariadne_resource_progress_stages_have_explicit_human_labels():
    assert (
        format_progress_stage("ariadne_resource_validation")
        == "Validating ARIADNE resource inputs"
    )
    assert (
        format_progress_stage("ariadne_task_map_validation")
        == "Validating ARIADNE task map"
    )
    assert (
        format_progress_stage("ariadne_trajectory_pool_validation")
        == "Validating ARIADNE trajectory pool"
    )
    assert (
        format_progress_stage("ariadne_current_model_validation")
        == "Validating current ARIADNE models"
    )
    assert (
        format_progress_stage("ariadne_resource_dimensions")
        == "Computing ARIADNE local subspace dimensions"
    )
    assert (
        format_progress_stage("ariadne_resource_bound")
        == "Applying configured ARIADNE resource bound"
    )
    assert (
        format_progress_stage("ariadne_resource_reuse")
        == "Reusing validated ARIADNE resource evidence"
    )
    assert (
        format_progress_stage("resource_rules")
        == "Applying scheduler resource rules"
    )
    assert cli._journal_operator_summary(
        {
            "event": "phase_activity_progress",
            "stage": "ariadne_resource_reuse",
            "status": "running",
            "completed": 196,
            "total": 196,
            "unit": "retry tasks",
        }
    ) == (
        "Reusing validated ARIADNE resource evidence: "
        "196/196 retry tasks"
    )
    status_record = {
        "stage": "ariadne_resource_reuse",
        "status": "running",
        "counters": {
            "completed": 196,
            "total": 196,
            "unit": "retry tasks",
        },
    }
    assert cli._format_generic_progress_activity(status_record) == (
        "Reusing validated ARIADNE resource evidence."
    )
    assert cli._format_generic_progress_count(status_record) == (
        "196/196 retry tasks"
    )


def test_scheduler_retirement_stage_is_explicit_and_scheduler_neutral():
    assert format_progress_stage("scheduler_retirement_wait") == (
        "Scientific publication complete; waiting for scheduler "
        "termination/accounting"
    )


def test_ferebus_publication_stages_have_explicit_human_labels():
    expected = {
        "ferebus_quality_context_binding": "Binding FEREBUS quality authority",
        "model_snapshot_inventory": "Inventorying staged FEREBUS publication",
        "model_snapshot_semantic_validation": (
            "Validating staged FEREBUS semantics"
        ),
        "model_snapshot_durability": "Synchronising staged FEREBUS publication",
        "model_commit_publication": "Publishing FEREBUS model version",
        "model_commit_validation": "Validating committed FEREBUS publication",
    }
    assert {
        stage: format_progress_stage(stage) for stage in expected
    } == expected


def test_scheduler_pending_diagnostics_render_in_journal_and_status():
    record = {
        "producer_kind": "scheduler",
        "stage": "scheduler_wait",
        "status": "running",
        "counters": {"completed": 0, "total": 200, "unit": "tasks"},
        "details": {
            "running": 0,
            "pending": 200,
            "pending_reason": "Resources",
            "pending_queue": "multicore",
            "scheduler_native_state": "PENDING",
        },
    }
    payload = {
        "phase": "ARIADNE_ARRAY",
        "runtime_progress": {
            "state": "current",
            "record": record,
            "age_seconds": 5,
        },
        "active_submission_intents": [],
    }

    assert ("recorded pending reason", "Resources") in cli._status_progress_rows(
        payload
    )
    assert ("recorded scheduler queue", "multicore") in cli._status_progress_rows(
        payload
    )
    assert (
        "recorded native scheduler state",
        "PENDING",
    ) in cli._status_progress_rows(payload)
    assert cli._journal_operator_summary(
        {
            "event": "scheduler_progress",
            "stage": "scheduler_wait",
            "status": "running",
            "completed": 0,
            "total": 200,
            "unit": "tasks",
            "running": 0,
            "pending": 200,
            "pending_reason": "Resources",
            "pending_queue": "multicore",
            "scheduler_native_state": "PENDING",
        }
    ).endswith(
        "; pending reason=Resources, queue=multicore, native state=PENDING"
    )


def test_matching_progress_rejects_worker_from_an_old_job():
    records = [
        {
            "campaign_uid": "uid",
            "phase": "PHASE_B_DIVERSITY",
            "iteration": 4,
            "replacement_round": 0,
            "producer_kind": "worker",
            "job_id": "old",
            "updated_at_iso": "2026-01-01T00:00:00+00:00",
        },
        {
            "campaign_uid": "uid",
            "phase": "PHASE_B_DIVERSITY",
            "iteration": 4,
            "replacement_round": 0,
            "producer_kind": "scheduler",
            "job_id": "current",
            "updated_at_iso": "2025-01-01T00:00:00+00:00",
        },
    ]

    selected = newest_matching_progress(
        records,
        campaign_uid="uid",
        phase="PHASE_B_DIVERSITY",
        iteration=4,
        replacement_round=0,
        job_ids=["current"],
    )

    assert selected["producer_kind"] == "scheduler"
    assert selected["job_id"] == "current"


def test_daemon_observes_worker_publication_and_throttles_queue_diagnostics(
    tmp_path,
):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase,
        fresh_campaign_state,
    )

    state = fresh_campaign_state()
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 3
    worker = PhaseProgressReporter(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase=state.phase.value,
        iteration=3,
        producer_kind="worker",
        identity={"job_id": "9001"},
    )
    worker.start("split_publication")
    worker.complete(stage="split_publication")
    worker.close()
    calls = []

    def diagnostics(job_id, **_kwargs):
        calls.append(str(job_id))
        return {
            "active": True,
            "inconclusive": False,
            "rows": [{
                "state": "PENDING",
                "reason": "Resources",
                "queue": "multicore",
            }],
        }

    daemon = Daemon(
        campaign_dir=tmp_path,
        config=CampaignConfig(),
        queue_diagnostics_collector=diagnostics,
    )

    assert daemon._worker_publication_complete(state, state.phase, "9001") is True
    assert daemon._worker_publication_complete(state, state.phase, "9002") is False
    first = daemon._pending_queue_diagnostics(
        job_id="9001",
        scheduler_identity={
            "expected_job_name": "job",
            "expected_owner": "user",
        },
    )
    second = daemon._pending_queue_diagnostics(
        job_id="9001",
        scheduler_identity={},
    )

    assert first == {
        "scheduler_native_state": "PENDING",
        "pending_reason": "Resources",
        "pending_queue": "multicore",
    }
    assert second == {}
    assert calls == ["9001"]


def test_scheduler_completion_preserves_retirement_wait_stage(tmp_path):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase,
        fresh_campaign_state,
    )

    state = fresh_campaign_state()
    state.phase = CampaignPhase.PHASE_B_DIVERSITY
    state.iteration = 3
    daemon = Daemon(campaign_dir=tmp_path, config=CampaignConfig())
    reporter = daemon._scheduler_progress_reporter(
        state,
        state.phase,
        "9001",
        expected_tasks=1,
    )
    reporter.update(
        stage="scheduler_retirement_wait",
        completed=1,
        total=1,
        unit="tasks",
    )

    daemon._finish_scheduler_progress(state.phase, "9001")

    record = newest_matching_progress(
        read_phase_progress_records(
            tmp_path,
            phase=state.phase.value,
            expected_campaign_uid=state.campaign_uid,
        ),
        campaign_uid=state.campaign_uid,
        phase=state.phase.value,
        iteration=3,
        replacement_round=0,
        job_ids=["9001"],
    )
    assert record["status"] == "completed"
    assert record["stage"] == "scheduler_retirement_wait"


def _reporter(tmp_path, clock, events, **overrides):
    options = {
        "campaign_uid": "campaign-uid",
        "phase": "AIMALL",
        "iteration": 3,
        "replacement_round": 1,
        "producer_kind": "local",
        "identity": {"daemon_pid": 1234, "daemon_start_id": "launch-a"},
        "journal_callback": lambda event, payload: events.append(
            (event, dict(payload))
        ),
        "write_interval_seconds": 100.0,
        "journal_interval_seconds": 30.0,
        "monotonic_fn": lambda: clock[0],
    }
    options.update(overrides)
    return PhaseProgressReporter(tmp_path, **options)


def test_progress_stage_changes_and_throttling_are_deterministic(tmp_path):
    clock = [0.0]
    events = []
    reporter = _reporter(tmp_path, clock, events)

    reporter.start("output_visibility", completed=0, total=200, unit="tasks")
    reporter.update(completed=20, total=200, unit="tasks")
    clock[0] = 31.0
    reporter.update(completed=40, total=200, unit="tasks")
    reporter.update(
        stage="structural_parsing",
        completed=40,
        total=200,
        unit="tasks",
    )
    reporter.complete()

    assert [event for event, _payload in events] == [
        "phase_activity_started",
        "phase_activity_progress",
        "phase_activity_progress",
        "phase_activity_completed",
    ]
    assert events[0][1]["iteration"] == 3
    assert events[0][1]["replacement_round"] == 1
    assert round(float(events[1][1]["throughput"]), 3) == round(40.0 / 31.0, 3)
    assert events[1][1]["stage_elapsed_seconds"] == 31.0
    assert events[2][1]["stage"] == "structural_parsing"
    assert events[2][1]["stage_elapsed_seconds"] == 0.0
    assert len(json.dumps(events[1][1]).encode("utf-8")) < 1024
    assert reporter.thread_alive is False


def test_scheduler_progress_uses_sixty_second_journal_cadence(tmp_path):
    clock = [0.0]
    events = []
    reporter = _reporter(
        tmp_path,
        clock,
        events,
        producer_kind="scheduler",
        identity={"job_id": "12345"},
        journal_interval_seconds=None,
    )

    reporter.start("scheduler_wait", completed=0, total=10, unit="tasks")
    clock[0] = 59.0
    reporter.update(completed=4, total=10, unit="tasks", running=3, pending=3)
    clock[0] = 60.0
    reporter.update(completed=5, total=10, unit="tasks", running=3, pending=2)
    reporter.complete()

    assert [event for event, _payload in events] == [
        "phase_activity_started",
        "scheduler_progress",
        "phase_activity_completed",
    ]
    assert events[1][1]["job_id"] == "12345"
    assert events[1][1]["running"] == 3


def test_reporting_failures_cannot_escape_or_leave_a_thread(monkeypatch, tmp_path):
    clock = [0.0]

    def fail_write(*_args, **_kwargs):
        raise OSError("injected progress write failure")

    def fail_journal(*_args, **_kwargs):
        raise RuntimeError("injected journal failure")

    monkeypatch.setattr(phase_progress, "atomic_write_json", fail_write)
    reporter = _reporter(tmp_path, clock, [], journal_callback=fail_journal)

    reporter.start("scientific_quality", completed=0, total=2, unit="points")
    reporter.update(completed=1, total=2, unit="points")
    reporter.complete()

    assert reporter.thread_alive is False


def test_reporter_thread_start_failure_is_best_effort(monkeypatch, tmp_path):
    clock = [0.0]

    def fail_start(_thread):
        raise RuntimeError("thread creation unavailable")

    monkeypatch.setattr(phase_progress.threading.Thread, "start", fail_start)
    reporter = _reporter(tmp_path, clock, [])

    reporter.start("scientific_quality", completed=0, total=2, unit="points")
    reporter.update(completed=1, total=2, unit="points")
    reporter.complete()

    assert reporter.thread_alive is False


def test_bounded_reader_reports_malformed_sidecars_without_traversal(tmp_path):
    clock = [0.0]
    events = []
    reporter = _reporter(tmp_path, clock, events)
    reporter.start("input_staging")
    reporter.close()
    malformed = progress_path(tmp_path, "AIMALL", "worker")
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("not-json", encoding="utf-8")
    errors = []

    records = read_phase_progress_records(
        tmp_path,
        phase="AIMALL",
        expected_campaign_uid="campaign-uid",
        errors=errors,
    )

    assert len(records) == 1
    assert records[0]["stage"] == "input_staging"
    assert len(errors) == 1
    assert "AIMALL.worker.json" in errors[0]


def test_progress_identity_matching_excludes_stale_daemon_and_job(tmp_path):
    del tmp_path
    base = {
        "campaign_uid": "campaign-uid",
        "phase": "GAUSSIAN",
        "iteration": 2,
        "replacement_round": 0,
        "updated_at_iso": "2026-07-21T10:00:00+00:00",
    }
    records = [
        {**base, "producer_kind": "local", "daemon_pid": 10},
        {
            **base,
            "producer_kind": "scheduler",
            "job_id": "200",
            "updated_at_iso": "2026-07-21T10:01:00+00:00",
        },
    ]

    assert newest_matching_progress(
        records,
        campaign_uid="campaign-uid",
        phase="GAUSSIAN",
        iteration=2,
        replacement_round=0,
        daemon_pid=11,
        daemon_start_id="launch-a",
        job_ids={"201"},
    ) is None
    assert newest_matching_progress(
        records,
        campaign_uid="campaign-uid",
        phase="GAUSSIAN",
        iteration=2,
        replacement_round=0,
        daemon_pid=10,
        daemon_start_id="launch-a",
        job_ids={"200"},
    )["job_id"] == "200"

    local = [{**base, "producer_kind": "local", "daemon_pid": 10,
              "daemon_start_id": "launch-old"}]
    assert newest_matching_progress(
        local,
        campaign_uid="campaign-uid",
        phase="GAUSSIAN",
        iteration=2,
        replacement_round=0,
        daemon_pid=10,
        daemon_start_id="launch-new",
    ) is None


def test_status_uses_generic_progress_stage_counters_and_age():
    payload = {
        "phase": "AIMALL",
        "runtime_progress": {
            "state": "current",
            "age_seconds": 4.0,
            "record": {
                "producer_kind": "local",
                "stage": "scientific_quality",
                "elapsed_seconds": 92.0,
                "throughput": 0.5,
                "counters": {
                    "completed": 81,
                    "total": 200,
                    "unit": "point directories",
                },
            },
        },
    }

    assert cli._status_current_activity(payload) == "Evaluating scientific quality."
    assert cli._status_progress_rows(payload) == [
        ("progress", "81/200 point directories"),
        ("elapsed", "1m 32s"),
        ("throughput", "0.50 point directories/s"),
        ("last update", "4s ago"),
    ]
    assert cli._format_generic_progress_activity(
        {"stage": "checkpoint_verification", "status": "completed"}
    ) == "Finished verifying checkpoint."


def test_status_preserves_detailed_seed_selection_progress_precedence():
    seed_record = {
        "stage": "d_optimal",
        "completed": 73,
        "total": 160,
        "shortlist_size": 1280,
        "elapsed_seconds": 90.0,
    }
    payload = {
        "phase": "SEED_SELECT",
        "seed_selection_progress": {
            "state": "current",
            "age_seconds": 3.0,
            "record": seed_record,
        },
        "runtime_progress": {
            "state": "current",
            "age_seconds": 3.0,
            "record": seed_record,
        },
    }

    assert cli._status_current_activity(payload) == (
        "Selecting D-optimal seeds 73/160 from 1280 shortlisted frames "
        "(1m 30s elapsed)."
    )
    assert cli._status_progress_rows(payload) == [
        ("progress", "73/160"),
        ("elapsed", "1m 30s"),
        ("last update", "3s ago"),
    ]


def test_scheduler_journal_summary_reports_all_array_counts():
    summary = cli._journal_operator_summary(
        {
            "event": "scheduler_progress",
            "stage": "scheduler_wait",
            "completed": 132,
            "total": 200,
            "running": 18,
            "pending": 50,
            "failed": 0,
            "missing": 0,
            "unit": "tasks",
        }
    )

    assert summary == (
        "Waiting for Slurm work: 132/200 completed, 18 running, 50 pending tasks"
    )
    assert cli._journal_operator_summary(
        {
            "event": "phase_activity_completed",
            "stage": "acceptance_publication",
            "status": "completed",
            "completed": 200,
            "total": 200,
            "unit": "point directories",
        }
    ) == "Finished publishing accepted outputs: 200/200 point directories"


def test_status_progress_includes_scientific_acceptance_counts():
    assert cli._format_generic_progress_count(
        {
            "producer_kind": "local",
            "counters": {
                "completed": 81,
                "total": 200,
                "unit": "point directories",
            },
            "details": {"accepted": 79, "rejected": 2},
        }
    ) == "81/200 point directories (79 accepted, 2 rejected)"


def test_reference_commit_ledger_is_exposed_as_generic_progress():
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase,
        fresh_campaign_state,
    )

    state = fresh_campaign_state(max_iterations=10)
    state.phase = CampaignPhase.REFERENCE_COMMIT
    state.iteration = 2
    payload = {
        "lock_held": True,
        "reference_commit_transactions": [
            {
                "state": "partially_moved",
                "ledger": {
                    "campaign_uid": state.campaign_uid,
                    "iteration": 2,
                    "point_bindings": [{}, {}, {}],
                    "moved_points": 2,
                    "moved_bytes": 500,
                    "shards_reused": 2,
                    "shards_repaired": 0,
                    "created_at_iso": "2026-07-21T09:00:00+00:00",
                    "updated_at_iso": "2026-07-21T09:01:00+00:00",
                },
            }
        ],
    }

    progress = cli._reference_commit_runtime_progress(payload, state)

    assert progress["state"] == "current"
    assert progress["record"]["stage"] == "reference_commit_move"
    assert progress["record"]["counters"] == {
        "completed": 2,
        "total": 3,
        "unit": "point directories",
    }


def test_status_prefers_progress_bound_to_the_active_slurm_job(tmp_path):
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase,
        fresh_campaign_state,
    )

    state = fresh_campaign_state(max_iterations=10)
    state.phase = CampaignPhase.GAUSSIAN
    state.iteration = 2
    state.pending_jobs[CampaignPhase.GAUSSIAN.value] = "9001"
    local = PhaseProgressReporter(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase=state.phase.value,
        iteration=2,
        producer_kind="local",
        identity={"daemon_pid": 123},
    )
    local.start("slurm_submission", completed=200, total=200, unit="tasks")
    local.complete()
    runtime = {"lock_held": True, "active_submission_intents": []}

    awaiting = cli._load_runtime_progress_status(tmp_path, state, runtime)

    assert awaiting["state"] == "stale"
    assert awaiting["reason"] == "awaiting_job_progress"

    scheduler = PhaseProgressReporter(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase=state.phase.value,
        iteration=2,
        producer_kind="scheduler",
        identity={"job_id": "9001"},
    )
    scheduler.start("scheduler_wait", completed=10, total=200, unit="tasks")
    scheduler.close()

    current = cli._load_runtime_progress_status(tmp_path, state, runtime)

    assert current["state"] == "current"
    assert current["record"]["producer_kind"] == "scheduler"
    assert current["record"]["job_id"] == "9001"

    worker = PhaseProgressReporter(
        tmp_path,
        campaign_uid=state.campaign_uid,
        phase=state.phase.value,
        iteration=2,
        producer_kind="worker",
        identity={"job_id": "9001"},
    )
    worker.start(
        "descriptor_construction",
        completed=40,
        total=200,
        unit="frames",
    )
    worker.close()

    detailed = cli._load_runtime_progress_status(tmp_path, state, runtime)

    assert detailed["state"] == "current"
    assert detailed["record"]["producer_kind"] == "worker"
    assert detailed["record"]["stage"] == "descriptor_construction"
