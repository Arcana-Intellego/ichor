"""Tests for ichor.hpc.active_learning.cli."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ichor.hpc.active_learning import cli as cli_mod
from ichor.hpc.active_learning.cli import build_parser, main
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import (
    DAEMON_LOCK_FILENAME,
    DEFAULT_DATA_SUBDIR,
)
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.job_names import live_job_name
from ichor.hpc.active_learning.daemon.journal import append_event
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.status_recommendations import (
    build_status_recommendations,
    recommendation_dicts,
)


def _campaign_with_config(tmp_path) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    CampaignConfig(max_iterations=2).to_yaml(campaign / "campaign.yaml")
    return campaign


def test_build_parser_has_all_subcommands():
    p = build_parser()
    # Parse a known subcommand to confirm registration.
    args = p.parse_args(["start", "--campaign-dir", "x", "--mock-ariadne"])
    assert args.command == "start"
    assert args.mock_ariadne is True


def test_parser_rejects_missing_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_preflight_prints_structured_backend_status(capsys, monkeypatch):
    from ichor.hpc.active_learning.daemon.preflight import BackendAvailability

    avail = BackendAvailability(
        profile=True,
        sbatch=True,
        sacct=True,
        gaussian=True,
        aimall=True,
        ferebus=True,
        ariadne=True,
        polus_rs=True,
        pyferebus=True,
        bc=True,
        gaussian_binary="jobscript:$g16root/g16/g16",
        sbatch_path="/usr/bin/sbatch",
        sacct_path="/usr/bin/sacct",
        bc_path="/usr/bin/bc",
        aimall_path="/home/user/AIMAll/aimqb.ish",
        ferebus_path="/home/user/.local/bin/ferebus",
        active_profile="csf3",
        profile_error="",
        python_executable="/home/user/.venv/ichor-al-csf3/bin/python",
    )
    monkeypatch.setattr(cli_mod, "check_backends", lambda: avail)

    rc = main(["preflight", "--campaign-dir", "."])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["active_profile"] == "csf3"
    assert payload["python_executable"].endswith("ichor-al-csf3/bin/python")


def test_recovery_dashboard_reports_missing_state(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Recovery dashboard" in text
    assert "state.json: missing" in text
    assert "start/resume is safe for clean first run" in text


def test_recovery_dashboard_reports_invalid_state(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / DEFAULT_STATE_FILENAME).write_text("{bad json", encoding="utf-8")

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "state.json: invalid" in text
    assert "run reconcile" in text


def test_recovery_dashboard_reports_last_exception_and_staging(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / "LAST_EXCEPTION.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "timestamp": "2026-01-02T03:04:05+00:00",
                "exception_type": "RuntimeError",
                "message": "boom",
                "phase": "GAUSSIAN",
                "iteration": 2,
            }
        ),
        encoding="utf-8",
    )
    staging = campaign / ".DATA" / "STAGING" / "GAUSSIAN"
    staging.mkdir(parents=True)
    (staging / "old.txt").write_text("old", encoding="utf-8")

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Last exception" in text
    assert "RuntimeError" in text
    assert "boom" in text
    assert "Staging" in text
    assert "non-empty top_level=1" in text
    assert "reconcile --archive-staging --apply" in text


def test_recovery_dashboard_reports_active_intents(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="uid123456789",
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Submission intents" in text
    assert "INITIAL_FEREBUS@0 PRE_SUBMIT" in text
    assert "stop --cancel-jobs first" in text


def test_live_job_name_rejects_control_characters_and_caps_length():
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    with pytest.raises(ValueError, match="unsafe Slurm job name"):
        live_job_name("bad\nuid", phase, 0)

    name = live_job_name("x" * 200, phase, 123456)
    assert len(name) <= 128
    assert "\n" not in name


def test_recovery_dashboard_reports_trajectory_pool_sha_mismatch(tmp_path):
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    campaign = _campaign_with_config(tmp_path)
    source = tmp_path / "pool.xyz"
    source.write_text(
        "1\n"
        "frame 0\n"
        "H 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    TrajectoryPool.import_from(
        source,
        campaign,
        overwrite=True,
        outlier_filter_enabled=False,
    )
    pool_xyz = campaign / ".DATA" / "TRAJECTORY" / "pool.xyz"
    with pool_xyz.open("a", encoding="utf-8") as f:
        f.write("# drift\n")

    text = cli_mod.format_recovery_dashboard(campaign)

    assert "Trajectory pool" in text
    assert "pool drift detected" in text


def test_journal_list_event_types_does_not_require_journal_file(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)

    rc = main(["journal", "--campaign-dir", str(campaign), "--list-event-types"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "reconcile_applied" in out
    assert "phase_submitted" in out


def test_cli_status_json_prints_state_payload(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    s.iteration = 3
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, s)
    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["iteration"] == 3
    assert payload["max_iterations"] == 5
    assert payload["state_path"].endswith(DEFAULT_STATE_FILENAME)
    assert payload["lock_file_exists"] is False
    assert payload["lock_held"] is False
    assert payload["active_submission_intents"] == []
    assert "artifact_manifest_status" in payload
    assert "state_artifact_contract_status" in payload
    assert payload["state_artifact_contract_status"]["ok"] is True
    assert payload["recommendations"][0]["code"] == "phase_init_ready"
    assert payload["next_action"] == payload["recommendations"][0]["primary"]


def test_cli_status_default_prints_operator_friendly_summary(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    s.iteration = 3
    s.phase = CampaignPhase.STOP_CHECK
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, s)

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Campaign\n" in out
    assert "  phase: STOP_CHECK" in out
    assert "  iteration: 3 / max 5" in out
    assert "Jobs\n" in out
    assert "  pending Slurm jobs: none recorded in state" in out
    assert "  active submission intents: none" in out
    assert "Runtime\n" in out
    assert "  foreground lock: free" in out
    assert "  daemon lease: none" in out
    assert "  background daemon: not running" in out
    assert "  shutdown requested: no" in out
    assert "Artifacts\n" in out
    assert "  state contract: problem - CommittedArtifactError:" in out
    assert "  training version: 0" in out
    assert "  training status: problem - CommittedArtifactError:" in out
    assert "  models version: 0" in out
    assert "  models status: problem - CommittedArtifactError:" in out
    assert "Recommendation\n" in out
    assert "  severity: required" in out
    assert "  primary: run reconcile; STOP_CHECK needs the latest coherent committed training/model pair" in out
    assert "  why: CommittedArtifactError:" in out
    assert "  command: ichor-al-daemon reconcile --campaign-dir " in out
    assert "training v0: problem" not in out
    assert "background_pid" not in out
    assert "shutdown_requested" not in out
    assert not out.lstrip().startswith("{")


def test_cli_status_reports_initial_ferebus_bootstrap_contract_problem(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    s.phase = CampaignPhase.INITIAL_FEREBUS
    s.training_set_version = -1
    s.models_version = -1
    write_state(data / DEFAULT_STATE_FILENAME, s)

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "  state contract: problem - CommittedArtifactError:" in out
    assert "initial_ferebus_bootstrap_manifest_invalid" in out
    assert "  training status: not required yet" in out
    assert "  models status: not required yet" in out


def test_reconcile_apply_contract_guard_rejects_invalid_state(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    state = fresh_campaign_state(max_iterations=5)
    state.phase = CampaignPhase.STOP_CHECK
    state.training_set_version = 0
    state.models_version = 0

    error = cli_mod._reconcile_apply_contract_error(campaign, state)

    assert error is not None
    assert "phase=STOP_CHECK" in error
    assert "training_set_version=0" in error
    assert "models_version=0" in error
    assert "state references missing committed training version" in error


def test_cli_status_default_summarises_active_submission_intents(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    s = fresh_campaign_state(max_iterations=5)
    write_state(data / DEFAULT_STATE_FILENAME, s)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=s.campaign_uid,
        phase_name="GAUSSIAN",
        iteration=0,
    )
    submission_intent.mark_submitted(
        campaign,
        "GAUSSIAN",
        0,
        "12345",
        expected_tasks=20,
    )

    rc = main(["status", "--campaign-dir", str(campaign)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "  active submission intents: 1 (GAUSSIAN@0 SUBMITTED job_id=12345)" in out
    assert "a submission intent is still active" in out


def test_cli_status_reports_stale_lock_file_as_not_held(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    (data / DAEMON_LOCK_FILENAME).write_text("stale\n", encoding="utf-8")

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_file_exists"] is True
    assert payload["lock_held"] is False


def test_cli_status_reports_actually_held_lock(tmp_path, capsys):
    pytest.importorskip("portalocker")
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    lock_path = data / DAEMON_LOCK_FILENAME

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import portalocker, sys, time\n"
                "lock = portalocker.Lock(sys.argv[1], mode='a', timeout=0, fail_when_locked=True)\n"
                "lock.acquire()\n"
                "print('ready', flush=True)\n"
                "time.sleep(10)\n"
                "lock.release()\n"
            ),
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_file_exists"] is True
    assert payload["lock_held"] is True


def test_cli_status_surfaces_lock_probe_error(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())

    monkeypatch.setattr(
        cli_mod,
        "_probe_daemon_lock",
        lambda lock_path: {
            "lock_file_exists": True,
            "lock_held": None,
            "lock_probe_error": "RuntimeError: boom",
        },
    )
    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lock_held"] is None
    assert payload["lock_probe_error"] == "RuntimeError: boom"


def test_cli_status_returns_4_when_state_missing(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["status", "--campaign-dir", str(campaign)])
    out = capsys.readouterr().out
    assert rc == 4
    assert "Recommendation" in out
    assert "initialise or reconcile the campaign" in out


def test_cli_status_returns_json_recommendation_when_state_missing(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["status", "--campaign-dir", str(campaign), "--json"])
    assert rc == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["status_error"] == "state_missing"
    assert payload["recommendations"][0]["code"] == "state_missing"
    assert payload["next_action"] == payload["recommendations"][0]["primary"]


def _recommendation_codes(campaign: Path, payload: dict) -> list[str]:
    return [item.code for item in build_status_recommendations(campaign, payload)]


def test_status_recommendations_cover_runtime_and_job_blockers(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(campaign, {"lock_held": True}) == [
        "daemon_running_lock"
    ]
    assert _recommendation_codes(
        campaign,
        {
            "active_submission_intents": [
                {"phase": "GAUSSIAN", "iteration": 0, "status": "SUBMITTED"}
            ]
        },
    ) == ["active_submission_intent"]
    assert _recommendation_codes(
        campaign,
        {"pending_jobs": {"INITIAL_GAUSSIAN": "12345"}},
    ) == ["pending_state_job"]
    assert _recommendation_codes(campaign, {"shutdown_requested": True}) == [
        "shutdown_requested"
    ]


def test_status_recommendations_cover_halted_reason_classes(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "NODE_FAIL during array"},
        },
    ) == ["halted_scheduler_transient"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "OUT_OF_MEMORY"},
        },
    ) == ["halted_scheduler_hard_failure"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "committed_model_contract_invalid"},
        },
    ) == ["halted_contract_failure"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.HALTED.value,
            "latest_halt_event": {"reason": "campaign.yaml changed"},
        },
    ) == ["halted_config_changed"]


def test_status_recommendations_cover_contract_failure_classes(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.STOP_CHECK.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: state references missing committed model version 0",
            },
        },
    ) == ["stop_check_no_committed_pair"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.GAUSSIAN.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: state training/model version skew",
            },
        },
    ) == ["version_skew"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.SEED_SELECT.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: training_version_invalid:0",
            },
        },
    ) == ["training_missing"]
    assert _recommendation_codes(
        campaign,
        {
            "phase": CampaignPhase.SEED_SELECT.value,
            "state_artifact_contract_status": {
                "ok": False,
                "error": "CommittedArtifactError: models_version_invalid:0",
            },
        },
    ) == ["models_missing"]


@pytest.mark.parametrize(
    ("phase", "code"),
    [
        (CampaignPhase.INIT, "phase_init_ready"),
        (CampaignPhase.PHASE_A_POLUS, "phase_phase_a_polus_ready"),
        (CampaignPhase.INITIAL_GAUSSIAN, "phase_initial_gaussian_ready"),
        (CampaignPhase.INITIAL_AIMALL, "phase_initial_aimall_ready"),
        (CampaignPhase.INITIAL_FEREBUS, "phase_initial_ferebus_ready"),
        (CampaignPhase.SEED_SELECT, "phase_seed_select_ready"),
        (CampaignPhase.ARIADNE_ARRAY, "phase_ariadne_array_ready"),
        (CampaignPhase.PHASE_B_POLUS, "phase_phase_b_polus_ready"),
        (CampaignPhase.SPLIT, "phase_split_ready"),
        (CampaignPhase.GAUSSIAN, "phase_gaussian_ready"),
        (CampaignPhase.AIMALL, "phase_aimall_ready"),
        (CampaignPhase.APPEND, "phase_append_ready"),
        (CampaignPhase.FEREBUS, "phase_ferebus_ready"),
        (CampaignPhase.STOP_CHECK, "phase_stop_check_ready"),
    ],
)
def test_status_recommendations_cover_all_idle_phases(tmp_path, phase, code):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(
        campaign,
        {
            "phase": phase.value,
            "state_artifact_contract_status": {"ok": True},
            "artifact_manifest_status": {},
        },
    ) == [code]


def test_status_recommendations_cover_done_and_unknown_phase(tmp_path):
    campaign = _campaign_with_config(tmp_path)

    assert _recommendation_codes(campaign, {"phase": CampaignPhase.DONE.value}) == [
        "campaign_done"
    ]
    assert _recommendation_codes(campaign, {"phase": "NOT_A_PHASE"}) == [
        "phase_unknown"
    ]


def test_status_recommendation_dicts_are_json_ready(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    payload = {
        "status_error": "state_schema_invalid",
        "state_error": "StateSchemaError: bad field",
    }

    data = recommendation_dicts(build_status_recommendations(campaign, payload))

    assert data[0]["code"] == "state_schema_invalid"
    assert data[0]["severity"] == "required"
    assert "primary" in data[0]


def test_cli_stop_sets_shutdown_flag(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    rc = main(["stop", "--campaign-dir", str(campaign)])
    assert rc == 0
    s = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert s.shutdown_requested is True


def test_cli_stop_without_cancel_jobs_does_not_call_scancel(tmp_path, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "123"
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: pytest.fail("plain stop must not call scancel"),
    )

    rc = main(["stop", "--campaign-dir", str(campaign)])

    assert rc == 0
    stopped = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is True
    assert stopped.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "123"


def test_cli_stop_cancel_jobs_cancels_pending_job_and_clears_state(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.campaign_uid = "abc123def456-uid"
    state.iteration = 0
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    state.pending_jobs[phase] = "123"
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    expected_name = live_job_name(state.campaign_uid, phase, 0)
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    out = capsys.readouterr().out

    assert rc == 0
    assert cancelled == ["123"]
    stopped = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is True
    assert stopped.pending_jobs[phase] is None
    assert "Cancelled Slurm jobs" in out


def test_cli_stop_cancel_jobs_cancels_ferebus_intent_with_expected_job_name(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    write_state(data / DEFAULT_STATE_FILENAME, state)
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )
    submission_intent.mark_submitted(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        "456",
        expected_tasks=12,
    )
    expected_name = live_job_name(
        state.campaign_uid,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "PENDING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])

    assert rc == 0
    assert cancelled == ["456"]
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "operator_cancelled_via_stop"


def test_cli_stop_cancel_jobs_uses_intents_when_state_missing(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="uid123456789",
        phase_name=phase,
        iteration=0,
    )
    submission_intent.mark_submitted(campaign, phase, 0, "999")
    expected_name = live_job_name("uid123456789", phase, 0)
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "state.json is missing" in captured.err
    assert cancelled == ["999"]
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "operator_cancelled_via_stop"


def test_cli_stop_cancel_jobs_uses_intents_when_state_corrupt(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    (data / DEFAULT_STATE_FILENAME).write_text("{bad json", encoding="utf-8")
    phase = CampaignPhase.INITIAL_FEREBUS.value
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid="uid123456789",
        phase_name=phase,
        iteration=0,
    )
    submission_intent.mark_submitted(campaign, phase, 0, "1001")
    expected_name = live_job_name("uid123456789", phase, 0)
    cancelled = []
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": expected_name}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: (cancelled.append(str(job_id)) or (True, "")),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "state.json invalid" in captured.err
    assert cancelled == ["1001"]
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "operator_cancelled_via_stop"


def test_cli_stop_cancel_jobs_refuses_inconclusive_scheduler_lookup(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "789"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": False,
            "inconclusive": True,
            "rows": [],
            "error": "squeue unavailable",
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: pytest.fail("inconclusive lookup must not scancel"),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    err = capsys.readouterr().err

    assert rc == 10
    assert "squeue lookup inconclusive" in err
    stopped = read_state(data / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is True
    assert stopped.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "789"


def test_cli_stop_cancel_jobs_skips_invalid_squeue_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] = "789"
    write_state(data / DEFAULT_STATE_FILENAME, state)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        if cmd[0] == "scancel":
            pytest.fail("invalid squeue job id must not call scancel")
        raise AssertionError("unexpected command: " + repr(cmd))

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Skipped Slurm jobs" in out
    assert "not active in squeue" in out
    stopped = read_state(data / DEFAULT_STATE_FILENAME)
    assert stopped.shutdown_requested is True
    assert stopped.pending_jobs[CampaignPhase.INITIAL_GAUSSIAN.value] == "789"


def test_cli_stop_cancel_jobs_refuses_campaign_job_name_mismatch(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    phase = CampaignPhase.INITIAL_GAUSSIAN.value
    state.pending_jobs[phase] = "321"
    write_state(data / DEFAULT_STATE_FILENAME, state)
    monkeypatch.setattr(
        cli_mod,
        "_lookup_active_slurm_job_for_cancel",
        lambda job_id: {
            "active": True,
            "inconclusive": False,
            "rows": [{"job_id": str(job_id), "state": "RUNNING", "job_name": "other-campaign"}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_scancel",
        lambda job_id: pytest.fail("job-name mismatch must not scancel"),
    )

    rc = main(["stop", "--campaign-dir", str(campaign), "--cancel-jobs"])
    err = capsys.readouterr().err

    assert rc == 10
    assert "scheduler job name mismatch" in err
    stopped = read_state(data / DEFAULT_STATE_FILENAME)
    assert stopped.pending_jobs[phase] == "321"


def test_cli_resume_explicitly_clears_shutdown_flag(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.shutdown_requested = True
    write_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME, state)
    rc = main([
        "resume", "--campaign-dir", str(campaign),
        "--mock-ariadne", "--max-ticks", "0",
    ])
    assert rc == 0
    s = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    assert s.shutdown_requested is False


def test_cli_start_background_spawns_child_without_shell(tmp_path, monkeypatch, capsys):
    campaign = _campaign_with_config(tmp_path)
    calls = []

    class FakePopen:
        pid = 4321

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))

        def poll(self):
            return None

    monkeypatch.setattr(cli_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda seconds: None)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mock-ariadne",
        "--max-ticks",
        "7",
        "--background",
    ])

    assert rc == 0
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:4] == [
        sys.executable,
        "-m",
        "ichor.hpc.active_learning.cli",
        "start",
    ]
    assert "--background" not in argv
    assert "--mock-ariadne" in argv
    assert "--max-ticks" in argv
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert "shell" not in kwargs
    assert kwargs["env"][cli_mod.BACKGROUND_CHILD_ENV] == "1"
    pid_path = campaign / DEFAULT_DATA_SUBDIR / cli_mod.BACKGROUND_PID_FILENAME
    payload = json.loads(pid_path.read_text(encoding="utf-8"))
    assert payload["pid"] == 4321
    assert payload["schema_version"] == cli_mod.BACKGROUND_PID_SCHEMA_VERSION
    assert payload["campaign_dir"] == str(campaign.resolve())
    assert payload["log_path"].endswith(cli_mod.BACKGROUND_LOG_FILENAME)
    assert "daemon started in background" in capsys.readouterr().out


def test_cli_resume_background_clears_shutdown_and_spawns_resume(tmp_path, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.shutdown_requested = True
    write_state(data / DEFAULT_STATE_FILENAME, state)
    calls = []

    class FakePopen:
        pid = 4322

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))

        def poll(self):
            return None

    monkeypatch.setattr(cli_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda seconds: None)

    rc = main([
        "resume",
        "--campaign-dir",
        str(campaign),
        "--mock-ariadne",
        "--background",
    ])

    assert rc == 0
    assert read_state(data / DEFAULT_STATE_FILENAME).shutdown_requested is False
    argv, _kwargs = calls[0]
    assert argv[3] == "resume"
    assert "--background" not in argv


def test_cli_background_refuses_recursive_child(tmp_path, monkeypatch, capsys):
    campaign = _campaign_with_config(tmp_path)
    monkeypatch.setenv(cli_mod.BACKGROUND_CHILD_ENV, "1")

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mock-ariadne",
        "--background",
    ])

    assert rc == 2
    assert "background child" in capsys.readouterr().err


def test_cli_background_refuses_live_pid_file(tmp_path, monkeypatch, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    pid_path = data / cli_mod.BACKGROUND_PID_FILENAME
    pid_path.write_text(
        json.dumps({"pid": 99999, "schema_version": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_pid_is_alive", lambda pid: True)

    rc = main([
        "start",
        "--campaign-dir",
        str(campaign),
        "--mock-ariadne",
        "--background",
    ])

    assert rc == 8
    assert "still alive" in capsys.readouterr().err


def test_cli_status_reports_background_pid_metadata(tmp_path, capsys, monkeypatch):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())
    pid_path = data / cli_mod.BACKGROUND_PID_FILENAME
    log_path = data / cli_mod.BACKGROUND_LOG_FILENAME
    pid = 12345
    pid_path.write_text(
        json.dumps({
            "schema_version": 1,
            "pid": pid,
            "log_path": str(log_path),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_pid_is_alive", lambda value: True)

    rc = main(["status", "--campaign-dir", str(campaign), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["background_pid"] == pid
    assert payload["background_pid_alive"] is True
    assert payload["background_log_path"] == str(log_path)


def test_cli_stop_when_no_state_returns_4(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["stop", "--campaign-dir", str(campaign)])
    assert rc == 4


def test_cli_journal_json_prints_filtered_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(journal, "alpha", x=1, ts="2026-06-27T14:32:10.123456+00:00")
    append_event(journal, "beta", x=2)
    append_event(journal, "alpha", x=3)
    rc = main([
        "journal", "--campaign-dir", str(campaign),
        "--event-type", "alpha", "--json",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert [payload["event"] for payload in payloads] == ["alpha", "alpha"]
    assert payloads[0]["ts"] == "2026-06-27T14:32:10.123456+00:00"


def test_cli_journal_default_prints_readable_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "alpha",
        phase="INITIAL_GAUSSIAN",
        iteration=0,
        job_id="111",
        ts="2026-06-27T14:32:10.123456+00:00",
    )
    append_event(
        journal,
        "beta",
        phase="INITIAL_AIMALL",
        iteration=1,
        n_completed=2,
        ts="2026-06-27T14:42:55+00:00",
    )
    append_event(
        journal,
        "alpha",
        phase="FEREBUS",
        ts="2026-06-27T14:43:02Z",
    )

    rc = main(["journal", "--campaign-dir", str(campaign), "--last-n", "2"])

    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "alpha" in out
    assert "beta" in out
    assert "2026-06-27 14:42:55" in out
    assert "2026-06-27 14:43:02" in out
    assert "iter=1" in lines[0]
    assert "iter=-" in lines[1]
    assert lines[0].count("iter=1") == 1
    assert lines[1].rstrip().endswith("-")
    assert not out.lstrip().startswith("{")


def test_cli_journal_left_justifies_columns_for_long_events(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "sbatch",
        phase="PHASE_A_POLUS",
        iteration=0,
        job_id="16175189",
        expected_tasks=1,
        ts="2026-06-27T14:32:10.123456+00:00",
    )
    append_event(
        journal,
        "sacct_rows_missing_but_squeue_active",
        phase="INITIAL_GAUSSIAN",
        iteration=12,
        job_id="16175294",
        ts="2026-06-27T14:42:55+00:00",
    )

    rc = main(["journal", "--campaign-dir", str(campaign)])

    assert rc == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("2026-06-27 14:32:10")
    assert lines[1].startswith("2026-06-27 14:42:55")
    assert lines[0].index("iter=0") == lines[1].index("iter=12")
    assert lines[0].index("PHASE_A_POLUS") == lines[1].index("INITIAL_GAUSSIAN")
    assert lines[0].index("sbatch") == lines[1].index("sacct_rows_missing_but_squeue_active")
    assert lines[0].index("job=16175189") == lines[1].index("job=16175294")


def test_cli_journal_verbose_prints_event_details(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    (campaign / DEFAULT_DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    journal = campaign / DEFAULT_DATA_SUBDIR / "journal.ndjson"
    append_event(
        journal,
        "sbatch",
        phase="INITIAL_AIMALL",
        iteration=0,
        job_id="123",
        expected_tasks=10,
        ts="not-a-real-timestamp",
    )

    rc = main(["journal", "--campaign-dir", str(campaign), "--verbose"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "sbatch" in out
    assert "INITIAL_AIMALL" in out
    assert "not-a-real-timestamp" in out
    assert "iter=0" in out
    assert "  job_id: 123" in out
    assert "  expected_tasks: 10" in out
    assert "  iteration: 0" not in out


def test_cli_journal_returns_4_when_no_journal(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["journal", "--campaign-dir", str(campaign)])
    assert rc == 4


def test_cli_reconcile_writes_proposed_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    rc = main(["reconcile", "--campaign-dir", str(campaign)])
    assert rc == 0
    proposed = campaign / DEFAULT_DATA_SUBDIR / (DEFAULT_STATE_FILENAME + ".proposed")
    assert proposed.exists()
    captured = capsys.readouterr()
    assert "Proposed state written" in captured.out
    assert "Committed training versions:" in captured.out
    assert "Valid training versions:" in captured.out
    assert "Committed model versions:" in captured.out
    assert "Valid model versions:" in captured.out


def test_cli_resume_refuses_halted_state(tmp_path, capsys):
    campaign = _campaign_with_config(tmp_path)
    data = campaign / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    state = fresh_campaign_state()
    state.phase = CampaignPhase.HALTED
    write_state(data / DEFAULT_STATE_FILENAME, state)

    rc = main(["resume", "--campaign-dir", str(campaign)])

    assert rc == 6
    assert "campaign is HALTED" in capsys.readouterr().err


def test_cli_start_with_mock_ariadne_drives_state_machine(tmp_path):
    campaign = _campaign_with_config(tmp_path)
    rc = main([
        "start", "--campaign-dir", str(campaign),
        "--mock-ariadne", "--max-ticks", "5",
        "--poll-interval", "1",
    ])
    assert rc == 0
    # state.json should now exist and be parseable.
    state = read_state(campaign / DEFAULT_DATA_SUBDIR / DEFAULT_STATE_FILENAME)
    # Within 5 ticks we should at least have advanced past INIT.
    assert state.phase.value != "INIT"


def test_cli_start_without_mode_refuses(tmp_path, capsys):
    """With no --live / --dry-run / --mock-ariadne flag the CLI must refuse
    rather than silently pick a default. Exit 3 + a message listing the
    three available modes."""
    campaign = _campaign_with_config(tmp_path)
    rc = main(["start", "--campaign-dir", str(campaign)])
    assert rc == 3
    captured = capsys.readouterr()
    assert "no execution mode selected" in captured.err
    # Each available mode is named in the help text.
    assert "--live" in captured.err
    assert "--dry-run" in captured.err
    assert "--mock-ariadne" in captured.err


def test_cli_start_with_mutually_exclusive_flags_refuses(tmp_path, capsys):
    """--live + --dry-run is a mutually-exclusive configuration; exit 2."""
    campaign = _campaign_with_config(tmp_path)
    rc = main(["start", "--campaign-dir", str(campaign), "--live", "--dry-run"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


def test_cli_start_live_on_windows_refuses_with_exit_12(tmp_path, capsys):
    """When --live is requested but the backends are absent (the off-cluster
    case), the CLI must refuse with exit 12 and a message naming the missing
    binaries -- not silently spin a daemon."""
    campaign = _campaign_with_config(tmp_path)
    rc = main(["start", "--campaign-dir", str(campaign), "--live"])
    # On a CSF4 host with all binaries present this test would skip; in our
    # CI / Windows environment, the backends are absent and exit 12 is the
    # expected refusal code.
    from ichor.hpc.active_learning.daemon.preflight import check_backends
    if check_backends().all_present:
        import pytest
        pytest.skip("all live backends present; refusal path not exercised here")
    assert rc == 12
    captured = capsys.readouterr()
    assert "backends are not available" in captured.err
