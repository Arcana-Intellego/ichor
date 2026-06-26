import argparse
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import ichor.hpc.active_learning.cli as cli_mod
from ichor.hpc.active_learning.cli import cmd_reconcile, cmd_start
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.daemon import config_lock as config_lock_mod
from ichor.hpc.active_learning.daemon.config_lock import (
    config_lock_path,
    review_config_changes,
    restore_config_from_lock_proposal,
    write_config_lock,
)
from ichor.hpc.active_learning.daemon.reconcile import ReconciliationReport
from ichor.hpc.active_learning.daemon import submission_intent
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.submit import sacct_poll
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


def _campaign(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    (campaign / "5_TRAINING").mkdir()
    (campaign / "6_TRAINED_MODELS").mkdir()
    return campaign


def _write_pool(campaign):
    src = campaign / "pool_source.xyz"
    src.write_text(
        "1\n"
        "frame 0\n"
        "H 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    TrajectoryPool.import_from(
        src,
        campaign,
        overwrite=True,
        outlier_filter_enabled=False,
    )


def _commit_training_version(campaign, version=0):
    _write_pool(campaign)
    tv = TrainingSetVersioning(campaign / "5_TRAINING")
    staging = tv.stage(None, version)
    (staging / "marker.txt").write_text("training", encoding="utf-8")
    tv.commit(version)


def _write_config(campaign, config):
    config.to_yaml(campaign / "campaign.yaml")


def _write_halted_pre_ferebus_state(campaign):
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.training_set_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    return state


def _write_submitted_initial_ferebus_intent(campaign, job_id="16218598"):
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )
    return submission_intent.mark_submitted(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
        job_id,
        expected_tasks=12,
    )


def _write_stale_pre_submit_intent(campaign, phase):
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    payload = submission_intent.write_pre_submit_intent(
        campaign,
        campaign_uid=state.campaign_uid,
        phase_name=phase,
        iteration=0,
    )
    path = submission_intent.intent_path(campaign, phase, 0)
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    payload["created_iso"] = old
    payload["updated_iso"] = old
    payload["updated_at_iso"] = old
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_ferebus_scaling_change_allowed_for_uncommitted_initial_ferebus(tmp_path):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    state = _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ferebus.scaling = False
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    proposed.training_set_version = 0
    proposed.models_version = -1
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == ["ferebus.scaling"]
    assert not review.blocked_changes
    assert state.phase is CampaignPhase.HALTED


def test_phase_walltime_changes_are_allowed_runtime_changes(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.resources.ferebus_walltime_hours = 2
    changed.resources.gaussian_walltime_hours = 3
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert sorted(c.path for c in review.allowed_changes) == [
        "resources.ferebus_walltime_hours",
        "resources.gaussian_walltime_hours",
    ]
    assert not review.blocked_changes


def test_ariadne_backtransform_changes_are_future_safe(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    changed.ariadne.trqn_geodesic_bt_mode = "matrix_free"
    changed.ariadne.trqn_geodesic_dt = 0.02
    changed.ariadne.trqn_geodesic_tol = 2.0e-8
    changed.ariadne.trqn_bt_ic_tol = 2.0e-6
    changed.ariadne.trqn_max_backtransform_iter = 75
    changed.ariadne.trqn_trust_min = 2.0e-4

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert sorted(c.path for c in review.allowed_changes) == [
        "ariadne.trqn_backtransform_mode",
        "ariadne.trqn_bt_ic_tol",
        "ariadne.trqn_geodesic_bt_mode",
        "ariadne.trqn_geodesic_dt",
        "ariadne.trqn_geodesic_tol",
        "ariadne.trqn_max_backtransform_iter",
        "ariadne.trqn_trust_min",
    ]
    assert {c.category for c in review.allowed_changes} == {"future_safe"}
    assert not review.blocked_changes


def test_gaussian_method_change_is_blocked_by_config_lock(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.gaussian.method = "PBE0"

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["gaussian.method"]


def test_reconcile_apply_promotes_state_and_cleans_ferebus_staging(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ferebus.scaling = False
    _write_config(campaign, changed)
    stale = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    stale.mkdir(parents=True)
    (stale / "runFerebus.sh").write_text("# stale\n", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.INITIAL_FEREBUS
    assert state.training_set_version == 0
    assert state.models_version == -1
    assert not stale.exists()
    assert "Applied proposed state" in out
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").exists()
    assert sorted(
        (campaign / ".DATA" / "ACTIVE_LEARNING").glob(
            "state.json.proposed.applied-*"
        )
    )
    assert sorted(
        (campaign / ".DATA" / "ACTIVE_LEARNING").glob(
            "state.json.before-reconcile-*"
        )
    )
    lock = json.loads(config_lock_path(campaign).read_text(encoding="utf-8"))
    assert lock["canonical_config"]["ferebus"]["scaling"] is False


def test_reconcile_apply_write_state_failure_keeps_old_state(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    old_state = _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    def fail_write_state(path, state):
        raise OSError("write failed")

    monkeypatch.setattr(cli_mod, "write_state", fail_write_state)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "failed to write recovered state.json" in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is old_state.phase
    assert state.training_set_version == old_state.training_set_version
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").is_file()


def test_reconcile_apply_config_lock_failure_restores_old_state(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    old_state = _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    def fail_config_lock(path, config):
        raise OSError("lock failed")

    monkeypatch.setattr(cli_mod, "apply_config_lock_update", fail_config_lock)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "failed to update config lock" in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is old_state.phase
    assert state.training_set_version == old_state.training_set_version
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").is_file()


def test_reconcile_apply_refuses_when_daemon_lock_may_be_active(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    monkeypatch.setattr(
        cli_mod,
        "_reconcile_runtime_status",
        lambda campaign: {
            "lock_held": True,
            "lease_heartbeat": None,
            "background_pid": None,
            "background_pid_alive": False,
            "reconcile_apply_blockers": ["daemon lock is held"],
        },
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "daemon may still be running" in err
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").exists()


def test_reconcile_non_apply_warns_when_daemon_lock_may_be_active(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    monkeypatch.setattr(
        cli_mod,
        "_reconcile_runtime_status",
        lambda campaign: {
            "lock_held": True,
            "lease_heartbeat": None,
            "background_pid": None,
            "background_pid_alive": False,
            "reconcile_apply_blockers": ["daemon lock is held"],
        },
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=False,
        )
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "daemon may still be running" in captured.err
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").exists()


def test_reconcile_restore_config_from_lock_writes_proposal(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    config.system_name = "RESTORED"
    write_config_lock(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=True,
        )
    )
    out = capsys.readouterr().out

    proposed = campaign / "campaign.yaml.proposed"
    assert rc == 0
    assert proposed.is_file()
    assert "Config proposal written" in out
    restored = CampaignConfig.from_yaml(proposed)
    assert restored.system_name == "RESTORED"


def test_reconcile_restore_config_from_lock_uses_current_directory(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    config.system_name = "RESTORED_CWD"
    write_config_lock(campaign, config)
    monkeypatch.chdir(campaign)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=None,
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=True,
        )
    )
    capsys.readouterr()

    proposed = campaign / "campaign.yaml.proposed"
    assert rc == 0
    assert proposed.is_file()
    restored = CampaignConfig.from_yaml(proposed)
    assert restored.system_name == "RESTORED_CWD"


def test_restore_config_from_lock_archives_existing_proposal(tmp_path):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    config.system_name = "RESTORED_ARCHIVE"
    write_config_lock(campaign, config)
    old_proposal = campaign / "campaign.yaml.proposed"
    old_proposal.write_text("old proposal marker\n", encoding="utf-8")

    target = restore_config_from_lock_proposal(campaign)

    archived = sorted(campaign.glob("campaign.yaml.proposed.before-*"))
    assert target == old_proposal
    assert target.is_file()
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "old proposal marker\n"
    restored = CampaignConfig.from_yaml(target)
    assert restored.system_name == "RESTORED_ARCHIVE"


def test_reconcile_restore_config_from_lock_refuses_existing_campaign_yaml(
    tmp_path,
    capsys,
):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            restore_config_from_lock=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "campaign.yaml already exists" in err


def test_reconcile_apply_archives_data_staging_for_ferebus_reentry(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    changed = CampaignConfig()
    changed.ferebus.scaling = False
    _write_config(campaign, changed)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "INITIAL_AIMALL" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    archived = sorted((campaign / ".DATA").glob("STAGING.before-reconcile-*"))
    assert len(archived) == 1
    assert (archived[0] / "INITIAL_AIMALL" / "old.txt").read_text(
        encoding="utf-8"
    ) == "old scratch"
    assert data_staging.is_dir()
    assert list(data_staging.iterdir()) == []
    assert "Archived stale .DATA/STAGING" in out


def test_reconcile_apply_archives_safe_dangling_training_staging(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    tv = TrainingSetVersioning(campaign / "5_TRAINING")
    dangling = tv.staging_path(1)
    dangling.mkdir(parents=True)
    (dangling / "partial.txt").write_text("partial\n", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    archived = sorted((campaign / "5_TRAINING").glob("iteration-0001.staging.before-reconcile-*"))
    assert len(archived) == 1
    assert (archived[0] / "partial.txt").read_text(encoding="utf-8") == "partial\n"
    assert "Archived stale training staging" in out


def test_reconcile_apply_cleans_transient_halted_ariadne_reentry(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.training_set_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    scripts = campaign / ".DATA" / "SCRIPTS"
    (scripts / "OUTPUTS").mkdir(parents=True)
    (scripts / "ERRORS").mkdir()
    (scripts / "ARIADNE_ARRAY-0.sh").write_text("# stale\n", encoding="utf-8")
    (scripts / "ERRORS" / "ARIADNE_ARRAY-0.e").write_text(
        "old error\n",
        encoding="utf-8",
    )
    model_staging = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    model_staging.mkdir(parents=True)
    (model_staging / "stale.model").write_text("stale\n", encoding="utf-8")
    committed_model = campaign / "6_TRAINED_MODELS" / "iteration-0000"
    committed_model.mkdir()
    (committed_model / "marker.txt").write_text("committed\n", encoding="utf-8")

    first_state = fresh_campaign_state(max_iterations=1)
    first_state.phase = CampaignPhase.HALTED
    first_state.training_set_version = 0
    first_state.models_version = 0
    second_state = fresh_campaign_state(max_iterations=1)
    second_state.phase = CampaignPhase.STOP_CHECK
    second_state.training_set_version = 0
    second_state.models_version = 0
    reports = iter([
        ReconciliationReport(
            proposed_state=first_state,
            committed_training_versions=[0],
            committed_model_versions=[0],
            last_phase_in_journal=CampaignPhase.ARIADNE_ARRAY.value,
            last_iteration_in_journal=0,
            notes=["re-entry HALTED because committed artefacts need operator review"],
            unsafe_reasons=[
                ".DATA/SCRIPTS contains sbatch scripts",
                "dangling model staging directories exist",
            ],
        ),
        ReconciliationReport(
            proposed_state=second_state,
            committed_training_versions=[0],
            committed_model_versions=[0],
            last_phase_in_journal=CampaignPhase.ARIADNE_ARRAY.value,
            last_iteration_in_journal=0,
            notes=["re-entry at STOP_CHECK (next tick decides loop/terminate)"],
            unsafe_reasons=[],
        ),
    ])

    monkeypatch.setattr(cli_mod, "propose_recovery", lambda *args, **kwargs: next(reports))
    monkeypatch.setattr(
        config_lock_mod,
        "verify_committed_model_version",
        lambda *args, **kwargs: None,
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.ARIADNE_ARRAY
    assert recovered.iteration == 0
    assert not model_staging.exists()
    assert (committed_model / "marker.txt").read_text(encoding="utf-8") == "committed\n"
    archived_scripts = sorted((campaign / ".DATA").glob("SCRIPTS.before-reconcile-*"))
    assert len(archived_scripts) == 1
    assert (archived_scripts[0] / "ARIADNE_ARRAY-0.sh").is_file()
    assert (archived_scripts[0] / "ERRORS" / "ARIADNE_ARRAY-0.e").is_file()
    assert (scripts / "OUTPUTS").is_dir()
    assert (scripts / "ERRORS").is_dir()
    assert "Archived stale .DATA/SCRIPTS" in out
    assert "Removed stale model staging" in out


def test_reconcile_apply_keeps_data_staging_blocked_for_non_ferebus_reentry(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.training_set_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    cfg = CampaignConfig()
    _write_config(campaign, cfg)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "INITIAL_GAUSSIAN" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert ".DATA/STAGING is non-empty" in err
    assert stale_file.exists()


def test_reconcile_apply_refuses_locked_config_change(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.gaussian.method = "PBE0"
    _write_config(campaign, changed)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "locked changes" in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.HALTED


def test_reconcile_apply_resolves_terminal_submission_intent(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda job_id: [
            sacct_poll.JobObservation(
                job_id=str(job_id) + "_[1-12]",
                status=sacct_poll.JobStatus.CANCELLED,
                exit_code=(0, 0),
                elapsed_seconds=0,
            )
        ],
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.INITIAL_FEREBUS
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_retry"


def test_reconcile_apply_resolves_cancelled_intent_when_squeue_invalid_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        if cmd[0] == "sacct":
            return SimpleNamespace(
                returncode=0,
                stdout="16218598_[1-12]|CANCELLED by 494098|0:0|00:00:00\n",
                stderr="",
            )
        raise AssertionError("unexpected command: " + repr(cmd))

    monkeypatch.setattr(sacct_poll.subprocess, "run", fake_run)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is CampaignPhase.INITIAL_FEREBUS
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_retry"


def test_reconcile_apply_supersedes_stale_pre_submit_without_job_id(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.INITIAL_FEREBUS.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_running_job_by_name_detailed",
        lambda name: sacct_poll.JobNameLookup(None, inconclusive=False, rows=[]),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Resolved terminal submission intents" in out
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "SUPERSEDED"
    assert intent["reason"] == "reconcile_apply_pre_submit_no_job_id"


def test_reconcile_apply_blocks_pre_submit_without_job_id_when_job_exists(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.INITIAL_FEREBUS.value
    payload = _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_running_job_by_name_detailed",
        lambda name: sacct_poll.JobNameLookup(
            "222",
            inconclusive=False,
            rows=[("222", "RUNNING")],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "matching scheduler job exists" in err
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "PRE_SUBMIT"
    assert intent["expected_job_name"] == payload["expected_job_name"]


def test_reconcile_apply_blocks_pre_submit_without_job_id_on_lookup_failure(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    config.runtime.lease_stale_seconds = 10
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    phase = CampaignPhase.INITIAL_FEREBUS.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_running_job_by_name_detailed",
        lambda name: sacct_poll.JobNameLookup(
            None,
            inconclusive=True,
            rows=[],
            error="sacct unavailable",
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "job-name lookup inconclusive" in err
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "PRE_SUBMIT"


def test_reconcile_apply_refuses_when_submission_intent_job_is_still_active(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(
            active=True,
            rows=[(str(job_id), "RUNNING")],
        ),
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "still live or scheduler status is inconclusive" in err
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_reconcile_apply_refuses_when_terminal_intent_sacct_is_inconclusive(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda job_id: [],
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "sacct returned no rows" in err
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_reconcile_apply_refuses_completed_intent_without_postprocess_verification(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    _write_halted_pre_ferebus_state(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    _write_submitted_initial_ferebus_intent(campaign)

    monkeypatch.setattr(
        sacct_poll,
        "find_active_job_by_id_detailed",
        lambda job_id: sacct_poll.JobQueueLookup(active=False, rows=[]),
    )
    monkeypatch.setattr(
        sacct_poll,
        "poll_job",
        lambda job_id: [
            sacct_poll.JobObservation(
                job_id=str(job_id) + "_[1-12]",
                status=sacct_poll.JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=60,
            )
        ],
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "without postprocess verification" in err
    intent = submission_intent.load_intent(
        campaign,
        CampaignPhase.INITIAL_FEREBUS.value,
        0,
    )
    assert intent["status"] == "SUBMITTED"


def test_start_refuses_config_drift_without_reconcile_apply(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _write_halted_pre_ferebus_state(campaign)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ferebus.scaling = False
    _write_config(campaign, changed)

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            preset=None,
            live=False,
            dry_run=True,
            mock_ariadne=False,
            poll_interval=None,
            max_ticks=1,
        )
    )
    err = capsys.readouterr().err

    assert rc == 7
    assert "campaign.yaml changed" in err


def test_start_allows_clean_first_run_with_config_and_pool(tmp_path):
    campaign = _campaign(tmp_path)
    config = CampaignConfig(max_iterations=1)
    _write_config(campaign, config)
    _write_pool(campaign)

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            preset=None,
            live=False,
            dry_run=True,
            mock_ariadne=False,
            poll_interval=None,
            max_ticks=1,
            background=False,
        )
    )

    assert rc == 0
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json").is_file()


def test_start_refuses_missing_state_in_nonempty_campaign(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    _write_config(campaign, config)
    _commit_training_version(campaign, 0)

    rc = cmd_start(
        argparse.Namespace(
            campaign_dir=str(campaign),
            config=None,
            preset=None,
            live=False,
            dry_run=True,
            mock_ariadne=False,
            poll_interval=None,
            max_ticks=1,
            background=False,
        )
    )
    err = capsys.readouterr().err

    assert rc == 8
    assert "state.json is missing but this campaign is not empty" in err
    assert "reconcile --campaign-dir" in err
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json").exists()
