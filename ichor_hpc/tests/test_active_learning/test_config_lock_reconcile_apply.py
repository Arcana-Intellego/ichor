import argparse
import json
from types import SimpleNamespace

from ichor.hpc.active_learning.cli import cmd_reconcile, cmd_start
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.config_lock import (
    config_lock_path,
    review_config_changes,
    write_config_lock,
)
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


def _commit_training_version(campaign, version=0):
    tv = TrainingSetVersioning(campaign / "5_TRAINING")
    staging = tv.stage(None, version)
    (staging / "marker.txt").write_text("training", encoding="utf-8")
    tv.commit(version)


def _write_config(campaign, config):
    config.to_yaml(campaign / "campaign.yaml")


def _write_halted_pre_ferebus_state(campaign):
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
    lock = json.loads(config_lock_path(campaign).read_text(encoding="utf-8"))
    assert lock["canonical_config"]["ferebus"]["scaling"] is False


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
