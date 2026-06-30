import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import ichor.hpc.active_learning.cli as cli_mod
import ichor.hpc.active_learning.daemon.reconcile as reconcile_mod
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
from ichor.hpc.active_learning.daemon.journal import iter_events
from ichor.hpc.active_learning.daemon.reconcile import ReconciliationReport
from ichor.hpc.active_learning.daemon import input_staging as stg
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


def _write_phase_a_sample(campaign):
    from ichor.hpc.active_learning.handoff_manifests import write_phase_a_sample_manifest

    initial = campaign / "3_DIVERSITY_SAMPLING" / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    sample = initial / "initial-SAMPLE-1.xyz"
    index = initial / "initial-INDEX-1.dat"
    sample.write_text("1\nframe 0\nH 0.0 0.0 0.0\n", encoding="utf-8")
    index.write_text("0\n", encoding="utf-8")
    write_phase_a_sample_manifest(initial, {
        "phase": "PHASE_A_POLUS",
        "iteration": -1,
        "sample_xyz": str(sample.resolve()),
        "index_path": str(index.resolve()),
        "n_select": 1,
        "n_frames": 1,
        "selected_indices": [0],
        "descriptor": "mass_weighted_rmsd",
        "n_pool_frames": 1,
    })


def _write_live_initial_gaussian_handoff(campaign):
    initial = campaign / ".DATA" / "STAGING" / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.wfn").write_text("wfn\n", encoding="utf-8")
    stg.write_points_file(initial, [pointdir])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=CampaignPhase.INITIAL_GAUSSIAN.value,
        iteration=0,
        accepted=[pointdir],
        rejected=[],
    )


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


def _write_archived_bootstrap_handoff(campaign, *, phase, suffix="20260627-201927"):
    initial = campaign / ".DATA" / ("STAGING.archived-" + suffix) / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.wfn").write_text("wfn\n", encoding="utf-8")
    stg.write_points_file(initial, [pointdir])
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=phase,
        iteration=0,
        accepted=[pointdir],
        rejected=[],
    )
    return initial


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


def test_reconcile_prints_recovery_contract_and_first_pass_guidance(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    _write_phase_a_sample(campaign)
    config = CampaignConfig()
    _write_config(campaign, config)
    write_config_lock(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            archive_staging=False,
            restore_config_from_lock=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "=== Recovery contract ===" in out
    assert "selected phase: INITIAL_GAUSSIAN" in out
    assert "contract:        ok" in out
    assert "Required inputs:" in out
    assert "Phase A sample" in out
    assert "=== Recovery guidance ===" not in out
    assert "=== First-pass recovery proposal ===" in out
    assert "Valid recovery candidates:" in out
    assert "ichor-al-daemon reconcile --campaign-dir " + str(campaign) + " --apply" in out
    assert "=== Inspect commands ===" in out


def test_reconcile_prints_protected_staging_handoff(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _write_live_initial_gaussian_handoff(campaign)
    config = CampaignConfig()
    _write_config(campaign, config)
    write_config_lock(campaign, config)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=False,
            archive_staging=False,
            restore_config_from_lock=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Protected staging handoffs:" in out
    assert "INITIAL_AIMALL@0 -> .DATA\\STAGING\\initial" in out or (
        "INITIAL_AIMALL@0 -> .DATA/STAGING/initial" in out
    )
    assert "Operator-review artefacts:" not in out
    assert "Hard blockers:" in out


def test_phase_walltime_changes_are_allowed_runtime_changes(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.resources.ferebus.walltime_hours = 2
    changed.resources.gaussian.walltime_hours = 3
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_FEREBUS
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert sorted(c.path for c in review.allowed_changes) == [
        "resources.ferebus.walltime_hours",
        "resources.gaussian.walltime_hours",
    ]
    assert not review.blocked_changes


def test_schema_v3_config_lock_migrates_before_diff(tmp_path):
    campaign = _campaign(tmp_path)
    current = CampaignConfig()
    old_v3 = {
        "schema_version": 3,
        "system_name": current.system_name,
        "resources": {
            "partition": "multicore",
            "default_walltime_hours": 24,
            "polus_walltime_hours": 2,
            "gaussian_walltime_hours": 24,
            "polus_cpus_per_task": "auto",
            "gaussian_cpus_per_task": "auto",
            "aimall_cpus_per_task": "auto",
            "ariadne_cpus_per_task": "auto",
            "ferebus_cpus_per_task": "auto",
            "polus_mem_per_cpu": "auto",
            "gaussian_mem_per_cpu": "auto",
            "aimall_mem_per_cpu": "auto",
            "ariadne_mem_per_cpu": "auto",
            "ferebus_mem_per_cpu": "auto",
            "gaussian_memory_mode": "slurm_env",
            "gaussian_link0_mem": "8GB",
            "gaussian_memory_fraction_of_slurm": 0.85,
            "array_concurrency_limit": None,
            "fail_on_memory_estimate_exceeds_request": True,
            "gradient_parallel_backend": "process",
        },
    }
    path = config_lock_path(campaign)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_schema_version": 3,
                "created_at_iso": "2026-01-01T00:00:00+00:00",
                "last_checked_at_iso": "2026-01-01T00:00:00+00:00",
                "canonical_config": old_v3,
                "fingerprint_sha256": "legacy",
                "field_policy_version": 1,
            }
        ),
        encoding="utf-8",
    )
    review = review_config_changes(campaign, current, fresh_campaign_state())
    assert review.allowed
    assert not review.changed


def test_retry_phase_requires_retryable_journal_event():
    state = fresh_campaign_state()
    success_report = ReconciliationReport(
        proposed_state=state,
        last_phase_in_journal=CampaignPhase.PHASE_B_POLUS.value,
        last_iteration_in_journal=0,
        last_phase_event_in_journal="phase_succeeded",
        last_phase_retryable=False,
    )
    halted_report = ReconciliationReport(
        proposed_state=state,
        last_phase_in_journal=CampaignPhase.PHASE_B_POLUS.value,
        last_iteration_in_journal=0,
        last_phase_event_in_journal="halt",
        last_phase_retryable=True,
    )

    assert cli_mod._retry_phase_from_cleaned_report(success_report) is None
    assert cli_mod._retry_phase_from_cleaned_report(halted_report) is CampaignPhase.PHASE_B_POLUS


def test_reconcile_apply_uses_safe_max_iterations_change(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    _commit_training_version(campaign, 0)
    state = fresh_campaign_state(max_iterations=1)
    state.phase = CampaignPhase.HALTED
    state.training_set_version = 0
    state.models_version = -1
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    original = CampaignConfig(max_iterations=1)
    write_config_lock(campaign, original)
    changed = CampaignConfig(max_iterations=7)
    _write_config(campaign, changed)

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
        )
    )

    assert rc == 0
    capsys.readouterr()
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.max_iterations == 7


def test_memory_estimate_guard_default_migration_is_allowed(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    path = config_lock_path(campaign)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["canonical_config"]["resources"]["fail_on_memory_estimate_exceeds_request"]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    changed = CampaignConfig()
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.INITIAL_AIMALL
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == [
        "resources.fail_on_memory_estimate_exceeds_request"
    ]
    assert not review.blocked_changes


def test_acquisition_driver_default_migration_is_future_safe(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    path = config_lock_path(campaign)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["canonical_config"]["acquisition"]["driver"]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    changed = CampaignConfig()
    _write_config(campaign, changed)

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    changed_paths = sorted(c.path for c in review.allowed_changes)
    assert changed_paths
    assert all(path.startswith("acquisition.driver.") for path in changed_paths)
    assert {c.category for c in review.allowed_changes} == {"future_safe"}
    assert not review.blocked_changes


def test_default_campaign_config_has_no_unclassified_lock_paths():
    config = CampaignConfig()
    proposed = fresh_campaign_state()
    campaign = Path("/tmp/campaign-placeholder")
    unexpected = []
    for path, value in sorted(config_lock_mod._flatten(config.to_dict()).items()):
        change = config_lock_mod._classify_change(
            campaign,
            proposed,
            path,
            None,
            value,
        )
        if change.category == "unclassified_locked":
            unexpected.append(path)

    assert unexpected == []


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


def test_ariadne_config_change_blocks_after_result_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool" / "seed_0000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.ARIADNE_ARRAY
    proposed.iteration = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "ariadne.trqn_backtransform_mode"
    ]
    assert review.blocked_changes[0].category == "postprocess_locked"


def test_ariadne_config_change_blocks_halted_uncommitted_result(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool" / "seed_0000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.HALTED
    proposed.iteration = 0
    proposed.training_set_version = 0
    proposed.models_version = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == [
        "ariadne.trqn_backtransform_mode"
    ]
    assert "uncommitted ARIADNE" in review.blocked_changes[0].reason


def test_ariadne_config_change_allows_committed_historical_result(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.ariadne.trqn_backtransform_mode = "newton"
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000" / "pool" / "seed_0000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "result.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.STOP_CHECK
    proposed.iteration = 0
    proposed.training_set_version = 1
    proposed.models_version = 1
    review = review_config_changes(campaign, changed, proposed)

    assert review.allowed
    assert [c.path for c in review.allowed_changes] == [
        "ariadne.trqn_backtransform_mode"
    ]
    assert not review.blocked_changes


def test_phase_b_config_change_blocks_after_selection_exists(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.phase_b.min_separation = 0.20
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "PHASE_B_SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.PHASE_B_POLUS
    proposed.iteration = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["phase_b.min_separation"]
    assert review.blocked_changes[0].category == "postprocess_locked"


def test_phase_b_config_change_blocks_halted_uncommitted_selection(tmp_path):
    campaign = _campaign(tmp_path)
    original = CampaignConfig()
    write_config_lock(campaign, original)
    changed = CampaignConfig()
    changed.phase_b.min_separation = 0.20
    iter_dir = campaign / "7_ACTIVE_LEARNING" / "iteration-0000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "PHASE_B_SELECTION.json").write_text("{}", encoding="utf-8")

    proposed = fresh_campaign_state()
    proposed.phase = CampaignPhase.HALTED
    proposed.iteration = 0
    proposed.training_set_version = 0
    proposed.models_version = 0
    review = review_config_changes(campaign, changed, proposed)

    assert not review.allowed
    assert [c.path for c in review.blocked_changes] == ["phase_b.min_separation"]
    assert "uncommitted Phase B" in review.blocked_changes[0].reason


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

    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    real_write_state = cli_mod.write_state
    write_state_calls = []

    def spy_write_state(path, state):
        write_state_calls.append(Path(path))
        return real_write_state(path, state)

    monkeypatch.setattr(cli_mod, "write_state", spy_write_state)
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
    state = read_state(state_path)
    assert state.phase is old_state.phase
    assert state.training_set_version == old_state.training_set_version
    assert (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json.proposed").is_file()
    assert write_state_calls.count(state_path) >= 2


def test_reconcile_config_lock_failure_reports_prior_staging_archive(
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
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "INITIAL_AIMALL" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

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
    assert "Some cleanup/archive operations already happened" in err
    assert "STAGING.before-reconcile-" in err
    assert "State/config lock was not applied." in err
    state = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert state.phase is old_state.phase


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
    committed_training = campaign / "5_TRAINING" / "iteration-0000"
    committed_training.mkdir()
    (committed_training / "marker.txt").write_text("committed\n", encoding="utf-8")

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
            last_phase_event_in_journal="halt",
            last_phase_retryable=True,
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
    monkeypatch.setattr(
        cli_mod,
        "_reconcile_apply_contract_error",
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


def test_reconcile_apply_archive_staging_explicitly_handles_non_ferebus_reentry(
    tmp_path,
    capsys,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)

    tv = TrainingSetVersioning(campaign / "5_TRAINING")
    staging = tv.stage(None, 0)
    (staging / "marker.txt").write_text("training", encoding="utf-8")
    tv.commit(0)
    mv = TrainingSetVersioning(campaign / "6_TRAINED_MODELS")
    staging = mv.stage(None, 0)
    (staging / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)
    monkeypatch.setattr(
        reconcile_mod,
        "verify_committed_model_version",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        reconcile_mod,
        "_validate_recovered_state_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_mod,
        "_reconcile_apply_contract_error",
        lambda *args, **kwargs: None,
    )

    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.training_set_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "GAUSSIAN" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    archived = sorted((campaign / ".DATA").glob("STAGING.archived-*"))
    assert len(archived) == 1
    assert (archived[0] / "GAUSSIAN" / "old.txt").read_text(
        encoding="utf-8"
    ) == "old scratch"
    assert data_staging.is_dir()
    assert list(data_staging.iterdir()) == []
    assert "Archived stale .DATA/STAGING" in out
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.SEED_SELECT
    events = list(iter_events(campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"))
    assert any(e.get("event") == "staging_archived" for e in events)


def test_operator_archive_staging_blocks_protected_active_handoff(tmp_path):
    campaign = _campaign(tmp_path)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.iteration = 0
    state.training_set_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    staging = campaign / ".DATA" / "STAGING" / "iter_1"
    pointdir = staging / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    stg.write_points_file(staging, [pointdir])
    stg.write_quantum_acceptance_manifest(
        staging,
        phase_name=CampaignPhase.GAUSSIAN.value,
        iteration=1,
        accepted=[pointdir],
        rejected=[],
    )
    report = ReconciliationReport(proposed_state=state)

    blockers = cli_mod._operator_staging_archive_blockers(campaign, report, {})

    assert any("protected handoff for AIMALL@1" in item for item in blockers)


def test_reconcile_apply_restores_archived_initial_gaussian_handoff(
    tmp_path,
    capsys,
):
    campaign = _campaign(tmp_path)
    _write_pool(campaign)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.STOP_CHECK
    state.training_set_version = 0
    state.models_version = 0
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    archived = _write_archived_bootstrap_handoff(
        campaign,
        phase=CampaignPhase.INITIAL_GAUSSIAN.value,
    )

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=False,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    recovered = read_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json")
    assert recovered.phase is CampaignPhase.INITIAL_AIMALL
    restored = campaign / ".DATA" / "STAGING" / "initial"
    assert (restored / "accepted_pointdirs.json").is_file()
    assert (restored / "POINT_0000.pointdir" / "input.wfn").read_text(
        encoding="utf-8"
    ) == "wfn\n"
    assert "Archived bootstrap handoff" in out
    assert "Restored bootstrap staging from archive" in out
    events = list(iter_events(campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"))
    restored_events = [
        e for e in events if e.get("event") == "staging_restored_from_archive"
    ]
    assert len(restored_events) == 1
    assert restored_events[0]["source_path"] == str(archived)
    assert restored_events[0]["source_phase"] == CampaignPhase.INITIAL_GAUSSIAN.value


def test_reconcile_apply_archive_staging_refuses_pending_job(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig()
    write_config_lock(campaign, config)
    _write_config(campaign, config)
    state = fresh_campaign_state(max_iterations=3)
    state.phase = CampaignPhase.HALTED
    state.pending_jobs[CampaignPhase.GAUSSIAN.value] = "12345"
    write_state(campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    data_staging = campaign / ".DATA" / "STAGING"
    stale_file = data_staging / "GAUSSIAN" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old scratch", encoding="utf-8")

    rc = cmd_reconcile(
        argparse.Namespace(
            campaign_dir=str(campaign),
            allow_fresh_init=False,
            apply=True,
            restore_config_from_lock=False,
            archive_staging=True,
        )
    )
    err = capsys.readouterr().err

    assert rc == 9
    assert "DATA/STAGING is non-empty" in err
    assert "pending job" in err
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
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            None,
            inconclusive=False,
            rows=[],
        ),
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


def test_reconcile_apply_uses_job_name_accounting_for_ferebus_pre_submit_without_job_id(
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

    def fail_running_lookup(name, **kwargs):
        raise AssertionError("PRE_SUBMIT recovery must use accounting lookup")

    monkeypatch.setattr(sacct_poll, "find_running_job_by_name_detailed", fail_running_lookup)
    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            None,
            inconclusive=False,
            rows=[],
        ),
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


def test_reconcile_apply_blocks_pre_submit_without_job_id_when_accounting_completed(
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
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            "222",
            terminal=True,
            successful=True,
            failed=False,
            inconclusive=False,
            rows=[("222", "COMPLETED")],
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
    assert "postprocess/operator review is required" in err
    intent = submission_intent.load_intent(campaign, phase, 0)
    assert intent["status"] == "PRE_SUBMIT"


def test_reconcile_apply_marks_failed_pre_submit_without_job_id_from_accounting(
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
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            "222",
            terminal=True,
            successful=False,
            failed=True,
            inconclusive=False,
            rows=[("222", "FAILED")],
        ),
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
    assert intent["status"] == "FAILED"
    assert intent["reason"] == "reconcile_apply_terminal_job:FAILED"


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
    phase = CampaignPhase.GAUSSIAN.value
    payload = _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
            "222",
            terminal=False,
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
    phase = CampaignPhase.GAUSSIAN.value
    _write_stale_pre_submit_intent(campaign, phase)

    monkeypatch.setattr(
        sacct_poll,
        "find_accounted_job_by_name_detailed",
        lambda name, **kwargs: sacct_poll.JobNameAccountingLookup(
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


def test_start_refuses_missing_state_when_config_lock_exists(tmp_path, capsys):
    campaign = _campaign(tmp_path)
    config = CampaignConfig(max_iterations=1)
    _write_config(campaign, config)
    write_config_lock(campaign, config)

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
    assert "state.json is missing" in err
    assert ".DATA/ACTIVE_LEARNING/config_lock.json" in err
    assert not (campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json").exists()


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
