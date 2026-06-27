"""Tests for ichor.hpc.active_learning.daemon.reconcile."""
import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.daemon.journal import append_event
from ichor.hpc.active_learning.daemon.reconcile import (
    RECONCILE_SUFFIX,
    ReconciliationReport,
    propose_recovery,
    stateful_campaign_artifacts,
    write_proposed_state,
)
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.daemon.state import (
    CampaignPhase,
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


def _campaign_dirs(tmp_path):
    campaign = tmp_path / "campaign"
    data = campaign / ".DATA" / "ACTIVE_LEARNING"
    training = campaign / "5_TRAINING"
    models = campaign / "6_TRAINED_MODELS"
    data.mkdir(parents=True, exist_ok=True)
    training.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return campaign, data, training, models


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


def _write_valid_initial_aimall_handoff(campaign, *, iteration=0):
    initial = campaign / ".DATA" / "STAGING" / "initial"
    pointdir = initial / "POINT_0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    stg.write_quantum_acceptance_manifest(
        initial,
        phase_name=CampaignPhase.INITIAL_AIMALL.value,
        iteration=iteration,
        accepted=[pointdir],
        rejected=[],
    )
    return initial


def test_propose_recovery_on_empty_campaign_returns_init(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    report = propose_recovery(campaign)
    assert report.proposed_state.phase is CampaignPhase.INIT
    assert report.proposed_state.training_set_version == 0
    assert report.proposed_state.models_version == 0
    assert report.committed_training_versions == []
    assert report.committed_model_versions == []
    assert report.existing_state_loaded is False


def test_propose_recovery_never_trusts_stop_check_without_committed_versions(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.training_set_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.committed_training_versions == []
    assert report.committed_model_versions == []
    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert report.proposed_state.training_set_version == -1
    assert report.proposed_state.models_version == -1
    assert "latest coherent committed training/model pair is trusted" not in report.decision
    assert "no committed versions" in report.decision


def test_propose_recovery_initial_aimall_handoff_reenters_initial_ferebus(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "phase_transition",
        from_phase=CampaignPhase.INITIAL_GAUSSIAN.value,
        to_phase=CampaignPhase.INITIAL_AIMALL.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.training_set_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_valid_initial_aimall_handoff(campaign)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.INITIAL_FEREBUS
    assert report.proposed_state.training_set_version == -1
    assert report.proposed_state.models_version == -1
    assert "valid initial AIMAll handoff" in report.decision
    assert "initial AIMAll acceptance manifest" in report.trusted_artifacts


def test_propose_recovery_initial_ferebus_journal_handoff_reenters_initial_ferebus(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "phase_transition",
        from_phase=CampaignPhase.INITIAL_AIMALL.value,
        to_phase=CampaignPhase.INITIAL_FEREBUS.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.training_set_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)
    _write_valid_initial_aimall_handoff(campaign)

    report = propose_recovery(campaign)

    assert report.last_phase_in_journal == CampaignPhase.INITIAL_FEREBUS.value
    assert report.proposed_state.phase is CampaignPhase.INITIAL_FEREBUS
    assert report.proposed_state.training_set_version == -1
    assert report.proposed_state.models_version == -1


def test_propose_recovery_initial_aimall_missing_handoff_halts(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    append_event(
        data / "journal.ndjson",
        "phase_transition",
        from_phase=CampaignPhase.INITIAL_GAUSSIAN.value,
        to_phase=CampaignPhase.INITIAL_AIMALL.value,
        iteration=0,
    )
    state = fresh_campaign_state(max_iterations=50)
    state.phase = CampaignPhase.STOP_CHECK
    state.training_set_version = 0
    state.models_version = 0
    write_state(data / DEFAULT_STATE_FILENAME, state)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert report.proposed_state.training_set_version == -1
    assert report.proposed_state.models_version == -1
    assert "INITIAL_AIMALL completed" in report.decision
    assert any("initial AIMAll handoff invalid or missing" in r for r in report.unsafe_reasons)
    assert ".DATA/STAGING/initial" in report.blocking_artifacts


def test_propose_recovery_missing_state_nonempty_staging_halts(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    staging = campaign / ".DATA" / "STAGING" / "iter_0"
    staging.mkdir(parents=True)
    (staging / "POINTS.txt").write_text("", encoding="utf-8")
    report = propose_recovery(campaign)
    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any("non-empty campaign" in n for n in report.notes)
    assert report.unsafe_reasons


def test_stateful_campaign_artifacts_include_config_lock(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    (data / "config_lock.json").write_text("{}", encoding="utf-8")

    findings = stateful_campaign_artifacts(campaign)

    assert ".DATA/ACTIVE_LEARNING/config_lock.json" in findings


def test_propose_recovery_active_submission_intent_is_adoption_ready(tmp_path):
    from ichor.hpc.active_learning.daemon.submission_intent import (
        mark_submitted,
        write_pre_submit_intent,
    )

    campaign, _, _, _ = _campaign_dirs(tmp_path)
    write_pre_submit_intent(
        campaign,
        campaign_uid="uid",
        phase_name="FEREBUS",
        iteration=3,
    )
    mark_submitted(campaign, "FEREBUS", 3, "123456")
    report = propose_recovery(campaign)
    assert report.proposed_state.phase is CampaignPhase.FEREBUS
    assert report.proposed_state.iteration == 3
    assert report.active_submission_intents
    reason = "\n".join(report.unsafe_reasons)
    assert "job_id=123456" in reason
    assert "expected_job_name=uid-FEREBUS-3" in reason


def test_propose_recovery_force_allows_fresh_init_on_nonempty_campaign(tmp_path):
    campaign, _, _, _ = _campaign_dirs(tmp_path)
    staging = campaign / ".DATA" / "STAGING" / "iter_0"
    staging.mkdir(parents=True)
    (staging / "POINTS.txt").write_text("", encoding="utf-8")
    report = propose_recovery(campaign, allow_fresh_init_on_nonempty=True)
    assert report.proposed_state.phase is CampaignPhase.INIT


def test_propose_recovery_finds_committed_training_versions(tmp_path):
    campaign, _, training, _ = _campaign_dirs(tmp_path)
    v = TrainingSetVersioning(training)
    for i in (0, 1, 2):
        s = v.stage(None, i)
        (s / "marker.txt").write_text(str(i))
        v.commit(i)
    report = propose_recovery(campaign)
    assert report.committed_training_versions == [0, 1, 2]
    assert report.proposed_state.training_set_version == 2
    assert report.proposed_state.models_version == -1
    assert report.proposed_state.phase is CampaignPhase.FEREBUS
    assert any("trajectory pool" in r for r in report.unsafe_reasons)


def test_propose_recovery_reports_decision_and_trusted_versions(tmp_path):
    campaign, _, training, models = _campaign_dirs(tmp_path)
    tv = TrainingSetVersioning(training)
    mv = TrainingSetVersioning(models)
    s = tv.stage(None, 0)
    (s / "marker.txt").write_text("training", encoding="utf-8")
    tv.commit(0)
    s = mv.stage(None, 0)
    (s / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert "coherent committed" in report.decision
    assert "training version 0" in report.trusted_artifacts
    assert "model version 0" in report.trusted_artifacts
    assert "trajectory pool" in report.blocking_artifacts


def test_propose_recovery_blocks_trajectory_pool_sha_drift(tmp_path):
    campaign, _, training, models = _campaign_dirs(tmp_path)
    _write_pool(campaign)
    tv = TrainingSetVersioning(training)
    mv = TrainingSetVersioning(models)
    s = tv.stage(None, 0)
    (s / "marker.txt").write_text("training", encoding="utf-8")
    tv.commit(0)
    s = mv.stage(None, 0)
    (s / "marker.txt").write_text("model", encoding="utf-8")
    mv.commit(0)

    pool_xyz = campaign / ".DATA" / "TRAJECTORY" / "pool.xyz"
    with pool_xyz.open("a", encoding="utf-8") as f:
        f.write("# drift\n")

    report = propose_recovery(campaign)

    assert report.proposed_state.phase is CampaignPhase.HALTED
    assert any("trajectory pool SHA mismatch" in r for r in report.unsafe_reasons)
    assert "trajectory pool" in report.blocking_artifacts


def test_propose_recovery_reports_unmanifested_committed_pointdir(tmp_path):
    campaign, _, training, _ = _campaign_dirs(tmp_path)
    v = TrainingSetVersioning(training)
    s = v.stage(None, 0)
    (s / "marker.txt").write_text("committed", encoding="utf-8")
    v.commit(0)
    rogue = v.iteration_path(0) / "POINT_9999.pointdir"
    rogue.mkdir()
    (rogue / "input.gjf").write_text("%chk=x\n", encoding="utf-8")
    report = propose_recovery(campaign)
    assert report.committed_training_versions == [0]
    assert report.valid_training_versions == []
    assert any("committed training version 0" in r for r in report.unsafe_reasons)
    assert report.proposed_state.phase is CampaignPhase.HALTED


def test_propose_recovery_preserves_existing_campaign_uid(tmp_path):
    campaign, data, training, _ = _campaign_dirs(tmp_path)
    v = TrainingSetVersioning(training)
    s = v.stage(None, 0); (s / "x.txt").write_text("hi"); v.commit(0)
    existing = fresh_campaign_state(max_iterations=42)
    existing.iteration = 5
    existing.last_acquisition_alpha0 = 0.41
    write_state(data / DEFAULT_STATE_FILENAME, existing)
    report = propose_recovery(campaign)
    assert report.proposed_state.campaign_uid == existing.campaign_uid
    assert report.proposed_state.max_iterations == 42
    assert report.proposed_state.last_acquisition_alpha0 == pytest.approx(0.41)
    assert report.existing_state_loaded is True


def test_propose_recovery_clears_pending_jobs(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    existing = fresh_campaign_state()
    existing.pending_jobs = {"FEREBUS": "9999"}
    write_state(data / DEFAULT_STATE_FILENAME, existing)
    report = propose_recovery(campaign)
    assert report.proposed_state.pending_jobs == {}


def test_propose_recovery_clears_shutdown_request(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    existing = fresh_campaign_state()
    existing.shutdown_requested = True
    write_state(data / DEFAULT_STATE_FILENAME, existing)
    report = propose_recovery(campaign)
    assert report.proposed_state.shutdown_requested is False


def test_propose_recovery_reads_last_journal_transition(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    journal = data / "journal.ndjson"
    append_event(journal, "phase_transition", from_phase="INIT", to_phase="PHASE_A_POLUS", iteration=0)
    append_event(journal, "phase_transition", from_phase="GAUSSIAN", to_phase="AIMALL", iteration=3)
    report = propose_recovery(campaign)
    assert report.last_phase_in_journal == "AIMALL"
    assert report.last_iteration_in_journal == 3


def test_propose_recovery_corrupt_state_does_not_crash(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    (data / DEFAULT_STATE_FILENAME).write_text("{not valid json")
    report = propose_recovery(campaign)
    # Falls through to a fresh state since existing did not load.
    assert report.existing_state_loaded is False
    assert any("failed validation" in n or "unreadable" in n for n in report.notes)


def test_write_proposed_state_creates_proposed_file(tmp_path):
    campaign, data, _, _ = _campaign_dirs(tmp_path)
    report = propose_recovery(campaign)
    target = write_proposed_state(campaign, report)
    assert target.exists()
    assert target.name == DEFAULT_STATE_FILENAME + RECONCILE_SUFFIX
    payload = read_state(target)
    assert payload.phase == report.proposed_state.phase
    # Does NOT clobber the live state.json.
    assert not (data / DEFAULT_STATE_FILENAME).exists()


# --- M15 F16: propose_recovery preserves M13/M14/M15 fields ------------


def test_propose_recovery_preserves_reference_scales_cache(tmp_path):
    """M13's reference_scales + reference_scales_iteration must survive
    a reconcile pass; otherwise STOP_CHECK alpha-trend gets reset and the
    cached scales get recomputed regardless of refresh policy."""
    from ichor.hpc.active_learning.daemon.reconcile import propose_recovery
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, CampaignState, fresh_campaign_state, write_state,
    )
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=20)
    state.iteration = 7
    state.phase = CampaignPhase.STOP_CHECK
    state.reference_scales = {"energy": 1.0e-3, "force": 1.0e-2, "omega": 1.0, "anh": 1.0, "anh_std": 1.0}
    state.reference_scales_iteration = 7
    write_state(cd / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    report = propose_recovery(cd)
    proposal = report.proposed_state
    assert proposal.reference_scales == {"energy": 1.0e-3, "force": 1.0e-2, "omega": 1.0, "anh": 1.0, "anh_std": 1.0}
    assert proposal.reference_scales_iteration == 7


def test_propose_recovery_preserves_alpha_history(tmp_path):
    """M14 alpha-trend history must survive reconcile."""
    from ichor.hpc.active_learning.daemon.reconcile import propose_recovery
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, fresh_campaign_state, write_state,
    )
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=20)
    state.iteration = 12
    state.phase = CampaignPhase.STOP_CHECK
    state.alpha_history = [0.1, 0.08, 0.05, 0.03, 0.02]
    write_state(cd / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    report = propose_recovery(cd)
    proposal = report.proposed_state
    assert proposal.alpha_history == [0.1, 0.08, 0.05, 0.03, 0.02]


def test_propose_recovery_preserves_M15_diagnostic_fields(tmp_path):
    """M15 F3 last_n_anti_overlap_flagged + F6 sacct_empty_streak must
    survive reconcile too -- otherwise the daemon would lose the in-flight
    sacct timeout state on every recovery."""
    from ichor.hpc.active_learning.daemon.reconcile import propose_recovery
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, fresh_campaign_state, write_state,
    )
    cd = tmp_path / "campaign"
    (cd / ".DATA" / "ACTIVE_LEARNING").mkdir(parents=True)
    state = fresh_campaign_state(max_iterations=20)
    state.iteration = 3
    state.phase = CampaignPhase.SEED_SELECT
    state.last_n_anti_overlap_flagged = 7
    state.sacct_empty_streak = {"99999": 4, "11111": 1}
    write_state(cd / ".DATA" / "ACTIVE_LEARNING" / "state.json", state)
    report = propose_recovery(cd)
    proposal = report.proposed_state
    assert proposal.last_n_anti_overlap_flagged == 7
    assert proposal.sacct_empty_streak == {"99999": 4, "11111": 1}
