"""Unit tests for the DryRunPhaseExecutor and DryRunSacctPoller."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.dry_run_executor import (
    DRYRUN_JOB_PREFIX,
    DryRunPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.dry_run_sacct import (
    DRYRUN_PREFIX,
    DryRunSacctPoller,
)
from ichor.hpc.active_learning.daemon.phase_executor import (
    FailureAction,
    PhaseResult,
)
from ichor.hpc.active_learning.daemon.state import CampaignPhase
from ichor.hpc.active_learning.point_allocation import (
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
    record_quantum_results,
)
from ichor.hpc.active_learning.submit.sacct_poll import JobObservation, JobStatus
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


def _make_exec(tmp_path: Path) -> DryRunPhaseExecutor:
    cfg = CampaignConfig(max_iterations=2)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 2
    return DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)


def _state(iteration=0, **updates):
    values = {
        "iteration": int(iteration),
        "campaign_uid": "dry-run-test-campaign",
        "replacement_round": 0,
        "training_set_version": -1,
        "models_version": -1,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _complete_bootstrap(e):
    state = _state()
    e.postprocess(state, CampaignPhase.PHASE_A_POLUS, observations=[])
    e.postprocess(state, CampaignPhase.INITIAL_GAUSSIAN, observations=[])
    e.postprocess(state, CampaignPhase.INITIAL_AIMALL, observations=[])
    check = e.submit_or_run(state, CampaignPhase.INITIAL_ALLOCATION_CHECK)
    assert check.next_phase_override == CampaignPhase.INITIAL_FEREBUS.value
    e.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
    return state


def _complete_active_quantum(e, *, iteration=0):
    state = _state(
        iteration,
        training_set_version=0,
        models_version=0,
    )
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    e.postprocess(state, CampaignPhase.PHASE_B_POLUS, observations=[])
    e.postprocess(state, CampaignPhase.GAUSSIAN, observations=[])
    e.postprocess(state, CampaignPhase.AIMALL, observations=[])
    check = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)
    assert check.next_phase_override == CampaignPhase.APPEND.value
    return state


def test_executor_creates_canonical_subdirs(tmp_path):
    e = _make_exec(tmp_path)
    base = tmp_path / "campaign"
    assert (base / "5_TRAINING").is_dir()
    assert (base / "6_TRAINED_MODELS").is_dir()
    assert (base / "3_DIVERSITY_SAMPLING").is_dir()
    assert (base / "7_ACTIVE_LEARNING").is_dir()
    assert (base / ".DATA" / "SCRIPTS").is_dir()


def test_submit_or_run_sbatch_phase_writes_stub_script(tmp_path):
    e = _make_exec(tmp_path)
    state = SimpleNamespace(iteration=0)
    result = e.submit_or_run(state, CampaignPhase.PHASE_A_POLUS)
    assert isinstance(result, PhaseResult)
    assert not result.is_complete
    assert result.submitted_job_id == DRYRUN_JOB_PREFIX + "PHASE_A_POLUS-0"
    script = tmp_path / "campaign" / ".DATA" / "SCRIPTS" / "PHASE_A_POLUS-0.sh"
    assert script.exists()
    content = script.read_text()
    assert "DRYRUN" in content
    assert "iteration=0" in content


def test_submit_or_run_inline_phase_runs_synchronously(tmp_path):
    e = _make_exec(tmp_path)
    state = SimpleNamespace(iteration=0)
    result = e.submit_or_run(state, CampaignPhase.SEED_SELECT)
    assert result.is_complete is True
    assert result.submitted_job_id is None
    # Seeds file should now exist in 7_ACTIVE_LEARNING/iteration-0000/
    seeds = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0000" / "seeds.xyz"
    assert seeds.exists()


def test_initial_ferebus_postprocess_commits_training_iteration_0(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    v = TrainingSetVersioning(tmp_path / "campaign" / "5_TRAINING")
    assert 0 in v.list_committed_versions()
    assert v.current_version() == 0


def test_initial_ferebus_postprocess_commits_models_iteration_0(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    v = TrainingSetVersioning(tmp_path / "campaign" / "6_TRAINED_MODELS")
    assert 0 in v.list_committed_versions()


def test_ariadne_postprocess_writes_per_seed_results(tmp_path):
    e = _make_exec(tmp_path)
    state = SimpleNamespace(iteration=3)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    pool = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0003" / "pool"
    assert pool.is_dir()
    seed_dirs = sorted(pool.iterdir())
    assert seed_dirs
    for sd in seed_dirs:
        result_json = sd / "result.json"
        assert result_json.exists()
        payload = json.loads(result_json.read_text())
        assert payload["mock"] is True
        assert "alpha_trajectory" in payload


def test_phase_b_polus_postprocess_writes_sample_xyz(tmp_path):
    e = _make_exec(tmp_path)
    state = _state(2)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    e.postprocess(state, CampaignPhase.PHASE_B_POLUS, observations=[])
    sample = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0002" / "phase_b_SAMPLE.xyz"
    assert sample.exists()
    assert "descriptor=hybrid_alf_rmsd" in sample.read_text()


def test_append_inline_stages_and_commits_next_training_iteration(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _complete_active_quantum(e)
    e.submit_or_run(state, CampaignPhase.APPEND)
    v = TrainingSetVersioning(tmp_path / "campaign" / "5_TRAINING")
    assert sorted(v.list_committed_versions()) == [0, 1]
    assert v.current_version() == 1


def test_ferebus_postprocess_commits_next_models_iteration(tmp_path):
    e = _make_exec(tmp_path)
    _complete_bootstrap(e)
    state = _complete_active_quantum(e)
    append = e.submit_or_run(state, CampaignPhase.APPEND)
    state.training_set_version = append.state_updates["training_set_version"]
    e.postprocess(state, CampaignPhase.FEREBUS, observations=[])
    v = TrainingSetVersioning(tmp_path / "campaign" / "6_TRAINED_MODELS")
    assert sorted(v.list_committed_versions()) == [0, 1]


def test_active_allocation_replaces_failed_candidate_from_finite_reserve(tmp_path):
    cfg = CampaignConfig(max_iterations=1)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 3
    e = DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)
    state = _state(training_set_version=0, models_version=0)
    e.postprocess(state, CampaignPhase.ARIADNE_ARRAY, observations=[])
    e.postprocess(state, CampaignPhase.PHASE_B_POLUS, observations=[])

    allocation_path = point_allocation_path(
        e.campaign_dir,
        context="active",
        iteration=0,
    )
    allocation = read_point_allocation(allocation_path)
    attempts = pending_attempts(allocation)
    failed = attempts[-1]
    record_quantum_results(
        allocation_path,
        [
            {
                "candidate_id": attempt["candidate_id"],
                "accepted": attempt["candidate_id"] != failed["candidate_id"],
                "pointdir": str(
                    e.campaign_dir
                    / ".DATA"
                    / "STAGING"
                    / "iter_0"
                    / ("synthetic-" + str(attempt["slot_id"]) + ".pointdir")
                ),
                "reason": (
                    "synthetic_qm_failure"
                    if attempt["candidate_id"] == failed["candidate_id"]
                    else None
                ),
            }
            for attempt in attempts
        ],
    )

    replacement = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)
    assert replacement.next_phase_override == CampaignPhase.REPLACEMENT_GAUSSIAN.value
    assert replacement.state_updates == {"replacement_round": 1}
    state.replacement_round = 1
    e.postprocess(state, CampaignPhase.REPLACEMENT_GAUSSIAN, observations=[])
    aimall = e.postprocess(state, CampaignPhase.REPLACEMENT_AIMALL, observations=[])
    assert aimall.next_phase_override == CampaignPhase.ALLOCATION_CHECK.value
    complete = e.submit_or_run(state, CampaignPhase.ALLOCATION_CHECK)

    final_allocation = read_point_allocation(allocation_path)
    assert final_allocation["summary"]["complete"] is True
    assert final_allocation["summary"]["accepted"] == {
        "train": 1,
        "int_val": 1,
        "ext_val": 0,
    }
    assert complete.next_phase_override == CampaignPhase.APPEND.value
    assert complete.state_updates == {"replacement_round": 0}


def test_handle_failure_returns_scrub_and_continue(tmp_path):
    e = _make_exec(tmp_path)
    action = e.handle_failure(SimpleNamespace(iteration=0), CampaignPhase.GAUSSIAN, observations=[])
    assert action is FailureAction.SCRUB_AND_CONTINUE


# --- DryRunSacctPoller -------------------------------------------------


def test_dry_run_sacct_returns_completed_for_dryrun_prefix():
    poller = DryRunSacctPoller(elapsed_seconds=0)
    obs = poller(DRYRUN_PREFIX + "GAUSSIAN-3")
    assert len(obs) == 1
    assert obs[0].status is JobStatus.COMPLETED
    assert obs[0].exit_code == (0, 0)


def test_dry_run_sacct_returns_every_expected_array_row():
    poller = DryRunSacctPoller(elapsed_seconds=0)

    observations = poller(
        DRYRUN_PREFIX + "AIMALL-0",
        expected_task_count=4,
    )

    assert [row.job_id for row in observations] == [
        DRYRUN_PREFIX + "AIMALL-0_0",
        DRYRUN_PREFIX + "AIMALL-0_1",
        DRYRUN_PREFIX + "AIMALL-0_2",
        DRYRUN_PREFIX + "AIMALL-0_3",
    ]
    assert all(row.status is JobStatus.COMPLETED for row in observations)


def test_dry_run_sacct_falls_back_for_non_dryrun_ids():
    def fake_fallback(job_id, **kw):
        return [JobObservation(job_id=job_id, status=JobStatus.RUNNING, exit_code=None, elapsed_seconds=10)]
    poller = DryRunSacctPoller(fallback_poller=fake_fallback)
    obs = poller("12345678")
    assert obs[0].status is JobStatus.RUNNING


def test_dry_run_sacct_records_invocations():
    poller = DryRunSacctPoller(elapsed_seconds=0)
    poller(DRYRUN_PREFIX + "FEREBUS-0")
    poller(DRYRUN_PREFIX + "AIMALL-1")
    assert poller.invocations == [DRYRUN_PREFIX + "FEREBUS-0", DRYRUN_PREFIX + "AIMALL-1"]
