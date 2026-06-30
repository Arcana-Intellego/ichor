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
from ichor.hpc.active_learning.submit.sacct_poll import JobObservation, JobStatus
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


def _make_exec(tmp_path: Path) -> DryRunPhaseExecutor:
    cfg = CampaignConfig(max_iterations=2)
    cfg.active_batch.final_batch_size = 2
    cfg.seed_selection.n_seeds_per_iteration = 2
    return DryRunPhaseExecutor(campaign_dir=tmp_path / "campaign", config=cfg)


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
    state = SimpleNamespace(iteration=0)
    e.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
    v = TrainingSetVersioning(tmp_path / "campaign" / "5_TRAINING")
    assert 0 in v.list_committed_versions()
    assert v.current_version() == 0


def test_initial_ferebus_postprocess_commits_models_iteration_0(tmp_path):
    e = _make_exec(tmp_path)
    state = SimpleNamespace(iteration=0)
    e.postprocess(state, CampaignPhase.INITIAL_FEREBUS, observations=[])
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
    state = SimpleNamespace(iteration=2)
    e.postprocess(state, CampaignPhase.PHASE_B_POLUS, observations=[])
    sample = tmp_path / "campaign" / "7_ACTIVE_LEARNING" / "iteration-0002" / "phase_b_SAMPLE.xyz"
    assert sample.exists()
    assert "descriptor=hybrid_alf_rmsd" in sample.read_text()


def test_append_inline_stages_and_commits_next_training_iteration(tmp_path):
    e = _make_exec(tmp_path)
    # Seed iteration 0 via INITIAL_FEREBUS
    e.postprocess(SimpleNamespace(iteration=0), CampaignPhase.INITIAL_FEREBUS, observations=[])
    # Now run an inline APPEND for iteration 0
    e.submit_or_run(SimpleNamespace(iteration=0), CampaignPhase.APPEND)
    v = TrainingSetVersioning(tmp_path / "campaign" / "5_TRAINING")
    assert sorted(v.list_committed_versions()) == [0, 1]
    assert v.current_version() == 1


def test_ferebus_postprocess_commits_next_models_iteration(tmp_path):
    e = _make_exec(tmp_path)
    # Seed iteration 0
    e.postprocess(SimpleNamespace(iteration=0), CampaignPhase.INITIAL_FEREBUS, observations=[])
    # Run FEREBUS post for iter 0
    e.postprocess(SimpleNamespace(iteration=0, training_set_version=1), CampaignPhase.FEREBUS, observations=[])
    v = TrainingSetVersioning(tmp_path / "campaign" / "6_TRAINED_MODELS")
    assert sorted(v.list_committed_versions()) == [0, 1]


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
