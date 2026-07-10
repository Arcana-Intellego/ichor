"""End-to-end dry-run campaign integration test.

Drives a Daemon through a 2-iteration campaign using DryRunPhaseExecutor +
DryRunSacctPoller and asserts on the on-disk artefacts. This is the M8
acceptance test from the migration plan:

    > On CSF4 scratch, run `ichor-al-daemon start --dry-run --mock-ariadne`
    > against a small fixture. Verify all directories, manifests, journal
    > entries, sbatch invocations, sacct polls, and atomic renames work
    > end-to-end without real Gaussian/AIMAll/FEREBUS.

The test runs locally in pytest's tmp_path, but the file-system flow is
the same one CSF4 would exercise.
"""
from pathlib import Path

import pytest

from ichor.hpc.active_learning.cli import main as cli_main
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import Daemon, DEFAULT_DATA_SUBDIR
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.dry_run_sacct import DryRunSacctPoller
from ichor.hpc.active_learning.daemon.journal import iter_events
from ichor.hpc.active_learning.daemon.state import CampaignPhase, read_state
from ichor.hpc.active_learning.versioning.manifest import (
    MANIFEST_FILENAME,
    verify_manifest,
)
from ichor.hpc.active_learning.versioning.training_set import TrainingSetVersioning


_SBATCH_PHASES = (
    "PHASE_A_POLUS", "INITIAL_GAUSSIAN", "INITIAL_AIMALL", "INITIAL_FEREBUS",
    "ARIADNE_ARRAY", "PHASE_B_POLUS", "GAUSSIAN", "AIMALL", "FEREBUS",
)


def _make_campaign(tmp_path, *, max_iterations=2):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = CampaignConfig(max_iterations=max_iterations, poll_interval_seconds=1)
    cfg.point_allocation.batch_training_size = 1
    cfg.point_allocation.batch_internal_validation_size = 1
    cfg.seed_selection.n_seeds_per_iteration = 2
    return campaign, cfg


def _run_two_iter_campaign(tmp_path):
    campaign, cfg = _make_campaign(tmp_path)
    executor = DryRunPhaseExecutor(campaign_dir=campaign, config=cfg)
    poller = DryRunSacctPoller(elapsed_seconds=0)
    d = Daemon(
        campaign_dir=campaign,
        config=cfg,
        executor=executor,
        sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0
    return campaign, d, executor, poller


def test_dry_run_finishes_in_DONE(tmp_path):
    campaign, d, _, _ = _run_two_iter_campaign(tmp_path)
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE


def test_dry_run_writes_one_stub_script_per_sbatch_phase_per_iter(tmp_path):
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    scripts_dir = campaign / ".DATA" / "SCRIPTS"
    scripts = sorted(scripts_dir.iterdir())
    # 4 initial sbatch phases at iter 0 + 5 per-iteration phases x 2 iters = 14
    assert len(scripts) == 4 + 5 * 2
    # Every name follows PHASE-iter.sh
    for s in scripts:
        assert s.suffix == ".sh"
        stem = s.stem
        phase_name, _, iter_part = stem.rpartition("-")
        assert phase_name in _SBATCH_PHASES
        assert iter_part.isdigit()


def test_dry_run_commits_three_training_versions(tmp_path):
    """Initial commit (iter 0) + APPEND at iter 0 + APPEND at iter 1."""
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    v = TrainingSetVersioning(campaign / "5_TRAINING")
    assert sorted(v.list_committed_versions()) == [0, 1, 2]
    assert v.current_version() == 2


def test_dry_run_commits_three_models_versions(tmp_path):
    """Initial models (iter 0) + FEREBUS post at iter 0 + FEREBUS at iter 1."""
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    v = TrainingSetVersioning(campaign / "6_TRAINED_MODELS")
    assert sorted(v.list_committed_versions()) == [0, 1, 2]


def test_dry_run_every_committed_iteration_has_manifest(tmp_path):
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    for root in (campaign / "5_TRAINING", campaign / "6_TRAINED_MODELS"):
        v = TrainingSetVersioning(root)
        for ver in v.list_committed_versions():
            iter_dir = v.iteration_path(ver)
            assert (iter_dir / MANIFEST_FILENAME).exists()
            verify_manifest(iter_dir)  # must not raise


def test_dry_run_active_learning_dirs_have_seeds_pool_and_phase_b(tmp_path):
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    for i in (0, 1):
        d = campaign / "7_ACTIVE_LEARNING" / ("iteration-" + str(i).zfill(4))
        assert d.is_dir()
        assert (d / "seeds.xyz").exists()
        assert (d / "pool").is_dir()
        assert (d / "phase_b_SAMPLE.xyz").exists()
        assert (d / "split.json").exists()
        # Every seed has a result.json
        seed_dirs = list((d / "pool").iterdir())
        assert seed_dirs
        for sd in seed_dirs:
            assert (sd / "result.json").exists()


def test_dry_run_journal_records_all_phases(tmp_path):
    campaign, d, _, _ = _run_two_iter_campaign(tmp_path)
    events = list(iter_events(d.journal_path()))
    types = set(e.get("event") for e in events)
    assert "campaign_started" in types
    assert "phase_transition" in types
    assert "sbatch" in types
    assert "phase_succeeded" in types
    assert "daemon_started" in types
    assert "daemon_stopped" in types
    # Final phase_transition target is DONE
    transitions = [e for e in events if e["event"] == "phase_transition"]
    last_transition_to_done = [
        t for t in transitions if t.get("to_phase") == "DONE"
    ]
    assert len(last_transition_to_done) == 1


def test_dry_run_sacct_polled_for_every_submitted_job(tmp_path):
    campaign, _, _, poller = _run_two_iter_campaign(tmp_path)
    # The poller is called >= once per submitted job (could be > if the
    # daemon polls a still-running job, but our dry sacct always says
    # COMPLETED so it's exactly once per submission).
    expected = 4 + 5 * 2
    assert len(poller.invocations) == expected
    assert all(j.startswith("DRYRUN-") for j in poller.invocations)


def test_dry_run_cli_drives_campaign_to_done(tmp_path):
    """Integration: invoke the CLI exactly as a user would and confirm the
    daemon completes a 2-iteration campaign."""
    campaign, cfg = _make_campaign(tmp_path)
    cfg.to_yaml(campaign / "campaign.yaml")
    assert cli_main(["init", "--campaign-dir", str(campaign)]) == 0
    rc = cli_main([
        "start",
        "--campaign-dir", str(campaign),
        "--dry-run",
        "--poll-interval", "1",
        "--max-ticks", "500",
    ])
    assert rc == 0
    state = read_state(campaign / DEFAULT_DATA_SUBDIR / "state.json")
    assert state.phase is CampaignPhase.DONE


def test_dry_run_state_persists_iteration_counter(tmp_path):
    campaign, d, _, _ = _run_two_iter_campaign(tmp_path)
    state = read_state(d.state_path())
    # max_iterations=2 -> the last iteration to run is iteration=1.
    assert state.iteration == 1
    # training_set_version was bumped by the APPEND inline at each iter.
    assert state.training_set_version == 2
    assert state.models_version == 2
