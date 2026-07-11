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
import json
import stat
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
    sha256_file,
    verify_manifest,
)
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory
from ichor.hpc.active_learning.versioning.sampling_iterations import (
    SamplingIterationError,
    active_iteration_manifest_path,
    bootstrap_manifest_path,
    verify_active_iteration,
    verify_sampling_chain,
)


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


def test_dry_run_commits_three_reference_data_versions(tmp_path):
    """Bootstrap version 0 plus active iterations 1 and 2."""
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    v = VersionedDirectory(campaign / "QM_REFERENCE_DATA")
    assert sorted(v.list_committed_versions()) == [0, 1, 2]
    assert v.current_version() == 2


def test_dry_run_commits_three_models_versions(tmp_path):
    """Bootstrap models 0 plus active iterations 1 and 2."""
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    v = VersionedDirectory(campaign / "TRAINED_MODELS")
    assert sorted(v.list_committed_versions()) == [0, 1, 2]


def test_dry_run_every_committed_iteration_has_manifest(tmp_path):
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    for root in (campaign / "QM_REFERENCE_DATA", campaign / "TRAINED_MODELS"):
        v = VersionedDirectory(root)
        for ver in v.list_committed_versions():
            iter_dir = v.iteration_path(ver)
            assert (iter_dir / MANIFEST_FILENAME).exists()
            verify_manifest(iter_dir)  # must not raise


def test_dry_run_active_learning_dirs_have_seeds_pool_and_phase_b(tmp_path):
    campaign, _, _, _ = _run_two_iter_campaign(tmp_path)
    bootstrap_manifest = bootstrap_manifest_path(campaign)
    assert bootstrap_manifest.is_file()
    for i in (1, 2):
        d = campaign / "ACTIVE_LEARNING" / ("iteration-" + str(i).zfill(6))
        assert d.is_dir()
        assert (d / "seed_selection" / "seeds.xyz").is_file()
        assert (d / "seed_selection" / "SELECTION.json").is_file()
        assert (d / "ariadne" / "TASK_MAP.json").is_file()
        assert (d / "ariadne" / "RESULTS.json").is_file()
        assert (d / "ariadne" / "AUDIT.json").is_file()
        assert (d / "phase_b" / "selected.xyz").is_file()
        assert (d / "phase_b" / "SELECTION.json").is_file()
        assert (d / "allocation" / "SPLIT_RECEIPT.json").is_file()
        assert (d / "ITERATION_MANIFEST.json").is_file()
        seed_dirs = sorted((d / "ariadne" / "seeds").iterdir())
        assert seed_dirs
        assert [seed_dir.name for seed_dir in seed_dirs] == [
            "seed-" + str(seed_id).zfill(6)
            for seed_id in range(1, len(seed_dirs) + 1)
        ]
        for seed_dir in seed_dirs:
            assert (seed_dir / "result.json").is_file()
            assert (seed_dir / "provenance.json").is_file()
            assert (seed_dir / "ARIADNE_OUTPUT_MANIFEST.json").is_file()
            assert (seed_dir / "trajectory" / "trajectory.xyz").is_file()
            assert (seed_dir / "trajectory" / "metrics.jsonl").is_file()
            assert (seed_dir / "trajectory" / "MANIFEST.json").is_file()
        manifest_path = active_iteration_manifest_path(campaign, i)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_parent = (
            bootstrap_manifest
            if i == 1
            else active_iteration_manifest_path(campaign, i - 1)
        )
        assert manifest["parent"]["sha256"] == sha256_file(expected_parent)
        assert manifest["input_head"]["version"] == i - 1
        assert manifest["output_head"]["version"] == i
        assert any(
            record["path"] == "phase_b/selected.xyz"
            and record["role"] == "derived_cache"
            for record in manifest["files"]
        )
        assert not (stat.S_IMODE(d.stat().st_mode) & stat.S_IWUSR)
    verify_sampling_chain(campaign, 2)

    first_iteration = campaign / "ACTIVE_LEARNING" / "iteration-000001"
    first_iteration.chmod(first_iteration.stat().st_mode | stat.S_IWUSR)
    rogue = first_iteration / "rogue.txt"
    rogue.write_text("not in the sealed inventory\n", encoding="utf-8")
    with pytest.raises(SamplingIterationError, match="exact inventory mismatch"):
        verify_active_iteration(campaign, 1)
    rogue.unlink()
    interrupted = first_iteration / ".tmp-interrupted-write"
    interrupted.write_text("partial\n", encoding="utf-8")
    with pytest.raises(SamplingIterationError, match="incomplete artefact"):
        verify_active_iteration(campaign, 1)


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
    # Active iterations are one-based; bootstrap alone is iteration 0.
    assert state.iteration == 2
    # Reference and model versions match the completed active iteration.
    assert state.reference_data_version == 2
    assert state.models_version == 2
