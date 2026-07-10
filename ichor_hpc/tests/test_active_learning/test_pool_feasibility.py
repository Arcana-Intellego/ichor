from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import pool_feasibility as pf


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"


def _config(*, bootstrap=12, seeds=8, final=4, max_iterations=1, skip=True):
    cfg = CampaignConfig()
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_training_size = int(bootstrap) - 4
    cfg.seed_selection.n_seeds_per_iteration = int(seeds)
    cfg.point_allocation.batch_internal_validation_size = 1 if int(final) > 1 else 0
    cfg.point_allocation.batch_training_size = (
        int(final) - cfg.point_allocation.batch_internal_validation_size
    )
    cfg.max_iterations = int(max_iterations)
    cfg.anti_overlap.skip_training_seeds = bool(skip)
    cfg._validate()
    return cfg


def test_pool_feasibility_passes_for_20_frame_first_live_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_pool_frame_count", lambda campaign: 20)

    result = pf.require_pool_feasibility(tmp_path, _config())

    assert result.ok is True
    assert result.required_pool_frames == 20
    assert result.reserve_after_bootstrap == 8


def test_pool_feasibility_fails_for_full_campaign_shortfall(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_pool_frame_count", lambda campaign: 20)

    with pytest.raises(pf.PoolFeasibilityError, match="12 \\+ 2 \\* 8 = 28"):
        pf.require_pool_feasibility(tmp_path, _config(max_iterations=2))


def test_pool_feasibility_fails_when_bootstrap_leaves_no_seed_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_pool_frame_count", lambda campaign: 20)

    with pytest.raises(pf.PoolFeasibilityError, match="16 \\+ 1 \\* 8 = 24"):
        pf.require_pool_feasibility(tmp_path, _config(bootstrap=16))


def test_pool_feasibility_reuse_mode_requires_only_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_pool_frame_count", lambda campaign: 20)

    result = pf.require_pool_feasibility(
        tmp_path,
        _config(bootstrap=20, seeds=50, final=4, max_iterations=50, skip=False),
    )

    assert result.ok is True
    assert result.required_pool_frames == 20


def test_pool_feasibility_counts_anchors_outside_pool_requirement(tmp_path):
    from ichor.core.files.xyz import Trajectory
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.sampling.polus_wrapper import _write_xyz_file

    campaign = tmp_path / "c"
    campaign.mkdir()
    TrajectoryPool.import_from(FIXTURE, campaign, overwrite=True)
    traj = Trajectory(FIXTURE)
    traj.read()
    base = [atoms.copy() for atoms in traj][0]
    anchors = [base.copy(), base.copy()]
    anchors[1][1].coordinates = [
        anchors[1][1].x + 0.1,
        anchors[1][1].y,
        anchors[1][1].z,
    ]
    _write_xyz_file(anchors, campaign / "anchor.xyz")
    cfg = _config(bootstrap=12, seeds=8, max_iterations=1, skip=True)
    cfg.point_allocation.bootstrap_external_validation_size = 2
    cfg.point_allocation.anchor = True

    result = pf.require_pool_feasibility(campaign, cfg)

    assert result.bootstrap_total_size == 12
    assert result.bootstrap_anchor_count == 2
    assert result.bootstrap_pool_frame_count == 10
    assert result.required_pool_frames == 18
    assert result.reserve_after_bootstrap == result.pool_n_frames - 10
