import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import pool_feasibility as pf


def _config(*, bootstrap=12, seeds=8, final=4, max_iterations=1, skip=True):
    cfg = CampaignConfig()
    cfg.bootstrap.initial_labelled_size = int(bootstrap)
    cfg.seed_selection.n_seeds_per_iteration = int(seeds)
    cfg.active_batch.final_batch_size = int(final)
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
