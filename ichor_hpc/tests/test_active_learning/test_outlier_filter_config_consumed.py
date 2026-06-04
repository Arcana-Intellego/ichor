"""Confirm cmd_import_pool now honours the outlier_filter block in
campaign.yaml instead of the dataclass defaults.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ichor.hpc.active_learning.cli import cmd_import_pool
from ichor.hpc.active_learning.config import CampaignConfig


def _write_minimal_xyz(path: Path, n_frames: int = 5):
    """Write a tiny water-monomer xyz, n_frames of it."""
    frame = "3\nwater monomer\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"
    path.write_text(frame * n_frames, encoding="utf-8")


def _make_args(campaign, source, no_filter=False, force=False):
    return Namespace(
        campaign_dir=str(campaign), source=str(source),
        no_outlier_filter=no_filter, force=force,
    )


def test_outlier_filter_thresholds_picked_up_from_yaml(tmp_path):
    """campaign.yaml with non-default thresholds reaches import_from."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.outlier_filter.energy_z_threshold = 2.5
    cfg.outlier_filter.per_atom_rmsd_z_threshold = 3.5
    cfg.to_yaml(campaign / "campaign.yaml")
    src = tmp_path / "traj.xyz"
    _write_minimal_xyz(src, n_frames=10)
    captured = {}

    real_import = __import__(
        "ichor.hpc.active_learning.acquisition.trajectory_pool",
        fromlist=["TrajectoryPool"],
    ).TrajectoryPool.import_from

    def spy(source, campaign_dir, **kw):
        captured.update(kw)
        return real_import(source, campaign_dir, **kw)

    with patch(
        "ichor.hpc.active_learning.acquisition.trajectory_pool.TrajectoryPool.import_from",
        side_effect=spy,
    ):
        rc = cmd_import_pool(_make_args(campaign, src))
    assert rc == 0
    assert captured["outlier_filter_enabled"] is True
    assert captured["energy_z_threshold"] == 2.5
    assert captured["per_atom_rmsd_z_threshold"] == 3.5


def test_cli_flag_wins_over_config(tmp_path):
    """--no-outlier-filter forces enabled=False regardless of config."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    cfg = CampaignConfig()
    cfg.outlier_filter.enabled = True
    cfg.to_yaml(campaign / "campaign.yaml")
    src = tmp_path / "traj.xyz"
    _write_minimal_xyz(src, n_frames=5)
    captured = {}

    real_import = __import__(
        "ichor.hpc.active_learning.acquisition.trajectory_pool",
        fromlist=["TrajectoryPool"],
    ).TrajectoryPool.import_from

    def spy(source, campaign_dir, **kw):
        captured.update(kw)
        return real_import(source, campaign_dir, **kw)

    with patch(
        "ichor.hpc.active_learning.acquisition.trajectory_pool.TrajectoryPool.import_from",
        side_effect=spy,
    ):
        rc = cmd_import_pool(
            _make_args(campaign, src, no_filter=True),
        )
    assert rc == 0
    assert captured["outlier_filter_enabled"] is False


def test_no_config_falls_back_to_defaults(tmp_path):
    """no campaign.yaml means we use the dataclass defaults."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    src = tmp_path / "traj.xyz"
    _write_minimal_xyz(src, n_frames=5)
    captured = {}

    real_import = __import__(
        "ichor.hpc.active_learning.acquisition.trajectory_pool",
        fromlist=["TrajectoryPool"],
    ).TrajectoryPool.import_from

    def spy(source, campaign_dir, **kw):
        captured.update(kw)
        return real_import(source, campaign_dir, **kw)

    with patch(
        "ichor.hpc.active_learning.acquisition.trajectory_pool.TrajectoryPool.import_from",
        side_effect=spy,
    ):
        rc = cmd_import_pool(_make_args(campaign, src))
    assert rc == 0
    # defaults: enabled True, 3.0, 4.0
    assert captured["outlier_filter_enabled"] is True
    assert captured["energy_z_threshold"] == 3.0
    assert captured["per_atom_rmsd_z_threshold"] == 4.0
