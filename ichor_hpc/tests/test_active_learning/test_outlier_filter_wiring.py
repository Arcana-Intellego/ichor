"""Regression tests for hard-deleting daemon trajectory outlier filtering."""

from pathlib import Path

import numpy as np
import pytest

from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.config import CampaignConfig, ConfigValidationError


def _write_xyz(path: Path, frames) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for coords in frames:
            handle.write("3\n\n")
            handle.write(f"O {coords[0, 0]} {coords[0, 1]} {coords[0, 2]}\n")
            handle.write(f"H {coords[1, 0]} {coords[1, 1]} {coords[1, 2]}\n")
            handle.write(f"H {coords[2, 0]} {coords[2, 1]} {coords[2, 2]}\n")


def test_outlier_filter_block_is_no_longer_valid_campaign_config():
    with pytest.raises(ConfigValidationError):
        CampaignConfig.from_dict(
            {
                "schema_version": 6,
                "outlier_filter": {
                    "enabled": True,
                    "energy_z_threshold": 3.0,
                    "per_atom_rmsd_z_threshold": 4.0,
                },
            }
        )


def test_import_from_keeps_geometric_outlier_and_writes_no_rejected_json(tmp_path):
    src_path = tmp_path / "src.xyz"
    rng = np.random.default_rng(1)
    base = rng.normal(scale=0.05, size=(30, 3, 3))
    outlier = np.array(
        [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    frames = list(base) + [outlier]
    _write_xyz(src_path, frames)

    campaign_dir = tmp_path / "campaign"
    pool = TrajectoryPool.import_from(src_path, campaign_dir)

    assert pool.n_frames() == len(frames)
    rejected_path = campaign_dir / ".DATA" / "TRAJECTORY" / "rejected.json"
    assert not rejected_path.exists()
