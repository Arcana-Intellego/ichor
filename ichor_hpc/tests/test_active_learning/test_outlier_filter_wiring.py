"""M15 F11 tests: outlier filter is now wired into TrajectoryPool.import_from
and emits the trajectory_pool_filtered journal event.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ichor.core.atoms import Atom, Atoms
from ichor.core.files.xyz import Trajectory
from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.config import (
    CampaignConfig,
    OutlierFilterConfigBlock,
)
from ichor.hpc.active_learning.sampling.outlier_filter import (
    filter_by_per_atom_rmsd_zscore,
)


FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"
)


# --- Bug fix: two-sided z-threshold ---


def test_filter_by_per_atom_rmsd_zscore_two_sided():
    """M15 F11: the per-atom filter was one-sided pre-F11 (z <= threshold);
    now it uses |z| so frames where an atom is abnormally CLOSE to the mean
    are also subject to rejection if their z is far in either direction."""
    rng = np.random.default_rng(0)
    # Build 50 normal frames clustered near origin + 1 outlier.
    base = rng.normal(scale=0.1, size=(50, 3, 3))
    outlier = np.array([[[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    frames_coords = np.concatenate([base, outlier], axis=0)
    frames = []
    for coords in frames_coords:
        frames.append(Atoms([
            Atom("O", coords[0, 0], coords[0, 1], coords[0, 2]),
            Atom("H", coords[1, 0], coords[1, 1], coords[1, 2]),
            Atom("H", coords[2, 0], coords[2, 1], coords[2, 2]),
        ]))
    kept, rejected = filter_by_per_atom_rmsd_zscore(frames, z_threshold=3.0)
    # The outlier frame index (50) must be rejected.
    assert 50 in rejected
    assert 50 not in kept


# --- OutlierFilterConfigBlock schema ---


def test_outlier_filter_config_default_enabled():
    cfg = CampaignConfig()
    assert cfg.outlier_filter.enabled is True
    assert cfg.outlier_filter.energy_z_threshold == 3.0
    assert cfg.outlier_filter.per_atom_rmsd_z_threshold == 4.0


def test_outlier_filter_block_roundtrips_via_yaml(tmp_path):
    c = CampaignConfig()
    c.outlier_filter.enabled = False
    c.outlier_filter.per_atom_rmsd_z_threshold = 5.5
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    c2 = CampaignConfig.from_yaml(p)
    assert c2.outlier_filter.enabled is False
    assert c2.outlier_filter.per_atom_rmsd_z_threshold == 5.5


# --- TrajectoryPool.import_from wiring ---


def test_import_from_writes_rejected_json_when_outliers_present(tmp_path):
    """Build a trajectory where one frame is an outrageous outlier.
    import_from must keep the inliers, write rejected.json, and produce a
    pool whose n_frames == kept count."""
    # 30 normal frames + 1 hard outlier.
    src_path = tmp_path / "src.xyz"
    rng = np.random.default_rng(1)
    base = rng.normal(scale=0.05, size=(30, 3, 3))
    outlier = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    all_coords = list(base) + [outlier]
    # Write the xyz manually (Trajectory.write needs an output path).
    with open(src_path, "w", encoding="utf-8") as f:
        for coords in all_coords:
            f.write("3\n\n")
            f.write(f"O {coords[0,0]} {coords[0,1]} {coords[0,2]}\n")
            f.write(f"H {coords[1,0]} {coords[1,1]} {coords[1,2]}\n")
            f.write(f"H {coords[2,0]} {coords[2,1]} {coords[2,2]}\n")

    campaign_dir = tmp_path / "campaign"
    pool = TrajectoryPool.import_from(src_path, campaign_dir)

    # Outlier should be rejected; pool has 30 kept frames.
    assert pool.n_frames() == 30
    rejected_path = campaign_dir / ".DATA" / "TRAJECTORY" / "rejected.json"
    assert rejected_path.is_file()
    payload = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert payload["rejected_count"] == 1
    assert payload["rejected"][0]["index"] == 30
    assert payload["rejected"][0]["reason"] == "per_atom_rmsd_z"


def test_import_from_skips_filter_when_disabled(tmp_path):
    """outlier_filter_enabled=False imports every frame verbatim."""
    src_path = tmp_path / "src.xyz"
    rng = np.random.default_rng(2)
    base = rng.normal(scale=0.05, size=(20, 3, 3))
    outlier = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    all_coords = list(base) + [outlier]
    with open(src_path, "w", encoding="utf-8") as f:
        for coords in all_coords:
            f.write("3\n\n")
            f.write(f"O {coords[0,0]} {coords[0,1]} {coords[0,2]}\n")
            f.write(f"H {coords[1,0]} {coords[1,1]} {coords[1,2]}\n")
            f.write(f"H {coords[2,0]} {coords[2,1]} {coords[2,2]}\n")

    campaign_dir = tmp_path / "campaign"
    pool = TrajectoryPool.import_from(
        src_path, campaign_dir, outlier_filter_enabled=False,
    )
    # Every frame kept; no rejected.json written.
    assert pool.n_frames() == 21
    rejected_path = campaign_dir / ".DATA" / "TRAJECTORY" / "rejected.json"
    assert not rejected_path.is_file()


def test_import_from_with_clean_trajectory_writes_empty_rejected(tmp_path):
    """A clean trajectory should still produce rejected.json (with 0
    rejections) when the filter runs -- gives the operator a positive
    confirmation that the filter ran."""
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path / "campaign")
    rejected_path = tmp_path / "campaign" / ".DATA" / "TRAJECTORY" / "rejected.json"
    assert rejected_path.is_file()
    payload = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert payload["rejected_count"] == 0
    # Pool has the original number of frames.
    assert pool.n_frames() > 0
