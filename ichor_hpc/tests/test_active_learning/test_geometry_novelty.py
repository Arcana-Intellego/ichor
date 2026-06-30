import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.geometry_novelty import (
    compute_geometry_novelty_scale,
    effective_phase_b_min_separation,
    geometry_novelty_scale_path,
    novelty_score,
    read_geometry_novelty_scale,
    scaled_distances,
    write_geometry_novelty_scale,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "water_tetramer.xyz"
)


def _iter_dir(campaign, iteration=0):
    path = campaign / "7_ACTIVE_LEARNING" / ("iteration-" + str(iteration).zfill(4))
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_geometry_novelty_falls_back_when_pool_history_missing(tmp_path):
    cfg = CampaignConfig()
    cfg.geometry_novelty.fallback_scale_angstrom = 0.04
    iter_dir = _iter_dir(tmp_path)

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=0)
    path = write_geometry_novelty_scale(iter_dir, payload)
    loaded = read_geometry_novelty_scale(iter_dir)

    assert path == geometry_novelty_scale_path(iter_dir)
    assert loaded["schema_version"] == 1
    assert loaded["scale_angstrom"] == pytest.approx(0.04)
    assert loaded["fallback_used"] is True
    assert "no_motion_values" in loaded["reasons"]


def test_geometry_novelty_uses_seed_neighbour_motion(tmp_path):
    cfg = CampaignConfig()
    TrajectoryPool.import_from(FIXTURE, tmp_path, outlier_filter_enabled=False)
    iter_dir = _iter_dir(tmp_path)
    (iter_dir / "seeds_picked.json").write_text(
        json.dumps(
            {
                "iteration": 0,
                "n_picked": 1,
                "frame_ids": [0],
                "indices": [0],
                "seed_records": [
                    {
                        "seed_index": 0,
                        "frame_id": 0,
                        "selection_index": 0,
                        "selection_origin": "bulk",
                        "variance_at_selection": 0.0,
                        "subspace_neighbour_frame_ids": [1, 2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = compute_geometry_novelty_scale(tmp_path, cfg, iteration=0)

    assert payload["fallback_used"] is False
    assert payload["n_values"] == 2
    assert payload["scale_angstrom"] > cfg.geometry_novelty.scale_floor_angstrom
    assert payload["local_motion_summary_angstrom"]["n"] == 2


def test_scaled_threshold_and_scores_are_dimensionless():
    cfg = CampaignConfig()
    cfg.phase_b.min_separation_scaled = 0.5
    payload = {"scale_angstrom": 0.04}

    threshold, mode = effective_phase_b_min_separation(cfg, payload)

    assert mode == "scaled"
    assert threshold == pytest.approx(0.02)
    assert scaled_distances([0.01, 0.04], 0.04) == pytest.approx([0.25, 1.0])
    assert novelty_score(0.02, 0.04, "linear_cap") == pytest.approx(0.5)
    assert novelty_score(10.0, 0.04, "linear_cap") == pytest.approx(1.0)
    assert novelty_score(0.04, 0.04, "exponential") == pytest.approx(1.0 - 2.718281828459045 ** -1)
