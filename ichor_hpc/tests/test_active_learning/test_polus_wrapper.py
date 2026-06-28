import numpy as np
import pytest

from ichor.hpc.active_learning.sampling.polus_wrapper import (
    DEFAULT_DESCRIPTORS,
    FPSResult,
    _phase_b_target_size,
    fps_select,
)
from ichor.hpc.active_learning.config import CampaignConfig


def test_phase_b_target_size_grows_with_iteration():
    # A40/A41: the batch USED to be pinned to floor forever (min(floor, n, cap), and since
    # cap >= floor that collapsed to min(floor, n)). it should now grow with iteration, capped.
    cfg = CampaignConfig()
    cfg.batch_sizing.floor = 5
    cfg.batch_sizing.cap = 30
    plenty = 1000  # never let the candidate pool be the binding constraint
    cfg.batch_sizing.policy = "linear"
    assert _phase_b_target_size(cfg, plenty, iteration=0) == 5     # floor at iter 0
    assert _phase_b_target_size(cfg, plenty, iteration=5) == 10    # floor + iteration
    assert _phase_b_target_size(cfg, plenty, iteration=100) == 30  # capped
    cfg.batch_sizing.policy = "fixed"
    assert _phase_b_target_size(cfg, plenty, iteration=50) == 5    # ignores iteration
    cfg.batch_sizing.policy = "sqrt"
    assert _phase_b_target_size(cfg, plenty, iteration=3) == 10    # round(5*sqrt(4)) = 10
    # never keep more than the candidate pool actually holds
    assert _phase_b_target_size(cfg, 3, iteration=100) == 3


def _line_distance_matrix(n, spacing=1.0):
    coords = np.arange(n, dtype=float) * spacing
    D = np.abs(coords[:, None] - coords[None, :])
    return D


def test_fps_select_picks_endpoints_first_on_line():
    D = _line_distance_matrix(5)
    out = fps_select(D, 3, seed_index=0)
    assert out.indices[0] == 0
    assert out.indices[1] == 4
    assert out.indices[2] == 2


def test_fps_select_zero_returns_empty():
    D = _line_distance_matrix(5)
    out = fps_select(D, 0)
    assert out.n == 0


def test_fps_select_too_many_raises():
    D = _line_distance_matrix(5)
    with pytest.raises(ValueError):
        fps_select(D, 6)


def test_fps_select_seed_index_out_of_range_raises():
    D = _line_distance_matrix(5)
    with pytest.raises(ValueError):
        fps_select(D, 3, seed_index=10)


def test_fps_select_seed_default_is_centroid():
    D = _line_distance_matrix(5)
    out = fps_select(D, 2)
    assert out.indices[0] == 2


def test_fps_select_default_seed_near_ties_prefer_lowest_index():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0 - 3.0e-13],
            [2.0, 2.0 - 3.0e-13, 0.0],
        ],
        dtype=float,
    )

    out = fps_select(D, 2)

    assert out.indices[0] == 0


def test_fps_select_diversities_monotone_non_increasing():
    rng = np.random.default_rng(0)
    n = 30
    coords = rng.standard_normal((n, 2))
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    out = fps_select(D, 10, seed_index=0)
    diffs = np.diff(out.diversities[1:])
    assert np.all(diffs <= 1.0e-9), f"diversities not monotone: {out.diversities}"


def test_fps_select_indices_unique():
    rng = np.random.default_rng(1)
    n = 20
    coords = rng.standard_normal((n, 3))
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    out = fps_select(D, 7, seed_index=0)
    assert len(set(out.indices)) == out.n


def test_fps_select_ties_prefer_lowest_candidate_index():
    D = np.ones((4, 4), dtype=float)
    np.fill_diagonal(D, 0.0)

    out = fps_select(D, 3, seed_index=0)

    assert out.indices == [0, 1, 2]


def test_fps_select_non_square_raises():
    with pytest.raises(ValueError):
        fps_select(np.zeros((5, 4)), 2)


def test_fps_select_non_symmetric_raises():
    D = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        fps_select(D, 2)


def test_default_descriptors_registry_has_phase_a_default():
    assert "rmsd_massweight" in DEFAULT_DESCRIPTORS
    cls = DEFAULT_DESCRIPTORS["rmsd_massweight"]
    inst = cls()
    assert inst.name == "rmsd_massweight"
