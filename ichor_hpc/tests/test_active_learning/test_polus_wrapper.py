import numpy as np
import pytest

from ichor.hpc.active_learning.sampling.polus_wrapper import (
    DEFAULT_DESCRIPTORS,
    FPSResult,
    _phase_b_refill_after_anti_overlap,
    _phase_b_target_size,
    fps_select,
)
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.core.atoms import Atom, Atoms


def _h2(length):
    return Atoms([Atom("H", 0.0, 0.0, 0.0), Atom("H", float(length), 0.0, 0.0)])


def test_phase_b_target_size_is_exact_active_allocation_total():
    cfg = CampaignConfig()
    cfg.point_allocation.batch_training_size = 4
    cfg.point_allocation.batch_internal_validation_size = 1
    plenty = 1000  # never let the candidate pool be the binding constraint
    assert _phase_b_target_size(cfg, plenty, iteration=0) == 5
    assert _phase_b_target_size(cfg, plenty, iteration=100) == 5
    with pytest.raises(ValueError, match="point_allocation_underfilled"):
        _phase_b_target_size(cfg, 3, iteration=100)


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


def test_phase_b_refill_uses_safe_reserve_before_underfill():
    frames = [_h2(1.0), _h2(2.0), _h2(3.0)]
    records = [{"seed_index": i} for i in range(3)]

    indices, considered, considered_records, report, refill = (
        _phase_b_refill_after_anti_overlap(
            ordered_indices=[0, 1, 2],
            candidate_frames=frames,
            candidate_records=records,
            training=[_h2(1.0)],
            min_separation=0.4,
            target_size=2,
        )
    )

    assert indices == [0, 1, 2]
    assert [rec["seed_index"] for rec in considered_records] == [0, 1, 2]
    assert len(considered) == 3
    assert report.kept_indices == (1, 2)
    assert report.dropped_indices == (0,)
    assert refill["refill_applied"] is True
    assert refill["reserve_exhausted"] is False
    assert refill["rejected_by_anti_overlap"] == 1


def test_phase_b_refill_reports_exhausted_reserve():
    frames = [_h2(1.0), _h2(1.05)]
    records = [{"seed_index": i} for i in range(2)]

    _indices, _considered, _records, report, refill = (
        _phase_b_refill_after_anti_overlap(
            ordered_indices=[0, 1],
            candidate_frames=frames,
            candidate_records=records,
            training=[_h2(1.0)],
            min_separation=0.5,
            target_size=1,
        )
    )

    assert report.n_kept == 0
    assert refill["reserve_exhausted"] is True
    assert refill["rejected_by_anti_overlap"] == 2


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
