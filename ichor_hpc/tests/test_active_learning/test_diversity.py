import numpy as np
import pytest

from ichor.hpc.active_learning.sampling.diversity import (
    DEFAULT_DESCRIPTORS,
    FPSResult,
    _phase_b_build_reserve,
    _phase_b_refill_after_anti_overlap,
    _phase_b_target_size,
    _relax_scaled_novelty,
    fps_select,
)
from ichor.hpc.active_learning.sampling.anti_overlap import (
    DedupReport,
    PhaseBNoveltyDistanceOracle,
    min_distance_to_training,
)
from ichor.hpc.active_learning.sampling.descriptors import CondensedDistanceStore
from ichor.hpc.active_learning.config import CampaignConfig
from ichor.core.atoms import Atom, Atoms


def _h2(length):
    return Atoms([Atom("H", 0.0, 0.0, 0.0), Atom("H", float(length), 0.0, 0.0)])


def test_phase_b_novelty_oracle_matches_exact_serial_order():
    references = [_h2(0.8), _h2(1.0), _h2(1.2)]
    candidates = [_h2(1.05), _h2(1.4), _h2(1.8), _h2(2.1)]

    serial = PhaseBNoveltyDistanceOracle(candidates, references, workers=1)
    parallel = PhaseBNoveltyDistanceOracle(candidates, references, workers=3)

    accepted = []
    for candidate_index, candidate in enumerate(candidates):
        expected = min_distance_to_training(
            candidate,
            references + [candidates[index] for index in accepted],
        )
        assert serial.nearest_distance(candidate_index, accepted) == expected
        assert parallel.nearest_distance(candidate_index, accepted) == expected
        accepted.append(candidate_index)
    assert serial.candidate_reference_pairs == 12
    assert serial.directed_candidate_pairs == 12


def test_phase_b_oracle_preserves_refill_relaxation_and_reserve_decisions():
    references = [_h2(1.0)]
    candidates = [_h2(1.05), _h2(1.4), _h2(1.8), _h2(2.2)]
    records = [{"seed_index": index} for index in range(len(candidates))]
    order = [0, 1, 2, 3]
    oracle = PhaseBNoveltyDistanceOracle(candidates, references, workers=2)

    legacy = _phase_b_refill_after_anti_overlap(
        ordered_indices=order,
        candidate_frames=candidates,
        candidate_records=records,
        training=references,
        min_separation=0.1,
        target_size=2,
    )
    optimised = _phase_b_refill_after_anti_overlap(
        ordered_indices=order,
        candidate_frames=candidates,
        candidate_records=records,
        training=(),
        min_separation=0.1,
        target_size=2,
        distance_oracle=oracle,
    )
    assert optimised[0] == legacy[0]
    assert optimised[2:] == legacy[2:]
    for observed, expected in zip(optimised[1], legacy[1]):
        np.testing.assert_array_equal(observed.coordinates, expected.coordinates)

    relaxed_legacy = _relax_scaled_novelty(
        considered_frames=legacy[1],
        report=legacy[3],
        training=references,
        target_size=3,
    )
    relaxed_optimised = _relax_scaled_novelty(
        considered_frames=optimised[1],
        report=optimised[3],
        training=(),
        target_size=3,
        considered_candidate_indices=optimised[0],
        distance_oracle=oracle,
    )
    assert relaxed_optimised == relaxed_legacy

    selected_candidate_indices = [
        optimised[0][index] for index in relaxed_optimised[0].kept_indices
    ]
    selected_frames = [candidates[index] for index in selected_candidate_indices]
    legacy_reserve = _phase_b_build_reserve(
        ordered_indices=order,
        already_considered_indices=optimised[0],
        candidate_frames=candidates,
        candidate_records=records,
        training=references,
        selected_frames=selected_frames,
        min_separation=0.1,
    )
    optimised_reserve = _phase_b_build_reserve(
        ordered_indices=order,
        already_considered_indices=optimised[0],
        candidate_frames=candidates,
        candidate_records=records,
        training=(),
        selected_frames=selected_frames,
        min_separation=0.1,
        selected_candidate_indices=selected_candidate_indices,
        distance_oracle=oracle,
    )
    assert optimised_reserve["candidate_indices"] == legacy_reserve[
        "candidate_indices"
    ]
    assert optimised_reserve["records"] == legacy_reserve["records"]
    assert optimised_reserve["distances_to_nearest_angstrom"] == legacy_reserve[
        "distances_to_nearest_angstrom"
    ]
    assert optimised_reserve["rejected"] == legacy_reserve["rejected"]


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


def test_fps_progress_is_best_effort_and_does_not_change_selection():
    distances = _line_distance_matrix(12)
    updates = []

    observed = fps_select(
        distances,
        10,
        seed_index=0,
        progress_callback=lambda **payload: updates.append(dict(payload)),
    )
    baseline = fps_select(distances, 10, seed_index=0)
    failed_reporting = fps_select(
        distances,
        10,
        seed_index=0,
        progress_callback=lambda **_payload: (_ for _ in ()).throw(
            OSError("injected progress failure")
        ),
    )

    assert observed.indices == baseline.indices == failed_reporting.indices
    assert updates[0]["completed"] == 1
    assert updates[-1]["completed"] == 10
    assert {update["stage"] for update in updates} == {
        "farthest_point_sampling"
    }


def test_fps_reports_medoid_as_a_distinct_stage():
    updates = []

    fps_select(
        _line_distance_matrix(5),
        3,
        progress_callback=lambda **payload: updates.append(dict(payload)),
    )

    assert [update["stage"] for update in updates[:3]] == [
        "medoid_selection",
        "medoid_selection",
        "farthest_point_sampling",
    ]
    assert [update["completed"] for update in updates[:3]] == [0, 1, 1]


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


@pytest.mark.parametrize("bad", [1.5, "1", True, None])
def test_fps_select_rejects_non_integer_selection_counts(bad):
    with pytest.raises((TypeError, ValueError)):
        fps_select(_line_distance_matrix(3), bad)


@pytest.mark.parametrize("bad", [0.5, "0", True])
def test_fps_select_rejects_non_integer_seed_identity(bad):
    with pytest.raises((TypeError, ValueError)):
        fps_select(_line_distance_matrix(3), 2, seed_index=bad)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
        np.array([[1.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
        np.array([[0.0, 1.0], [1.0 + 2.0e-12, 0.0]]),
    ],
)
def test_fps_select_rejects_invalid_metric_matrices(matrix):
    with pytest.raises(ValueError):
        fps_select(matrix, 1)


def test_fps_select_validates_condensed_cardinality_and_values():
    with pytest.raises(ValueError, match="expected"):
        fps_select(CondensedDistanceStore(n=3, values=np.asarray([1.0])), 1)
    with pytest.raises(ValueError, match="finite non-negative"):
        fps_select(
            CondensedDistanceStore(n=2, values=np.asarray([np.nan])),
            1,
        )


def test_scaled_novelty_relaxation_fills_with_non_duplicates():
    considered = [_h2(1.1), _h2(1.2), _h2(1.3)]
    report = DedupReport(
        kept_indices=(),
        dropped_indices=(0, 1, 2),
        distances_to_nearest=(0.1, 0.2, 0.3),
        min_separation=0.5,
    )

    updated, diagnostic = _relax_scaled_novelty(
        considered_frames=considered,
        report=report,
        training=[_h2(1.0)],
        target_size=2,
    )

    assert updated.n_kept == 2
    assert diagnostic["target_satisfied"] is True
    assert diagnostic["n_admitted"] == 2


def test_default_descriptors_registry_has_phase_a_default():
    assert "rmsd_massweight" in DEFAULT_DESCRIPTORS
    cls = DEFAULT_DESCRIPTORS["rmsd_massweight"]
    inst = cls()
    assert inst.name == "rmsd_massweight"
