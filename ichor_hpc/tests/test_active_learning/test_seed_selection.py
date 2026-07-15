"""Tests for ichor.hpc.active_learning.acquisition.seed_selection."""
import numpy as np
import pytest

from ichor.hpc.active_learning.acquisition.seed_selection import (
    SeedSelection,
    select_seeds,
)


class _StubPosterior:
    """Returns the variance from a pre-set list keyed by id(atoms)."""

    def __init__(self, var_by_id):
        self._var_by_id = dict(var_by_id)

    def variance(self, atoms):
        return float(self._var_by_id[id(atoms)])


class _BatchedPosterior(_StubPosterior):
    def variances(self, atoms_list):
        return np.array([self.variance(atoms) for atoms in atoms_list], dtype=float)


class _ChunkedPosterior(_StubPosterior):
    def __init__(self, var_by_id):
        super().__init__(var_by_id)
        self.chunk_sizes = []

    def variances(self, atoms_list, *, chunk_size=None):
        self.chunk_sizes.append(chunk_size)
        return np.array([self.variance(atoms) for atoms in atoms_list], dtype=float)


class _CountingPosterior(_StubPosterior):
    def __init__(self, var_by_id):
        super().__init__(var_by_id)
        self.n_variance_calls = 0
        self.n_variances_calls = 0

    def variance(self, atoms):
        self.n_variance_calls += 1
        return super().variance(atoms)

    def variances(self, atoms_list, *, chunk_size=None):
        self.n_variances_calls += 1
        return np.array([self.variance(atoms) for atoms in atoms_list], dtype=float)


class _CovariancePosterior:
    def __init__(self, atoms, covariance):
        self._pos = {id(atom): i for i, atom in enumerate(atoms)}
        self._cov = np.asarray(covariance, dtype=float)
        self.covariance_matrix_called = False

    def variance(self, atoms):
        idx = self._pos[id(atoms)]
        return float(self._cov[idx, idx])

    def variances(self, atoms_list, *, chunk_size=None):
        return np.array([self.variance(atoms) for atoms in atoms_list], dtype=float)

    def covariance(self, atoms_a, atoms_b):
        ia = self._pos[id(atoms_a)]
        ib = self._pos[id(atoms_b)]
        return float(self._cov[ia, ib])

    def covariance_matrix(self, _atoms_list):
        self.covariance_matrix_called = True
        raise AssertionError("seed selection must not build a full pool covariance matrix")


def _atoms_with_indexed_variances(n):
    """Return (atoms_list, posterior, variances) where variance(atoms[i]) = i."""
    atoms = [object() for _ in range(n)]   # opaque marker; posterior keys on id
    variances = list(range(n))
    posterior = _StubPosterior({id(a): v for a, v in zip(atoms, variances)})
    return atoms, posterior, variances


def test_n_seeds_zero_returns_empty():
    atoms, posterior, _ = _atoms_with_indexed_variances(10)
    out = select_seeds(atoms, posterior, n_seeds=0)
    assert out.n == 0
    assert out.seeds == []
    assert out.indices == []


def test_n_seeds_exceeds_training_returns_all():
    atoms, posterior, _ = _atoms_with_indexed_variances(5)
    out = select_seeds(atoms, posterior, n_seeds=100)
    assert out.n == 5
    assert set(out.indices) == set(range(5))
    assert out.variances == [None] * 5
    assert out.diagnostics["posterior_variance_evaluations"] == 0


def test_half_half_split_sizes_default():
    """With default bulk_fraction=0.5 and n_seeds=10, expect 5 bulk + 5 variance."""
    atoms, posterior, _ = _atoms_with_indexed_variances(50)
    out = select_seeds(atoms, posterior, n_seeds=10, rng_seed=42)
    assert len(out.bulk_indices) == 5
    assert len(out.variance_indices) == 5
    assert set(out.bulk_indices).isdisjoint(out.variance_indices)


def test_variance_indices_are_actually_top_variance():
    """Once the bulk indices are removed, the variance pick should match the
    top-variance ranking of the remaining atoms."""
    atoms, posterior, _ = _atoms_with_indexed_variances(50)
    out = select_seeds(atoms, posterior, n_seeds=10, rng_seed=42)
    # variance i = index i; the highest-variance unused indices must be picked.
    remaining = [i for i in range(50) if i not in set(out.bulk_indices)]
    expected = sorted(remaining, reverse=True)[: len(out.variance_indices)]
    assert out.variance_indices == expected


def test_bulk_fraction_zero_pure_variance():
    atoms, posterior, _ = _atoms_with_indexed_variances(20)
    out = select_seeds(atoms, posterior, n_seeds=5, bulk_fraction=0.0, rng_seed=0)
    assert len(out.bulk_indices) == 0
    assert out.variance_indices == [19, 18, 17, 16, 15]


def test_variance_near_ties_are_quantised_then_broken_by_index():
    atoms = [object() for _ in range(4)]
    values = [
        1.0,
        2.0 + 3.0e-13,
        2.0 + 4.0e-13,
        0.5,
    ]
    posterior = _BatchedPosterior({id(a): v for a, v in zip(atoms, values)})

    out = select_seeds(atoms, posterior, n_seeds=2, bulk_fraction=0.0)

    assert out.variance_indices == [1, 2]
    assert out.diagnostics["ranking_tie_break_policy"] == "quantised_score_then_index"
    assert out.diagnostics["n_variance_score_ties_after_quantisation"] >= 2


def test_bulk_fraction_one_pure_random():
    atoms = [object() for _ in range(20)]
    posterior = _CountingPosterior({id(atom): idx for idx, atom in enumerate(atoms)})
    out = select_seeds(atoms, posterior, n_seeds=5, bulk_fraction=1.0, rng_seed=0)
    assert len(out.variance_indices) == 0
    assert len(out.bulk_indices) == 5
    assert out.variances == [None] * 5
    assert posterior.n_variance_calls == 0
    assert posterior.n_variances_calls == 0
    assert out.diagnostics["posterior_variance_evaluations"] == 0


def test_reproducibility_with_same_rng_seed():
    atoms, posterior, _ = _atoms_with_indexed_variances(30)
    a = select_seeds(atoms, posterior, n_seeds=6, rng_seed=7)
    b = select_seeds(atoms, posterior, n_seeds=6, rng_seed=7)
    assert a.indices == b.indices


def test_different_rng_seed_changes_bulk_selection():
    atoms, posterior, _ = _atoms_with_indexed_variances(30)
    a = select_seeds(atoms, posterior, n_seeds=6, rng_seed=1)
    b = select_seeds(atoms, posterior, n_seeds=6, rng_seed=2)
    # Variance picks should still rank by the same order (modulo bulk holes);
    # but at least the bulk indices should be different in expectation.
    assert a.bulk_indices != b.bulk_indices


def test_invalid_bulk_fraction_raises():
    atoms, posterior, _ = _atoms_with_indexed_variances(5)
    with pytest.raises(ValueError):
        select_seeds(atoms, posterior, n_seeds=2, bulk_fraction=-0.1)
    with pytest.raises(ValueError):
        select_seeds(atoms, posterior, n_seeds=2, bulk_fraction=1.5)


def test_empty_training_raises():
    posterior = _StubPosterior({})
    with pytest.raises(ValueError):
        select_seeds([], posterior, n_seeds=3)


def test_returned_variances_match_picked_indices():
    atoms, posterior, _ = _atoms_with_indexed_variances(20)
    out = select_seeds(atoms, posterior, n_seeds=4, rng_seed=11)
    expected = [float(i) for i in out.indices]
    assert out.variances == expected


def test_non_finite_scalar_variance_rejected_before_selection():
    atoms = [object() for _ in range(4)]
    posterior = _StubPosterior({id(a): float("nan") for a in atoms})
    with pytest.raises(ValueError, match="non-finite"):
        select_seeds(atoms, posterior, n_seeds=3, bulk_fraction=0.0)


def test_non_finite_batched_variance_rejected_before_ranking():
    atoms = [object() for _ in range(6)]
    values = [0.0, 1.0, float("inf"), 3.0, 4.0, 5.0]
    posterior = _BatchedPosterior({id(a): v for a, v in zip(atoms, values)})
    with pytest.raises(ValueError, match="non-finite"):
        select_seeds(atoms, posterior, n_seeds=3, bulk_fraction=0.0)


# --- M11: forbidden_frame_ids + frame_ids -------------------------------


def test_frame_ids_default_to_none_when_unspecified():
    atoms, posterior, _ = _atoms_with_indexed_variances(10)
    out = select_seeds(atoms, posterior, n_seeds=4, rng_seed=0)
    assert out.frame_ids == [None] * 4


def test_frame_ids_returned_in_pick_order():
    atoms, posterior, _ = _atoms_with_indexed_variances(10)
    fids = list(range(100, 110))  # frame_id i+100 for training row i
    out = select_seeds(
        atoms, posterior, n_seeds=4, rng_seed=0,
        training_frame_ids=fids,
    )
    # frame_ids must be parallel to indices
    expected = [fids[i] for i in out.indices]
    assert out.frame_ids == expected


def test_forbidden_frame_ids_filters_bulk_and_variance():
    atoms, posterior, _ = _atoms_with_indexed_variances(10)
    fids = list(range(100, 110))
    forbidden = frozenset({100, 101, 102, 103, 104})  # ban rows 0..4
    out = select_seeds(
        atoms, posterior, n_seeds=4, rng_seed=0,
        training_frame_ids=fids,
        forbidden_frame_ids=forbidden,
    )
    # No picked frame_id may be in the forbidden set.
    assert all(f not in forbidden for f in out.frame_ids)
    # Equivalently: no picked index may have a forbidden frame_id.
    for i in out.indices:
        assert fids[i] not in forbidden


def test_forbidden_set_eating_all_returns_empty():
    atoms, posterior, _ = _atoms_with_indexed_variances(5)
    fids = list(range(5))
    out = select_seeds(
        atoms, posterior, n_seeds=3, rng_seed=0,
        training_frame_ids=fids,
        forbidden_frame_ids=frozenset(range(5)),
    )
    assert out.n == 0
    assert out.frame_ids == []


def test_none_frame_id_rows_are_excluded_when_forbidden_filter_is_active():
    """Rows whose frame_id is None are unsafe when a forbidden-frame ledger is active:
    they might be old training rows that predate provenance metadata, so do not let them
    bypass the repeat-frame guard.
    """
    atoms, posterior, _ = _atoms_with_indexed_variances(6)
    # Half the rows have None frame_id; the other half are 100..102.
    fids = [None, None, None, 100, 101, 102]
    out = select_seeds(
        atoms, posterior, n_seeds=3, rng_seed=0, bulk_fraction=1.0,
        training_frame_ids=fids,
        forbidden_frame_ids=frozenset({100, 101, 102}),
    )
    assert out.n == 0
    assert out.skipped_unknown_provenance == 3


def test_none_frame_id_rows_remain_eligible_without_forbidden_filter():
    atoms, posterior, _ = _atoms_with_indexed_variances(4)
    out = select_seeds(
        atoms,
        posterior,
        n_seeds=2,
        rng_seed=0,
        training_frame_ids=[None, None, 10, 11],
        forbidden_frame_ids=frozenset(),
    )
    assert out.n == 2
    assert out.skipped_unknown_provenance == 0


def test_mismatched_frame_ids_length_raises():
    atoms, posterior, _ = _atoms_with_indexed_variances(5)
    with pytest.raises(ValueError, match="length"):
        select_seeds(
            atoms, posterior, n_seeds=2,
            training_frame_ids=[1, 2, 3],  # too short
        )


def test_pick_skips_forbidden_when_n_seeds_exceeds_eligible():
    """When n_seeds >= n_eligible (after filtering), every eligible row is
    returned; the bulk_indices contain them all and variance_indices is
    empty (matches the legacy `n_seeds >= n` branch)."""
    atoms, posterior, _ = _atoms_with_indexed_variances(6)
    fids = list(range(10, 16))
    out = select_seeds(
        atoms, posterior, n_seeds=10, rng_seed=0,
        training_frame_ids=fids,
        forbidden_frame_ids=frozenset({10, 11}),  # 4 eligible rows remain
    )
    assert out.n == 4
    assert out.variance_indices == []
    assert set(out.frame_ids) == {12, 13, 14, 15}


def test_variance_chunk_size_passed_to_batched_posterior():
    atoms = [object() for _ in range(8)]
    posterior = _ChunkedPosterior({id(a): i for i, a in enumerate(atoms)})
    out = select_seeds(
        atoms,
        posterior,
        n_seeds=4,
        bulk_fraction=0.0,
        variance_chunk_size=3,
    )
    assert out.variance_indices == [7, 6, 5, 4]
    assert posterior.chunk_sizes == [3, 3]


def test_variance_chunk_size_is_backward_compatible_with_old_batched_posterior():
    atoms, _posterior, _ = _atoms_with_indexed_variances(6)
    posterior = _BatchedPosterior({id(a): i for i, a in enumerate(atoms)})
    out = select_seeds(
        atoms,
        posterior,
        n_seeds=3,
        bulk_fraction=0.0,
        variance_chunk_size=2,
    )
    assert out.variance_indices == [5, 4, 3]


def test_d_optimal_avoids_redundant_high_variance_candidate():
    atoms = [object() for _ in range(4)]
    cov = np.diag([10.0, 9.0, 8.0, 1.0])
    cov[0, 1] = cov[1, 0] = np.sqrt(10.0 * 9.0) * 0.999
    posterior = _CovariancePosterior(atoms, cov)

    out = select_seeds(
        atoms,
        posterior,
        n_seeds=2,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=4,
    )

    assert out.indices == [0, 2]
    assert out.selection_origins == ["d_optimal", "d_optimal"]
    assert out.variance_indices == [0, 2]
    assert out.selection_diagnostics[1]["d_optimal_conditional_variance"] == pytest.approx(8.0)
    assert posterior.covariance_matrix_called is False


def test_d_optimal_is_deterministic_and_reports_gain_diagnostics():
    atoms = [object() for _ in range(5)]
    posterior = _CovariancePosterior(atoms, np.diag([5.0, 4.0, 3.0, 2.0, 1.0]))

    a = select_seeds(
        atoms,
        posterior,
        n_seeds=3,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=2,
    )
    b = select_seeds(
        atoms,
        posterior,
        n_seeds=3,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=2,
    )

    assert a.indices == b.indices == [0, 1, 2]
    assert len(set(a.indices)) == len(a.indices)
    assert all(row["d_optimal_gain"] >= 0.0 for row in a.selection_diagnostics)
    assert a.diagnostics["prefilter_pool_size"] == 5


def test_d_optimal_ties_prefer_lowest_candidate_index():
    atoms = [object() for _ in range(3)]
    posterior = _CovariancePosterior(atoms, np.eye(3))

    out = select_seeds(
        atoms,
        posterior,
        n_seeds=2,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=3,
        d_optimal_score_power=0.0,
    )

    assert out.indices == [0, 1]
    assert out.selection_origins == ["d_optimal", "d_optimal"]


def test_d_optimal_prefilter_near_ties_are_broken_by_index():
    atoms = [object() for _ in range(4)]
    cov = np.diag([1.0, 2.0 + 3.0e-13, 2.0 + 4.0e-13, 0.5])
    posterior = _CovariancePosterior(atoms, cov)

    out = select_seeds(
        atoms,
        posterior,
        n_seeds=2,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=4,
    )

    assert out.indices[:2] == [1, 2]
    assert out.diagnostics["ranking_tie_break_policy"] == "quantised_score_then_index"


def test_d_optimal_skips_degenerate_zero_pivot_candidate():
    atoms = [object() for _ in range(3)]
    posterior = _CovariancePosterior(atoms, np.diag([2.0, 1.0, 0.0]))

    out = select_seeds(
        atoms,
        posterior,
        n_seeds=2,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=3,
        d_optimal_score_power=0.0,
    )

    assert out.indices == [0, 1]
    assert 2 not in out.indices
    assert out.diagnostics["d_optimal_skipped_degenerate"] == 1


def test_d_optimal_can_rank_by_transformed_score():
    atoms = [object() for _ in range(4)]
    posterior = _CovariancePosterior(atoms, np.diag([10.0, 9.0, 8.0, 1.0]))

    out = select_seeds(
        atoms,
        posterior,
        n_seeds=1,
        bulk_fraction=0.0,
        strategy="d_optimal",
        score_transform=lambda idx, var: 100.0 if idx == 2 else var,
    )

    assert out.indices == [2]
    assert out.selection_diagnostics[0]["raw_score"] == pytest.approx(100.0)


def test_d_optimal_respects_bulk_context():
    atoms = [object() for _ in range(5)]
    cov = np.diag([10.0, 9.0, 8.0, 7.0, 6.0])
    cov[0, 1] = cov[1, 0] = np.sqrt(10.0 * 9.0) * 0.999
    posterior = _CovariancePosterior(atoms, cov)

    out = select_seeds(
        atoms,
        posterior,
        n_seeds=2,
        bulk_fraction=0.5,
        rng_seed=12,
        strategy="d_optimal",
        d_optimal_pool_multiplier=4,
    )

    assert len(out.bulk_indices) == 1
    assert len(out.variance_indices) == 1
    assert out.selection_origins == ["bulk", "d_optimal"]
    assert out.bulk_indices[0] not in out.variance_indices


def test_d_optimal_requires_covariance_contract():
    atoms, posterior, _ = _atoms_with_indexed_variances(6)
    with pytest.raises(ValueError, match="requires posterior.covariance"):
        select_seeds(
            atoms,
            posterior,
            n_seeds=3,
            bulk_fraction=0.0,
            strategy="d_optimal",
        )


def test_d_optimal_score_backfill_is_deterministic_for_rank_one_pool():
    atoms = [object() for _ in range(4)]
    posterior = _CovariancePosterior(atoms, np.ones((4, 4), dtype=float))

    output = select_seeds(
        atoms,
        posterior,
        n_seeds=3,
        bulk_fraction=0.0,
        strategy="d_optimal",
        d_optimal_pool_multiplier=4,
        d_optimal_score_power=0.0,
        d_optimal_degenerate_policy="score_backfill",
    )

    assert output.indices == [0, 1, 2]
    assert output.selection_origins == [
        "d_optimal",
        "d_optimal_backfill",
        "d_optimal_backfill",
    ]
    assert output.diagnostics["d_optimal_selected"] == 1
    assert output.diagnostics["d_optimal_backfilled"] == 2
    assert all(
        row.get("d_optimal_backfill_reason") == "model_space_degeneracy"
        for row in output.selection_diagnostics[1:]
    )
    assert all(
        row.get("d_optimal_backfill_rank_source") == "raw_posterior_variance"
        for row in output.selection_diagnostics[1:]
    )


def test_d_optimal_fail_policy_reports_model_space_degeneracy():
    atoms = [object() for _ in range(3)]
    posterior = _CovariancePosterior(atoms, np.ones((3, 3), dtype=float))

    with pytest.raises(ValueError, match="model-space degeneracy selected 1 of 2"):
        select_seeds(
            atoms,
            posterior,
            n_seeds=2,
            bulk_fraction=0.0,
            strategy="d_optimal",
            d_optimal_pool_multiplier=3,
            d_optimal_score_power=0.0,
            d_optimal_degenerate_policy="fail",
        )


def test_invalid_seed_selection_strategy_raises():
    atoms, posterior, _ = _atoms_with_indexed_variances(6)
    with pytest.raises(ValueError, match="strategy"):
        select_seeds(atoms, posterior, n_seeds=2, strategy="not_real")
