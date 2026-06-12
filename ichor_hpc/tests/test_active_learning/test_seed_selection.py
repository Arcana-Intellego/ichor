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


def test_bulk_fraction_one_pure_random():
    atoms, posterior, _ = _atoms_with_indexed_variances(20)
    out = select_seeds(atoms, posterior, n_seeds=5, bulk_fraction=1.0, rng_seed=0)
    assert len(out.variance_indices) == 0
    assert len(out.bulk_indices) == 5


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
        select_seeds(atoms, posterior, n_seeds=4)


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
